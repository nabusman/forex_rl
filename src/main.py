import os
import math
import random
import argparse
import pickle
from datetime import datetime

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import fx_env
import fx_model
import helpers


def select_action(config, state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = config['eps_end'] + (config['eps_start'] - config['eps_end']) * \
        math.exp(-1 * steps_done / config['eps_decay'])
    if sample > eps_threshold:
        with torch.no_grad():
            action = torch.argmax(policy_net(torch.unsqueeze(state,0))).item()
    else:
        action = random.choice(range(len(config['actions'])))
    return action


def calc_loss(policy_net, target_net, transitions, gamma, device):
    non_final_mask = [True if x['next_state'] is not None else False 
        for x in transitions]
    non_final_next_states = torch.cat([torch.unsqueeze(x['next_state'], 0) 
        for x in transitions if x['next_state'] is not None]).to(device)
    states_batch = torch.cat([torch.unsqueeze(x['state'], 0) for x in transitions]).to(device)
    rewards_batch = torch.tensor([x['reward'] for x in transitions]).to(device)
    actions_batch = torch.tensor([x['action'] for x in transitions]).to(device)
    state_action_values = torch.squeeze(
        policy_net(states_batch).gather(1, actions_batch.view(-1,1)).to(device))
    next_state_values = torch.zeros(len(transitions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + rewards_batch
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    return loss


def calc_metrics(rewards, info, risk_free_rate = 0.05):
    roi = (info['account_balance'] - info['starting_balance']) / info['starting_balance']
    mean_reward = np.mean(rewards)
    std = np.std(rewards)
    median_reward = np.median(rewards)
    mean_returns = np.mean(rewards / info['starting_balance'] - risk_free_rate)
    sharpe = mean_returns / std if std != 0 else 0
    downside_deviation = np.sqrt(np.sum(rewards[np.argwhere(rewards < 0)] ** 2)) \
        / len(rewards) if len(rewards) != 0 else 0
    sortino = (roi - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    metrics = {
        'sharpe' : sharpe, 
        'sortino' : sortino, 
        'mean' : mean_reward, 
        'median' : median_reward,
        'downside_deviation' : downside_deviation,
        'standard_deviation' : std,
        'roi' : roi,
        'account_balance' : info['account_balance'],
    }
    return metrics


def test(model, device, writer, data_dir, config_path, **args):
    config = helpers.get_config(config_path)
    env = fx_env.ForexEnv(
        aggregation = config['aggregation'], 
        n_samples = config['n_samples'],
        actions = config['actions'],
        fx_pair = config['fx_pair'],
        pip_size = config['pip_size'][config['fx_pair']],
        data_dir = data_dir,
        tech_indicators = config['tech_indicators'],
        is_test = True,
    )
    rewards = []
    steps = 0
    state = env.reset()
    while True:
        with torch.no_grad():
            action = torch.argmax(model(torch.unsqueeze(state,0))).item()
        next_state, reward, done, info = env.step(action)
        if done or steps == 100:
            break
        steps += 1
        rewards.append(reward)
        writer.add_scalar('test_reward', reward, steps)
    metrics = calc_metrics(np.array(rewards), info)
    return metrics


def save(model, metrics, now_str, model_dir, config_path, **args):
    # Save Model
    file_name = f"{now_str}_model_{os.path.basename(config_path).split('.')[0]}.pt"
    torch.save(model, os.path.join(model_dir, file_name))
    # Save metrics
    file_name = f"{now_str}_metrics_{os.path.basename(config_path).split('.')[0]}.pkl"
    with open(os.path.join(model_dir, file_name), 'wb') as f:
        pickle.dump(metrics, f, protocol = pickle.HIGHEST_PROTOCOL)


def main(device, writer, data_dir, config_path, **args):
    # Setup
    config = helpers.get_config(config_path)
    env = fx_env.ForexEnv(
        aggregation = config['aggregation'], 
        n_samples = config['n_samples'],
        actions = config['actions'],
        fx_pair = config['fx_pair'],
        pip_size = config['pip_size'][config['fx_pair']],
        data_dir = data_dir,
        tech_indicators = config['tech_indicators'],
        neutral_cost = config['neutral_cost']
    )
    policy_net = fx_model.Agent(
        n_features = env.observation_space.shape[0],
        n_samples = env.observation_space.shape[1],
        actions = env.actions,
        dense_params = config['dense_params'],        
        conv_params = config['conv_params'],        
    ).to(device)
    target_net = fx_model.Agent(
        n_features = env.observation_space.shape[0],
        n_samples = env.observation_space.shape[1],
        actions = env.actions,
        dense_params = config['dense_params'],        
        conv_params = config['conv_params'],        
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.AdamW(policy_net.parameters(), lr = config['learning_rate'])
    memory = fx_model.ReplayMemory(config['memory'])
    steps = 0
    state = env.reset()

    print('Starting training...')
    # Training Loop
    while True:    
        # Pick action
        action = select_action(config, state, policy_net, steps)
        # Get reward
        next_state, reward, done, info = env.step(action)
        writer.add_scalar('reward', reward, steps)
        # add to memory
        memory.add(state, action, next_state, reward)
        if done:
            state = env.reset()
        # if memory is not big enough continue
        if len(memory) < config['start_memory']:
            continue
        transitions = memory.sample(config['batch_size'])
        # Calculate loss
        loss = calc_loss(policy_net, target_net, transitions, config['gamma'], device)
        writer.add_scalar('loss', loss.item(), steps)
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calc metrics, if above a level quit training
        rewards = np.array([x['reward'] for x in transitions if x['reward'] is not None])
        metrics = calc_metrics(rewards, info)
        for metric,value in metrics.items():
            writer.add_scalar(metric, metrics[metric], steps)
        print(f'Step number {steps} - metrics: {metrics}')
        # Sync with target if X trades have happened
        if steps % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        steps += 1
        to_break = False
        if steps == config['max_steps']:
            to_break = True
        elif metrics[config['stopping_metric']['type']] >= config['stopping_metric']['threshold'] \
            and metrics[config['stopping_metric']['type']] is not np.inf:
            to_break = True
        if to_break:
            print(f'Solved in {steps} steps with metrics: {metrics}')
            target_net.load_state_dict(policy_net.state_dict())
            break
    return target_net


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, 
        help = 'Full path to config file to use')
    parser.add_argument('--model_dir', type = str, 
        help = 'Full path to the model directory to use',
        default = '/home/nabs/Projects/forex_rl/models')
    parser.add_argument('--data_dir', type = str, 
        help = 'Full path to the data directory to use',
        default = '/Data/foreign_exchange_data/')
    parser.add_argument('--log_dir', type = str, 
        help = 'Full path to the log directory to use',
        default = '/home/nabs/Projects/forex_rl/runs')
    args = parser.parse_args()
    # General setup
    now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]
    writer = SummaryWriter(os.path.join(args.log_dir, now_str))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Main
    model = main(device, writer, **vars(args))
    metrics = test(model, device, writer, **vars(args))
    save(model, metrics, now_str, **vars(args))
    writer.close()
