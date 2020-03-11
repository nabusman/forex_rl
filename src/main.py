import os
import math
import random
import argparse

import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import fx_env
import fx_model
import helpers

def select_action(config, state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = config['eps_end'] + (config['eps_start'] - config['eps_end']) * \
        math.exp(-1 * steps_done / config['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            action = torch.argmax(policy_net(state)).item()
    else:
        action = random.choice(range(len(config['actions'])))
    return action, steps_done

def calc_loss(policy_net, target_net, device, transitions, gamma):
    non_final_mask = [True if x['next_state'] is not None else False 
        for x in transitions]
    non_final_next_states = torch.cat([torch.unsqueeze(x['next_state'], 0) 
        for x in transitions if x['next_state'] is not None])
    states_batch = torch.cat([torch.unsqueeze(x['state'], 0) for x in transitions])
    rewards_batch = torch.tensor([x['reward'] for x in transitions])
    actions_batch = torch.tensor([x['action'] for x in transitions])
    state_action_values = policy_net(states_batch).gather(1, actions_batch.view(-1,1))
    next_state_values = torch.zeros(len(transitions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + rewards_batch
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    return loss


def calc_metrics(transitions, info, risk_free_rate = 0.05):
    rewards = np.array([x['reward'] for x in transitions if rewards is not None])
    roi = (info['account_balance'] - info['starting_balance']) / info['starting_balance']
    mean_reward = np.mean(rewards)
    std = np.std(rewards)
    median_reward = np.median(rewards)
    sharpe = (roi - risk_free_rate) / np.std(rewards)
    downside_deviation = np.sqrt(np.sum(rewards[np.argwhere(rewards < 0)] ** 2)) / len(rewards)
    sortino = (roi - risk_free_rate) / downside_deviation
    metrics = {
        'sharpe' : sharpe, 
        'sortino' : sortino, 
        'mean' : mean_reward, 
        'median' : median_reward,
        'downside_deviation' : downside_deviation,
        'standard_deviation' : std,
    }
    return metrics


def main(device, writer, data_dir, model_dir, config_path):
    # Setup
    config = helpers.get_config(config_path)
    env = fx_env.ForexEnv(
        aggregation = config['aggregation'], 
        n_samples = config['n_samples'],
        actions = config['actions'],
        fx_pair = config['fx_pair'],
        pip_size = config['pip_size'][config['fx_pair']],
        data_dir = data_dir,
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
    steps_done = 0

    # Training Loop
    # Pick action
    # Get reward
    # add to memory
    # if memory is not big enough continue
    # Calc loss
    # optimizer step
    # Sync with target if X trades have happened
    # Calc metrics, if above a level quit training




if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, 
        help = 'Full path to config file to use')
    parser.add_argument('--model_dir', type = str, 
        help = 'Full path to the model directory to use',
        default = '/home/nabs/Projects/trading_fx_rl/models')
    parser.add_argument('--data_dir', type = str, 
        help = 'Full path to the data directory to use',
        default = '/Data/foreign_exchange_data/')
    args = parser.parse_args()
    # General setup
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Main
    main(device, writer, **vars(args))
