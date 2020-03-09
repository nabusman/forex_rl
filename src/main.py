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


def optimize_model(policy_net, memory, batch_size):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    state_batch = torch.cat([torch.unsqueeze(x['state'], 0) for x in transitions])



def main(device, writer, data_dir, model_dir, config_path):
    # Setup
    config = helpers.get_config(config_path)
    env = fx_env.ForexEnv(
        aggregation = config['aggregation'], 
        n_samples = config['n_samples'],
        actions = config['actions'],
        fx_pair = config['fx_pair'],
        spread = config['pip_size'][config['fx_pair']] * config['spread'],
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
