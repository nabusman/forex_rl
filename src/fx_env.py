import os

import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from tqdm import tqdm
from sklearn.preprocessing import normalize

import indicators
import helpers


class ForexEnv(gym.Env):
    """
    OpenAI Gym environment for forex
    Params:
    - aggregation [string]: aggregation level of data to be returned
    two part, first is a number second is a time aggregation e.g. 5 minute
    - n_samples [int]: how many number of samples of the aggregation to include in
    the observation provided
    - actions [list]: list of actions ['long', 'neutral', 'short']
    - tech_indicators [dict]: provides the details of the technical indicators
    to be included in the observations e.g. {'macd' : [12,26], 'rsi' : [14]}
    - stop_loss [int]: the trailing stop loss to set for this environment
    - is_test [bool]: flag to enable test data
    - neutral_cost [int]: the cost of picking the neutral action, decreases this
    to force more aggressive trading behavior
    - max_slippage [int]: the maximum number of pips of slippage in exiting a 
    position, this will be multiplied by a random number so slippage is between 
    [0,max_slippage] pips with a uniform distribution.
    - dollars_per_pip [int]: how many dollars per pip the trade is, this is normally
    dependent on the amount of money invested but here we keep it constant. Increase
    this value to create a more risky agent.
    - enable_mmap [bool]: If mmap_mode is enabled for the numpy data loads, turn this
    on if you are getting system memory OOM errors.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, aggregation, n_samples, actions, fx_pair, pip_size, 
        data_dir, max_slippage = 0, tech_indicators = {}, stop_loss = 50, 
        is_test = False, neutral_cost = 0, dollars_per_pip = 10, enable_mmap = False):
        super(ForexEnv, self).__init__()
        self.n_samples = n_samples
        self.stop_loss = stop_loss
        self.max_slippage = max_slippage
        self.actions = actions
        self.pip_size = pip_size
        self.is_test = is_test
        self.neutral_cost = neutral_cost
        self.dollars_per_pip = dollars_per_pip
        # Parse aggregation
        if 'min' in aggregation.lower():
            agg_level = 'Min'
        elif 'hour' in aggregation.lower():
            agg_level = 'H'
        elif 'day' in aggregation.lower():
            agg_level = 'D'
        else:
            raise Exception(f'Unrecognized aggregation level {aggregation}')
        # Load Data
        file_path = helpers.get_fx_file_paths(fx_pair, agg_level, 
            int(aggregation.split(' ')[0]), tech_indicators, data_dir, is_test)
        if enable_mmap:
            self.data = np.load(file_path, mmap_mode = 'r')
            self.tick_data = np.load(
                os.path.join(data_dir, 'npy', f'{fx_pair.upper()}_tick_level.npy'), 
                mmap_mode = 'r')
        else:
            self.data = np.load(file_path)
            self.tick_data = np.load(
                os.path.join(data_dir, 'npy', f'{fx_pair.upper()}_tick_level.npy'))
        self.action_space = spaces.Discrete(n = len(self.actions))
        # Note: "4" is for OHLC
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, 
            shape = (4 + indicators.get_return_cols(tech_indicators.keys()), self.n_samples), 
            dtype = np.float32)


    def reset(self):
        self.enter_price = None
        self.start_index = None
        self.position = None
        self.starting_balance = 100000
        self.max_reward = 10000 # Scale rewards by this number
        self.account_balance = self.starting_balance
        self.max_balance = self.account_balance * 100
        self.obs, self.end_time = self._get_observation()
        return self.obs


    def step(self, action):
        reward = 0 if self.is_test else self.neutral_cost
        net_pips = None
        self.position = self.actions[action]
        self.start_index = np.searchsorted(self.tick_data[:,0], self.end_time)
        if self.position == 'long':
            self.enter_price = self.tick_data[self.start_index,2]
            raw_exit_price = helpers.calc_exit_price(
                tick_data = self.tick_data, 
                start_index = self.start_index, 
                position = self.position, 
                stop_loss = self.stop_loss, 
                pip_size = self.pip_size, 
                enter_price = self.enter_price,
            )
            self.exit_price = raw_exit_price - (self.pip_size * self.max_slippage * np.random.random())
            net_pips = (self.exit_price - self.enter_price) / self.pip_size
        elif self.position == 'short':
            self.enter_price = self.tick_data[self.start_index,1]
            raw_exit_price = helpers.calc_exit_price(
                tick_data = self.tick_data, 
                start_index = self.start_index, 
                position = self.position, 
                stop_loss = self.stop_loss, 
                pip_size = self.pip_size, 
                enter_price = self.enter_price,
            )
            self.exit_price = raw_exit_price + (self.pip_size * self.max_slippage * np.random.random())
            net_pips = (self.enter_price - self.exit_price) / self.pip_size
        reward = net_pips * self.dollars_per_pip if net_pips is not None else reward
        scaled_reward = reward / self.max_reward
        scaled_reward = 1.0 if scaled_reward > 1 else scaled_reward
        scaled_reward = -1.0 if scaled_reward < -1 else scaled_reward
        self.account_balance += reward
        done = True if self.account_balance < 0 or self.account_balance > self.max_balance else False
        if done:
            self.obs = None
        else:
            self.obs, self.end_time = self._get_observation()
        info = {'account_balance' : self.account_balance, 
            'starting_balance' : self.starting_balance}
        return self.obs, scaled_reward, reward, done, info


    def _get_observation(self):
        # Get random index number between 0 and len of data minus 2X n_samples
        i = np.random.randint(0, self.data.shape[0] - (2 * self.n_samples))
        obs = self.data[i : i + self.n_samples, 1:].T.copy()
        end_time = self.data[i + self.n_samples, 0]
        # normalize data
        obs = normalize(obs.T).T # Shape: (n_features, n_samples)
        return torch.from_numpy(obs).float(), end_time


    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]