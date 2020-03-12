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
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, aggregation, n_samples, actions, fx_pair, pip_size, 
        data_dir, max_slippage = 0.02, tech_indicators = {}, stop_loss = 50):
        super(ForexEnv, self).__init__()
        self.n_samples = n_samples
        self.stop_loss = stop_loss
        self.max_slippage = max_slippage
        self.actions = actions
        self.pip_size = pip_size
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
            int(aggregation.split(' ')[0]), tech_indicators, data_dir)
        self.data = np.load(file_path, mmap_mode = 'r')
        self.tick_data = np.load(
            os.path.join(data_dir, 'npy', f'{fx_pair.upper()}_tick_level.npy'), 
            mmap_mode = 'r')
        self.action_space = spaces.Discrete(n = len(self.actions))
        # Note: "4" is for OHLC
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, 
            shape = (4 + indicators.get_return_cols(tech_indicators.keys()), self.n_samples), 
            dtype = np.float32)


    def reset(self):
        self.enter_price = None
        self.start_index = None
        self.position = None
        self.dollars_per_pip = 100 # Change this to allow more failures
        self.starting_balance = 100000
        self.account_balance = self.starting_balance
        self.max_balance = self.account_balance * 100
        self.obs, self.end_time = self._get_observation()
        return self.obs


    def step(self, action):
        reward = 0
        self.position = self.actions[action]
        self.start_index = np.searchsorted(self.tick_data[:,0], self.end_time)
        if self.position == 'long':
            self.enter_price = self.tick_data[self.start_index,2] \
                * ((np.random.random() - 0.5) * self.max_slippage)
            self.exit_price = self._calc_exit_price() * ((np.random.random() - 0.5) * self.max_slippage)
            reward = (self.exit_price - self.enter_price) / self.pip_size * self.dollars_per_pip
        elif self.position == 'short':
            self.enter_price = self.tick_data[self.start_index,1] \
                * ((np.random.random() - 0.5) * self.max_slippage)
            self.exit_price = self._calc_exit_price() * ((np.random.random() - 0.5) * self.max_slippage)
            reward = (self.enter_price - self.exit_price) / self.pip_size * self.dollars_per_pip
        self.account_balance += reward
        done = True if self.account_balance < 0 or self.account_balance > self.max_balance else False
        if done:
            self.obs = None
        else:
            self.obs, self.end_time = self._get_observation()
        info = {'account_balance' : self.account_balance, 
            'starting_balance' : self.starting_balance}
        return self.obs, reward, done, info


    def _get_observation(self):
        # Get random index number between 0 and len of data minus n_samples
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

    def _calc_exit_price(self):
        if self.position == 'long':
            max_price = self.enter_price
        elif self.position == 'short':
            min_price = self.enter_price
        else:
            raise Exception(f'Unknown self.position {self.position}')
        stop_loss_amount = self.stop_loss * self.pip_size
        for i in range(self.start_index,self.tick_data.shape[0]):
            _, bid, ask = self.tick_data[i,:]
            if self.position == 'long':
                stop_loss = max_price - stop_loss_amount
                if max_price < bid:
                    max_price = bid
                    continue
                elif bid <= stop_loss or i == (self.tick_data.shape[0] - 1):
                    return bid
            elif self.position == 'short':
                stop_loss = min_price + stop_loss_amount
                if min_price > ask:
                    min_price = ask
                    continue
                elif ask >= stop_loss or i == (self.tick_data.shape[0] - 1):
                    return ask