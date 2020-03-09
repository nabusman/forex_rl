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

    def __init__(self, aggregation, n_samples, actions, fx_pair, spread, data_dir,
        max_slippage = 0.02, tech_indicators = {}, stop_loss = 50):
        super(ForexEnv, self).__init__()
        self.n_samples = n_samples
        self.stop_loss = stop_loss
        self.max_slippage = max_slippage
        self.actions = actions
        # self.spread = helpers.get_config()['pip'][fx_pair] * spread
        self.spread = spread
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
        file_path = self._get_fx_file_paths(fx_pair, agg_level, 
            int(aggregation.split(' ')[0]), tech_indicators, data_dir)
        self.data = np.load(file_path, mmap_mode = 'r')
        self.action_space = spaces.Discrete(n = len(self.actions))
        # Note: "4" is for OHLC
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, 
            shape = (4 + indicators.get_return_cols(tech_indicators.keys()), self.n_samples), 
            dtype = np.float64)


    def reset(self):
        self.position = None
        self.steps = 0
        self.enter_price = None
        self.trx_size = 100000 # Change this to allow more failures
        self.account_balance = 100000
        self.max_balance = self.account_balance * 100
        self.obs, self.obs_prices = self._get_observation()
        return self.obs


    def step(self, action):
        self.steps += 1
        reward = 0
        close_position = False
        if self.position is None and self.actions[action] != 'neutral':
            self.position = self.actions[action]
            self.enter_price = self.obs_prices[3,-1] * ((np.random.random() - 0.5) * self.max_slippage) # Close price + slippage
        elif self.position is not None and self.actions[action] == 'neutral':
            close_position = True
        elif self.position != self.actions[action]:
            close_position = True

        if close_position:
            if self.position == 'long':
                reward = (self.obs_prices[3,-1] * ((np.random.random() - 0.5) * self.max_slippage)) - \
                    (self.enter_price + self.spread)
            elif self.position == 'short':
                reward = (self.enter_price - self.spread) - \
                    (self.obs_prices[3,-1] * ((np.random.random() - 0.5) * self.max_slippage))
            self.enter_price = None
            self.position = None
        self.account_balance += self.trx_size * reward
        done = True if self.account_balance < 0 or self.account_balance > self.max_balance else False
        self.obs, self.obs_prices = self._get_observation()
        if done:
            self.obs = None
        return self.obs, reward, done, {'account_balance' : self.account_balance, 'steps' : self.steps}


    def _get_observation(self):
        # Get random index number between 0 and len of data minus n_samples
        _index = np.random.randint(0, self.data.shape[0] - self.n_samples)
        obs = self.data[_index : _index+self.n_samples,:].T.copy()
        real_prices = obs.copy()
        # normalize data
        obs = normalize(obs.T).T
        return torch.from_numpy(obs), torch.from_numpy(real_prices) # Size of both: (n_features, n_samples)


    def _get_fx_file_paths(self, fx_pair, agg_level, agg_magnitude, tech_indicators, data_dir,
        train_end = '2019-02-01'):
        input_path = os.path.join(data_dir, 'csv')
        output_path = os.path.join(data_dir, 'npy')
        train_end = np.datetime64(train_end)
        phases = ['train', 'test']
        file_paths = {phase : 
            os.path.join(output_path, 
                f"""{fx_pair.upper()}_{str(agg_magnitude)}_{agg_level}_{phase}_{'_'.join([str(y) 
                    for x in list(tech_indicators.items()) 
                    for y in x]).replace(', ', '_')}.npy""") 
            for phase in phases}
        # Return if data exists
        if os.path.exists(file_paths['train']):
            return file_paths['train']
        # Otherwise create it
        print(f'Creating data {file_paths}')
        file_list = [x for x in os.listdir(input_path) if fx_pair in x]
        fx_npy = {x : None for x in phases}
        for file_name in tqdm(file_list, total = len(file_list)):
            # Read file and setup
            df = pd.read_csv(os.path.join(input_path, file_name), header = None, names = ['fx_pair', 'timestamp', 'bid', 'ask'])
            df = df.drop('fx_pair', axis = 1)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%Y%m%d %H:%M:%S.%f')
            df = df.dropna()
            df = df[['timestamp', 'bid', 'ask']]
            df['price'] = df[['bid', 'ask']].mean(axis = 1)
            df = df.drop(['bid', 'ask'], axis = 1)
            # Set phase
            if df['timestamp'].max() < train_end:
                phase = 'train'
            else:
                phase = 'test'
            # Aggregate data at level and magnitude
            df = df.set_index('timestamp')
            df = df.resample(f'{agg_magnitude}{agg_level}').ohlc()['price'].reset_index(drop = True)
            # Calc technical indicators
            for indicator,params in tech_indicators.items():
                if indicator == 'macd':
                    df = indicators.macd(df, *params)
                elif indicator == 'rsi':
                    df = indicators.rsi(df, *params)
                else:
                    raise Exception(f'Unrecognized technical indicator {indicator}')
            # Add to phase numpy array
            df_np = df.dropna().to_numpy()
            assert np.sum(np.isnan(df_np)) == 0            
            if fx_npy[phase] is None:
                fx_npy[phase] = df_np
            else:
                fx_npy[phase] = np.vstack([fx_npy[phase], df_np])
        
        for phase in fx_npy.keys():
            np.save(file_paths[phase], fx_npy[phase])
        return file_paths['train']


    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    # def trailing_stop_loss(self, pip_size = 0.0001):
    #     """
    #     data [np.array]: tick level future data, after the transaction (time, bid, ask)
    #     enter_price [float]: transaction enter price, minus spread
    #     self.position [str]: one of 'long' or 'short'
    #     self.stop_loss [int]: number of pips for the stop loss
    #     pip_size [float]: the size of the pip for given currency
    #     """
    #     if self.position == 'long':
    #         max_price = self.enter_price
    #     elif self.position == 'short':
    #         min_price = self.enter_price
    #     else:
    #         raise Exception(f'Unknown self.position {self.position}')
    #     for time,bid,ask in self.tick_data:
    #         if self.position == 'long':
    #             stop_loss = (max_price - (self.stop_loss * pip_size))
    #             if max_price < bid:
    #                 max_price = bid
    #                 continue
    #             elif bid <= stop_loss:
    #                 print(f'Closing position: enter_price: {enter_price}, stop_loss {stop_loss}, exit price: {bid}, profit/loss: {(bid - enter_price) / pip_size} pips')
    #                 return bid
    #         elif self.position == 'short':
    #             stop_loss = (min_price + (self.stop_loss * pip_size))
    #             if min_price > ask:
    #                 min_price = ask
    #                 continue
    #             elif ask >= stop_loss:
    #                 print(f'Closing position: enter_price: {enter_price}, stop_loss {stop_loss}, exit price: {ask}, profit/loss: {(enter_price - ask) / pip_size} pips')
    #                 return ask