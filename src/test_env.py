import random

import torch

import fx_env

n_features = 4
n_samples = 100
batch_size = 128
aggregation = '1 minute'
actions = ['long', 'neutral', 'short']
fx_pair = 'EURUSD'
spread = 0.0004
data_dir = '/Data/foreign_exchange_data/'


input_data = torch.randn([batch_size, n_features, n_samples])

env = fx_env.ForexEnv(aggregation, n_samples, actions, fx_pair, spread, data_dir)

obs = env.reset()

while True:
	action = random.choice(range(len(actions)))
	obs, reward, done, info = env.step(action)
	print(f'Reward: {reward}\nInfo: {info}')
	if done: break