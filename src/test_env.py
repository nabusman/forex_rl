import os
import random

import torch

import fx_env

n_features = 4
n_samples = 100
batch_size = 16
aggregation = '1 minute'
actions = ['long', 'neutral', 'short']
fx_pair = 'AUDJPY'
pip_size = 0.01
dollars_per_pip = 10
stop_loss = 10
data_dir = '/Data/foreign_exchange_data/'
is_test = True
enable_mmap = True
max_slippage = 0

input_data = torch.randn([batch_size, n_features, n_samples])

env = fx_env.ForexEnv(
	aggregation = aggregation, 
	n_samples = n_samples, 
	actions = actions, 
	fx_pair = fx_pair, 
	pip_size = pip_size, 
	data_dir = data_dir, 
	is_test = is_test, 
	enable_mmap = enable_mmap,
	dollars_per_pip = dollars_per_pip,
	stop_loss = stop_loss,
	max_slippage = max_slippage,
)

obs = env.reset()

while True:
	action = random.choice(range(len(actions)))
	obs, scaled_reward, reward, done, info = env.step(action)
	if reward < -dollars_per_pip * stop_loss:
		raise Exception('WTF?')
	print(f'Reward: {reward}\nInfo: {info}')
	if done: break