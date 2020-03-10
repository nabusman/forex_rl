import torch
import random

import fx_env
import fx_model

n_features = 4
n_samples = 100
batch_size = 128
aggregation = '1 minute'
actions = ['long', 'neutral', 'short']
fx_pair = 'AUDJPY'
pip_size = 0.01
data_dir = '/Data/foreign_exchange_data/'


input_data = torch.randn([batch_size, n_features, n_samples])

env = fx_env.ForexEnv(aggregation, n_samples, actions, fx_pair, pip_size, data_dir)
memory = fx_model.ReplayMemory()
state = env.reset()

while True:
	action = random.choice(range(len(actions)))
	next_state, reward, done, info = env.step(action)
	if state is None:
		raise Exception('state is None')
	memory.add(state, action, next_state, reward)
	if done:
		print('Done! Resetting...')
		state = env.reset()
	else:
		state = next_state
	if len(memory) == batch_size:
		break
