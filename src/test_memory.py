import torch
import random

import fx_env
import fx_model

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
memory = fx_model.ReplayMemory()
model = fx_model.Agent(n_features, n_samples, actions)
state = env.reset()

while True:
	action = random.choice(range(len(actions)))
	next_state, reward, done, info = env.step(action)
	memory.add(state, action, next_state, reward)
	state = next_state
	if len(memory) == batch_size:
		break
