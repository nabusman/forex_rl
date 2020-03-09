import torch

import fx_model

n_features = 4
n_samples = 100

input_data = torch.randn([1, n_features, n_samples])
actions = ['long', 'neutral', 'short']

model = fx_model.Agent(n_features, n_samples, actions)
model(input_data)

model = fx_model.Agent(n_features, n_samples, actions, 
	dense_params = [{'out_features' : 64}, {'out_features' : 32}])
model(input_data)

model = fx_model.Agent(n_features, n_samples, actions, 
	dense_params = [{'out_features' : 64}, {'out_features' : 32, 'dropout' : 0.5}])
model(input_data)

model = fx_model.Agent(n_features, n_samples, actions, 
	dense_params = [{'out_features' : 64}, {'out_features' : 32}],
	conv_params = [{'out_channels' : 32, 'kernel_size' : 4}])
model(input_data)

model = fx_model.Agent(n_features, n_samples, actions, 
	dense_params = [{'out_features' : 64}, {'out_features' : 32, 'dropout' : 0.5}],
	conv_params = [{'out_channels' : 64, 'kernel_size' : 4}, {'out_channels' : 32, 'kernel_size' : 2}])
model(input_data)
