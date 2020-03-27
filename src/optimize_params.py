import os
import pickle
from datetime import datetime

import torch
from ax.service.managed_loop import optimize
from torch.utils.tensorboard import SummaryWriter

from main import train, evaluate, save
import helpers

# General setup
config_path = '/home/nabs/Projects/forex_rl/config/config.yaml'
model_dir = '/home/nabs/Projects/forex_rl/models'
data_dir = '/Data/foreign_exchange_data/'
log_dir = '/home/nabs/Projects/forex_rl/runs'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forex_eval(params):
    print(f'Using params: {params}')
    now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]
    writer = SummaryWriter(os.path.join(log_dir, now_str))
    config = helpers.get_config(config_path)
    if 'n_dense_layers' in params and 'n_nodes_dense_layers' in params:
        config['dense_params'] = [{'out_features' : params['n_nodes_dense_layers']} 
            for _ in range(params['n_dense_layers'])]
    if 'n_conv_layers' in params and 'conv_filter_size' in params and 'conv_kernel_size' in params:
        config['conv_params'] = [{'out_channels' : params['conv_filter_size'], 
            'kernel_size' : params['conv_kernel_size']} 
            for _ in range(params['n_conv_layers'])]
    for param,value in params.items():
        if param in config: config[param] = value
    model = train(device, writer, config, data_dir)
    metrics = evaluate(model, device, writer, config, data_dir, config_path)
    save(model, device, metrics, now_str, config, model_dir, config_path)
    return {k : (v, 0.0) for k,v in metrics.items() if v is not tuple}

best_parameters, values, experiment, model = optimize(
    parameters = [
        {
            'name' : 'n_conv_layers',
            'type' : 'range',
            'bounds' : [2,12],
        },
        {
            'name' : 'learning_rate',
            'type' : 'choice',
            'values' : [1 * 10 ** -x for x in list(range(3,7))], # [0.001, 0.0001, 1e-05, 1e-06]
            'is_ordered' : True,
        },
        {
            'name' : 'conv_filter_size',
            'type' : 'choice',
            'values' : [2 ** x for x in range(6,10)], # [64, 128, 256, 512]
            'is_ordered' : True,
        },
        {
            'name' : 'conv_kernel_size',
            'type' : 'choice',
            'values' : list(range(1,6)), # [1, 2, 3, 4, 5]
            'is_ordered' : True,
        },
        {
            'name' : 'neutral_cost',
            'type' : 'choice',
            'values' : [-2 ** x for x in range(8,0,-2)] + [0], # [-256, -64, -16, -4, 0]
            'is_ordered' : True,
        },
        {
            'name' : 'stop_loss',
            'type' : 'choice',
            'values' : list(range(30,200, 30)), # [30, 60, 90, 120, 150, 180]
            'is_ordered' : True,
        },
        {
            'name' : 'aggregation',
            'type' : 'choice',
            'values' : ['1 min', '5 min', '15 min', '30 min', '1 hour'],
            'is_ordered' : True,
        },
        {
            'name' : 'n_dense_layers',
            'type' : 'range',
            'bounds' : [2,12],
        },
        {
            'name' : 'n_nodes_dense_layers',
            'type' : 'choice',
            'values' : [2 ** x for x in range(9,13)], # [512, 1024, 2048, 4096]
            'is_ordered' : True,
        },
    ],
    evaluation_function = forex_eval,
    objective_name = 'sum',
    total_trials = 20,
    parameter_constraints=["n_dense_layers + n_conv_layers <= 16"],
)

print(f'Values are: {values}')
print(f'Best parameters are: {best_parameters}')

# Save results
now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]

with open(os.path.join(model_dir, f'{now_str}_best_parameters.pkl'), 'wb') as f:
    pickle.dump(best_parameters, f, protocol = pickle.HIGHEST_PROTOCOL)

with open(os.path.join(model_dir, f'{now_str}_values.pkl'), 'wb') as f:
    pickle.dump(values, f, protocol = pickle.HIGHEST_PROTOCOL)

with open(os.path.join(model_dir, f'{now_str}_experiment.pkl'), 'wb') as f:
    pickle.dump(experiment, f, protocol = pickle.HIGHEST_PROTOCOL)

with open(os.path.join(model_dir, f'{now_str}_model.pkl'), 'wb') as f:
    pickle.dump(model, f, protocol = pickle.HIGHEST_PROTOCOL)
