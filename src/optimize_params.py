import os
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
            'kernel_size' : params['conv_kernel_size']} for _ in range(params['n_conv_layers'])]
    for param,value in params.items():
        if param in config: config[param] = value
    model = train(device, writer, config, data_dir)
    metrics = evaluate(model, device, writer, config, data_dir, config_path)
    save(model, device, metrics, now_str, config, model_dir, config_path)
    return {k : (v, 0.0) for k,v in metrics.items()}

best_parameters, values, experiment, model = optimize(
    parameters = [
        {
            'name' : 'learning_rate',
            'type' : 'range',
            'bounds' : [0.000001,0.01],
        },
        {
            'name' : 'n_conv_layers',
            'type' : 'range',
            'bounds' : [0,6],
        },
        {
            'name' : 'conv_filter_size',
            'type' : 'range',
            'bounds' : [10,500],
        },
        {
            'name' : 'conv_kernel_size',
            'type' : 'range',
            'bounds' : [1,10],
        },
        {
            'name' : 'n_dense_layers',
            'type' : 'range',
            'bounds' : [8,15],
        },
        {
            'name' : 'n_nodes_dense_layers',
            'type' : 'range',
            'bounds' : [2000,5000],
        },
    ],
    evaluation_function = forex_eval,
    objective_name = 'sortino',
    total_trials = 20,
)
