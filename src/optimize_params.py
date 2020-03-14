import os
from datetime import datetime

import torch
from ax.service.managed_loop import optimize
from torch.utils.tensorboard import SummaryWriter

from main import train, evaluate
import helpers

# General setup
config_path = '/home/nabs/Projects/forex_rl/config/config.yaml'
model_dir = '/home/nabs/Projects/forex_rl/models'
data_dir = '/Data/foreign_exchange_data/'
log_dir = '/home/nabs/Projects/forex_rl/runs'
now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]
writer = SummaryWriter(os.path.join(log_dir, now_str))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forex_eval(params):
    config = helpers.get_config(config_path)
    if 'n_dense_layers' in params and 'n_nodes_dense_layers' in params:
        config['dense_params'] = [{'out_features' : params['n_nodes_dense_layers']} 
            for _ in range(params['n_dense_layers'])]
    for param,value in params.items():
        if param in config: config[param] = value
    metrics = evaluate(train(device, writer, config, data_dir), 
        device, writer, config, data_dir, config_path,)
    return metrics

best_parameters, values, experiment, model = optimize(
    parameters = [
        {
            'name' : 'batch_size',
            'type' : 'range',
            'bounds' : [16,512],
        },
        {
            'name' : 'max_steps',
            'type' : 'range',
            'bounds' : [5000,25000],
        },
        {
            'name' : 'n_samples',
            'type' : 'range',
            'bounds' : [100,500],
        },
        {
            'name' : 'neutral_cost',
            'type' : 'range',
            'bounds' : [-1000,0],
        },
        {
            'name' : 'learning_rate',
            'type' : 'range',
            'bounds' : [0.000001,0.01],
        },
        {
            'name' : 'n_dense_layers',
            'type' : 'range',
            'bounds' : [1,10],
        },
        {
            'name' : 'n_nodes_dense_layers',
            'type' : 'range',
            'bounds' : [512,4048],
        },
    ],
    evaluation_function = forex_eval,
    objective_name = 'sortino',
    total_trials = 100,
)
