import argparse
import os
import yaml

import helpers


def main(config_path):
    config = helpers.get_config(config_path)
    i = 0
    for batch_size in [128,512]:
        config['batch_size'] = batch_size
        for max_steps in [5000,10000]:
            config['max_steps'] = max_steps
            for n_samples in [100,500]:
                config['n_samples'] = n_samples
                for neutral_cost in [-100,-500]:
                    config['neutral_cost'] = neutral_cost
                    for learning_rate in [0.001, 0.00001]:
                        config['learning_rate'] = learning_rate
                        for dense_params in [[{'out_features' : 2048}, {'out_features' : 2048}, {'out_features' : 2048}, {'out_features' : 2048}], [{'out_features' : 4096}, {'out_features' : 4096}, {'out_features' : 4096}, {'out_features' : 4096}]]:
                            config['dense_params'] = dense_params
                            with open(os.path.join(os.path.dirname(config_path), f'config_{i}.yaml'), 'w') as f:
                                yaml.dump(config, f, default_flow_style=False)
                            i += 1

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, 
        help = 'Full path to config file to use',
        default = '/home/nabs/Projects/forex_rl/config/config.yaml')
    args = parser.parse_args()
    main(**vars(args))
