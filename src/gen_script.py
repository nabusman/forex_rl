import argparse
import os
from random import shuffle


def main(config_dir, scripts_dir, main_path):
    configs = os.listdir(config_dir)
    shuffle(configs)
    with open(os.path.join(scripts_dir, 'train.sh'), 'w') as f:
        for config_file in configs:
            f.write(f'python {main_path} --config_path {os.path.join(config_dir, config_file)}\n')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type = str, 
        help = 'Full path to config file to use',
        default = '/home/nabs/Projects/forex_rl/config/')
    parser.add_argument('--scripts_dir', type = str, 
        help = 'Full path to config file to use',
        default = '/home/nabs/Projects/forex_rl/scripts/')
    parser.add_argument('--main_path', type = str, 
        help = 'Full path to config file to use',
        default = '/home/nabs/Projects/forex_rl/src/main.py')
    args = parser.parse_args()
    main(**vars(args))