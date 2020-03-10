import os

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_config(config_path):
	with open(config_path) as f:
		config = yaml.load(f, Loader = yaml.FullLoader)
	return config


def get_fx_file_paths(fx_pair, agg_level, agg_magnitude, tech_indicators, data_dir,
    train_end = '2019-02-01'):
    input_path = os.path.join(data_dir, 'csv')
    output_path = os.path.join(data_dir, 'npy')
    train_end = np.datetime64(train_end)
    phases = ['train', 'test']
    file_paths = {phase : 
        os.path.join(output_path, 
            f"""{fx_pair.upper()}_{str(agg_magnitude)}_{agg_level}_{phase}_{'_'.join([str(y) 
                for x in list(tech_indicators.items()) 
                for y in x]).replace(', ', '_')}.npy""") 
        for phase in phases}
    # Return if data exists
    if os.path.exists(file_paths['train']):
        return file_paths['train']
    # Otherwise create it
    print(f'Creating data {file_paths}')
    file_list = [x for x in os.listdir(input_path) if fx_pair in x]
    fx_npy = {x : None for x in phases}
    for file_name in tqdm(file_list, total = len(file_list)):
        # Read file and setup
        df = pd.read_csv(os.path.join(input_path, file_name), header = None, 
            names = ['fx_pair', 'timestamp', 'bid', 'ask'])
        df = df.drop('fx_pair', axis = 1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], 
            format = '%Y%m%d %H:%M:%S.%f')
        df = df.dropna()
        df = df[['timestamp', 'bid', 'ask']]
        df['price'] = df[['bid', 'ask']].mean(axis = 1)
        df = df.drop(['bid', 'ask'], axis = 1)
        # Set phase
        if df['timestamp'].max() < train_end:
            phase = 'train'
        else:
            phase = 'test'
        # Aggregate data at level and magnitude
        df = df.set_index('timestamp')
        df = df.resample(f'{agg_magnitude}{agg_level}', 
            label = 'right').ohlc()['price'].reset_index()
        df['timestamp'] = df['timestamp'].astype(int)
        # Calc technical indicators
        for indicator,params in tech_indicators.items():
            if indicator == 'macd':
                df = indicators.macd(df, *params)
            elif indicator == 'rsi':
                df = indicators.rsi(df, *params)
            else:
                raise Exception(f'Unrecognized technical indicator {indicator}')
        # Add to phase numpy array
        df_np = df.dropna().to_numpy()
        assert np.sum(np.isnan(df_np)) == 0            
        if fx_npy[phase] is None:
            fx_npy[phase] = df_np
        else:
            fx_npy[phase] = np.vstack([fx_npy[phase], df_np])
    
    for phase in fx_npy.keys():
        np.save(file_paths[phase], fx_npy[phase])
    return file_paths['train']