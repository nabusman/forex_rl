import os

import numpy as np
import pandas as pd
from tqdm import tqdm

currencies = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

csv_path = '/Data/foreign_exchange_data/csv/'
npy_path = '/Data/foreign_exchange_data/npy/'
df_path = '/Data/foreign_exchange_data/df/'
# train_end = np.datetime64('2018-02-01') # Not inclusive end date
# val_end = np.datetime64('2019-02-01') # Not inclusive end date

for fx in currencies:
    print(f'Working on {fx}')
    file_list = [x for x in os.listdir(csv_path) if fx in x]
    # fx_dict = {p : None for p in ['train', 'val', 'test']}
    fx_npy = None
    dropped_rows = 0
    for file_name in tqdm(file_list, total = len(file_list)):
        df = pd.read_csv(os.path.join(csv_path, file_name), header = None, names = ['fx_pair', 'timestamp', 'bid', 'ask'])
        df = df.drop('fx_pair', axis = 1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%Y%m%d %H:%M:%S.%f')
        pre_drop_len = df.shape[0]
        df = df.dropna()
        dropped_rows += pre_drop_len - df.shape[0]
        # if df['timestamp'].max() < train_end:
        #     phase = 'train'
        # elif df['timestamp'].max() >= train_end and df['timestamp'].max() < val_end:
        #     phase = 'val'
        # else:
        #     phase = 'test'
        df = df[['timestamp', 'bid', 'ask']]
        current_npy = df.to_numpy()
        current_npy[:,1] = current_npy[:,1].astype(np.float16)
        current_npy[:,2] = current_npy[:,2].astype(np.float16)
        if fx_npy is None:
            fx_npy = current_npy
        else:
            fx_npy = np.vstack([fx_npy, current_npy])
        # if fx_dict[phase] is None:
        #     fx_dict[phase] = current_npy
        # else:
        #     fx_dict[phase] = np.vstack([fx_dict[phase], current_npy])
    if dropped_rows != 0: print(f'Warning: dropped {dropped_rows} rows in converting timestamp')
    np.save(os.path.join(npy_path, f'{fx}_tick_level.npy'), fx_npy)
    # for phase in fx_dict:
    #     print(f'{phase} set size: {fx_dict[phase].shape}')
    #     np.save(os.path.join(npy_path, f'{fx}_{phase}.npy'), fx_dict[phase])
