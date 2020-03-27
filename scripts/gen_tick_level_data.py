import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# currencies = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
currencies = ['AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

csv_path = '/Data/foreign_exchange_data/csv/'
npy_path = '/Data/foreign_exchange_data/npy/'
df_path = '/Data/foreign_exchange_data/df/'

for fx in currencies:
    print(f'Working on {fx}')
    file_list = [x for x in os.listdir(csv_path) if fx in x]
    fx_npy = None
    dropped_rows = 0
    for file_name in tqdm(file_list, total = len(file_list)):
        df = pd.read_csv(os.path.join(csv_path, file_name), header = None, names = ['fx_pair', 'timestamp', 'bid', 'ask'])
        df = df.drop('fx_pair', axis = 1)
        df['timestamp'] = pd.to_numeric(pd.to_datetime(df['timestamp'], format = '%Y%m%d %H:%M:%S.%f'), errors = 'coerce')
        pre_drop_len = df.shape[0]
        df = df.dropna()
        dropped_rows += pre_drop_len - df.shape[0]
        df = df[['timestamp', 'bid', 'ask']]
        current_npy = df.to_numpy()
        current_npy[:,1] = current_npy[:,1].astype(np.float16)
        current_npy[:,2] = current_npy[:,2].astype(np.float16)
        if fx_npy is None:
            fx_npy = current_npy
        else:
            fx_npy = np.vstack([fx_npy, current_npy])
    fx_npy = fx_npy[fx_npy[:,0].argsort()]
    if dropped_rows != 0: print(f'Warning: dropped {dropped_rows} rows in converting timestamp')
    np.save(os.path.join(npy_path, f'{fx}_tick_level.npy'), fx_npy)
