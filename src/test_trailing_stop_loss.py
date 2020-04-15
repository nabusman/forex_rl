import os
import numpy as np

from helpers import calc_exit_price

fx_pair = 'AUDJPY'
data_dir = '/Data/foreign_exchange_data/'
tick_data = np.load(os.path.join(data_dir, 'npy', f'{fx_pair.upper()}_tick_level.npy'), mmap_mode = 'r')
pip_size = 0.01
stop_loss = 10
is_test = True
max_slippage = 0
start_index = np.random.randint(0,tick_data.shape[0])
position = 'short'
enter_price = 96.6875
exit_price = None

print(f'Start index: {start_index}')

exit_price = calc_exit_price(tick_data, start_index, position, stop_loss, pip_size, enter_price)

if position == 'long':
    print(f'Net pips: {(exit_price - enter_price) / pip_size}')
elif position == 'short':
    print(f'Net pips: {(enter_price - exit_price) / pip_size}')