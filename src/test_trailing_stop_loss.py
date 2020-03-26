import os
 import numpy as np

fx_pair = 'AUDJPY'
pip_size = 0.01
stop_loss = 10
data_dir = '/Data/foreign_exchange_data/'
is_test = True
enable_mmap = True
max_slippage = 0
start_index = 155028745
position = 'short'
enter_price = 96.6875
exit_price = None

tick_data = np.load(os.path.join(data_dir, 'npy', f'{fx_pair.upper()}_tick_level.npy'), mmap_mode = 'r')

if position == 'long':
    max_price = enter_price
elif position == 'short':
    min_price = enter_price
else:
    raise Exception(f'Unknown position {position}')
stop_loss_amount = stop_loss * pip_size
for i in range(start_index,tick_data.shape[0]):
    _, bid, ask = tick_data[i,:]
    print(f'i: {i} | bid: {bid} | ask: {ask}')
    if position == 'long':
        stop_loss = max_price - stop_loss_amount
        if max_price < bid:
            max_price = bid
            continue
        elif bid <= stop_loss or i == (tick_data.shape[0] - 1):
            exit_price =  bid
            break
    elif position == 'short':
        stop_loss = min_price + stop_loss_amount
        if min_price > ask:
            print(f'min price updated old: {min_price} | new: {ask}')
            min_price = ask
            continue
        elif ask >= stop_loss or i == (tick_data.shape[0] - 1):
            exit_price = ask
            break