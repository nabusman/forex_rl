import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayMemory(object):
    """ReplayMemory for storing history"""
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.memory = deque()
    
    def add(self, state, action, next_state, reward):
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append({'state' : state, 'action' : action, 
            'next_state' : next_state, 'reward' : reward})

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_history(self):
        return self.memory

    def __len__(self):
        return len(self.memory)


class Agent(nn.Module):
    """
    Deep Q-Learning Neural Network for trading

    Parameters:
    n_features [int]: number of input features for OHLC + technical indicators, etc.
    n_samples [int]: size of the sample length taken for each observation
    actions [list]: the actions that can be taken e.g. ['long', 'neutral', 'short']
    dense_params [list]: list of dicts of parameters for dense 
        layers not including final layer e.g. [{out_features = 128, dropout = 0.5}, {out_features = 64}]
    conv_params [list]: list of dicts for the convolution layers parameters
        e.g. [{'out_channels' : 32, 'kernel_size' : 4, 'stride' : 2}]
    """
    def __init__(self, n_features, n_samples, actions, 
        dense_params = [{'out_features' : 32}], conv_params = []):
        super(Agent, self).__init__()
        assert isinstance(n_features, int)
        assert isinstance(n_samples, int)
        assert isinstance(actions, list)
        assert isinstance(dense_params, list)
        assert isinstance(conv_params, list)
        assert dense_params != []
        self.actions = actions
        self.dense_params = dense_params
        self.conv_params = conv_params
        self.dense_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.flatten = nn.Flatten()
        conv_output = (0,0)
        for i,conv_param in enumerate(conv_params):
            if i == 0:
                l = nn.Conv1d(n_features, **conv_param)
                conv_output = self._calc_conv_output(l, (n_features, n_samples))
                self.conv_layers.append(l)
            else:
                l = nn.Conv1d(conv_output[0], **conv_param)
                conv_output = self._calc_conv_output(l, conv_output)
                self.conv_layers.append(l)
        for i,dense_param in enumerate(dense_params):
            if i == 0 and conv_params == []:
                self.dense_layers.append(nn.Linear(n_features * n_samples, 
                    dense_param['out_features']))
            elif i == 0 and conv_params != []:
                self.dense_layers.append(nn.Linear(conv_output[0] * conv_output[1], 
                    dense_param['out_features']))
            else:
                self.dense_layers.append(nn.Linear(dense_params[i-1]['out_features'], 
                    dense_param['out_features']))
            if 'dropout' in dense_param:
                self.dense_layers.append(nn.Dropout(dense_param['dropout']))
        self.output = nn.Linear(dense_params[-1]['out_features'], len(actions))


    def forward(self, x):
        for l in self.conv_layers:
            x = F.relu(l(x))
        x = self.flatten(x)
        for l in self.dense_layers:
            if 'Dropout' in str(l):
                x = l(x)
            else:
                x = F.relu(l(x))
        scores = self.output(x)
        return F.softmax(scores, dim = 1)


    def _calc_conv_output(self, conv_layer, input_size):
        """
        Calculates the output shape of a conv layer
        Parameters:
        conv_layer [nn.Conv1d]: conv layer
        input_size [tuple]: tuple of the input to layer excluding the batch size 
            e.g. (rows, columns)
        """
        output_shape = conv_layer(torch.zeros([100, input_size[0], input_size[1]])).shape
        return (output_shape[1], output_shape[2])


