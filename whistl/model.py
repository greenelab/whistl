'''This file contains pytorch models to be used for classifying gene expression data'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerNet(nn.Module):
    '''A simple feed-forward neural network with three layers'''
    def __init__(self, input_size):
        super(ThreeLayerNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).view(-1)
        return x
