'''This file contains pytorch models to be used for classifying gene expression data'''

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
        # Sigmoid function is handle by BCEWithLogitsLoss
        x = self.fc3(x).view(-1)
        return x


class ExpressionRepresentation(nn.Module):
    '''The encoding portion of a multitask network/the weights shared between the head nodes'''
    def __init__(self, input_size):
        super(ExpressionRepresentation, self).__init__()
        self.final_size = 64

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, self.final_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class MultitaskHead(nn.Module):
    '''A head node to do binary classification on the encoding portion of a multitask network'''
    def __init__(self, input_size):
        super(MultitaskHead, self).__init__()

        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc1(x).view(-1)
        return x
