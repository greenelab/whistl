import argparse
import logging
import numpy as np
import random
import sys

import torch

# Add whistl modules to the path
import classifier
import model
import plot_util
import util

# Tell pytorch to use the gpu
device = torch.device('cuda')
# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
# Ensure the models train deterministically
seed = 42

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Select a classifier architecture
label_to_encoding = {'sepsis': 1, 'healthy': 0}
net = model.ThreeLayerNet

# Split train and test data
train_dirs, tune_dirs = util.train_tune_split('../data/', 2)

# Initialize arguments to use in training the models
map_file = '../data/sample_classifications.pkl'
gene_file = '../data/intersection_genes.csv'
num_epochs = 3000
loss_scaling_factor = 1

irm_results = classifier.train_with_irm(net, map_file, train_dirs, tune_dirs, gene_file, num_epochs,
                                        loss_scaling_factor, label_to_encoding, device, logger)
