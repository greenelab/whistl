'''This script trains a classifier to differentiate between sepsis and healthy gene expresssion'''

import argparse
import numpy as np
import os
import pickle
import random
import sys

import torch
from torch.utils.data import DataLoader

import dataset


def parse_map_file(map_file_path):
    '''Create a sample: label mapping from the pickled file output by label_samples.py

    Arguments
    ---------
    map_file_path: str or Path object
        The path to a pickled file created by label_samples.py

    Returns
    -------
    sample_to_label: dict
        A string to string dict mapping sample ids to their corresponding label string.
        E.g. {'GSM297791': 'sepsis'}
    '''
    sample_to_label = {}
    label_to_sample = None
    with open(map_file_path, 'rb') as map_file:
        label_to_sample, _ = pickle.load(map_file)

    for label in label_to_sample:
        for sample in label_to_sample[label]:
            sample_to_label[sample] = label

    return sample_to_label


def train_tune_split(data_dir, tune_study_count):
    '''Split the data directories into train and tune directories

    Arguments
    ---------
    data_dir: str or Path
        The directory containing subdirectories with gene expression data
    tune_study_count:
        The number of studies to put in the tuning set

    Returns
    -------
    train_dirs: list of strs
        The directories to be used as training data
    tune_dirs: list of strs
        The directories to be used for model tuning
    '''
    # List everything in data_dir
    subfiles = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    # Keep only data directories, not anything else that might be in data_dir
    data_dirs = [f for f in subfiles if ('SRP' in f or 'GSE' in f) and os.path.isdir(f)]

    # Pull out directories for tuning, then put everything else in train_dirs
    tune_dirs = random.sample(data_dirs, tune_study_count)
    train_dirs = [dir_ for dir_ in data_dirs if dir_ not in tune_dirs]

    return train_dirs, tune_dirs


if __name__ == '__main__':
    # Load mapping file
    # Load data
    # Combine all dataframes into a single dataframe
    # Convert the dataframe into a matrix
    # Create the label vector based on the mapping file and matrix
    # Create pytorch dataset from the matrix and labels
    # train model on dataset

    parser = argparse.ArgumentParser(description='This script trains a classifier to differentiate'
                                                 ' between sepsis and healthy gene expresssion')
    parser.add_argument('map_file',
                        help='The label: sample mapping file created by label_samples.py')
    parser.add_argument('data_dir',
                        help='The directory containing gene expression data from refine.bio. '
                             'This directory should contain only the results from unzipping the '
                             'file downloaded from refine.bio.')
    parser.add_argument('gene_file',
                        help='A file containing a list of genes to be used in the analysis.')
    parser.add_argument('--tune_study_count',
                        help='The number of studies to put in the tuning set',
                        default=2)
    parser.add_argument('--seed', help='The random seed to use', default=42)
    parser.add_argument('--num_epochs', help='The number of epochs to train the model',
                        default=500)
    parser.add_argument('--no_gpu', help='Use the CPU to train the network', action='store_false')
    args = parser.parse_args()

    device = None
    if args.no_gpu:
        device = torch.device('cpu')
    else:
        # We'll use the default GPU, this will need to change later if using multiple GPUs
        device = torch.device('cuda')

    # set random seeds
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    sample_to_label = parse_map_file(args.map_file)

    # Prepare data
    train_dirs, tune_dirs = train_tune_split(args.data_dir, args.tune_study_count)
    label_to_encoding = {'sepsis': 1, 'healthy': 0}

    sys.stderr.write('Generating training dataset...\n')
    train_dataset = dataset.ExpressionDataset(train_dirs, sample_to_label, label_to_encoding,
                                              args.gene_file)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                              pin_memory=True)
    sys.stderr.write('Generating tuning dataset...\n')
    tune_dataset = dataset.ExpressionDataset(tune_dirs, sample_to_label, label_to_encoding,
                                             args.gene_file)
    tune_loader = DataLoader(tune_dataset, batch_size=16, shuffle=True, num_workers=2,
                             pin_memory=True)

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(train_loader):
            expression, labels = batch
            expression = expression.to(device)
            labels = labels.to(device)
            print(expression.shape)

        # TODO evaluate tune set error
        # TODO plot learning curve
        # TODO handle checkpoints
        # TODO actually load model
