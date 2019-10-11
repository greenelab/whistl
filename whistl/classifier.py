'''This script trains a classifier to differentiate between sepsis and healthy gene expresssion'''

import argparse
import numpy as np
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
import model
import util


def count_correct(output, labels):
    '''Calculate the number of correct predictions in the given batch'''
    # This could be more efficient with a hard sigmoid or something,
    # Performance impact should be negligible though
    correct = 0
    predictions = [1 if p > .5 else 0 for p in output]
    for y, y_hat in zip(predictions, labels):
        if y == y_hat:
            correct += 1
    return correct


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


def train_model(classifier, train_loader, tune_loader, train_dataset, tune_dataset, num_epochs):
    ''' Train the provided classifier on the data from train_loader, evaluating the performance
    along the way with the data from tune_loader

    Arguments
    ---------
    classifier: pytorch.nn.Module
        The model to train
    train_loader: pytorch.DataLoader
        An object that loads training data into batches
    tune_data: pytorch.DataLoader
        An object that loads tuning data into batches
    train_dataset: dataset.ExpressionDataset
        A pytorch Dataset object containing expression data to train the model on
    tune_dataset: dataset.ExpressionDataset
        A pytorch Dataset object containing expression data to evaluate the model
    num_epochs: int
        The number of times the model should be trained on all the data
    '''
    optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

    print('Training with {} training samples'.format(len(train_dataset)))
    print('Tuning with {} tuning samples'.format(len(tune_dataset)))

    class_weights = util.get_class_weights(train_loader)

    # Calculate baseline tune set prediction accuracy (just pick the largest class)
    tune_label_counts, _ = util.get_value_counts(tune_loader)
    baseline = max(list(tune_label_counts.values())) / len(tune_dataset)

    results = {'train_loss': [], 'tune_loss': [], 'train_acc': [], 'tune_acc': [],
               'baseline': baseline}

    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        # Set training mode
        classifier = classifier.train()
        for batch in train_loader:
            expression, labels, ids = batch
            expression = expression.to(device)
            labels = labels.to(device).double()

            # Get weights to handle the class imbalance
            batch_weights = [class_weights[int(label)] for label in labels]
            batch_weights = torch.DoubleTensor(batch_weights).to(device)

            loss_function = nn.BCELoss(weight=batch_weights)
            optimizer.zero_grad()
            output = classifier(expression)
            loss = loss_function(output, labels)
            train_loss += float(loss)

            train_correct += count_correct(output, labels)

            loss.backward()
            optimizer.step()

        # Disable the gradient and switch into model evaluation mode
        with torch.no_grad():
            classifier = classifier.eval()

            tune_loss = 0
            tune_correct = 0
            for tune_batch in tune_loader:
                expression, labels, ids = tune_batch
                expression = expression.to(device)
                tune_labels = labels.to(device).double()

                loss_function = nn.BCELoss()

                tune_output = classifier(expression)

                loss = loss_function(tune_output, tune_labels)
                tune_loss += float(loss)
                tune_correct += count_correct(tune_output, tune_labels)

        train_accuracy = train_correct / len(train_dataset)
        tune_accuracy = tune_correct / len(tune_dataset)

        sys.stderr.write('Epoch {}'.format(epoch) + '\n')
        sys.stderr.write('Train loss: {}'.format(train_loss / len(train_dataset)) + '\n')
        sys.stderr.write('Tune loss: {}'.format(tune_loss / len(tune_dataset)) + '\n')
        sys.stderr.write('Train accuracy: {}'.format(train_accuracy) + '\n')
        sys.stderr.write('Tune accuracy: {}'.format(tune_accuracy) + '\n')
        sys.stderr.write('Baseline accuracy: {}'.format(baseline) + '\n')

        results['train_loss'].append(train_loss / len(train_dataset))
        results['tune_loss'].append(tune_loss / len(tune_dataset))
        results['train_acc'].append(train_accuracy)
        results['tune_acc'].append(tune_accuracy)

    return results


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

    # Get the number of genes in the data
    input_size = train_dataset[0][0].shape[0]
    classifier = model.ThreeLayerNet(input_size).double()

    results = train_model(classifier, train_loader, tune_loader, train_dataset,
                          tune_dataset, args.num_epochs)

    # TODO log results to a file
    # TODO model checkpoints/early stopping
