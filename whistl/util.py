'''This file contains useful functions for processing data'''

import os
import pickle
import random

import numpy as np


def generate_encoding(classes):
    '''Given a list of class names, generate a one-hot encoding for each class

    Arguments
    ---------
    classes: list of str
        The classes to generate an encoding for

    Returns
    -------
    label_to_encoding: dict
        A dictionary mapping each class to its encoding
    '''

    # Handle binary classification by encoding as 0/1 instead of one-hot
    if len(classes) == 2:
        label_to_encoding = {}
        for i in range(len(classes)):
            label_to_encoding[classes[i]] = i

        return label_to_encoding

    else:
        label_to_encoding = {}
        zero_matrix = np.zeros((len(classes), len(classes)))

        for i in range(len(classes)):
            encoding = zero_matrix.copy()
            encoding[i, i] = 1
            label_to_encoding[classes[i]] = encoding

        return label_to_encoding


def save_results(out_file_path, results):
    '''Write the results of a classifier training run to a file

    Arguments
    ---------
    out_file_path: str or Path
        The path to save the results to
    results: dict
        The results to save to the file
    '''
    with open(out_file_path, 'wb') as out_file:
        pickle.dump(results, out_file)


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


def count_correct(output, labels):
    '''Calculate the number of correct predictions in the given batch'''
    # This could be more efficient with a hard sigmoid or something,
    # Performance impact should be negligible though
    correct = 0
    predictions = [1 if p > 0 else 0 for p in output]
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


def get_labels(df, sample_to_label, label_to_encoding):
    ''' Retrieve the labels for the given dataset

    Arguments
    ---------
    df: pandas.DataFrame
        The DataFrame to generate labels for. Each column in the DataFrame should be a
        sample contained in sample_to_label
    sample_to_label: dict
        A dictionary mapping sample ids to their label
    label_to_encoding: dict
        A dictionary mapping the string version of a label e.g. 'sepsis' to the int encoded version
        e.g. 0

    Returns
    ------
    labels: list of ints
        The labels to be used in training a model
    '''
    labels = []
    for column in df.columns:
        labels.append(label_to_encoding[sample_to_label[column]])

    return labels


def remove_samples_with_label(df, sample_to_label, label_to_remove):
    ''' Remove all samples with a given label from the DataFrame

    Arguments
    ---------
    df: pandas.DataFrame
        The DataFrame to be filtered. Each column in the DataFrame should be a
        sample contained in sample_to_label
    sample_to_label: dict
        A dictionary mapping sample ids to their label
    label_to_remove: str
        The name of the label to remove. For example, 'other'

    Returns
    -------
    df: pandas.DataFrame
        The filtered version of the dataframe passed in
    '''
    keep_columns = [col for col in df.columns if sample_to_label[col] != label_to_remove]
    df = df[keep_columns]

    return df


def parse_gene_file(gene_file_path):
    '''Read a list of genes from a file

    Arguments
    ---------
    gene_file_path: str
        The path to the csv file to be read from

    Returns
    -------
    genes: list of str
        The genes found in the file
    '''
    with open(gene_file_path, 'r') as gene_file:
        genes = []
        for line in gene_file:
            genes.append(line.strip().strip(','))

    return genes


def get_class_weights(train_loader):
    '''Calculate class weights for better training performance on unbalanced data

    Arguments
    ---------
    train_loader: pytorch.DataLoader
        The data loader for a ExpressionDataset containing the labels to calculate weights for

    Returns
    -------
    weights: dict
        A dictionary mapping encoded labels to weights
    '''
    value_counts, total = get_value_counts(train_loader)

    # Weight labels according to the inverse of their frequency
    weights = {}
    for label in value_counts:
        weights[label] = 1 - value_counts[label] / total

    return weights


def get_value_counts(data_loader):
    '''Get the number of instances of each label in the dataset

    Arguments
    ---------
    data_loader: pytorch.DataLoader
        The data loader for a ExpressionDataset containing the labels to count

    Returns
    -------
    value_counts: dict
        A dictionary mapping the labels to their number of occurences in the dataset
    total: int
        The total number of data points in the dataset
    '''
    value_counts = {}
    total = 0

    # Count number of instances of each label
    for batch in data_loader:
        _, labels, _ = batch
        for label in labels:
            if int(label) not in value_counts:
                value_counts[int(label)] = 1
            else:
                value_counts[int(label)] += 1
            total += 1

    return value_counts, total
