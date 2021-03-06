'''This file contains useful functions for processing data'''

import os
import pickle
import random

import numpy as np


def filter_invalid_samples(sample_metadata: dict, sample_ids: list) -> dict:
    '''Remove invalid samples from a list of sample accessions

    Arguments
    ---------
    sample_metadata:
        The 'samples' subtree of the metadata for the whole compendium
    sample_ids:
        The accessions for each gene expression sample

    Returns
    -------
    valid_samples:
        The samples that meet the filtering criteria
    '''
    valid_samples = []
    # Remove beadchip samples (see https://github.com/AlexsLemonade/refinebio/issues/2114)
    for sample in sample_ids:
        if 'beadchip' not in sample_metadata[sample]['refinebio_platform'].lower():
            valid_samples.append(sample)

    return valid_samples


def map_sample_to_platform(metadata_json: dict, sample_ids: list) -> dict:
    '''Create a dict mapping each sample id to its corresponding gene expression measurement
    platform

    Arguments
    ---------
    metadata_json:
        The metadata for the whole compendium
    sample_ids:
        The accessions for each sample

    Returns
    -------
    sample_to_platform:
        The mapping from sample accessions to the expression platform the sample was measured by
    '''

    sample_metadata = metadata_json['samples']

    sample_to_platform = {}
    for sample in sample_ids:
        sample_to_platform[sample] = sample_metadata[sample]['refinebio_platform'].lower()

    return sample_to_platform


def map_sample_to_study(metadata_json: dict, sample_ids: list) -> dict:
    '''Create a dict mapping each sample id to its corresponding gene expression measurement
    platform

    Arguments
    ---------
    metadata_json:
        The metadata for the whole compendium
    sample_ids:
        The accessions for each sample

    Returns
    -------
    sample_to_study:
        The mapping from sample accessions to the study they are a member of
    '''
    experiments = metadata_json['experiments']
    id_set = set(sample_ids)

    sample_to_study = {}
    for study in experiments:
        for accession in experiments[study]['sample_accession_codes']:
            if accession in id_set:
                sample_to_study[accession] = study

    return sample_to_study


def count_items_in_dataloader(dataloader):
    '''Calculate the total number of items in a dataset by iterating through a dataloader

    Arguments
    ---------
    dataloader: pytorch DataLoader object
        The dataloader to be iterated over

    Returns
    -------
    item_count: int
        The number of items in the dataset
    '''
    item_count = 0
    for batch in dataloader:
        try:
            X = batch[0]

            # Add the size of the batch to the item count
            item_count += X.shape[0]
        except KeyError:
            # The dataloader shouldn't return an empty batch, so this may be overkil
            pass

    return item_count


def get_gene_count(gene_file):
    '''Count the number of genes in a file produced by microarray_rnaseq_gene_intersection.py

    Arguments
    ---------
    gene_file: string or Path
        The path to a file containing the list of genes created by
        microarray_rnaseq_gene_intersection.py

    Returns
    -------
    num_genes: int
        The number of genes present in the gene file
    '''
    num_genes = 0
    with open(gene_file) as in_file:
        for line in in_file:
            num_genes += 1

    return num_genes


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


def add_genes_to_results(results, gene_file):
    '''Add the ids of the genes used to train the model to the results dictionary

    Arguments
    ---------
    results: dict
        The dictionary containing metrics about the run
    gene_file: str or Path
        The file containing the list of genes to train the model on

    Returns
    -------
    results: dict
        The dictionary passed in with the list of genes added
    '''
    genes = parse_gene_file(gene_file)
    results['genes'] = genes

    return results


def add_study_ids_to_results(results, train_dirs, tune_dirs):
    '''Add the ids of the training and tuning set studies used to the results dictionary

    Arguments
    ---------
    results: dict
        The dictionary containing metrics about the run
    train_dirs: list of strings
        The paths to each directory containing a study that was used in training the model
    tune_dirs: list of strings
        The paths to each directory containing a study that was used in tuning the model

    Returns
    -------
    results: dict
        The dictionary passed in, with the train and tune study ids added
    '''
    train_ids = []
    tune_ids = []

    for dir_ in train_dirs:
        study_id = os.path.split(os.path.normpath(dir_))[-1]
        train_ids.append(study_id)
    for dir_ in tune_dirs:
        study_id = os.path.split(os.path.normpath(dir_))[-1]
        tune_ids.append(study_id)

    if 'train_ids' in results:
        results['train_ids'].extend(train_ids)
    else:
        results['train_ids'] = train_ids

    if 'tune_ids' in results:
        results['tune_ids'].extend(tune_ids)
    else:
        results['tune_ids'] = tune_ids

    return results


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


def train_tune_split(data_dirs, tune_study_count):
    '''Split the data directories into train and tune directories

    Arguments
    ---------
    data_dirs: str or Path
        The directories corresponding to different transcriptomic studies
    tune_study_count:
        The number of studies to put in the tuning set

    Returns
    -------
    train_dirs: list of strs
        The directories to be used as training data
    tune_dirs: list of strs
        The directories to be used for model tuning
    '''
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


def keep_samples_with_labels(df, sample_to_label, labels_to_keep):
    ''' Remove all samples from a dataframe except those matching one of the provided labels

    Arguments
    ---------
    df: pandas.DataFrame
        The DataFrame to be filtered. Each column in the DataFrame should be a
        sample contained in sample_to_label
    sample_to_label: dict
        A dictionary mapping sample ids to their label
    labels_to_keep: dict.dict_keys (or list of strings, depending on python version)
        The labels to be kept in the dataframe

    Returns
    -------
    df: pandas.DataFrame
        The filtered version of the dataframe passed in
    '''
    keep_columns = [col for col in df.columns if sample_to_label[col] in labels_to_keep]
    # Some studies will only contain a disease you aren't currently working with. If that
    # is the case, return None to signal that the Dataset shouldn't include this study
    if len(keep_columns) == 0:
        return None

    df = df[keep_columns]

    return df


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


def get_class_weights(train_loaders):
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
    weights = {}
    if type(train_loaders) == list:
        total_count = 0
        all_value_counts = {}
        for train_loader in train_loaders:
            value_counts, total = get_value_counts(train_loader)
            total_count += total
            for label in value_counts:
                all_value_counts[label] = all_value_counts.get(label, 0) + value_counts[label]

        for label in all_value_counts:
            weights[label] = 1 - all_value_counts[label] / total_count
    else:
        value_counts, total = get_value_counts(train_loaders)

        # Weight labels according to the inverse of their frequency
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
            if label not in value_counts:
                value_counts[int(label)] = 1
            else:
                value_counts[int(label)] += 1
            total += 1

    return value_counts, total
