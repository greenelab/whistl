'''Functions for creating dataset objects from the raw data directories/compendia and otherwise
decoupling the data access, dataset, and model training portions of the code'''

import os

import numpy as np
import pandas as pd

import util


# WORKFLOW
# Get all directories given root
# Get all directories with label
# Split directories to train/tune/test
# Get dataframe from directories
# Create dataset from directories
# Create data loader from directories

# TODO simplify logic in classifier.py
# TODO refactor training functions in classifier.py
# TODO rename classifier.py to train.py
# TODO rename data_util to something else
# TODO change analysis dirname to notebook
# TODO create model dir


def get_dataframe_from_dirs(data_dirs, classes, map_file):
    '''
    Arguments
    ---------
    data_dirs: list of str
        The directories to create a dataframe for
    classes: list of str
        The labels to load data for
    map_file: str or map
        A file containing the labels for all samples

    Returns
    -------
    selected_studies_df: pandas.DataFrame
        A dataframe containing the expression data for the studies in data_dirs
    labels: numpy.array
    '''
    # Get mapping from sample ids to disease labels
    sample_to_label = util.parse_map_file(map_file)
    # Select which genes to keep
    intersection_genes = get_gene_intersection(data_dirs)
    # Get label to encoding (via generate encoding, don't hardcode)
    label_to_encoding = util.generate_encoding(classes.extend(['healthy']))

    # Extract data from each directory
    df_list = []
    labels = []
    for data_dir in data_dirs:
        curr_df, study_labels = parse_study_dir(data_dir, sample_to_label, label_to_encoding,
                                                intersection_genes)
        if curr_df is None or study_labels is None:
            continue

        df_list.append(curr_df)
        labels.extend(study_labels)

    # Combine dataframes into a single df
    selected_studies_df = pd.concat(df_list, axis=1, join='inner')
    labels = np.array(labels)

    assert len(labels) == len(selected_studies_df.columns)

    return selected_studies_df, labels


def get_dataframe_from_compendium():
    '''

    Arguments
    ---------

    Returns
    -------
    '''
    raise NotImplementedError


def parse_study_dir(data_dir, sample_to_label, label_to_encoding, genes_to_keep):
    '''This function extracts the gene expression data and labels for a single study

    Arguments
    ---------
    data_dir: str
        The path to the directories where the data are stored. These are generally directories
        within the unzipped main directory downloaded from refine.bio, and will contain
        data for a single study.
    sample_to_label: dict
        A dictionary mapping sample identifiers to their corresponding labels
    label_to_encoding: dict
        A dictionary mapping the string label (e.g. 'sepsis') to a numerical target like 0
    genes_to_keep: list of strs
        The list of gene identifiers to be kept in the dataframe

    Returns
    -------
    curr_df: pandas.DataFrame
        A single dataframe containing the expression data of all genes in genes_to_keep for all
        samples in the study
    study_labels: list of ints
        Labels corresponding to whether each sample contains to septic or healthy gene expression
    '''
    study = os.path.basename(os.path.normpath(data_dir))
    study_file_name = study + '.tsv'
    data_file = os.path.join(data_dir, study_file_name)
    curr_df = pd.read_csv(data_file, sep='\t')

    curr_df = curr_df.set_index('Gene')

    # Remove samples that don't fall into a class of interest
    labels_to_keep = label_to_encoding.keys()
    curr_df = util.keep_samples_with_labels(curr_df, sample_to_label, labels_to_keep)

    # If keep_samples_with_labels returns None, we should return None for the labels as well
    if curr_df is None:
        return (None, None)

    # Retrieve labels for each sample
    study_labels = util.get_labels(curr_df, sample_to_label, label_to_encoding)

    curr_df = curr_df.loc[genes_to_keep, :]

    return curr_df, study_labels


def get_gene_intersection(data_dirs):
    '''Find the set of genes present in all samples in data_dirs

    Arguments
    ---------
    data_dirs: str or Path
        The path to the directories containing the studies used in model training

    Returns
    -------
    intersection_genes: list of str
        The list of gene ids for genes present in all samples in all data directories
    '''
    df_list = []

    for data_dir in data_dirs:
        study = os.basename(os.normpath(data_dir))
        study_file_name = study + '.tsv'
        data_file = os.path.join(data_dir, study_file_name)
        curr_df = pd.read_csv(data_file, sep='\t')

        curr_df = curr_df.set_index('Gene')
        df_list.append(curr_df)

    combined_df = pd.concat(df_list, axis=1, join='inner')
    intersection_genes = list(combined_df.index)

    return intersection_genes
