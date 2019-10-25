''' This file contains pytorch Dataset objects for use in processing data'''
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import util


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
    study = data_dir.rstrip('/').split('/')[-1]
    study_file_name = study + '.tsv'
    data_file = os.path.join(data_dir, study_file_name)
    curr_df = pd.read_csv(data_file, sep='\t')

    curr_df = curr_df.set_index('Gene')
    # Remove samples that don't fall into a class of interest
    curr_df = util.remove_samples_with_label(curr_df, sample_to_label, 'other')
    # Retrieve labels for each sample
    study_labels = util.get_labels(curr_df, sample_to_label, label_to_encoding)

    curr_df = curr_df.loc[genes_to_keep, :]

    return curr_df, study_labels


class ExpressionDataset(Dataset):
    ''' A dataset to parse data output by refine.bio'''
    def __init__(self, data_dirs, sample_to_label, label_to_encoding, gene_file_path):
        ''' The dataset's constructor function

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
        gene_file_path: string
            The path to a file containing the genes to use in the dataset
        '''
        genes_to_keep = util.parse_gene_file(gene_file_path)

        df_list = []
        labels = []
        for data_dir in data_dirs:
            curr_df, study_labels = parse_study_dir(data_dir, sample_to_label, label_to_encoding,
                                                    genes_to_keep)

            df_list.append(curr_df)
            labels.extend(study_labels)

        all_studies_df = pd.concat(df_list, axis=1, join='inner')

        assert len(labels) == len(all_studies_df.columns)

        labels = np.array(labels)

        self.gene_expression = all_studies_df
        self.labels = labels

    def __getitem__(self, idx):
        '''

        Arguments
        ---------
        idx: int
            The index of the sample to retrieve

        Returns
        -------
        sample: numpy.array
            The gene expression information for the sample at index idx
        label: int
            The label for the sample at index idx
        id_: string
            The sample identifier for the given sample
        '''
        sample = self.gene_expression.iloc[:, idx].values
        label = np.array(self.labels[idx])
        id_ = self.gene_expression.columns[idx]

        return sample, label, id_

    def __len__(self):
        '''Provides the number of samples in the dataset'''
        return len(self.labels)


class SingleStudyDataset(Dataset):
    '''A DataSet designed to store gene expression data for all samples in a single study.
    Being able to create a list of these datasets will be useful in implementing invariant
    risk minimization and other methods that don't treat studies as IID'''
    def __init__(self, data_dir, sample_to_label, label_to_encoding, gene_file_path):
        ''' The dataset's constructor function

        Arguments
        ---------
        data_dir: str
            The path to the directory where the study data is stored. this is generally a dir
            within the unzipped main directory downloaded from refine.bio, and will contain
            data for a single study.
        sample_to_label: dict
            A dictionary mapping sample identifiers to their corresponding labels
        label_to_encoding: dict
            A dictionary mapping the string label (e.g. 'sepsis') to a numerical target like 0
        gene_file_path: string
            The path to a file containing the genes to use in the dataset
        '''
        genes_to_keep = util.parse_gene_file(gene_file_path)
        study_df, labels = parse_study_dir(data_dir, sample_to_label, label_to_encoding,
                                           genes_to_keep)
        assert len(labels) == len(study_df.columns)

        labels = np.array(labels)
        self.gene_expression = study_df
        self.labels = labels

    def __getitem__(self, idx):
        '''

        Arguments
        ---------
        idx: int
            The index of the sample to retrieve

        Returns
        -------
        sample: numpy.array
            The gene expression information for the sample at index idx
        label: int
            The label for the sample at index idx
        id_: string
            The sample identifier for the given sample
        '''
        sample = self.gene_expression.iloc[:, idx].values
        label = np.array(self.labels[idx])
        id_ = self.gene_expression.columns[idx]

        return sample, label, id_

    def __len__(self):
        '''Provides the number of samples in the dataset'''
        return len(self.labels)
