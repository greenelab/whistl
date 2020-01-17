''' This file contains pytorch Dataset objects for use in processing data'''
import numpy as np
from torch.utils.data import Dataset


class ExpressionDataset(Dataset):
    ''' A dataset of one or more studies from refine.bio'''
    def __init__(self, data_df, labels):
        ''' The dataset's constructor function

        Arguments
        ---------
        data_df: pandas.DataFrame
            The data (genes x samples) to be used to train a model
        labels: np.array
            The disease labels for each sample in data_df
        '''
        self.gene_expression = data_df
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
