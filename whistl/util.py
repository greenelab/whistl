'''This file contains useful functions for processing data'''


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
