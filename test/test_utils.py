'''A set of tests for functions in util.py. Each function is named according to the function that
it tests in util.py.'''

import os
import pickle
import sys

import pandas as pd
import pytest
import numpy as np

whistl_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(whistl_path + '/../whistl')
import utils


data = {'sample1': [1, 1], 'sample2': [2, 2], 'sample3': [3, 3]}
df = pd.DataFrame.from_dict(data)

sample_to_label_1 = {'sample1': 'good', 'sample2': 'bad', 'sample3': 'good'}
sample_to_label_2 = {'sample1': 'bad', 'sample2': 'bad', 'sample3': 'good'}
label_to_encoding_1 = {'good': 1, 'bad': 0}
label_to_encoding_2 = {'good': 0, 'bad': 1}


@pytest.mark.parametrize('df,sample_to_label,label_to_encoding,correct_output',
                         [(df, sample_to_label_1, label_to_encoding_1, [1, 0, 1]),
                          (df, sample_to_label_1, label_to_encoding_2, [0, 1, 0]),
                          (df, sample_to_label_2, label_to_encoding_1, [0, 0, 1]),
                          (df, sample_to_label_2, label_to_encoding_2, [1, 1, 0]),
                          ])
def test_get_labels(df, sample_to_label, label_to_encoding, correct_output):
    assert correct_output == utils.get_labels(df, sample_to_label, label_to_encoding)


@pytest.mark.parametrize('df,sample_to_label,label_to_remove,correct_cols',
                         [(df, sample_to_label_1, 'bad', ['sample1', 'sample3']),
                          (df, sample_to_label_1, 'good', ['sample2']),
                          ])
def test_remove_samples_with_label(df, sample_to_label, label_to_remove, correct_cols):
    filtered_df = utils.remove_samples_with_label(df, sample_to_label, label_to_remove)
    assert len(filtered_df.columns) == len(correct_cols)
    for test_column, correct_column in zip(filtered_df.columns, correct_cols):
        assert test_column == correct_column


class1_encoding = np.zeros((3, 3))
class1_encoding[0, 0] = 1
class2_encoding = np.zeros((3, 3))
class2_encoding[1, 1] = 1
class3_encoding = np.zeros((3, 3))
class3_encoding[2, 2] = 1


@pytest.mark.parametrize('classes,correct_encodings',
                         [(['class1', 'class2', 'class3'],
                           [class1_encoding, class2_encoding, class3_encoding]),
                          (['class1', 'class2'], [0, 1]),
                          ])
def test_generate_encoding(classes, correct_encodings):
    generated_encodings = utils.generate_encoding(classes)

    for i, curr_class in enumerate(generated_encodings.keys()):
        generated_encoding = generated_encodings[curr_class]
        assert np.array_equal(generated_encoding, correct_encodings[i])


dir_list_1 = ['../data/GSE1', '../data/GSE2/']
dir_list_2 = ['../data/SRP1', '../data/SRP2', '../data/SRP3']


@pytest.mark.parametrize('train_dirs, tune_dirs, train_len, tune_len, train_last_id, tune_last_id',
                         [(dir_list_1, dir_list_2, 2, 3, 'GSE2', 'SRP3'),
                          (dir_list_2, dir_list_1, 3, 2, 'SRP3', 'GSE2'),
                          ])
def test_add_study_ids(train_dirs, tune_dirs, train_len, tune_len, train_last_id, tune_last_id):
    init_results = {'losses': [5] * 50, 'final_acc': .1}

    results = utils.add_study_ids_to_results(init_results, train_dirs, tune_dirs)

    assert 'train_ids' in results
    assert 'tune_ids' in results
    assert len(results['train_ids']) == train_len
    assert len(results['tune_ids']) == tune_len
    assert results['train_ids'][-1] == train_last_id
    assert results['tune_ids'][-1] == tune_last_id
