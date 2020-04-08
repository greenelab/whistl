'''A set of tests for functions in datasets.py. Each function is named according to the function
that it tests in datasets.py.'''

import json
import os
import sys
import unittest

import pytest

from whistl import datasets


example_metadata = json.loads('''{"experiments": {"GSE14844": {"sample_accession_codes": [
                        "GSM371398",
                        "GSM371375"]},
                    "GSE-ABC": {"sample_accession_codes": [
                        "GSMA",
                        "GSMB",
                        "GSMC"]}}}''')
sample_to_study_example = {'GSM371398': 'GSE14844', 'GSM371375': 'GSE14844',
                           'GSMA': 'GSE-ABC', 'GSMB': 'GSE-ABC', 'GSMC': 'GSE-ABC'}
study_to_sample_example = {'GSE14844': ['GSM371398', 'GSM371375'],
                           'GSE-ABC': ['GSMA', 'GSMB', 'GSMC']}


@pytest.mark.parametrize('metadata,correct_dict',
                         [(example_metadata, sample_to_study_example),
                          ])
def test_create_sample_to_study_mapping(metadata, correct_dict):
    pred_dict = datasets.create_sample_to_study_mapping(metadata)
    tc = unittest.TestCase()
    tc.assertDictEqual(correct_dict, pred_dict)


@pytest.mark.parametrize('metadata,correct_dict',
                         [(example_metadata, study_to_sample_example),
                          ])
def test_create_study_to_sample_mapping(metadata, correct_dict):
    pred_dict = datasets.create_study_to_sample_mapping(metadata)
    tc = unittest.TestCase()
    tc.assertDictEqual(correct_dict, pred_dict)
