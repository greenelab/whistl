'''Save the subset of hte compendium that has already been labeled to allow faster loading'''
import argparse

import pandas as pd

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser('This script saves the subset of the compendium that has '
                                     'already been labeled to allow faster loading and training')
    parser.add_argument('compendium', help='The refine.bio file containing the gene expression '
                                            'data to subset')
    parser.add_argument('map_file', help='The file produced by label_samples with sample labels.')
    parser.add_argument('out_path', help='The location to write the resulting tsv to')
    args = parser.parse_args()

    # Next ~20 lines graciously lifted from comendium_eda.ipynb
    sample_to_label = utils.parse_map_file(args.map_file)
    sample_ids = sample_to_label.keys()

    compendium_path = args.compendium

    # Not all labeled samples show up in the compendium, which causes pandas to panic. To fix this
    # we have to take the intersection of the accessions in sample_ids and the accessions in the
    # compendium
    header_ids = None
    with open(compendium_path) as in_file:
        header = in_file.readline()
        header_ids = header.split('\t')

    valid_sample_ids = [id_ for id_ in sample_ids if id_ in header_ids]

    # This is hacky, but the devs explicitly want to index_col to pull from only cols in usecols so
    # there isn't a better way to do it without manually changing the original file
    # Further reading https://github.com/pandas-dev/pandas/issues/9098 and
    # https://github.com/pandas-dev/pandas/issues/2654
    valid_sample_ids.append('Unnamed: 0')

    compendium_df = pd.read_csv(compendium_path, sep='\t', index_col=0, usecols=valid_sample_ids, nrows=5)

    compendium_df.to_csv(args.out_path, sep='\t')
