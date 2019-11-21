'''This script looks at gene expression files and checks how much data is missing to find samples
that should be removed from the analysis.'''

import argparse
import os

import pandas as pd

if __name__ == '__main__':
    # Get arguments dir, num_genes
    parser = argparse.ArgumentParser(description='''This script looks at gene expression files
                                     and checks how much data is missing to find samples
                                     that should be removed from the analysis.''')
    parser.add_argument('study_dir', help='The path to the directory containing subdirectories '
                                          'corresponding to studies')
    parser.add_argument('missing_data_dir',
                        help='The directory to which to move files that do not meet the threshold '
                             'number of genes')
    parser.add_argument('--num_genes',
                        help='The number of genes without missing data required to '
                             'avoid being flagged for further examination',
                        type=int, default=15000)

    args = parser.parse_args()

    for dir_ in os.listdir(args.study_dir):
        if 'SRP' in dir_ or 'GSE' in dir_:
            study = os.path.normpath(dir_)
            expression_file = os.path.join(args.study_dir, study, study + '.tsv')

            expression_df = pd.read_csv(expression_file, sep='\t')
            expression_df.set_index('Gene')

            expression_df = expression_df.dropna(axis='index')

            if len(expression_df.index) < args.num_genes:
                print('Study {} only has {} genes'.format(study, len(expression_df.index)))
                print('Moving to missing_data/')

                os.rename(os.path.join(args.study_dir, study),
                          os.path.join(args.missing_data_dir, study))
