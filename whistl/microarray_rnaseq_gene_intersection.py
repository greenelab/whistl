'''This script finds the list of genes that are present both in the microarray and rnaseq
experiments downloaded from refine.bio. As it is data source specific and will likely be
replaced with something better down the line, the current version is hacked together from various
logic from classifier.py and dataset.py. Since the gene list will be part of the repository itself,
this doesn't need to be run by end users, and exists only as a record of how the list was generated
'''

import pandas as pd
import os

# Grab representative studies

train_dir = '../data'
test_dir = '../data/test'

# List everything in data_dir
subfiles = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
# Keep only data directories, not anything else that might be in data_dir
train_dirs = [f for f in subfiles if ('SRP' in f or 'GSE' in f or 'E-M' in f)
              and os.path.isdir(f)]

subfiles = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
# Keep only data directories, not anything else that might be in data_dir
test_dirs = [f for f in subfiles if ('SRP' in f or 'GSE' in f or 'E-M' in f)
             and os.path.isdir(f)]

df_list = []
for data_dir in train_dirs:
    study = data_dir.rstrip('/').split('/')[-1]
    study_file_name = study + '.tsv'
    data_file = os.path.join(data_dir, study_file_name)
    curr_df = pd.read_csv(data_file, sep='\t')

    curr_df = curr_df.set_index('Gene')
    df_list.append(curr_df)

for data_dir in test_dirs:
    study = data_dir.rstrip('/').split('/')[-1]
    study_file_name = study + '.tsv'
    data_file = os.path.join(data_dir, study_file_name)
    curr_df = pd.read_csv(data_file, sep='\t')

    curr_df = curr_df.set_index('Gene')
    df_list.append(curr_df)


combined_df = pd.concat(df_list, axis=1, join='inner')

# Write genes to a file
combined_df.to_csv('../data/intersection_genes.csv', columns=[], header=False)
