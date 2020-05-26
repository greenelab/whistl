#!/usr/bin/env python
# coding: utf-8

# # Semi-supervision Feasibility
# This notebook evaluates the data to see how much unlabeled blood data exists in the refine.bio human compendium. If enough exists, it will be important to evaluate whether semi-supervision helps model performance

# In[1]:


import collections
import json
import os
import pickle

import pandas as pd

from whistl import utils


# In[2]:


data_dir = '../../data/'
map_file = os.path.join(data_dir, 'sample_classifications.pkl')

sample_to_label = utils.parse_map_file(map_file)
with open(map_file, 'rb') as in_file:
    label_to_sample = pickle.load(in_file)[0]


# In[3]:


metadata_path = os.path.join(data_dir, 'human_compendium/aggregated_metadata.json')
with open(metadata_path) as json_file:
    metadata = json.load(json_file)


# In[4]:


def get_tissue(sample_metadata, sample):
    '''Take a sample as input and return the tissue if that information is 
       present, otherwise return None
    '''
    try:
        characteristics = sample_metadata[sample]['refinebio_annotations'][0]['characteristics_ch1']
        for characteristic in characteristics:
            if 'tissue:' in characteristic:
                tissue = characteristic.split(':')[1]
                tissue = tissue.strip().lower()
                return tissue
            
    # Catch exceptions caused by a field not being present
    except KeyError:
        return None
    
    # 'refinebio_annotations' is usually a length 1 list containing a dictionary.
    # Sometimes it's a length 0 list indicating there aren't annotations
    except IndexError:
        return None


# In[5]:


sample_metadata = metadata['samples']

tissues = []
for sample in sample_metadata:
    tissue = get_tissue(sample_metadata, sample)
    if tissue is not None:
        tissues.append(tissue)


# In[6]:


tissue_counts = collections.Counter(tissues)
tissue_counts.most_common()[:5]


# In[7]:


keys = tissue_counts.keys()
blood_keys = []
for key in keys:
    if 'blood' in key or 'pbmc' in key:
        blood_keys.append(key)
sorted(blood_keys)


# In[8]:


# Keep whole blood and pbmcs, leave out samples containing a single cell type
# Also leave out umbilical cord blood because it's not quite the same thing
# https://pubmed.ncbi.nlm.nih.gov/12634410/
keys_to_keep = ['blood',
                'blood (buffy coat)',
                'blood cells',
                'blood monocytes',
                'blood sample',
                'cells from whole blood',
                'fresh venous blood anticoagulated with 50 g/ml thrombin-inhibitor lepirudin',
                'healthy human blood',
                'host peripheral blood',
                'leukemic peripheral blood',
                'monocytes isolated from pbmc',
                'normal peripheral blood cells',
                'pbmc',
                'pbmcs',
                'peripheral blood',
                'peripheral blood (pb)',
                'peripheral blood mononuclear cell',
                'peripheral blood mononuclear cell (pbmc)',
                'peripheral blood mononuclear cells',
                'peripheral blood mononuclear cells (pbmc)',
                'peripheral blood mononuclear cells (pbmcs)',
                'peripheral blood mononuclear cells (pbmcs) from healthy donors',
                'peripheral maternal blood',
                'peripheral whole blood',
                'periphral blood',
                'pheripheral blood',
                'whole blood',
                'whole blood (wb)',
                'whole blood, maternal peripheral',
                'whole venous blood'
               ]


# In[9]:


blood_counts = dict((k, tissue_counts[k]) for k in keys_to_keep)


# In[10]:


total_samples = 0
for key in blood_counts:
    total_samples += blood_counts[key]
total_samples


# ## Count unlabeled blood cells
# ~25k blood samples is around 3x as many samples as we have labeled. Let's find exactly how much overlap there is between these samples and our labeled samples

# In[11]:


labeled_samples = sample_to_label.keys()
print(len(labeled_samples))
other_samples = label_to_sample['other']
print(len(other_samples))
labeled_samples = [sample for sample in labeled_samples if sample not in other_samples]
print(len(labeled_samples))


# In[12]:


unlabeled_samples = []

for sample in sample_metadata:
    tissue = get_tissue(sample_metadata, sample)
    if (tissue is not None and tissue in keys_to_keep 
                           and sample not in labeled_samples
       ):
        unlabeled_samples.append(sample)

print(len(unlabeled_samples))
print(len(labeled_samples))
        
# Get samples corresponding to blood
# Find the intersection and disjunction of the sets


# ## Conclusion
# There is a large number of blood samples that don't have labels. These samples can be used for semi-supervised learning, and the number of samples is large enough to make it worth trying.
