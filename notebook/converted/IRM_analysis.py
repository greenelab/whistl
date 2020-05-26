#!/usr/bin/env python
# coding: utf-8

# # IRM Analysis
# This notebook will compare the performance of IRM on an unseen platform's worth of gene expression to that of ERM. These results will be used for the preliminary data section for Aim 2 in my prelim proposal.
# 
# The EDA code is [here](#EDA), or to skip to the analysis, go [here](#eval)

# <a id='eda'></a>
# ## Sepsis EDA
# 
# To have a good measure of training performance, ideally we'll have one platform's data held out as a validation set. To see how possible that is, we'll do exploratory data analysis on the sepsis studies in the dataset. 

# In[1]:


import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import torch
from plotnine import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whistl import datasets
from whistl.datasets import CompendiumDataset
from whistl import models
from whistl import train
from whistl import utils


# In[2]:


import random
import numpy as np
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)


# In[3]:


curr_path = str(Path('.'))

map_file = str(Path('../../data/sample_classifications.pkl'))
sample_to_label = utils.parse_map_file(map_file)
sample_ids = sample_to_label.keys()

metadata_file = str(Path('../../data/all_metadata.json'))
metadata_json = json.load(open(metadata_file))
sample_metadata = metadata_json['samples']

sample_ids = utils.filter_invalid_samples(sample_metadata, sample_ids)

sample_to_platform = utils.map_sample_to_platform(metadata_json, sample_ids)
sample_to_study = utils.map_sample_to_study(metadata_json, sample_ids)


# In[4]:


compendium_path = str(Path('../../data/subset_compendium.tsv'))

compendium_df = datasets.load_compendium_file(compendium_path)
compendium_df.head()


# In[5]:


sepsis_samples = [sample for sample in sample_ids if sample_to_label[sample] == 'sepsis']
sepsis_platforms = [sample_to_platform[sample] for sample in sepsis_samples]
sepsis_studies = [sample_to_study[sample] for sample in sepsis_samples]
print(len(sepsis_samples))
print(len(sepsis_platforms))
print(len(sepsis_studies))


# In[6]:


sepsis_metadata_dict = {'sample': sepsis_samples, 'platform': sepsis_platforms, 'study': sepsis_studies}
sepsis_metadata_df = pd.DataFrame(sepsis_metadata_dict)
sepsis_metadata_df = sepsis_metadata_df.set_index('sample')
sepsis_metadata_df.head()


# In[7]:


sepsis_metadata_df['platform'].value_counts()


# In[8]:


sepsis_metadata_df[sepsis_metadata_df['platform'] == 'affymetrix human genome u133a array (hgu133a)']


# In[9]:


# Remove platform with only one sample to reduce downstream variance
sepsis_metadata_df = sepsis_metadata_df.drop(labels='GSM301847', axis=0)
print(len(sepsis_metadata_df.index))


# In[10]:


sepsis_metadata_df['study'].value_counts()


# <a id='eval'></a>
# ## IRM Evaluation

# ### Setup

# In[11]:


curr_path = os.path.dirname(os.path.abspath(os.path.abspath('')))

map_file = str(Path('../../data/sample_classifications.pkl'))
sample_to_label = utils.parse_map_file(map_file)

metadata_path = str(Path('../../data/all_metadata.json'))

compendium_path = str(Path('../../data/subset_compendium.tsv'))


# ### More setup
# Initialize the model and encoder for the training process

# In[12]:


classes = ['sepsis', 'healthy']
encoder = preprocessing.LabelEncoder()
encoder.fit(classes)


# ### Tune split
# We will get a rough estimate of performance with leave-one-out cross-validation. To know when to stop training, though, we will need a tuning dataset.

# In[13]:


tune_df = sepsis_metadata_df[sepsis_metadata_df['platform'] == 'affymetrix human genome u219 array (hgu219)']
train_df = sepsis_metadata_df[sepsis_metadata_df['platform'] != 'affymetrix human genome u219 array (hgu219)']
print(len(tune_df.index))
print(len(train_df.index))

tune_studies = tune_df['study'].unique()
tune_dataset = CompendiumDataset(tune_studies, classes, 
                                 sample_to_label, metadata_path, 
                                 compendium_path, encoder)
tune_loader = DataLoader(tune_dataset, batch_size=1)


# ### Filter Platforms
# Remove a platform that corresponds to a study present in the labeled data, but not the human compendium

# In[14]:


platforms = train_df['platform'].unique()
platforms = [p 
             for p in platforms 
             if p != 'affymetrix human human exon 1.0 st array (huex10st)'
            ]
num_seeds = 5


# ## Training
# 
# The models are trained with two platforms held out. 
# One platform (huex10st) is left out in all runs, and is used as a tuning set to determine which version of the model should be saved.
# The second platform (referred to going forward as the 'held-out platform') is held out during training, then the trained model's performance is evaluated by trying to predict whether each sample corresponds to sepsis or healthy expression.

# In[15]:


irm_result_list = []
erm_result_list = []


for hold_out_platform in platforms:
    train_platforms = train_df[train_df['platform'] != hold_out_platform]['platform'].unique()

    train_loaders = []
    total_irm_samples = 0
    for platform in train_platforms:
        studies = train_df[train_df['platform'] == platform]['study']
        train_dataset = CompendiumDataset([platform], classes, sample_to_label, metadata_path, compendium_path, 
                                          encoder, mode='platform')
        total_irm_samples += len(train_dataset)

        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            train_loaders.append(train_loader)

    platform_file = hold_out_platform.split('(')[-1].strip(')')

    full_train_studies = train_df[train_df['platform'] != hold_out_platform]['study'].unique()
    full_train_dataset = CompendiumDataset(train_platforms, classes, sample_to_label, metadata_path,
                                           compendium_path, encoder, mode='platform')
    full_train_loader = DataLoader(full_train_dataset, batch_size=8, shuffle=True)

    assert total_irm_samples == len(full_train_dataset)
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        net = models.ThreeLayerNet(len(compendium_df.index))
        
        writer_path = Path('./logs/erm_analysis_{}_{}.tfrecord'.format(platform_file, seed))
        writer = SummaryWriter(writer_path)

        save_file = Path('./logs/erm_analysis_{}_{}.pkl'.format(platform_file, seed))
        results = train.train_with_erm(net, full_train_loader, 
                                       tune_loader, num_epochs=400, 
                                       save_file=save_file, writer=writer)
        erm_result_list.append(results)

        net = models.ThreeLayerNet(len(compendium_df.index))
        
        writer_path = Path('./logs/irm_analysis_{}_{}.tfrecord'.format(platform_file, seed))
        writer = SummaryWriter(writer_path)

        
        save_file = Path('./logs/irm_analysis_{}_{}.pkl'.format(platform_file, seed))
        results = train.train_with_irm(net, train_loaders, 
                                       tune_loader, num_epochs=400, 
                                       loss_scaling_factor=1, save_file=save_file, 
                                       writer=writer, burn_in_epochs=0)
        irm_result_list.append(results)


# In[16]:


def eval_model(net, loader):
    all_labels = []
    all_preds = []
    for batch in loader:
        expression, labels, ids = batch
        expression = expression.float().to('cuda')
        labels = labels.numpy()
        all_labels.extend(labels)
                
        output = net(expression)
        preds = [1 if p > 0 else 0 for p in output]
        all_preds.extend(preds)
        
    f1 = metrics.f1_score(all_labels, all_preds)
        
    return f1
        


# In[17]:


irm_f1_scores = []
erm_f1_scores = []
for hold_out_platform in platforms:
    for seed in range(num_seeds):
        # Load data
        try:
            hold_out_studies = train_df[train_df['platform'] == hold_out_platform]['study']
            hold_out_dataset = CompendiumDataset(hold_out_studies, classes, sample_to_label, metadata_path, compendium_path, encoder)
            hold_out_loader = DataLoader(hold_out_dataset, batch_size=1, shuffle=False)

            # Load IRM model
            platform_file = hold_out_platform.split('(')[-1].strip(')')
            save_file = Path('./logs/irm_analysis_{}_{}.pkl'.format(platform_file, seed))
            net = torch.load(save_file, 'cuda')

            #Evaluate ERM model
            f1_score = eval_model(net, hold_out_loader)
            irm_f1_scores.append(f1_score)

            # Load ERM model
            save_file = Path('./logs/erm_analysis_{}_{}.pkl'.format(platform_file, seed))
            net = torch.load(save_file, 'cuda')

            # Evaluate IRM model
            f1_score = eval_model(net, hold_out_loader)
            erm_f1_scores.append(f1_score)
        except FileNotFoundError as e:
            print(e)


# In[18]:


print(irm_f1_scores)
print(erm_f1_scores)
held_out_platform_list = []
for platform in platforms:
    p = [platform] * 2 * num_seeds
    held_out_platform_list.extend(p)
#print(held_out_platform_list)

score_list = list(itertools.chain(*zip(irm_f1_scores, erm_f1_scores)))
print(score_list)
label_list = (['irm'] + ['erm']) *  (len(score_list) // 2)
print(label_list)


# In[29]:


held_out_platform_list = [plat.split('(')[-1].strip(')') for plat in held_out_platform_list]
result_dict = {'f1_score': score_list, 'irm/erm': label_list, 'held_out_platform': held_out_platform_list}
result_df = pd.DataFrame(result_dict)
result_df.head()


# ## Results
# 
# The first figures measure the models' performance on the held out platform. These figures measure the model's ability to generalize.
# 
# The second set of figures measure the models' performance no the tuning set to measure the model's training behavior (and to a lesser extend the models' ability to predict a held-out set). 

# In[30]:


(ggplot(result_df, aes('irm/erm', 'f1_score', color='held_out_platform')) +
 geom_jitter(size=3) +
 ggtitle('F1 Score on held-out platform')
)


# In[21]:


(ggplot(result_df, aes('irm/erm', 'f1_score')) +
 geom_violin() +
 ggtitle('F1 Score on held-out platform')
)


# In[38]:


irm_accs = [result['tune_acc'] for result in irm_result_list]
irm_mean_accs = [sum(accs) / len(accs) for accs in irm_accs]
print(irm_mean_accs)
[acc.sort() for acc in irm_accs]
irm_median_accs = [acc[len(acc) //2] for acc in irm_accs]
print(irm_median_accs)
irm_max_accs = [max(accs) for accs in irm_accs]


# In[39]:


erm_accs = [result['tune_acc'] for result in erm_result_list]
erm_mean_accs = [sum(accs) / len(accs) for accs in erm_accs]
print(erm_mean_accs)
[acc.sort() for acc in erm_accs]
erm_median_accs = [acc[len(acc) //2] for acc in erm_accs]
print(erm_median_accs)
erm_max_accs = [max(accs) for accs in erm_accs]


# In[40]:


mean_list = list(itertools.chain(*zip(irm_mean_accs, erm_mean_accs)))
median_list = list(itertools.chain(*zip(irm_median_accs, erm_median_accs)))
max_list = list(itertools.chain(*zip(irm_max_accs, erm_max_accs)))
label_list = (['irm'] + ['erm']) *  (len(mean_list) // 2)

held_out_platform_list = []
for platform in platforms:
    plat = platform.split('(')[-1].strip(')')
    p = [plat] * 2 * num_seeds
    held_out_platform_list.extend(p)
held_out_platform_list = [plat.split('(')[-1].strip(')') for plat in held_out_platform_list]

result_dict = {'mean_acc': mean_list, 'median_acc': median_list, 'max_acc': max_list, 'irm/erm': label_list, 
               'held_out_platform': held_out_platform_list}
result_df = pd.DataFrame(result_dict)


# In[32]:


(ggplot(result_df, aes('irm/erm', 'mean_acc', color='held_out_platform')) + 
 geom_jitter(size=3) +
 ggtitle('Mean accuracies in IRM and ERM on tune set')
)


# In[33]:


(ggplot(result_df, aes('irm/erm', 'mean_acc')) + 
 geom_violin() +
 ggtitle('Mean accuracies in IRM and ERM on tune set')
)


# ### Median accuracies
# IRM is somewhat unstable during training, and will occasionally have sudden spikes of poor performance.
# Since we save the best version of the model based on the model's loss, it makes more sense to look at the median accuracy than the mean as a measure of central tendency

# In[34]:


(ggplot(result_df, aes('irm/erm', 'median_acc', color='held_out_platform')) + 
geom_jitter(size=3) +
ggtitle('Median accuracies in ERM and IRM on tune set')
)


# In[28]:


(ggplot(result_df, aes('irm/erm', 'median_acc')) + 
geom_violin() +
ggtitle('Median accuracies in ERM and IRM on tune set')
)


# ## Max accuracies

# In[43]:


(ggplot(result_df, aes('irm/erm', 'max_acc', color='held_out_platform')) + 
geom_point(size=3) +
ggtitle('Max accuracies in ERM and IRM on tune set')
)


# In[ ]:




