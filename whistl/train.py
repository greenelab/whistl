'''This script trains a classifier to differentiate between sepsis and healthy gene expresssion'''

import argparse
import logging
import numpy as np
import random
import sys
import time

import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

import datasets
import models
import utils


def compute_irm_penalty(loss, dummy_w):
    '''Calculate the invariance penalty for the classifier. This penalty is the norm of the
    gradient of the loss function multiplied by a dummy classifier with the value 1. This penalty
    constrains the model to perform well across studies. A more detailed explanation on why the
    dummy classifier is used can be found in section 3.1 of https://arxiv.org/abs/1907.02893
    '''
    dummy_grad = abs(grad(loss, dummy_w, create_graph=True)[0])

    return dummy_grad


def train_with_irm(classifier, train_loaders, tune_loader, num_epochs, loss_scaling_factor,
                   logger=None, save_file=None, burn_in_epochs=100):
    '''Train the provided classifier using invariant risk minimization
    IRM looks for features in the data that are invariant between different environments, as
    they are more likely to be predictive of true causal signals as opposed to spurious
    correlations. For more information, read https://arxiv.org/abs/1907.02893

    Arguments
    ---------
    classifier: pytorch.nn.Module
        The class of model to train
    map_file: string or Path
        The file created by label_samples.py to be used to match samples to labels
    train_dirs: list of str
        The directories containing training data
    tune_dirs: list of str
        The directories containing tuning data
    gene_file: string or Path
        The path to the file containing the list of genes to use in the model
    num_epochs: int
        The number of times the model should be trained on all the data
    loss_scaling_factor: float
        A hyperparameter that balances the classification loss penalty with the study invariance
        penalty. A larger value of loss_scaling_factor will cause the loss to count for more and
        the invariance penalty to count for less
    label_to_encoding: dict
        A dictionary mapping string labels like 'sepsis' to an encoded form like 0 or 1
    device: torch.device
        The device to train the model on (either a gpu or a cpu)
    logger: logging.logger
        The python logger object to handle printing logs
    save_file: string or Path object
        The file to save the model to. If save_file is None, the model won't be saved
    burn_in_epochs: int
        The number of epochs at the beginning of training to not save the model

    Returns
    -------
    results: dict
        A dictionary containing lists tracking different loss metrics across epochs
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

    # TODO make a function equivalent to util.get_class_weights
    class_weights = {0: .9, 1: .1}

    tune_label_counts, _ = util.get_value_counts(tune_loader)
    baseline = max(list(tune_label_counts.values())) / len(tune_dataset)

    results = {'train_loss': [], 'tune_loss': [], 'train_acc': [], 'tune_acc': [],
               'baseline': baseline, 'train_penalty': [], 'train_raw_loss': []}

    train_sample_count = sum([util.count_items_in_dataloader(dl) for dl in train_loaders])

    try:
        dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0])).to(device)
        best_tune_loss = None

        for epoch in tqdm_notebook(range(num_epochs)):
            train_correct = 0
            train_loss = 0
            train_penalty = 0
            train_raw_loss = 0
            for study_loader in train_loaders:
                for batch in study_loader:
                    expression, labels, ids = batch
                    expression = expression.float().to(device)
                    labels = labels.to(device).float()

                    # Set weights for this batch according to their labels
                    batch_weights = [class_weights[label.item()] for label in labels]
                    batch_weights = torch.FloatTensor(batch_weights).to(device)

                    # Note: as of 10/17/19, nn.BCELoss doesn't have a second derivative.
                    # At some point it will be possible to switch to BCELoss, follow this issue
                    # for more details: https://github.com/pytorch/pytorch/issues/18945
                    loss_function = nn.BCEWithLogitsLoss(weight=batch_weights)

                    # Bread and butter pytorch: make predictions, calculate loss and accuracy
                    pred = classifier(expression)
                    loss = loss_function(pred * dummy_w, labels)
                    train_raw_loss += loss
                    train_correct += util.count_correct(pred, labels)

                # This penalty is the norm of the gradient of 1 * the loss function.
                # The penalty helps keep the model from ignoring one study to the benefit
                # of the others, and the theoretical basis can be found in the Invariant
                # Risk Minimization paper
                penalty = compute_irm_penalty(loss, dummy_w)
                train_penalty += penalty.item()

                optimizer.zero_grad()
                # Calculate the gradient of the combined loss function
                train_loss += float(loss_scaling_factor * loss + penalty)
                (loss_scaling_factor * loss + penalty).backward(retain_graph=False)
                optimizer.step()

            tune_loss = 0
            tune_correct = 0
            # Speed up validation by telling torch not to worry about computing gradients
            with torch.no_grad():
                for tune_batch in tune_loader:
                    expression, labels, ids = tune_batch
                    tune_expression = expression.float().to(device)
                    tune_labels = labels.to(device).float()

                    loss_function = nn.BCEWithLogitsLoss()

                    tune_preds = classifier(tune_expression)
                    loss = loss_function(tune_preds, tune_labels)
                    tune_loss += loss.item()
                    tune_correct += util.count_correct(tune_preds, tune_labels)

                # Save the model
                if save_file is not None:
                    if best_tune_loss is None or tune_loss < best_tune_loss:
                        best_tune_loss = tune_loss
                        if epoch > burn_in_epochs:
                            torch.save(classifier, save_file)

            tune_loss = tune_loss / len(tune_dataset)
            tune_acc = tune_correct / len(tune_dataset)
            train_loss = train_loss / train_sample_count
            train_acc = train_correct / train_sample_count
            # We cast these to floats to avoid having to pickle entire Tensor objects
            train_penalty = float(train_penalty / train_sample_count)
            train_raw_loss = float(train_raw_loss / train_sample_count)

            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['tune_loss'].append(tune_loss)
            results['tune_acc'].append(tune_acc)
            results['train_penalty'].append(train_penalty)
            results['train_raw_loss'].append(train_raw_loss)

            if logger is not None:
                logger.info('Epoch {}'.format(epoch))
                logger.info('Train loss: {}'.format(train_loss))
                logger.info('Tune loss: {}'.format(tune_loss))
                logger.info('Train accuracy: {}'.format(train_acc))
                logger.info('Tune accuracy: {}'.format(tune_acc))
                logger.info('Baseline accuracy: {}'.format(baseline))

    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        # results = util.add_genes_to_results(results, gene_file)
        results = util.add_study_ids_to_results(results, train_dirs, tune_dirs)
        return results


def train_multitask(train_loaders, tune_loaders, representation, heads, num_epochs, logger):
    ''' Given a set of disease data loaders and disease model heads, do multitask training
    treating each disease as a task

    Arguments
    ---------
    train_loaders: list of DataLoader
        The data loaders that provide training data
    tune_loaders: list of DataLoader
        The data loaders that provide tuning data
    representation: nn.Module
        A neural network that will learn a representation of gene expression by taking input
        data and feeding it to disease specific head networks
    heads: list of nn.Module
        A list of neural networks that take the output of the representation network and use
        it to differentiate between disease and healthy samples
    num_epochs: int
        The number of epochs to train the model for
    logger: logging.logger
        The python logger object to handle printing logs

    Returns
    -------
    results: dict
        A dictionary containing lists tracking different loss metrics across epochs
    '''
    for train_loader, tune_loader, head in zip(train_loaders, tune_loaders, heads):
        classifier = nn.Sequential(representation, head)

        results = train_with_erm(classifier, train_loader, tune_loader, num_epochs, logger)

    # TODO combine results well
    return results


def train_with_erm(classifier, train_loader, tune_loader, num_epochs, logger=None, save_file=None):
    ''' Train the provided classifier on the data from train_loader, evaluating the performance
    along the way with the data from tune_loader

    Arguments
    ---------
    classifier: pytorch.nn.Module
        The model to train
    train_loader: pytorch.utils.data.DataLoader
        The DataLoader containing training data
    tune_loader: pytorch.utils.data.DataLoader
        The DataLoader containing tuning data
    num_epochs: int
        The number of times the model should be trained on all the data
    logger: logging.logger
        The python logger object to handle printing logs
    save_file: string or Path object
        The file to save the model to. If save_file is None, the model won't be saved

    Returns
    -------
    results: dict
        A dictionary containing lists tracking different loss metrics across epochs
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

    class_weights = util.get_class_weights(train_loader)

    # Calculate baseline tune set prediction accuracy (just pick the largest class)
    tune_label_counts, _ = util.get_value_counts(tune_loader)
    baseline = max(list(tune_label_counts.values())) / len(tune_dataset)

    results = {'train_loss': [], 'tune_loss': [], 'train_acc': [], 'tune_acc': [],
               'baseline': baseline}
    try:
        best_tune_loss = None

        for epoch in tqdm_notebook(range(num_epochs)):
            train_loss = 0
            train_correct = 0
            # Set training mode
            classifier = classifier.train()
            for batch in train_loader:
                expression, labels, ids = batch
                expression = expression.float().to(device)
                labels = labels.float().to(device)

                # Get weights to handle the class imbalance
                batch_weights = [class_weights[int(label)] for label in labels]
                batch_weights = torch.FloatTensor(batch_weights).to(device)

                loss_function = nn.BCEWithLogitsLoss(weight=batch_weights)

                # Standard update step
                optimizer.zero_grad()
                output = classifier(expression)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += util.count_correct(output, labels)

            # Disable the gradient and switch into model evaluation mode
            with torch.no_grad():
                classifier = classifier.eval()

                tune_loss = 0
                tune_correct = 0
                for tune_batch in tune_loader:
                    expression, labels, ids = tune_batch
                    expression = expression.float().to(device)
                    tune_labels = labels.float().to(device)

                    batch_weights = [class_weights[int(label)] for label in labels]
                    batch_weights = torch.FloatTensor(batch_weights).to(device)

                    loss_function = nn.BCEWithLogitsLoss(weight=batch_weights)

                    tune_output = classifier(expression)
                    loss = loss_function(tune_output, tune_labels)
                    tune_loss += loss.item()
                    tune_correct += util.count_correct(tune_output, tune_labels)

                # Save the model
                if save_file is not None:
                    if best_tune_loss is None or tune_loss < best_tune_loss:
                        best_tune_loss = tune_loss
                        torch.save(classifier, save_file)

            train_accuracy = train_correct / len(train_dataset)
            tune_accuracy = tune_correct / len(tune_dataset)

            if logger is not None:
                logger.info('Epoch {}'.format(epoch))
                logger.info('Train loss: {}'.format(train_loss / len(train_dataset)))
                logger.info('Tune loss: {}'.format(tune_loss / len(tune_dataset)))
                logger.info('Train accuracy: {}'.format(train_accuracy))
                logger.info('Tune accuracy: {}'.format(tune_accuracy))
                logger.info('Baseline accuracy: {}'.format(baseline))

            results['train_loss'].append(train_loss / len(train_dataset))
            results['tune_loss'].append(tune_loss / len(tune_dataset))
            results['train_acc'].append(train_accuracy)
            results['tune_acc'].append(tune_accuracy)
    except Exception as e:
        # Print error
        logger.error(e, exc_info=True)
    finally:
        results = util.add_study_ids_to_results(results, train_dirs, tune_dirs)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script trains a classifier to differentiate'
                                                 ' between sepsis and healthy gene expresssion')
    parser.add_argument('map_file',
                        help='The label: sample mapping file created by label_samples.py')
    parser.add_argument('data_dir',
                        help='The directory containing gene expression data from refine.bio. '
                             'This directory should contain only the results from unzipping the '
                             'file downloaded from refine.bio.')
    parser.add_argument('--mode', help='The method to use to train the model. Options include '
                                       'erm, irm, and multitask', default='erm', required=True)
    parser.add_argument('--gene_file', help='The file containing the genes to run the analysis on')
    parser.add_argument('--tune_study_count',
                        help='The number of studies to put in the tuning set',
                        default=2)
    parser.add_argument('--out_file', help='The file to save the results of the run to',
                        default='../logs/{}'.format(int(time.time())))
    parser.add_argument('--seed', help='The random seed to use', default=42, type=int)
    parser.add_argument('--num_epochs', help='The number of epochs to train the model',
                        default=500, type=int)
    parser.add_argument('--loss_scaling_factor', help='A hyperparameter that balances the '
                        'classification loss penalty with the study invariance penalty. A larger '
                        'value of loss_scaling_factor will cause the loss to count for more and '
                        'the invariance penalty to count for less', default=1e0)
    parser.add_argument('--no_gpu', help='Use the CPU to train the network', action='store_true')
    parser.add_argument('--multitask', help='Use multitask learning instead of '
                                            'training on one dataset', action='store_true')
    args = parser.parse_args()

    device = None
    if args.no_gpu:
        device = torch.device('cpu')
    else:
        # We'll use the default GPU, this will need to change later if using multiple GPUs
        device = torch.device('cuda')

    # Set up a logger to write logs to
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.gene_file is None:
        logger.error('A gene file is required for training without a compendium')
        sys.exit()

    # set random seeds
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_dirs = dataset.get_data_dirs(args.data_dir)

    # TODO do this better
    classes = ['tb', 'sepsis']
    sample_to_label = util.parse_map_file(args.map_file)

    mode = args.mode.lower().strip()

    if mode == 'irm':
        intersection_genes = util.parse_gene_file(args.gene_file)

        disease_train_data_dirs = []
        disease_tune_data_dirs = []
        for disease in classes:
            _, disease_dirs = dataset.extract_dirs_with_label(data_dirs, disease, sample_to_label)
            train_dirs, tune_dirs = util.train_tune_split(disease_dirs, args.tune_study_count)

            disease_train_data_dirs.extend(train_dirs)
            disease_tune_data_dirs.extend(tune_dirs)

        train_loaders = []
        for data_dir in disease_train_data_dirs:
            train_dataset = dataset.RefineBioDataset([data_dir], classes, sample_to_label,
                                                     intersection_genes)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
            train_loaders.append(train_loader)

        tune_dataset = dataset.RefineBioDataset(disease_tune_data_dirs, classes, sample_to_label,
                                                intersection_genes)
        tune_loader = DataLoader(tune_dataset, batch_size=16, num_workers=2, pin_memory=True)

        classifier = model.ThreeLayerNet(len(intersection_genes))

        results = train_with_irm(classifier, train_loaders, tune_loader, args.num_epochs,
                                 args.loss_scaling_factor, logger)

    if mode == 'erm':
        disease_train_data_dirs = []
        disease_tune_data_dirs = []
        for disease in classes:
            _, disease_dirs = dataset.extract_dirs_with_label(data_dirs, disease, sample_to_label)
            train_dirs, tune_dirs = util.train_tune_split(disease_dirs, args.tune_study_count)

            disease_train_data_dirs.extend(train_dirs)
            disease_tune_data_dirs.extend(tune_dirs)

            # TODO check number of tune data dirs

        intersection_genes = util.parse_gene_file(args.gene_file)
        train_dataset = dataset.RefineBioDataset(disease_train_data_dirs, classes, sample_to_label,
                                                 intersection_genes)
        tune_dataset = dataset.RefineBioDataset(disease_tune_data_dirs, classes, sample_to_label,
                                                intersection_genes)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                                  pin_memory=True)
        tune_loader = DataLoader(tune_dataset, batch_size=16, num_workers=2, pin_memory=True)

        classifier = model.ThreeLayerNet(len(intersection_genes))
        results = train_with_erm(classifier, train_loader, tune_loader, args.num_epochs, logger)

    if mode == 'multitask':
        train_loaders = []
        tune_loaders = []
        heads = []

        intersection_genes = util.parse_gene_file(args.gene_file)

        representation = model.ExpressionRepresentation(len(intersection_genes))

        for disease in classes:
            _, disease_dirs = dataset.extract_dirs_with_label(data_dirs, disease, sample_to_label)
            train_dirs, tune_dirs = util.train_tune_split(disease_dirs, args.tune_study_count)

            # Instead of making a list with all the directories, make a list of lists where each
            # entry is a list of directories corresponding to a disease
            train_dataset = dataset.RefineBioDataset(train_dirs, [disease], sample_to_label,
                                                     intersection_genes)
            tune_dataset = dataset.RefineBioDataset(tune_dirs, [disease], sample_to_label,
                                                    intersection_genes)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                                      pin_memory=True)
            tune_loader = DataLoader(tune_dataset, batch_size=16, num_workers=2, pin_memory=True)

            train_loaders.append(train_loader)
            tune_loaders.append(tune_loader)

            head = model.MultitaskHead(representation.final_size)
            heads.append(head)

        results = train_multitask(train_loaders, tune_loaders, representation, heads,
                                  args.num_epochs, logger)
