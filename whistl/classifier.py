'''This script trains a classifier to differentiate between sepsis and healthy gene expresssion'''

import argparse
import logging
import numpy as np
import random
import time

import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model
import util


def compute_irm_penalty(loss, dummy_w):
    '''Calculate the invariance penalty for the classifier. This penalty is the norm of the
    gradient of the loss function multiplied by a dummy classifier with the value 1. This penalty
    constrains the model to perform well across studies. A more detailed explanation on why the
    dummy classifier is used can be found in section 3.1 of https://arxiv.org/abs/1907.02893
    '''
    dummy_grad = abs(grad(loss, dummy_w, create_graph=True)[0])

    return dummy_grad


def train_with_irm(classifier, map_file, train_dirs, tune_dirs, gene_file,
                   num_epochs, loss_scaling_factor, label_to_encoding, device, logger=None):
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

    Returns
    -------
    results: dict
        A dictionary containing lists tracking different loss metrics across epochs
    '''
    sample_to_label = util.parse_map_file(map_file)

    # Prepare data
    logger.info('Generating training dataset...')

    train_study_loaders = []
    train_study_counts = []
    for curr_dir in train_dirs:
        data = dataset.SingleStudyDataset(curr_dir, sample_to_label, label_to_encoding,
                                          gene_file)
        loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        train_study_loaders.append(loader)
        train_study_counts.append(len(data))

    tune_dataset = dataset.ExpressionDataset(tune_dirs, sample_to_label, label_to_encoding,
                                             gene_file)
    tune_loader = DataLoader(tune_dataset, batch_size=16, num_workers=2, pin_memory=True)

    input_size = tune_dataset[0][0].shape[0]
    classifier = classifier(input_size).double()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

    # TODO make a function equivalent to util.get_class_weights
    class_weights = {0: .9, 1: .1}

    tune_label_counts, _ = util.get_value_counts(tune_loader)
    baseline = max(list(tune_label_counts.values())) / len(tune_dataset)

    results = {'train_loss': [], 'tune_loss': [], 'train_acc': [], 'tune_acc': [],
               'baseline': baseline, 'train_penalty': [], 'train_raw_loss': []}

    try:
        # TODO there is probably a more concise way of handling this
        train_samples = sum(train_study_counts)

        dummy_w = torch.nn.Parameter(torch.DoubleTensor([1.0]))

        for epoch in tqdm(range(num_epochs)):
            train_correct = 0
            train_loss = 0
            train_penalty = 0
            train_raw_loss = 0
            for study_loader in train_study_loaders:
                for batch in study_loader:
                    expression, labels, ids = batch
                    expression = expression.to(device)
                    labels = labels.to(device).double()

                    # Set weights for this batch according to their labels
                    batch_weights = [class_weights[int(label)] for label in labels]
                    batch_weights = torch.DoubleTensor(batch_weights).to(device)

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
                train_penalty += penalty

                optimizer.zero_grad()
                # Calculate the gradient of the combined loss function
                train_loss += loss_scaling_factor * loss + penalty
                (loss_scaling_factor * loss + penalty).backward()
                optimizer.step()

            tune_loss = 0
            tune_correct = 0
            # Speed up validation by telling torch not to worry about computing gradients
            with torch.no_grad():
                for tune_batch in tune_loader:
                    expression, labels, ids = tune_batch
                    expression = expression.to(device)
                    tune_labels = labels.to(device).double()

                    loss_function = nn.BCEWithLogitsLoss()

                    tune_preds = classifier(expression)
                    loss = loss_function(tune_preds, tune_labels)
                    tune_loss += float(loss)
                    tune_correct += util.count_correct(tune_preds, tune_labels)

            tune_loss = tune_loss / len(tune_dataset)
            tune_acc = tune_correct / len(tune_dataset)
            train_loss = train_loss / train_samples
            train_acc = train_correct / train_samples
            # We cast these to floats to avoid having to pickle entire Tensor objects
            train_penalty = float(train_penalty / train_samples)
            train_raw_loss = float(train_raw_loss / train_samples)

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
        print(e)
        raise e
    finally:
        return results


def train_model_vanilla(classifier, map_file, train_dirs, tune_dirs, gene_file,
                        num_epochs, label_to_encoding, device, logger=None):
    ''' Train the provided classifier on the data from train_loader, evaluating the performance
    along the way with the data from tune_loader

    Arguments
    ---------
    classifier: pytorch.nn.Module
        The model to train
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
    label_to_encoding: dict
        A dictionary mapping string lables lieke 'sepsis' to an encoded form like 0 or 1
    device: torch.device
        The device to train the model on (either a gpu or a cpu)
    logger: logging.logger
        The python logger object to handle printing logs

    Returns
    -------
    results: dict
        A dictionary containing lists tracking different loss metrics across epochs
    '''

    sample_to_label = util.parse_map_file(map_file)

    # Prepare data
    logger.info('Generating training dataset...')
    train_dataset = dataset.ExpressionDataset(train_dirs, sample_to_label, label_to_encoding,
                                              gene_file)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                              pin_memory=True)
    logger.info('Generating tuning dataset...')
    tune_dataset = dataset.ExpressionDataset(tune_dirs, sample_to_label, label_to_encoding,
                                             gene_file)
    tune_loader = DataLoader(tune_dataset, batch_size=16, num_workers=2, pin_memory=True)

    # Get the number of genes in the data
    input_size = train_dataset[0][0].shape[0]
    classifier = classifier(input_size).double()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

    if logger is not None:
        logger.info('Training with {} training samples'.format(len(train_dataset)))
        logger.info('Tuning with {} tuning samples'.format(len(tune_dataset)))

    class_weights = util.get_class_weights(train_loader)

    # Calculate baseline tune set prediction accuracy (just pick the largest class)
    tune_label_counts, _ = util.get_value_counts(tune_loader)
    baseline = max(list(tune_label_counts.values())) / len(tune_dataset)

    results = {'train_loss': [], 'tune_loss': [], 'train_acc': [], 'tune_acc': [],
               'baseline': baseline}
    try:
        for epoch in tqdm(range(num_epochs)):
            train_loss = 0
            train_correct = 0
            # Set training mode
            classifier = classifier.train()
            for batch in train_loader:
                expression, labels, ids = batch
                expression = expression.to(device)
                labels = labels.to(device).double()

                # Get weights to handle the class imbalance
                batch_weights = [class_weights[int(label)] for label in labels]
                batch_weights = torch.DoubleTensor(batch_weights).to(device)

                loss_function = nn.BCEWithLogitsLoss(weight=batch_weights)
                optimizer.zero_grad()
                output = classifier(expression)
                loss = loss_function(output, labels)
                train_loss += float(loss)

                train_correct += util.count_correct(output, labels)

                loss.backward()
                optimizer.step()

            # Disable the gradient and switch into model evaluation mode
            with torch.no_grad():
                classifier = classifier.eval()

                tune_loss = 0
                tune_correct = 0
                for tune_batch in tune_loader:
                    expression, labels, ids = tune_batch
                    expression = expression.to(device)
                    tune_labels = labels.to(device).double()

                    loss_function = nn.BCEWithLogitsLoss()

                    tune_output = classifier(expression)

                    loss = loss_function(tune_output, tune_labels)
                    tune_loss += float(loss)
                    tune_correct += util.count_correct(tune_output, tune_labels)

                    if epoch > 70:
                        for out, label, id_ in zip(tune_output, tune_labels, ids):
                            pred = 0
                            if out > 0:
                                pred = 1

                            if pred != label:
                                print('True: {}, Pred: {}, ID: {}'.format(label, pred, id_))

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
        raise e
    finally:
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
    parser.add_argument('gene_file',
                        help='A file containing a list of genes to be used in the analysis.')
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
    parser.add_argument('--no_gpu', help='Use the CPU to train the network', action='store_false')
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

    # set random seeds
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    label_to_encoding = {'sepsis': 1, 'healthy': 0}
    classifier = model.ThreeLayerNet

    train_dirs, tune_dirs = util.train_tune_split(args.data_dir, args.tune_study_count)

    results = train_with_irm(classifier, args.map_file, train_dirs, tune_dirs, args.gene_file,
                             args.num_epochs, args.loss_scaling_factor,
                             label_to_encoding, device, logger)
    out_path = args.out_file + '_irm.pkl'
    util.save_results(args.out_file, results)

    results = train_model_vanilla(classifier, args.map_file, train_dirs, tune_dirs, args.gene_file,
                                  args.num_epochs, label_to_encoding, device, logger)

    out_path = args.out_file + '_vanilla.pkl'
    util.save_results(out_path, results)

    # TODO model checkpoints/early stopping
