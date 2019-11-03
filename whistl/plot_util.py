''' This file contains functions for use in plotting loss curves from the results returned by
classifier.py. I decided to create a different plotting function for each metric both to make
plotting individual metrics super simple (just pass in results), and to allow consistant plotting
style across analyses.
'''
import pandas
from plotnine import ggplot, aes
from plotnine.geoms import geom_line, geom_hline
from plotnine.labels import ggtitle


def create_data_df(metric_name, metric_list):
    '''Create a pandas Dataframe for a given metric that can be used in plotting by plotnine

    Arguments
    ---------
    metric_name: string
        The human understandable name of the metric being plotted
    metric_list: list
        A list containing the various values of the metric over the epochs

    Returns
    -------
    data_df: pandas.DataFrame
        The dataframe containing the metric list and the list of epochs
    '''
    epochs = [i for i in range(len(metric_list))]
    data_df = pandas.DataFrame({metric_name: metric_list, 'epochs': epochs})

    return data_df


def plot_metric(metric_name, metric_list):
    '''This function is the generic plotting function for plotting a given metric

    Arguments
    ---------
    metric_name: string
        The human understandable name of the metric being plotted
    metric_list: list
        A list containing the various values of the metric over the epochs

    Returns
    -------
    plot: plotnine.Plot
        A plot showing the change in the metric over the epochs
    '''
    data_df = create_data_df(metric_name, metric_list)

    plot = ggplot(data_df, aes(x='epochs', y=metric_name)) + geom_line() +\
        ggtitle('{} vs epochs'.format(metric_name))

    return plot


def make_baseline_geom(results):
    '''Create a line to mark a baseline prediction in a plotnine plot

    Arguments
    ---------
    results: dict
        The dictionary containing the baseline

    Returns
    -------
    geom: plotnine.geoms.geom_hline
        The horizontal line to add to the plot
    '''
    return geom_hline(yintercept=results['baseline'], color='red')


def plot_train_acc(results):
    '''Plot the model's training accuracy over time'''
    metric_list = results['train_acc']
    metric_name = 'Training Accuracy'

    plot = plot_metric(metric_name, metric_list)
    plot = plot + ggtitle('{} vs epochs'.format(metric_name)) + make_baseline_geom(results)

    return plot


def plot_tune_acc(results):
    '''Plot the model's tuning set accuracy over time'''
    metric_list = results['tune_acc']
    metric_name = 'Tuning Set Accuracy'

    plot = plot_metric(metric_name, metric_list)
    plot = plot + ggtitle('{} vs epochs'.format(metric_name)) + make_baseline_geom(results)

    return plot


def plot_train_penalty(results):
    '''Plot the penalty for a model not giving invariant predictions across environments'''
    metric_list = results['train_penalty']
    metric_list = [float(i) for i in metric_list]
    metric_name = 'Variant Predictor Penalty'

    plot = plot_metric(metric_name, metric_list)

    return plot


def plot_raw_train_loss(results):
    '''Plot the portion of the training loss in invariant risk minimization that comes from
    the model making inaccurate predictions. This standard loss, along with the penalty
    for failing to give invariant predictions, are combined to give the full loss.'''
    metric_list = results['train_raw_loss']
    metric_list = [float(i) for i in metric_list]
    metric_name = 'Empirical Risk Minimization (Cross Entropy) loss'

    plot = plot_metric(metric_name, metric_list)

    return plot
