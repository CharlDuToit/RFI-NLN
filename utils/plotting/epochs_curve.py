from matplotlib import pyplot as plt
import os
import numpy as np


def save_epochs_curve(dir_path, metrics, labels, metrics_std=None, file_name='training_metrics', ylabel='Loss', ylim_bottom=None, ylim_top=None):
    """
        Shows line plot of the training curves

        dir_path  (str): path to save image
        metrics (list of lists): list of metrics
        metric_labels (list): name of each metrics

    """
    if not isinstance(metrics[0], (list, np.ndarray)):
        metrics = [metrics]
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels]
    for epochs_metric, label in zip(metrics, labels):
        if len(epochs_metric) == 0:
            metrics.remove(epochs_metric)
            labels.remove(label)
    if metrics_std is None:
        metrics_std = [0] * len(metrics)

    #epochs = [e for e in range(len(metrics[0]))]
    fig = plt.figure(figsize=(10,10))
    for metric, label, std in zip(metrics, labels, metrics_std):
        epochs = [e for e in range(len(metric))]
        plt.plot(epochs, metric, label=label)
        fill_lo = np.array(metric) - np.array(std)
        fill_hi = np.array(metric) + np.array(std)
        plt.fill_between(epochs, fill_lo, fill_hi, alpha=0.1)
    plt.legend(fontsize=20)
    plt.xlabel('Epochs', fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.ylabel(ylabel, fontsize=30)
    plt.tick_params(axis='y', labelsize=20)


    if ylim_top is not None:
        plt.ylim(top=ylim_top)
    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    #plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file = os.path.join(dir_path, f'{file_name}.png')
    plt.savefig(file)
    plt.close('all')