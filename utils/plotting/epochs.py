from matplotlib import pyplot as plt
import os
import numpy as np

# from common import apply_plot_settings
from .common import apply_plot_settings


# logx = False, logy = False, dir_path = './',
# file_name = 'scatter', xlabel = 'x-axis', ylabel = 'y-axis', title = None, axis_fontsize = 20, xtick_size = 20,
# ytick_size = 20, legend_fontsize = 20, title_fontsize = 20, legend_title = None, ** kwargs)
# def save_epochs_curve(dir_path, metrics, labels, metrics_std=None,
#                       file_name='training_metrics', ylabel='Loss', ylim_bottom=None, ylim_top=None):
def save_epochs_curve(dir_path, metrics, labels, metrics_std=None, figsize=(10, 10),
                      logx=False, logy=False,
                      file_name='training_metrics', xlabel='Epochs', ylabel='Loss', title=None, axis_fontsize=30,
                      xtick_size=20, ytick_size=20, legend_fontsize=20, title_fontsize=20, legend_title=None, linewidth=10,
                      show_legend=True, xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None, show=False, **kwargs):
    """
        Shows line plot of the training curves

        dir_path  (str): path to save image
        metrics (list of lists): list of metrics
        metric_labels (list): name of each metrics

    """
    fig, ax = plt.subplots(figsize=figsize)

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

    for metric, label, std in zip(metrics, labels, metrics_std):
        epochs = [e for e in range(len(metric))]
        ax.plot(epochs, metric, label=label, linewidth=linewidth)
        fill_lo = np.array(metric) - np.array(std)
        fill_hi = np.array(metric) + np.array(std)
        ax.fill_between(epochs, fill_lo, fill_hi, alpha=0.1)

    apply_plot_settings(fig, ax,
                        xlim_top=xlim_top,
                        xlim_bottom=xlim_bottom,
                        ylim_top=ylim_top,
                        ylim_bottom=ylim_bottom,
                        logx=logx,
                        logy=logy,
                        dir_path=dir_path,
                        file_name=file_name,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=title,
                        title_fontsize=title_fontsize,
                        axis_fontsize=axis_fontsize,
                        xtick_size=xtick_size,
                        ytick_size=ytick_size,
                        show_legend=show_legend,
                        legend_fontsize=legend_fontsize,
                        legend_title=legend_title,
                        show=show)

    # plt.legend(fontsize=20)
    # plt.xlabel('Epochs', fontsize=30)
    # plt.tick_params(axis='x', labelsize=20)
    # plt.ylabel(ylabel, fontsize=30)
    # plt.tick_params(axis='y', labelsize=20)
    #
    # if ylim_top is not None:
    #     plt.ylim(top=ylim_top)
    # if ylim_bottom is not None:
    #     plt.ylim(bottom=ylim_bottom)
    #
    # # plt.tight_layout()
    #
    # if len(file_name) < 4 or file_name[-4:] != '.png':
    #     file_name += '.png'
    #
    # if dir_path is not None:
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
    #     file = os.path.join(dir_path, f'{file_name}')
    # else:
    #     file = file_name
    # plt.savefig(file)
    # plt.close('all')


def main():
    # Testing
    # x_vals = np.array([0, 0, 3, 4, 5, 1, 2, 10, 100])
    # y_vals = np.array([1, 2, 3, 4, 5, 5, 4, 2, 1])

    x_vals = [[1, 2, 3], [4, 5, 6]]
    metrics = [[1, 2, 3], [7, 8, 9]]
    metrics_std = [[1, 2, 3], [7, 8, 9]]
    labels = ['a', 'b']
    save_epochs_curve('./', metrics, labels, metrics_std=metrics_std,
                      logx=False,
                      logy=False,
                      file_name='epochs',
                      xlabel='xlabel',
                      ylabel='ylabel',
                      title=None,
                      title_fontsize=20,
                      axis_fontsize=20,
                      xtick_size=20,
                      ytick_size=20,
                      show_legend=True,
                      legend_fontsize=20,
                      legend_title=None,
                      )


if __name__ == '__main__':
    main()
