import numpy as np
import matplotlib.pyplot as plt

# from common import apply_plot_settings
from .common import apply_plot_settings
###from utils import apply_plot_settings


def save_scatter(list_of_list_of_x, list_of_list_of_y, labels=None, size=10, figsize=(20,10),
                 logx=False, logy=False, dir_path='./',
                 file_name='scatter', xlabel='x-axis', ylabel='y-axis', title=None, axis_fontsize=20, xtick_size=20,
                 ytick_size=20, legend_fontsize=20, title_fontsize=20, legend_title=None,
                 xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None, show=False, **kwargs):
    """
    Every list in list_of_list_of_x has its own label


    Parameters
    ----------
    legend_size
    legend_title
    list_of_list_of_x: List of List of float
    list_of_list_of_y: List of List of float
    size: List of sizes
    logx: x-axis logarithmic ?
    logy: y-axis logarithmic ?
    dir_path: path to save .png
    file_name: name of .png
    xlabel: x-axis label
    ylabel: y-axis label
    title: Title
    axis_fontsize: Axis font size
    xtick_size: x-axis tick size
    ytick_size: y-axis tick size

    Returns
    -------

    """

    fig, ax = plt.subplots(figsize=figsize)
    # ax.scatter(list_of_list_of_x, list_of_list_of_y, s=size)

    if not isinstance(list_of_list_of_x[0], (list, np.ndarray)):
        list_of_list_of_x = [list_of_list_of_x]
    if not isinstance(list_of_list_of_y[0], (list, np.ndarray)):
        list_of_list_of_y = [list_of_list_of_y]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]*len(list_of_list_of_x)

    show_legend = False
    for x_vals, y_vals, label in zip(list_of_list_of_x, list_of_list_of_y, labels):
        ax.scatter(x_vals, y_vals, s=size, label=label)
        show_legend = show_legend or label

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

    # if show_legend:
    #     ax.legend(fontsize=legend_size, title=legend_title)
    #
    # if title is not None:
    #     ax.set_title(title, fontsize=axis_fontsize)
    # if logx:
    #     ax.set_xscale('log')
    # if logy:
    #     ax.set_yscale('log')
    #
    #
    # ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    # ax.tick_params(axis='x', labelsize=xtick_size, which='minor')
    # ax.tick_params(axis='x', labelsize=xtick_size, which='major')
    # ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    # ax.tick_params(axis='y', labelsize=ytick_size, which='minor')
    # ax.tick_params(axis='y', labelsize=ytick_size, which='major')
    #
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # if len(file_name) < 4 or file_name[-4:] != '.png':
    #     file_name += '.png'
    # file = os.path.join(dir_path, f'{file_name}')
    #
    # fig.savefig(file)
    #
    # plt.close('all')


def main():
    # Testing
    #x_vals = np.array([0, 0, 3, 4, 5, 1, 2, 10, 100])
    #y_vals = np.array([1, 2, 3, 4, 5, 5, 4, 2, 1])

    x_vals = [1, 2, 3, 4, 5, 1, 2, 4, 5]
    y_vals = [1, 2, 3, 4, 5, 5, 4, 2, 1]
    save_scatter(x_vals, y_vals, labels='lol')

    x_vals = [[1, 2, 3], [4, 5, 6]]
    y_vals = [[1, 2, 3], [7, 8, 9]]
    save_scatter(x_vals, y_vals, labels=['lol', 'jol'], size=500,
                 logx=False,
                 logy=False,
                 dir_path='./',
                 file_name='4',
                 xlabel='xlabel',
                 ylabel='ylabel',
                 title='title',
                 title_fontsize=20,
                 axis_fontsize=20,
                 xtick_size=20,
                 ytick_size=20,
                 legend_fontsize=20,
                 legend_title='leg'
                 )


if __name__ == '__main__':
    main()





