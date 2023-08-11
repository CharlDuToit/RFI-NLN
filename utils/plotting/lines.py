import matplotlib.pyplot as plt
import numpy as np

#from common import apply_plot_settings
from .common import apply_plot_settings


def sort_by_x(x_vals, y_vals):
    sort_indexes = np.argsort(x_vals)
    return np.array(x_vals)[sort_indexes], np.array(y_vals)[sort_indexes]


def save_lines(list_of_list_of_x, list_of_list_of_y, list_of_list_of_sizes=None, scatter=False, labels=None, size=10, figsize=(10,10),
               logx=False, logy=False, dir_path='./',
               file_name='lines', xlabel='x-axis', ylabel='y-axis', title=None, axis_fontsize=30,
               xtick_size=20, ytick_size=20, legend_fontsize=20, title_fontsize=20, legend_title=None,
               xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None, show=False, linewidth=10,
               **kwargs):

    # fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # if not isinstance(list_of_list_of_sizes[0], list):
    #     list_of_list_of_sizes = [list_of_list_of_sizes]
    if not isinstance(list_of_list_of_x[0], list):
        list_of_list_of_x = [list_of_list_of_x]
    if not isinstance(list_of_list_of_y[0], list):
        list_of_list_of_y = [list_of_list_of_y]
    if not isinstance(labels, list):
        labels = [labels]*len(list_of_list_of_x)

    show_legend = False
    for x_vals, y_vals, label in zip(list_of_list_of_x, list_of_list_of_y, labels):
        x_vals, y_vals = sort_by_x(x_vals, y_vals)
        ax.plot(x_vals, y_vals, label=label, linewidth=linewidth)
        if scatter:
            ax.scatter(x_vals, y_vals, s=size)
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

    # if title is not None:
    #     plt.title(title, fontsize=axis_fontsize)
    # if logx:
    #     plt.xscale('log')
    # if logy:
    #     plt.yscale('log')
    # if show_legend:
    #     plt.legend(fontsize=legend_size)
    #
    # plt.xlabel(xlabel, fontsize=axis_fontsize)
    # plt.tick_params(axis='x', labelsize=tick_size)
    # plt.ylabel(ylabel, fontsize=axis_fontsize)
    # plt.tick_params(axis='y', labelsize=tick_size)
    # #plt.tight_layout()
    #
    # # if ylim_top is not None:
    # #     plt.ylim(top=ylim_top)
    # # if ylim_bottom is not None:
    # #     plt.ylim(bottom=ylim_bottom)
    #
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # file = os.path.join(dir_path, f'{file_name}.png')
    # plt.savefig(file)
    # plt.close('all')

if __name__ == '__main__':
    x1_vals = [5,4,3,2,1]
    y1_vals = [5,4,3,2,1]

    x2_vals = [1,2,3,4,5]
    y2_vals = [5,4,3,2,1]

    x3_vals = [6 ,8 ,10]
    y3_vals = [7, 9, 11]

    x = []
    x.append(x1_vals)
    x.append(x2_vals)
    x.append(x3_vals)

    y = []
    y.append(y1_vals)
    y.append(y2_vals)
    y.append(y3_vals)

    #save_lines(x, y, file_name='1')
    #save_lines(x, y, scatter=True, file_name='2')
    save_lines(x, y, labels=['a','b', 'c'], scatter=True,
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
               legend_title=None)


    #save_lines(x, y, labels='a', scatter=True, file_name='4')
    #save_lines(x1_vals, y1_vals, labels='', scatter=True, file_name='5')
