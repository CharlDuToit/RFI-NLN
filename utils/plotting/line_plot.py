import matplotlib.pyplot as plt
import numpy as np
import os


def sort_by_x(x_vals, y_vals):
    sort_indexes = np.argsort(x_vals)
    return np.array(x_vals)[sort_indexes], np.array(y_vals)[sort_indexes]


def save_lines(list_of_list_of_x, list_of_list_of_y, scatter=False, labels=None, logx=False, logy=False, dir_path='./',
               file_name='line_plot', xlabel='x-axis', ylabel='y-axis', title=None, axis_fontsize=30, tick_size=20, legend_size=20):

    if not isinstance(list_of_list_of_x[0], list):
        list_of_list_of_x = [list_of_list_of_x]
    if not isinstance(list_of_list_of_y[0], list):
        list_of_list_of_y = [list_of_list_of_y]
    if not isinstance(labels, list):
        labels = [labels]*len(list_of_list_of_x)

    fig = plt.figure(figsize=(10, 10))

    show_legend = False
    for x_vals, y_vals, label in zip(list_of_list_of_x, list_of_list_of_y, labels):
        x_vals, y_vals = sort_by_x(x_vals, y_vals)
        plt.plot(x_vals, y_vals, label=label)
        if scatter:
            plt.scatter(x_vals, y_vals, s=10)
        show_legend = show_legend or label

    if title is not None:
        plt.title(title, fontsize=axis_fontsize)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if show_legend:
        plt.legend(fontsize=legend_size)

    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.tick_params(axis='x', labelsize=tick_size)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    plt.tick_params(axis='y', labelsize=tick_size)
    #plt.tight_layout()

    # if ylim_top is not None:
    #     plt.ylim(top=ylim_top)
    # if ylim_bottom is not None:
    #     plt.ylim(bottom=ylim_bottom)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file = os.path.join(dir_path, f'{file_name}.png')
    plt.savefig(file)
    plt.close('all')

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

    save_lines(x, y, file_name='1')
    save_lines(x, y, scatter=True, file_name='2')
    save_lines(x, y, labels=['a','b', 'c'], scatter=True, file_name='3')
    save_lines(x, y, labels='a', scatter=True, file_name='4')
    save_lines(x1_vals, y1_vals, labels='', scatter=True, file_name='5')
