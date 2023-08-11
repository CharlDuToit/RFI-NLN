import matplotlib.pyplot as plt
import numpy as np
import os


def save_percentile(list_of_list_of_values, labels=None, scatter=False, logy=False, dir_path='./',
                    file_name='percentile_plot', scatter_size=10, xlabel='Percentiles', ylabel='y-axis', title=None,
                    axis_fontsize=20, tick_size=20, legend_size=20):

    fig, ax = plt.subplots(figsize=(20, 10))

    if not isinstance(list_of_list_of_values[0], (list, np.ndarray)):
        list_of_list_of_values = [list_of_list_of_values]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]*len(list_of_list_of_values)

    show_legend = False
    for values, label in zip(list_of_list_of_values, labels):
        n_points = 1000
        plot_x = 100*(np.arange(n_points)/n_points)
        plot_y = [np.percentile(values, p) for p in plot_x]

        ax.plot(plot_x, plot_y, label=label)
        if scatter:
            scatter_x = [100 * np.mean(values < x) for x in values]
            ax.scatter(scatter_x, values, s=scatter_size, label=label)
        show_legend = show_legend or label

    ax.grid(visible=True, which='both')
    if title is not None:
        ax.title(title, fontsize=axis_fontsize)
    if logy:
        ax.set_yscale('log')
    if show_legend:
        ax.legend(fontsize=legend_size)

    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.tick_params(axis='x', labelsize=tick_size, which='minor')
    ax.tick_params(axis='x', labelsize=tick_size, which='major')
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.tick_params(axis='y', labelsize=tick_size, which='minor')
    ax.tick_params(axis='y', labelsize=tick_size, which='major')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(file_name) < 4 or file_name[-4:] != '.png':
        file_name += '.png'
    file = os.path.join(dir_path, f'{file_name}')
    plt.savefig(file)
    plt.close('all')


if __name__ == '__main__':
    x_vals1 = np.random.random(10)
    x_vals2 = np.random.random(100)

    save_percentile( (x_vals1, x_vals2), scatter=False, labels=(1,2))
