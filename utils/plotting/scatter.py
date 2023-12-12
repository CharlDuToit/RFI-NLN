import numpy as np
import matplotlib.pyplot as plt

# from common import apply_plot_settings
from .common import apply_plot_settings
###from utils import apply_plot_settings

def save_scatter(list_of_list_of_x_scatter,
                 list_of_list_of_y_scatter,
                 list_of_list_of_x_lines=None,
                 list_of_list_of_y_lines=None,
                 labels_scatter=None,
                 size=10,
                 layout_rect=None,
                 figsize=(20,15),
                 linewidth=4,
                 logx=False,
                 logy=False,
                 dir_path='./',
                 file_name='scatter',
                 xlabel='x-axis',
                 ylabel='y-axis',
                 title=None,
                 axis_fontsize=55,
                 xtick_size=55,
                 ytick_size=55,
                 legend_fontsize=55,

                 show_legend=False,
                 legendspacing=None,
                 legend_borderpad=None,
                 legend_title=None,
                 legend_bbox=None,
                 legend_loc=None,

                 title_fontsize=55,
                 xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None,
                 show=False, **kwargs):

    fig, ax = plt.subplots(figsize=figsize)
    # ax.scatter(list_of_list_of_x, list_of_list_of_y, s=size)

    if not isinstance(list_of_list_of_x_scatter[0], (list, np.ndarray)):
        list_of_list_of_x_scatter = [list_of_list_of_x_scatter]
    if not isinstance(list_of_list_of_y_scatter[0], (list, np.ndarray)):
        list_of_list_of_y_scatter = [list_of_list_of_y_scatter]
    if not isinstance(labels_scatter, (list, tuple)):
        labels_scatter = [labels_scatter] * len(list_of_list_of_x_scatter)

    # show_legend = False
    for x_vals, y_vals, label in zip(list_of_list_of_x_scatter, list_of_list_of_y_scatter, labels_scatter):
        ax.scatter(x_vals, y_vals, s=size, label=label)
        # show_legend = show_legend or label
    if list_of_list_of_x_lines is not None and list_of_list_of_y_lines is not None:
        for x_vals, y_vals in zip(list_of_list_of_x_lines, list_of_list_of_y_lines):
            ax.plot(x_vals, y_vals, c='black', linewidth=linewidth, label=None)


    apply_plot_settings(fig, ax,
                        layout_rect=layout_rect,
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
                        legendspacing=legendspacing,
                        legend_loc=legend_loc,
                        legend_borderpad=legend_borderpad,
                        legend_bbox=legend_bbox,
                        show=show)


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





