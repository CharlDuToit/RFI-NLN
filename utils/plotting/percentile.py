import matplotlib.pyplot as plt
import numpy as np
import os
from .common import apply_plot_settings


def save_percentile(list_of_list_of_values,
                    labels=None,
                    scatter=False,
                    size=10,
                    figsize=(20, 15),

                    xlabel='Percentiles',
                    ylabel='y-axis',
                    legendspacing=None,
                    legend_borderpad=None,
                    legend_title=None,
                    legend_bbox=None,
                    legend_loc=None,
                    title=None,
                    xtick_size=55,
                    ytick_size=55,
                    axis_fontsize=55,
                    legend_fontsize=55,
                    title_fontsize=55,
                    linewidth=5,
                    ylim_top=None,
                    ylim_bottom=None,
                    xlim_top=None,
                    xlim_bottom=None,
                    logx=False,
                    logy=False,
                    show=False,
                    show_legend=False,
                    layout_rect=None,
                    dir_path='./',
                    file_name='percentile_plot',
                    ):

    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(list_of_list_of_values[0], (list, np.ndarray)):
        list_of_list_of_values = [list_of_list_of_values]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]*len(list_of_list_of_values)

    # show_legend = False
    for values, label in zip(list_of_list_of_values, labels):
        n_points = 1000
        plot_x = 100*(np.arange(n_points)/n_points)
        plot_y = [np.percentile(values, p) for p in plot_x]

        ax.plot(plot_x, plot_y, label=label, linewidth=linewidth)
        if scatter:
            scatter_x = [100 * np.mean(values < x) for x in values]
            ax.scatter(scatter_x, values, s=size, label=label)
        # show_legend = show_legend or label

    ax.grid(visible=True, which='both')
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
                        legendspacing=legendspacing,
                        legend_loc=legend_loc,
                        legend_borderpad=legend_borderpad,
                        legend_fontsize=legend_fontsize,
                        legend_title=legend_title,
                        legend_bbox=legend_bbox,
                        show=show,
                        layout_rect=layout_rect)


if __name__ == '__main__':
    x_vals1 = np.random.random(10)
    x_vals2 = np.random.random(100)

    save_percentile( (x_vals1, x_vals2), scatter=False, labels=(1,2))
