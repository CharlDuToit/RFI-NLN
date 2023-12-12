import numpy as np
import matplotlib.pyplot as plt

from .common import apply_plot_settings

def plot_pdf(list_of_samples,
             labels,
             bins=100,
             invert_axis=False,

             show_legend=True,
             # annotate_fontsize=52,
             figsize=(20, 15),
             legend_fontsize=52,
             legendspacing=0,
             legend_borderpad=0,
             dir_path='./',
             file_name='pdf',
             show=False,
             legend_loc=None,
             xlim_top=None,
             xlim_bottom=None,
             ylim_top=None,
             ylim_bottom=None,
             axis_fontsize=55,
             title=None,
             xlabel=None,
             legend_title=None,
             legend_bbox=None,
             grid=True,
             ylabel=None,
             title_fontsize=55,
             ytick_size=55,
             xtick_size=55,
             layout_rect=None,
             **kwargs
             ):
    fig, ax = plt.subplots(figsize=figsize)

    for samples, label in zip(list_of_samples, labels):
    # Create a histogram of the samples
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)

        # Calculate bin widths
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Calculate PDF by normalizing the histogram
        pdf = hist / np.sum(hist * bin_widths)

        # Create a plot
        # ax.plot(bin_edges[:-1], pdf, marker='o', linestyle='-', label=label)

        if invert_axis:
            ax.plot(pdf, bin_edges[:-1], marker='o', linestyle='-', label=label)
        else:
            ax.plot(bin_edges[:-1], pdf, marker='o', linestyle='-', label=label)

        # if scatter:
        #     scatter_x = [100 * np.mean(values < x) for x in values]
        #     ax.scatter(scatter_x, values, s=size, label=label)
    if invert_axis:
        temp = xlabel
        xlabel = ylabel
        ylabel = temp

        temp = xlim_bottom
        xlim_bottom = ylim_bottom
        ylim_bottom = temp

        temp = xlim_top
        xlim_top = ylim_top
        ylim_top = temp

   #  plt.xlabel('X')
    # plt.ylabel('Probability Density')
    # plt.title(title)
    # plt.grid(True)

    apply_plot_settings(fig, ax,
                        grid=grid,
                        ylim_top=ylim_top,
                        xlim_top=xlim_top,
                        xlim_bottom=xlim_bottom,
                        ylim_bottom=ylim_bottom,
                        dir_path=dir_path,
                        file_name=file_name,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=title,
                        xtick_size=xtick_size,
                        legend_loc=legend_loc,
                        title_fontsize=title_fontsize,
                        axis_fontsize=axis_fontsize,
                        ytick_size=ytick_size,
                        show_legend=show_legend,
                        legend_fontsize=legend_fontsize,
                        legendspacing=legendspacing,
                        legend_borderpad=legend_borderpad,
                        legend_title=legend_title,
                        legend_bbox=legend_bbox,
                        layout_rect=layout_rect,
                        show=show)

    # Show the plot
    # plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate some random samples for demonstration
    np.random.seed(0)
    samples = np.random.normal(loc=0, scale=1, size=10000)

    samples_2 = np.random.normal(loc=-0.1, scale=0.8, size=10000)


    # Plot the PDF of the samples
    plot_pdf(
        [samples, samples_2],
        ['set 1', 'set 2'],
        bins=100,
        invert_axis=True
    )
