import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def load_csv():
    """Loads every csv. Can filter it afterwards"""
    df_lofar_train = pd.read_csv('lofar_train_july.csv')
    df_lofar_train['dataset'] = 'LOFAR train'

    df_lofar_test = pd.read_csv('lofar_test_july.csv')
    df_lofar_test['dataset'] = 'LOFAR test'

    df_hera = pd.read_csv('HERA_CHARL.csv')
    df_hera['dataset'] = 'HERA'

    df_all = pd.concat([df_lofar_train, df_lofar_test, df_hera], ignore_index=True)
    df_all['rfi_mean_over_nonrfi_mean'] = df_all['rfi_mean'] / df_all['nonrfi_mean']
    return df_all


def main(
        y_axis_names=('rfi_overlap_ratio', 'rfi_ratio', 'nonrfi_perc90_minus_rfi_perc10', 'rfi_std_over_rfi_mean', 'rfi_mean_over_nonrfi_mean')
):
    df = load_csv()
    invert = False
    ylabel_dict = dict(rfi_overlap_ratio='RFI Overlap Ratio',
                       nonrfi_perc90_minus_rfi_perc10='Overlap Range',
                       rfi_ratio='RFI Ratio',
                       rfi_std_over_rfi_mean='std(RFI) / mean(RFI)',
                        rfi_mean_over_nonrfi_mean='mean(RFI) / mean(non-RFI)',
                       )
    filename_dict = dict(rfi_overlap_ratio='RFI Overlap Ratio',
                         nonrfi_perc90_minus_rfi_perc10='Overlap Range',
                         rfi_ratio='RFI Ratio',
                         rfi_std_over_rfi_mean='std(RFI) div mean(RFI)',
                         rfi_mean_over_nonrfi_mean = 'mean(RFI) div mean(non-RFI)',
                         )
    legend_loc_dict = dict(
        rfi_overlap_ratio=None, # 'lower right' if not invert else 'upper left',
        nonrfi_perc90_minus_rfi_perc10='lower right',
        rfi_ratio='upper left',
        rfi_mean_over_nonrfi_mean='upper left',
        rfi_std_over_rfi_mean=None)

    if not isinstance(y_axis_names, (list, np.ndarray, tuple)):
        y_axis_names = tuple([y_axis_names])

    y_axis_names = ['rfi_mean_over_nonrfi_mean']

    # ------------------ Percentile plots
    # for y_axis in y_axis_names:
    #     hera = list(df.query('dataset == "HERA"')[y_axis])
    #     lofar_train = list(df.query('dataset == "LOFAR train"')[y_axis])
    #     lofar_test = list(df.query('dataset == "LOFAR test"')[y_axis])
    #
    #
    #     # save_percentile(
    #     #     [hera, lofar_test, lofar_train, ],
    #     #     labels=['HERA', 'LOFAR test', 'LOFAR train', ],
    #     #     layout_rect=(0.13, 0.09, 0.97, 0.99),
    #     #     ylabel=ylabel_dict[y_axis],
    #     #     show_legend=True,
    #     #     file_name=filename_dict[y_axis],
    #     #     legend_loc=legend_loc_dict[y_axis],
    #     #     logy=True if y_axis == 'rfi_ratio' else False,
    #     #     invert_axis=True if (y_axis == 'rfi_overlap_ratio' and invert) else False,
    #     #     xlim_top=100.0, xlim_bottom=-0.1,
    #     #     ylim_bottom= -0.1 if y_axis != 'rfi_ratio' else None,
    #     #     ylim_top=1.0 if y_axis == 'rfi_overlap_ratio' else (15.0 if y_axis == 'rfi_std_over_rfi_mean' else None)
    #     # )
    #
    #     plot_pdf(
    #         [hera, lofar_test, lofar_train, ],
    #         labels=['HERA', 'LOFAR test', 'LOFAR train', ],
    #         bins=25,
    #         linewidth=8,
    #         layout_rect=(0.13, 0.09, 0.96, 0.99),
    #         # ylabel=ylabel_dict[y_axis],
    #         xlabel=ylabel_dict[y_axis],
    #         ylabel='Probability Likelihood',
    #         show_legend=True,
    #         file_name='pdf ' + filename_dict[y_axis], # + 'inverted',
    #         legend_loc=legend_loc_dict[y_axis],
    #         logy=True if y_axis == 'rfi_ratio' else False,
    #         # invert_axis=True if (y_axis == 'rfi_overlap_ratio' and invert) else False,
    #         xlim_top =15 if y_axis == 'rfi_std_over_rfi_mean' else None,
    #        # xlim_top=100,
    #
    #         # invert_axis=True,
    #         # xlim_top=100.0, xlim_bottom=-0.1,
    #         # xlim_top=100.0, xlim_bottom=-0.1,
    #        # ylim_bottom= -0.1 if y_axis != 'rfi_ratio' else None,
    #       #  ylim_top=1.0 if y_axis == 'rfi_overlap_ratio' else (15.0 if y_axis == 'rfi_std_over_rfi_mean' else None)
    #     )

    # ------------------ Scatter plots

    save_scatter_gmm(df, groupby=['dataset'],
                     x_axis='rfi_overlap_ratio', y_axis='rfi_std_over_rfi_mean',
                     xlabel='RFI Overlap Ratio', ylabel='std(RFI) / mean(RFI)',
                     line_ellipse=True,
                     file_name='scatter_fs42',
                     size=25,
                     linewidth=5,
                     mean_size_factor=10,
                     show_legend=True,
                     logy=False,
                     logx=False,
                     axis_fontsize=42, xtick_size=42, ytick_size=42, legend_fontsize=42, title_fontsize=42,
                     include_legend_titles=False,
                     color_legend_bbox=(0.05, 0.85),
                     # layout_rect=(0.12, 0.09, 0.97, 0.99), # fontsize = 55
                     layout_rect=(0.09, 0.07, 0.97, 0.99),  # fontsize = 42
                     ylim_top=15, ylim_bottom=-0.1, xlim_top=1.0, xlim_bottom=-0.1
                     )




def save_percentile(list_of_list_of_values,
                    labels=None,
                    scatter=False,
                    size=10,
                    figsize=(20, 15),
                    invert_axis=False,

                    xlabel='Percentiles',
                    ylabel='y-axis',
                    show_legend=False,
                    legend_loc='lower right',
                    legendspacing=None,
                    legend_borderpad=None,
                    legend_title=None,
                    legend_bbox=None,
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
                    layout_rect=None,
                    dir_path='./',
                    file_name='percentile_plot',
                    ):
    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(list_of_list_of_values[0], (list, np.ndarray)):
        list_of_list_of_values = [list_of_list_of_values]
    if not isinstance(labels, (list, tuple)):
        labels = [labels] * len(list_of_list_of_values)

    # show_legend = False
    for values, label in zip(list_of_list_of_values, labels):
        n_points = 1000
        plot_x = 100 * (np.arange(n_points) / n_points)
        plot_y = [np.percentile(values, p) for p in plot_x]

        if invert_axis:
            ax.plot(plot_y, plot_x, label=label, linewidth=linewidth)
        else:
            ax.plot(plot_x, plot_y, label=label, linewidth=linewidth)

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
                        legend_loc=legend_loc,
                        legendspacing=legendspacing,
                        legend_borderpad=legend_borderpad,
                        legend_fontsize=legend_fontsize,
                        legend_title=legend_title,
                        legend_bbox=legend_bbox,
                        show=show,
                        layout_rect=layout_rect)


def save_scatter(list_of_list_of_x,
                 list_of_list_of_y,
                 labels=None,
                 size=10,
                 figsize=(20,15),
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

    if not isinstance(list_of_list_of_x[0], (list, np.ndarray)):
        list_of_list_of_x = [list_of_list_of_x]
    if not isinstance(list_of_list_of_y[0], (list, np.ndarray)):
        list_of_list_of_y = [list_of_list_of_y]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]*len(list_of_list_of_x)

    # show_legend = False
    for x_vals, y_vals, label in zip(list_of_list_of_x, list_of_list_of_y, labels):
        ax.scatter(x_vals, y_vals, s=size, label=label)
        # show_legend = show_legend or label

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
                        legendspacing=legendspacing,
                        legend_loc=legend_loc,
                        legend_borderpad=legend_borderpad,
                        legend_bbox=legend_bbox,
                        show=show)


def save_scatter_gmm(df: pd.DataFrame, groupby, x_axis: str, y_axis: str, figsize=(20, 15), size=10,
                     hatch_ellipse: bool = False, line_ellipse: bool = False, means: bool = True, points: bool = True,
                     legend_titles=None,
                     file_name='test', title=None, xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None,
                     dir_path='./', show=False, show_legend=False, logx=False, logy=False, xlabel=None, ylabel=None,
                     axis_fontsize=55, xtick_size=55, ytick_size=55, legend_fontsize=55, title_fontsize=55,
                     linewidth=10, layout_rect=None, color_legend_bbox=None, line_legend_bbox=None,
                     mean_size_factor=3, marker_legend_bbox=None, legendspacing=None, legend_borderpad=None,
                     include_legend_titles=True,
                     **kwargs):
    from sklearn.mixture import GaussianMixture
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=figsize)

    # Grouping the dataframe based on the provided column names
    grouped = df.groupby(groupby)

    # Define color, line type, and hatching styles
    colors = plt.cm.get_cmap('tab10').colors  # Use tab10 colormap for colors
    line_types = ['-', '--', ':', '-.']
    marker_types = ['o', 'v', 's', '*']
    hatches = ['//', '\\\\', '||', '---']
    # https://matplotlib.org/stable/api/markers_api.html

    # unique_colors has the column value as key and the color index as the value
    if len(groupby) >= 2:
        unique_colors = np.unique([group_keys[0] for i, group_keys in enumerate(grouped.groups.keys())])
    else:
        unique_colors = np.unique([group_keys for i, group_keys in enumerate(grouped.groups.keys())])
    unique_colors = {val: i % len(colors) for i, val in enumerate(unique_colors)}

    if len(groupby) >= 2:
        unique_lines = np.unique([group_keys[1] for i, group_keys in enumerate(grouped.groups.keys())])
        unique_lines = {val: i % len(line_types) for i, val in enumerate(unique_lines)}
        unique_markers = np.unique([group_keys[1] for i, group_keys in enumerate(grouped.groups.keys())])
        unique_markers = {val: i % len(marker_types) for i, val in enumerate(unique_markers)}

    if len(groupby) >= 3:
        unique_hatch = np.unique([group_keys[2] for i, group_keys in enumerate(grouped.groups.keys())])
        unique_hatch = {val: i % len(hatches) for i, val in enumerate(unique_hatch)}

    # mean_size_factor = 3

    # Create legends for colors, line types, and hatching
    if show_legend:
        if len(groupby) >= 1:
            color_legend_handles = []
            if include_legend_titles:
                tit = legend_titles[0] if legend_titles and len(legend_titles) >= 1 else groupby[0]
            else:
                tit = None
            for k in unique_colors.keys():
                color_legend_handles.append(
                    ax.scatter([], [], s=size * mean_size_factor, color=colors[unique_colors[k]], label=k))
            # Fontsize 30
            #color_legend = ax.legend(handles=color_legend_handles, title=tit, loc='center left',
            #                         bbox_to_anchor=(1, 0.85), fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            # Fontsize 60 L1: figsize (20,20) all models, each loss, all reg
            bbox = color_legend_bbox if color_legend_bbox else (1, 0.85)

            color_legend = ax.legend(handles=color_legend_handles, title=tit, loc='center left',
                                     bbox_to_anchor=bbox, fontsize=legend_fontsize,
                                     labelspacing=legendspacing,
                                     title_fontsize=legend_fontsize,
                                     borderpad=legend_borderpad)
            # color_legend = ax.legend(handles=color_legend_handles, title=tit, loc='center left',
            #                           fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            ax.add_artist(color_legend)

        if line_ellipse and len(groupby) >= 2:
            line_legend_handles = []
            if include_legend_titles:
                tit = legend_titles[1] if legend_titles and len(legend_titles) >= 2 else groupby[1]
            else:
                tit = None
            for k in unique_lines.keys():
                line_legend_handles.append(
                    ax.plot([], [], color='black', linewidth=linewidth, linestyle=line_types[unique_lines[k]], label=k)[
                        0])
            bbox = line_legend_bbox if line_legend_bbox else (1, 0.55)
            line_legend = ax.legend(handles=line_legend_handles, title=tit, loc='center left',
                                    labelspacing=legendspacing,
                                    borderpad=legend_borderpad,
                                    bbox_to_anchor=bbox, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            ax.add_artist(line_legend)

        if len(groupby) >= 2:
            marker_legend_handles = []
            if include_legend_titles:
                tit = legend_titles[1] if legend_titles and len(legend_titles) >= 2 else groupby[1]
            else:
                tit = None
            for k in unique_markers.keys():
                marker_legend_handles.append(
                    ax.scatter([], [], color='black', s=size * mean_size_factor, marker=marker_types[unique_markers[k]],
                               label=k))
            bbox = marker_legend_bbox if marker_legend_bbox else (1, 0.25)
            marker_legend = ax.legend(handles=marker_legend_handles, title=tit, loc='center left',
                                      bbox_to_anchor=bbox, fontsize=legend_fontsize,
                                      title_fontsize=legend_fontsize,
                                      labelspacing=legendspacing,
                                      borderpad=legend_borderpad)
            ax.add_artist(marker_legend)

        if len(groupby) >= 3:
            hatch_legend_handles = []
            if include_legend_titles:
                tit = legend_titles[2] if legend_titles and len(legend_titles) >= 3 else groupby[2]
            else:
                tit = None
            for k in unique_hatch.keys():
                hatch_legend_handles.append(ax.add_patch(
                    Ellipse((0, 0), 0, 0, edgecolor='black', facecolor='white', hatch=hatches[unique_hatch[k]],
                            label=k)))
            hatch_legend = ax.legend(handles=hatch_legend_handles, title=tit, loc='center left',
                                     bbox_to_anchor=(1, 0.00), fontsize=legend_fontsize,
                                     borderpad=legend_borderpad,
                                     title_fontsize=legend_fontsize)
            ax.add_artist(hatch_legend)

    # Iterate over each group
    for i, (group_keys, group) in enumerate(grouped):
        # Extract x and y values for the current group
        x = group[x_axis]
        y = group[y_axis]

        enough_points = len(group[x_axis]) > 1
        # Fit Gaussian Mixture Model to the group
        if enough_points:
            gmm = GaussianMixture(n_components=1)
            gmm.fit(group[[x_axis, y_axis]])

            # Get the mean and covariance of the fitted Gaussian
            mean = gmm.means_[0]
            cov = gmm.covariances_[0]

        # Determine the index for color, line type, and hatch based on grouping elements
        color_idx = unique_colors[group_keys[0]] if len(groupby) > 1 else unique_colors[group_keys]

        line_type_idx = unique_lines[group_keys[1]] if line_ellipse and len(groupby) > 1 else 0
        marker_type_idx = unique_markers[group_keys[1]] if len(groupby) > 1 else 0

        hatch_idx = unique_hatch[group_keys[2]] if hatch_ellipse and len(groupby) > 2 else 0
        hatch_marker_idx = unique_hatch[group_keys[2]] if len(groupby) > 2 else 0

        # Plotting the individual row values with color
        if points:
            # print(hatches[hatch_marker_idx])
            s = size/10 if group_keys == 'LOFAR train' else size
            # print(group_keys, s)
            ax.scatter(x, y, color=colors[color_idx], label=group_keys[0], s=s,
                       marker=marker_types[marker_type_idx], hatch=hatches[hatch_marker_idx])
            # ax.scatter(x, y, color=colors[color_idx], label=group_keys[0], s=size,
            #            marker=marker_types[marker_type_idx], hatch='|')

        # Plotting the mean point with color and line type
        if means and enough_points:
            # ax.plot(mean[0], mean[1], marker='o', markersize=10, color=colors[color_idx],
            #         linestyle=line_types[line_type_idx], label=f'{group_keys[0]} (Mean)')
            ax.scatter(mean[0], mean[1], color=colors[color_idx], label=group_keys[0], s=size * mean_size_factor,
                       marker=marker_types[marker_type_idx], hatch=hatches[hatch_marker_idx])

        # Plotting the ellipse with color and hatching
        if line_ellipse and enough_points:
            linestyle = line_types[line_type_idx]
            edgecolor = colors[color_idx]

            if hatch_ellipse:
                facecolor = 'none'
                hatch_pattern = hatches[hatch_idx]
            else:
                # facecolor = colors[color_idx]
                facecolor = 'none'
                hatch_pattern = None

            # if line_ellipse:
            #     linestyle = line_types[line_type_idx]
            # else:
            #     linestyle = '-'

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(2 * eigenvalues)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              edgecolor=edgecolor, linestyle=linestyle, linewidth=linewidth,
                              facecolor=facecolor, hatch=hatch_pattern)
            ax.add_patch(ellipse)

    # plt.tight_layout(rect=(0.06, 0.6, 0.85, 0.95))
    if layout_rect:
        plt.tight_layout(rect=layout_rect)
    elif show_legend:
        # fontsize 30
        # plt.tight_layout(rect=(0.06, 0.06, 0.75, 0.95))
        # fontsize 60
        plt.tight_layout(rect=(0.08, 0.08, 0.7, 0.95))
    else:
        plt.tight_layout(rect=(0.11, 0.11, 0.96, 0.98))
    # increase: more left space
    # increase: more bottom space
    # decrease: more right space
    apply_plot_settings(fig, ax,
                        xlim_top=xlim_top,
                        xlim_bottom=xlim_bottom,
                        ylim_top=ylim_top,
                        ylim_bottom=ylim_bottom,
                        logx=logx,
                        logy=logy,
                        dir_path=dir_path,
                        file_name=file_name,
                        xlabel=xlabel if xlabel else x_axis,
                        ylabel=ylabel if ylabel else y_axis,
                        title=title,
                        title_fontsize=title_fontsize,
                        axis_fontsize=axis_fontsize,
                        xtick_size=xtick_size,
                        ytick_size=ytick_size,
                        show_legend=False,
                        legend_fontsize=None,
                        legend_title=None,
                        show=show)


def plot_pdf(list_of_samples,
             labels,
             bins=100,
             invert_axis=False,

             linewidth=10,
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
            ax.plot(pdf, bin_edges[:-1], linewidth=linewidth, marker='o', linestyle='-', label=label)
        else:
            ax.plot(bin_edges[:-1], pdf, linewidth=linewidth, marker='o', linestyle='-', label=label)

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
def apply_plot_settings(fig, ax,
                        grid=False,
                        xlabel=None,
                        ylabel=None,
                        ylim_top=None,
                        ylim_bottom=None,
                        xlim_top=None,
                        xlim_bottom=None,
                        logx=False,
                        logy=False,
                        title=None,
                        title_fontsize=None,
                        axis_fontsize=None,
                        xtick_size=None,
                        ytick_size=None,
                        show=False,
                        show_legend=False,
                        legend_loc='lower right',
                        legend_fontsize=None,
                        legend_title=None,
                        legend_bbox=None,
                        legend_borderpad=None,
                        legendspacing=None,
                        layout_rect=None,
                        dir_path='./',
                        file_name='test', ):
    if layout_rect:
        fig.tight_layout(rect=layout_rect)
    # else:
    # fig.tight_layout()

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if grid:
        ax.grid(visible=True, which='both')
    if show_legend:
        ax.legend(title=legend_title,
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox,
                  fontsize=legend_fontsize,
                  labelspacing=legendspacing,
                  title_fontsize=legend_fontsize,
                  borderpad=legend_borderpad)
    if ylim_top:
        ax.set_ylim(top=ylim_top)
    if ylim_bottom:
        ax.set_ylim(bottom=ylim_bottom)
    if xlim_top:
        ax.set_xlim(right=xlim_top)
    if xlim_bottom:
        ax.set_xlim(left=xlim_bottom)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    if xtick_size:
        ax.tick_params(axis='x', labelsize=xtick_size, which='minor', length=xtick_size / 8, width=xtick_size / 16)
        ax.tick_params(axis='x', labelsize=xtick_size, which='major', length=xtick_size / 8, width=xtick_size / 16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    if ytick_size:
        ax.tick_params(axis='y', labelsize=ytick_size, which='minor', length=ytick_size / 8, width=ytick_size / 16)
        ax.tick_params(axis='y', labelsize=ytick_size, which='major', length=ytick_size / 8, width=ytick_size / 16)

    # print(ax.get_xlim())
    # print(ax.get_ylim())

    if show:
        # fig.show()
        plt.show()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(file_name) < 4 or file_name[-4:] != '.png':
        file_name += '.png'
    file = os.path.join(dir_path, f'{file_name}')

    fig.savefig(file)

    plt.close('all')


if __name__ == '__main__':
    main()
