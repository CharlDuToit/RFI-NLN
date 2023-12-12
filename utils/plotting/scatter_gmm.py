import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from typing import Tuple, List

# from common import apply_plot_settings
from .common import apply_plot_settings


def save_scatter_gmm(df: pd.DataFrame, groupby: Tuple[str, ...], x_axis: str, y_axis: str, figsize=(20, 15), size=10,
                     hatch_ellipse: bool = False, line_ellipse: bool = False, means: bool = True, points: bool = True,
                     legend_titles: Tuple = None,
                     file_name='test', title=None, grid=True, xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None,
                     dir_path='./', show=False, show_legend=False, logx=False, logy=False, xlabel=None, ylabel=None,
                     axis_fontsize=55, xtick_size=55, ytick_size=55, legend_fontsize=55, title_fontsize=55,
                     linewidth=10, layout_rect=None, color_legend_bbox=None, line_legend_bbox=None,
                     mean_size_factor=3, marker_legend_bbox=None, legendspacing=None, legend_borderpad=None,
                     include_legend_titles=True,
                     **kwargs):
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
            bbox = color_legend_bbox # if color_legend_bbox else (1, 0.85)

            color_legend = ax.legend(handles=color_legend_handles, title=tit, loc='upper left',
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
            ax.scatter(x, y, color=colors[color_idx], label=group_keys[0], s=size,
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
            #if group_keys != 'aof':
            #    width = width / 3.5
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              edgecolor=edgecolor, linestyle=linestyle, linewidth=linewidth,
                              facecolor=facecolor, hatch=hatch_pattern)
            ax.add_patch(ellipse)

    #     plt.tight_layout(rect=(0.11, 0.11, 0.96, 0.98))
    # increase: more left space
    # increase: more bottom space
    # decrease: more right space
    # decreate: more top space
    apply_plot_settings(fig, ax,
                        grid=grid,
                        layout_rect=layout_rect,
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


if __name__ == '__main__':
    # Example DataFrame

    data = {
        'Group1': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'] * 4,
        'Group2': ['a', 'b', 'c', 'd', 'd', 'c', 'b', 'a'] * 4,
        'Group3': ['1', '2', '3', '4', '1', '2', '3', '4'] * 4,
        'X': np.random.random_integers(-100, 100, 8 * 4),
        'Y': np.random.random_integers(-100, 100, 8 * 4)
    }

    data = {
        'Group1': ['A', 'A', 'B', 'B'] * 4,
        'Group2': ['a', 'b', 'a', 'b'] * 4,
        'Group3': ['1', '2', '3', '4'] * 4,
        'X': np.random.random_integers(-100, 100, 4 * 4),
        'Y': np.random.random_integers(-100, 100, 4 * 4)
    }

    df = pd.DataFrame(data)

    save_scatter_gmm(df, ['Group1', 'Group2', 'Group3'], 'X', 'Y', size=50,
                     hatch_ellipse=False, line_ellipse=True, means=True, points=True,
                     legend_titles=('G1', 'G2', 'G3'),
                     file_name='test',
                     title='Test',
                     xlim_top=None,
                     xlim_bottom=None,
                     ylim_top=None,
                     ylim_bottom=None,
                     dir_path='./',
                     show=False,
                     show_legend=True,
                     logx=False,
                     logy=False,
                     xlabel='x-axis',
                     ylabel='y-axis',
                     axis_fontsize=20,
                     xtick_size=20,
                     ytick_size=20,
                     legend_fontsize=20,
                     title_fontsize=20)
