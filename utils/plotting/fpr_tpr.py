import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from typing import Tuple, List

# from common import apply_plot_settings
from .common import apply_plot_settings


# val or test?
# threshold values for ellipsis
# f1 values for contours

# def calculate_f1(recall, precision):
#     return 2 * (precision * recall) / (precision + recall)

def resample_linear_1d(original, target_len):
    original = np.array(original, dtype='float')
    index_arr = np.linspace(0, len(original) - 1, num=target_len, dtype='float')
    index_floor = np.array(index_arr, dtype='int') #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == target_len)
    return interp

def save_fpr_tpr_curve(df: pd.DataFrame,
                       groupby: Tuple[str, ...],
                       #x_axis: str,
                       #y_axis: str,
                       gmm_thresholds=(0.5,),
                       # f1_contours=(0.6, ),
                       data_subset='test',
                       scatter_thresholds=True,
                       color_legend_bbox=None,
                       line_legend_bbox=None,
                       marker_legend_bbox=None,
                       hatch_ellipse: bool = False,
                       line_ellipse: bool = False,
                       mean_size_factor=3,
                       scatter_gmm_means: bool = True,
                       scatter_gmm_points: bool = True,
                       legend_titles: Tuple = None,
                       layout_rect=None,
                       file_name='test', title=None, xlim_top=None, xlim_bottom=None, ylim_top=None,
                       ylim_bottom=None,
                       dir_path='./', show=False, show_legend=False, logx=False, logy=False, xlabel='False Positive Rate',
                       ylabel='True Positive Rate',
                       axis_fontsize=20, xtick_size=20, ytick_size=20, legend_fontsize=20, title_fontsize=20,
                       linewidth=10,
                       figsize=(10, 10), size=10,
                       legendspacing=None, legend_borderpad=None,
                       **kwargs):

    dfq = df.query(f'{data_subset}_tpr_vals == {data_subset}_tpr_vals')
    dfq = dfq.query(f'{data_subset}_fpr_vals == {data_subset}_fpr_vals')
    dfq = dfq.query(f'{data_subset}_fpr_tpr_thr_vals == {data_subset}_fpr_tpr_thr_vals')

    # dfq = df.query(f'{data_subset}_recall_vals.notna()', engine="python")
    if len(dfq) == 0:
        print('df does not recall precision arrays in any rows')
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Grouping the dataframe based on the provided column names
    grouped = dfq.groupby(groupby)

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

    # Countours
    # recall_range = np.linspace(0, 1, 100)
    # precision_range = np.linspace(0, 1, 100)
    # recall_mesh, precision_mesh = np.meshgrid(recall_range, precision_range)
    # f1_values = calculate_f1(recall_mesh, precision_mesh)
    # # Overlay contour lines for constant F1 values
    # contour_levels = np.linspace(0, 1, 10)  # Adjust the number of levels as needed
    # contour_levels = sorted(list(contour_levels) + list(f1_contours))
    # contour = ax.contour(recall_mesh, precision_mesh, f1_values, levels=contour_levels, colors='gray',
    #                      linestyles='dashed')
    # ax.clabel(contour, inline=True, fontsize=20)

    # Create legends for colors, line types, and hatching
    if show_legend:
        if len(groupby) >= 1:
            color_legend_handles = []
            tit = legend_titles[0] if legend_titles and len(legend_titles) >= 1 else groupby[0]
            for k in unique_colors.keys():
                color_legend_handles.append(
                    ax.scatter([], [], s=size * mean_size_factor, color=colors[unique_colors[k]], label=k))
            # Fontsize 30
            # color_legend = ax.legend(handles=color_legend_handles, title=tit, loc='center left',
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
            tit = legend_titles[1] if legend_titles and len(legend_titles) >= 2 else groupby[1]
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
            tit = legend_titles[1] if legend_titles and len(legend_titles) >= 2 else groupby[1]
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
            tit = legend_titles[2] if legend_titles and len(legend_titles) >= 3 else groupby[2]
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
        tpr_vals = group[f'{data_subset}_tpr_vals']
        fpr_vals = group[f'{data_subset}_fpr_vals']
        thr_vals = group[f'{data_subset}_fpr_tpr_thr_vals']

        #--------------------
        tpr_lens = [len(a) for a in tpr_vals]
        unique_lens = np.unique(tpr_lens)
        print('Recall Array Lengths for ', group_keys, ' : ', np.unique(tpr_lens))

        if len(unique_lens) == 1:
            tpr_vals = np.array([a for a in tpr_vals])
            fpr_vals = np.array([a for a in fpr_vals])
            thr_vals = np.array([a for a in thr_vals])
        else:
            first_thr = [a[0] for a in thr_vals]
            same_first_thr = np.all(np.isclose(first_thr, first_thr[0]))
            last_thr = [a[-1] for a in thr_vals]
            same_last_thr = np.all(np.isclose(last_thr, last_thr[0]))
            print('same_first_thr: ', same_first_thr, ', same_last_thr: ', same_last_thr)
            if same_first_thr and same_last_thr:
                new_length = np.max(unique_lens)
                print('Resampling to length: ', new_length)
                tpr_vals = np.array([resample_linear_1d(a, new_length) for a in tpr_vals])
                fpr_vals = np.array([resample_linear_1d(a, new_length)  for a in fpr_vals])
                thr_vals = np.array([resample_linear_1d(a, new_length)  for a in thr_vals])
            else:
                print('First and last threshold of different lengthed lists are not the same')
                return

        # --------------------

        mean_tpr_vals = tpr_vals.mean(axis=0)
        mean_fpr_vals = fpr_vals.mean(axis=0)
        mean_thr_vals = thr_vals.mean(axis=0)


        # Determine the index for color, line type, and hatch based on grouping elements
        color_idx = unique_colors[group_keys[0]] if len(groupby) > 1 else unique_colors[group_keys]
        color = colors[color_idx]

        line_type_idx = unique_lines[group_keys[1]] if line_ellipse and len(groupby) > 1 else 0
        linestyle = line_types[line_type_idx]

        marker_type_idx = unique_markers[group_keys[1]] if len(groupby) > 1 else 0
        marker = marker_types[marker_type_idx]

        hatch_idx = unique_hatch[group_keys[2]] if hatch_ellipse and len(groupby) > 2 else 0
        hatch_marker_idx = unique_hatch[group_keys[2]] if len(groupby) > 2 else 0

        # Plot average curve
        ax.plot( mean_fpr_vals,mean_tpr_vals ,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,)
        # Scatter individual thresholds
        if scatter_thresholds:
            ax.scatter(mean_fpr_vals, mean_tpr_vals,
                       color=color,
                       # label=group_keys[0],
                       s=size*0.25,
                       marker=marker,)

        enough_points = tpr_vals.shape[0] > 1
        # Fit Gaussian Mixture Model to the group at specified thresholds
        if enough_points and (scatter_gmm_means or scatter_gmm_points):
            for thr in gmm_thresholds:
                idx = np.argmin(np.abs(thr - mean_thr_vals))
                gmm_tpr_vals = tpr_vals[:, idx]
                gmm_fpr_vals = fpr_vals[:, idx]

                gmm = GaussianMixture(n_components=1)
                # gmm.fit(group[[x_axis, y_axis]])
                # gmm.fit([gmm_tpr_vals, gmm_prec_vals])
                gmm.fit(np.vstack([gmm_fpr_vals, gmm_tpr_vals]).T)


                # Get the mean and covariance of the fitted Gaussian
                mean = gmm.means_[0]
                cov = gmm.covariances_[0]

                # Plotting the mean point with color and line type
                if scatter_gmm_means:
                    # ax.plot(mean[0], mean[1], marker='o', markersize=10, color=colors[color_idx],
                    #         linestyle=line_types[line_type_idx], label=f'{group_keys[0]} (Mean)')
                    ax.scatter(mean[0], mean[1],
                               color=color,
                               # label=group_keys[0],
                               s=size * mean_size_factor,
                               # hatch=hatches[hatch_marker_idx],
                               marker=marker)


                # Plotting the individual row values with color
                if scatter_gmm_points:
                    # print(hatches[hatch_marker_idx])
                    ax.scatter(gmm_fpr_vals, gmm_tpr_vals,
                               color=color,
                               #label=group_keys[0],
                               s=size,
                               #hatch=hatches[hatch_marker_idx],
                               marker=marker,
                               )
                    # ax.scatter(x, y, color=colors[color_idx], label=group_keys[0], s=size,
                    #            marker=marker_types[marker_type_idx], hatch='|')


                # Plotting the ellipse with color and hatching
                if line_ellipse:
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
    ax.grid()
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
                        #xlabel=xlabel if xlabel else x_axis,
                        #ylabel=ylabel if ylabel else y_axis,
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

    save_fpr_tpr_curve(df, ['Group1', 'Group2', 'Group3'], 'X', 'Y', size=50,
                       hatch_ellipse=False, line_ellipse=True, scatter_gmm_means=True, scatter_gmm_points=True,
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
