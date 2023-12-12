import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from typing import Tuple, List
import copy

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


def find_thresholds_and_averages(thr_lists, x_lists, y_lists):
    # Determine the list of thresholds with the most elements

    # x and/or y can have 1 extra values
    # thr_lists = [thr_list[:len(x_list)] for x_list, thr_list in zip(x_lists, thr_lists)
    min_lens = [min(len(x_list), len(y_list), len(thr_list)) for x_list, y_list, thr_list in zip(x_lists, thr_lists, y_lists)]
    thr_lists = [lst[0:le] for lst, le in zip(thr_lists, min_lens)]
    x_lists = [lst[0:le] for lst, le in zip(x_lists, min_lens)]
    y_lists = [lst[0:le] for lst, le in zip(y_lists, min_lens)]

    # Flatten
    flat_thr = np.array([value for sublist in thr_lists for value in sublist])
    flat_x = np.array([value for sublist in x_lists for value in sublist])
    flat_y = np.array([value for sublist in y_lists for value in sublist])

    #x_lists = [np.array(x) for x in x_lists]
    #y_lists = [np.array(y) for y in y_lists]
    assert len(flat_thr) == len(flat_x)

    # Each unique threshold, sorted
    thr_bins = np.sort(np.unique(flat_thr))
    # flattened_list = np.unique(sorted([value for sublist in thresholds for value in sublist]))
    num_bins = len(thr_bins)

    # Initialize counters for averaging
    x_bin_arrays = [0] * num_bins
    y_bin_arrays = [0] * num_bins
    # bin_counts = [0] * num_bins
    x_bin_means = [0] * num_bins
    y_bin_means = [0] * num_bins

    for i in range(num_bins):
        # indexes = [np.logical_and(a > 1, a <= 6)]
        indexes = flat_thr == thr_bins[i]
        # bin_counts[i] = np.sum(indexes)
        x_bin_arrays[i] = flat_x[indexes]
        y_bin_arrays[i] = flat_y[indexes]
        x_bin_means[i] = np.mean(x_bin_arrays[i] )
        y_bin_means[i] = np.mean(y_bin_arrays[i] )

    return thr_bins, x_bin_arrays, y_bin_arrays, x_bin_means, y_bin_means


def save_fpr_tpr_curve(df: pd.DataFrame,
                       groupby: Tuple[str, ...],
                       #x_axis: str,
                       #y_axis: str,
                       gmm_thresholds=(0.5,),
                       aof_score='None',
                       # f1_contours=(0.6, ),
                       data_subset='test',
                       val_and_test=False,
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
                       file_name='test', title=None,
                       grid=True,
                       xlim_top=1, xlim_bottom=0, ylim_top=1, ylim_bottom=0,
                       dir_path='./', show=False, show_legend=False, logx=False, logy=False, xlabel='False Positive Rate',
                       ylabel='True Positive Rate',
                       axis_fontsize=20, xtick_size=20, ytick_size=20, legend_fontsize=20, title_fontsize=20,
                       linewidth=10,
                       figsize=(10, 10), size=10,
                       legendspacing=None, legend_borderpad=None,
                       **kwargs):

    if not val_and_test:
        dfq = df.query(f'{data_subset}_recall_vals == {data_subset}_recall_vals')
        dfq = dfq.query(f'{data_subset}_prec_vals == {data_subset}_prec_vals')
        dfq = dfq.query(f'{data_subset}_prec_recall_thr_vals == {data_subset}_prec_recall_thr_vals')
    else:
        for ds in ('val', 'test'):
            dfq = df.query(f'{ds}_recall_vals == {ds}_recall_vals')
            dfq = dfq.query(f'{ds}_prec_vals == {ds}_prec_vals')
            dfq = dfq.query(f'{ds}_prec_recall_thr_vals == {ds}_prec_recall_thr_vals')

    # dfq = df.query(f'{data_subset}_recall_vals.notna()', engine="python")
    if len(dfq) == 0:
        print('df does not recall precision arrays in any rows')
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Copy dataframe and add new column to each, then merge
    if val_and_test:
        #groupby = tuple([g for g in groupby] + ['data_subset'])
        groupby = [g for g in groupby] + ['data_subset']
        dfq2 = copy.deepcopy(dfq)
        dfq['data_subset'] = 'val'
        dfq2['data_subset'] = 'test'
        dfq = pd.concat([dfq, dfq2], ignore_index=True)

    # Grouping the dataframe based on the provided column names
    grouped = dfq.groupby(groupby)

    # Define color, line type, and hatching styles
    colors = plt.cm.get_cmap('tab10').colors  # Use tab10 colormap for colors
    line_types = ['-', ':', '--', '-.']
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

        if len(groupby) >= 2:
        #if line_ellipse and len(groupby) >= 2:
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
        if not val_and_test:
            tpr_vals = group[f'{data_subset}_tpr_vals']
            fpr_vals = group[f'{data_subset}_fpr_vals']
            thr_vals = group[f'{data_subset}_fpr_tpr_thr_vals']
        else:
            ds = group_keys[-1]
            tpr_vals = group[f'{ds}_tpr_vals']
            fpr_vals = group[f'{ds}_fpr_vals']
            thr_vals = group[f'{ds}_fpr_tpr_thr_vals']


        thr_vals, tpr_vals, fpr_vals, mean_tpr_vals, mean_fpr_vals = find_thresholds_and_averages(
            [a for a in thr_vals],
            [a for a in tpr_vals],
            [a for a in fpr_vals],
        )

        # ---------------------------------------------------
        # Determine the index for color, line type, and hatch based on grouping elements
        color_idx = unique_colors[group_keys[0]] if len(groupby) > 1 else unique_colors[group_keys]
        color = colors[color_idx]

        line_type_idx = unique_lines[group_keys[1]] if len(groupby) > 1 else 0
        # line_type_idx = unique_lines[group_keys[1]] if line_ellipse and len(groupby) > 1 else 0
        linestyle = line_types[line_type_idx]

        marker_type_idx = unique_markers[group_keys[1]] if len(groupby) > 1 else 0
        marker = marker_types[marker_type_idx]

        hatch_idx = unique_hatch[group_keys[2]] if hatch_ellipse and len(groupby) > 2 else 0
        hatch_marker_idx = unique_hatch[group_keys[2]] if len(groupby) > 2 else 0

        # ---------------------------------------------------
        # Plot average curve
        ax.plot( mean_fpr_vals,mean_tpr_vals ,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,)
        # ---------------------------------------------------
        # Scatter individual thresholds
        if scatter_thresholds:
            ax.scatter(mean_fpr_vals, mean_tpr_vals,
                       color=color,
                       # label=group_keys[0],
                       s=size,
                       marker=marker,)
        # ---------------------------------------------------
        # AOFLAGGER
        if aof_score in ('LOFAR', 'HERA_CHARL'):
            if aof_score == 'LOFAR':
                tpr, fpr = 0.58019, 0.00352173
            if aof_score == 'HERA_CHARL':
                tpr, fpr = 0.8635519709326119, 0.0103579157538425

            ax.scatter(fpr, tpr,
                       color='black',
                       # label=group_keys[0],
                       s=size*mean_size_factor,
                       marker='o', )
            ax.annotate('AOF',
                        (fpr, tpr),  # Position for bar value
                        fontsize=legend_fontsize,
                        ha='center', va='center',
                        xytext=(0, 20),
                        textcoords='offset points')
            ax.plot( (xlim_bottom, xlim_top), (tpr, tpr), color='black', linewidth=linewidth, linestyle='--')
            ax.plot( (fpr, fpr), (ylim_bottom, ylim_top), color='black', linewidth=linewidth, linestyle='--')

        # ---------------------------------------------------
        # Fit Gaussian Mixture Model to the group at specified thresholds
        if scatter_gmm_means or scatter_gmm_points:
            for thr in gmm_thresholds:
                idx = np.argmin(np.abs(thr - thr_vals))
                gmm_tpr_vals = tpr_vals[idx]
                gmm_fpr_vals = fpr_vals[idx]
                mean_tpr = mean_tpr_vals[idx]
                mean_fpr= mean_fpr_vals[idx]

                # print(group_keys, 'thr=',thr, '#vals=', len(gmm_tpr_vals))
                # if group_keys == 'R5' and thr==1e-5: print('tpr=', mean_tpr, ' fpr=', mean_fpr)


                # Plotting the mean point with color and line type
                if scatter_gmm_means:
                    ax.scatter(mean_fpr, mean_tpr,
                               color=color,
                               # label=group_keys[0],
                               s=size * mean_size_factor,
                               # hatch=hatches[hatch_marker_idx],
                               marker=marker)

                # Plotting the individual row values with color
                if scatter_gmm_points:
                    ax.scatter(gmm_fpr_vals, gmm_tpr_vals,
                               color=color,
                               # label=group_keys[0],
                               s=size,
                               # hatch=hatches[hatch_marker_idx],
                               marker=marker,
                               )

                if len(gmm_tpr_vals) > 1 and line_ellipse:
                    gmm = GaussianMixture(n_components=1)
                    gmm.fit(np.vstack([gmm_fpr_vals, gmm_tpr_vals]).T)

                    # Get the mean and covariance of the fitted Gaussian
                    mean = gmm.means_[0]
                    cov = gmm.covariances_[0]

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

                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
                        width, height = 2 * np.sqrt(2 * eigenvalues)
                        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                          edgecolor=edgecolor, linestyle=linestyle, linewidth=linewidth,
                                          facecolor=facecolor, hatch=hatch_pattern)
                        ax.add_patch(ellipse)

    # plt.tight_layout(rect=(0.06, 0.6, 0.85, 0.95))
    # if layout_rect:
    #     plt.tight_layout(rect=layout_rect)
    # elif show_legend:
    #     # fontsize 30
    #     # plt.tight_layout(rect=(0.06, 0.06, 0.75, 0.95))
    #     # fontsize 60
    #     plt.tight_layout(rect=(0.08, 0.08, 0.7, 0.95))
    # else:
    #     plt.tight_layout(rect=(0.11, 0.11, 0.96, 0.98))
    # increase: more left space
    # increase: more bottom space
    # decrease: more right space
    # ax.grid()
    apply_plot_settings(fig, ax,
                        layout_rect=layout_rect,
                        grid=grid,
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
