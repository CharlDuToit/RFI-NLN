import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .common import apply_plot_settings


def save_bar(df, column_name, group_1, group_2, group_3=None,
             bar_width=0.3,
             show_legend=True,
             annotate_group_2=True,
             annotate_group_2_mean=True,
             show_group_2_legend=True,
             show_group_3_legend=True,
             mean_fontsize=52,
             annotate_fontsize=52,
             figsize=(20,15),
             legend_fontsize=52,
             legendspacing=0,
             legend_borderpad=0,
             dir_path='./',
             file_name='test',
             show=False,
             ylim_top=None,
             ylim_bottom=None,
             axis_fontsize=52,
             title=None,
             ylabel=None,
             title_fontsize=52,
             ytick_size=52,
             layout_rect=None,
             plot_min_max=True,
             color_legend_bbox=None,
             hatch_legend_bbox=None,
             label_1_ytext_factor=-2,
             **kwargs
             ):
    fig, ax = plt.subplots(figsize=figsize)

    group_1_labels = df[group_1].unique()  # Separated by gaps between collection of bars
    group_2_labels = df[group_2].unique()  # Unique color
    group_3_labels = df[group_3].unique() if group_3 is not None else ['nothing'] # Unique hatching

    colors = plt.cm.get_cmap('tab10').colors  # Use tab10 colormap for colors
    hatches = [None, '//', '\\\\', '||', '---']

    label_2_color_dict = {label_2: colors[i] for i, label_2 in enumerate(group_2_labels)}
    label_3_hatch_dict = {label_3: hatches[i] for i, label_3 in enumerate(group_3_labels)}

    if show_legend:
        # -------------- label 2 color legend
        if len(group_2_labels) > 1 and show_group_2_legend:
            color_legend_handles = []
            # tit = legend_titles[0] if legend_titles and len(legend_titles) >= 1 else groupby[0]
            for k in label_2_color_dict.keys():
                color_legend_handles.append(
                    #ax.bar([], [], color=label_2_color_dict[k], label=k))
                    ax.bar([-100], [0], color=label_2_color_dict[k], label=k))

            color_legend = ax.legend(handles=color_legend_handles, title=None, loc='upper right',
                                     bbox_to_anchor=color_legend_bbox,
                                     fontsize=legend_fontsize,
                                     labelspacing=legendspacing,
                                     # title_fontsize=legend_fontsize,
                                     borderpad=legend_borderpad)
            ax.add_artist(color_legend)
        # -------------- label 3 hatch legend
        if len(group_3_labels) > 1 and show_group_3_legend:
            hatch_legend_handles = []
            for k in label_3_hatch_dict.keys():
                hatch_legend_handles.append(
                    ax.bar([-100], [0], hatch=label_3_hatch_dict[k], label=k, edgecolor='white', color='black'))

            hatch_legend = ax.legend(handles=hatch_legend_handles, title=None, loc='center right',
                                     bbox_to_anchor=hatch_legend_bbox,
                                     fontsize=legend_fontsize,
                                     labelspacing=legendspacing,
                                     # title_fontsize=legend_fontsize,
                                     borderpad=legend_borderpad)
            ax.add_artist(hatch_legend)
    ax.legend=None
    smallest_y = df[column_name].min() if ylim_bottom is None else ylim_bottom
    x = 0
    for i, label_1 in enumerate(group_1_labels):
        dfq = df[df[group_1] == label_1]
        x_1_start = x
        for label_2 in group_2_labels:
            dfq_2 = dfq[dfq[group_2] == label_2]
            if len(dfq_2) == 0: continue

            for label_3 in group_3_labels:
                if len(group_3_labels) > 1:
                    dfq_3 = dfq_2[dfq_2[group_3] == label_3]
                    if len(dfq_3) == 0: continue
                else:
                    dfq_3 = dfq_2

                mean = dfq_3[column_name].mean()
                min = dfq_3[column_name].min()
                max = dfq_3[column_name].max()
                color=label_2_color_dict[label_2]
                hatch=label_3_hatch_dict[label_3]
                ax.bar(x, mean, align='edge', width=bar_width, color=color, # label=label_2,
                             hatch=hatch, edgecolor='white')
                if plot_min_max:
                    ax.plot([x, x+bar_width], [max, max], linewidth=1, color='black')
                    ax.plot([x, x+bar_width], [min, min], linewidth=1, color='black')
                    ax.plot([x+bar_width/2, x+bar_width/2], [min, max], linewidth=1, color='black')
                if annotate_group_2_mean:
                    ax.annotate(format(mean, '.2f'),
                                (x+bar_width/2, smallest_y),  # Position for bar value
                                fontsize=mean_fontsize,
                                ha='center', va='bottom',
                                xytext=(0, 10),
                                textcoords='offset points')
                if annotate_group_2:
                    ax.annotate(label_2,
                                (x+bar_width/2, smallest_y),  # Position for bar value
                                fontsize=annotate_fontsize,
                                ha='center', va='bottom',
                                xytext=(0, -annotate_fontsize),
                                textcoords='offset points')
                x += bar_width
            # -- for label_3 in group_3_labels:
        # -- for label_2 in group_2_labels
        x_1_end = x
        trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        ax.annotate(label_1,
                    xy = ((x_1_start + x_1_end) / 2, smallest_y),  # Position for bar value
                    # annotation_clip=True,
                    fontsize=annotate_fontsize,
                    ha='center', va='bottom',
                    xytext=(0, label_1_ytext_factor*annotate_fontsize),
                    xycoords=trans,
                    textcoords='offset points')

        x = x + bar_width

    ax.set_xlim(left= -bar_width/2)
    ax.set_xlim(right= x)
    ax.set_xticks([])  # Remove x-axis tick labels

    apply_plot_settings(fig, ax,
                        ylim_top=ylim_top,
                        ylim_bottom=ylim_bottom,
                        dir_path=dir_path,
                        file_name=file_name,
                        # xlabel=xlabel,
                        ylabel=ylabel if ylabel else column_name,
                        title=title,
                        title_fontsize=title_fontsize,
                        axis_fontsize=axis_fontsize,
                        ytick_size=ytick_size,
                        # show_legend=show_legend,
                        legend_fontsize=legend_fontsize,
                        layout_rect=layout_rect,
                        show=show)




if __name__ == '__main__':
    # Example usage
    data = {'group_1': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A', 'B', 'C']*2,
            'group_2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']*2,
            'group_3': ['0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1']*2,
            'value': np.random.random((24,))}

    #'value': [  10,   15,   8,   12, 5,   7,   13,  9,   4,   2,   1,   6]*2}

    # 'value': np.random.random((12,))}

    np.random.random((10,1))

    df = pd.DataFrame(data)
    # create_grouped_bar_plot(df, 'value', 'group_1', 'group_2')
    save_bar(df, 'value', 'group_1', 'group_2', 'group_3',
             annotate_group_2_mean=False,
             annotate_group_2=False,
             layout_rect=(0.09, 0.02, 0.98, 0.98),
             show_group_2_legend=True,
             )
