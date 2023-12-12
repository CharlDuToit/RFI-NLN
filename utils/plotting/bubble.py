import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from .common import apply_plot_settings
# from common import apply_plot_settings

def rescale(vals, new_min, new_max, num_points):
    """
    Returns [new_min] * num_points if:
      vals is None or
      new_max - new_min == 0.0 or
      range of vals == 0.0
    Otherwise rescales vals and returns vals and the scaling parameters
    """
    if vals is None:
        vals = [new_min] * num_points
        sub, mul, div = 0.0, 0.0, 0.0
    else:
        div = np.max(vals) - np.min(vals)
        sub = np.min(vals)
        mul = new_max - new_min
        if div == 0.0:
            vals = [new_min] * num_points
        elif mul == 0.0:
            vals = [new_min] * num_points
        else:
            vals = vals - sub
            vals = vals / div
            vals = vals * mul
            vals = vals + new_min
    if div == 0.0 or mul == 0.0:
        return vals, sub, 0.0
    return vals, sub, mul / div


def rescale_with_args(val, sub, mul, plus):
    if mul == 0.0:
        return plus
    return (val - sub) * mul + plus


def num2str(num):
    if 1e-2 < np.absolute(num) < 1e6:
        return str(np.round(num, 2))
    else:
        return "{:.2e}".format(num)


def get_labels(labels_, sizes_, label_on_point, size_on_point, num_points):
    """
    Will combines label and size into one string if label_on_point and size_on_point is True
    Will return list of empty labels_new if label_on_point is True
    Will return list of empty point_labels if label_on_point and size_on_point is False

    Parameters
    ----------
    labels_: labels given to save_scatter
    sizes_: sizes given to save_scatter
    label_on_point: Place label on markers ?
    size_on_point: Place size on markers ?
    num_points: Number of points. Used of sizes is None

    Returns
    -------
    labels_new: Labels to place on the legend
    point_labels: Labels to place on markers
    """
    if labels_ is not None:
        labels_new = copy.deepcopy(labels_)
    else:
        labels_new = [''] * num_points

    point_labels = [''] * num_points
    if label_on_point:
        point_labels = copy.deepcopy(labels_new)
        labels_new = [''] * num_points
    if size_on_point and sizes_ is not None:
        for i in range(num_points):
            size_str = num2str(sizes_[i])
            if point_labels[i] == '':
                point_labels[i] = size_str
            else:
                if len(point_labels[i]) > len(size_str):
                    spaces = (len(point_labels[i]) - len(size_str)) // 2
                    point_labels[i] = point_labels[i] + '\n' + ' ' * spaces + size_str
                else:
                    spaces = (len(size_str) - len(point_labels[i])) // 2
                    point_labels[i] = ' ' * spaces + point_labels[i] + '\n' + size_str
    return labels_new, point_labels


def save_bubble(x_vals, y_vals, labels=None, sizes=None, label_on_point=True, size_on_point=True, size_legend=True,
                point_label_size=20, size_min=100, size_max=550, legendspacing=2, figsize=(20,10),
                logx=False, logy=False, dir_path='./', file_name='bubble', xlabel='x-axis', ylabel='y-axis', title=None,
                legend_title=None, grid=True, xlim_top=None, xlim_bottom=None, ylim_top=None, ylim_bottom=None, show=False,
                axis_fontsize=20, xtick_size=20, ytick_size=20, legend_fontsize=20, title_fontsize=20,
                size_legend_title='Parameters', legend_size_labels=((5, 'Five'),(10, 'Ten')),
                layout_rect=None, legend_bbox=None, adjustment_set=1, **kwargs):
    """
    Rescales sizes to size_min and size_max.
    Combines labels and sizes to one string to place on marker, if so desired.
    Your custom size legend must be hardcoded below.

    Parameters
    ----------
    x_vals: List of float
    y_vals: List of float
    labels: List of str
    sizes: List of sizes
    label_on_point: Place label on markers ? if False then will create a label legend
    size_on_point: Place size on markers ?
    logx: x-axis logarithmic ?
    size_legend: Create legend for sizes ? Will have to tweak them in code below
    logy: y-axis logarithmic ?
    dir_path: path to save .png
    file_name: name of .png
    xlabel: x-axis label
    ylabel: y-axis label
    title: Title
    legend_title: Title of legend
    size_min: Minimum marker size
    size_max: Maximim marker size
    axis_fontsize: Axis font size
    xtick_size: x-axis tick size
    ytick_size: y-axis tick size
    point_label_size: Sizes of labels on markers
    legend_fontsize: Font size of legend

    Returns
    -------

    """
    plot_sizes, sub, mul = rescale(sizes, size_min, size_max, len(x_vals))
    labels_, point_labels = get_labels(labels, sizes, label_on_point, size_on_point, len(x_vals))

    fig, ax = plt.subplots(figsize=figsize)

    for x, y, label, point_label, s in zip(x_vals, y_vals, labels_, point_labels, plot_sizes):
        ax.scatter(x, y, s=s, label=label)
        if adjustment_set == 3:  # MNRAS
            if point_label == 'U':
                x = x * 0.95
                y = y + 0.004
            if point_label == 'RFI':
                x = x * 0.95
            if point_label == 'DUAL':
                y = y + 0.005
            if point_label == 'AC':
                y = y - 0.01
        if adjustment_set == 1: # LOFAR
            if point_label == 'U' and logx:  # TEMPORARY
                x = x * 0.95
            if point_label == 'RFI' and logx:  # TEMPORARY
                x = x * 0.94
            if point_label == 'RFI':
                if not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                    x = x - 0.005
                if not logx and (xlabel == 'Validation FPR' or xlabel == 'Test FPR'):  # TEMPORARY
                    x = x - 0.00003
            if point_label == 'R5' and ylabel != 'Test FPR': # TEMPORARY
                y = y - 0.0001
            if point_label == 'DUAL' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                x = x - 0.003
            if point_label == 'AC' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                x = x - 0.0028
            if point_label == 'ASPP' and not logx:  # TEMPORARY

                if xlabel == 'Validation AUROC':
                    x = x - 0.0083
                if xlabel == 'Validation AUPRC':
                    x = x - 0.011
                if ylabel == 'Test AUROC':
                    y = y + 0.001
        if adjustment_set == 2:
            if xlabel != 'Validation AUROC':
                if point_label == 'U' and logx:  # TEMPORARY
                    x = x * 0.95
                if point_label == 'U' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':
                    x = x - 0.0008
                if point_label == 'U':
                    y = y - 0.001
                if point_label == 'RFI' and logx:  # TEMPORARY
                    x = x * 0.94
                if point_label == 'RFI' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                    x = x -  0.001
                if point_label == 'R5' and ylabel != 'Validation FPR' and ylabel != 'Test FPR': # TEMPORARY
                    y = y + 0.0001
                    if logx:
                        x = x*0.9
                    elif xlabel != 'Validation FPR' and xlabel != 'Test FPR':
                        x = x - 0.0008
                if point_label == 'DSC_DUAL_RESUNET' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                    x = x - 0.01
                if point_label == 'AC' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                    x = x - 0.0008
                if point_label == 'AC' and ylabel != 'Validation FPR' and ylabel != 'Test FPR':
                    y = y - 0.0017
                if point_label == 'ASPP' and not logx and (xlabel == 'Validation FPR' or xlabel == 'Test FPR'):  # TEMPORARY
                    if xlabel == 'Test FPR':
                        x = x - 0.5e-5
                if point_label == 'ASPP' and not logx and xlabel != 'Validation FPR' and xlabel != 'Test FPR':  # TEMPORARY
                    if xlabel == 'Validation AUROC':
                        x = x - 0.0023
                    if xlabel == 'Validation AUPRC':
                        x = x + 0.0004
                        y = y - 0.0011
                    if ylabel == 'Test AUROC':
                        x = x + 0.0024
                        y = y - 0.0011
        ax.annotate(point_label, (x, y), fontsize=point_label_size)

    if labels is not None and label_on_point is False:
        legend = ax.legend(fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)
    elif size_legend is True and size_on_point is False:
        temp_fig, temp_ax = plt.subplots(figsize=(10, 10))
        handles = [
            temp_ax.scatter(0, 0, c='white', edgecolors='black', s=rescale_with_args(size_label[0], sub, mul, size_min), label=size_label[1])
            for size_label in legend_size_labels
        ]
        bbox = legend_bbox if legend_bbox else None

        ax.legend(handles=handles,labelspacing=legendspacing, title=size_legend_title, fontsize=legend_fontsize,
                  title_fontsize=legend_fontsize, bbox_to_anchor=bbox)


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
                        xlabel=xlabel,
                        ylabel=ylabel,
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
    # Testing
    # x_vals = np.array([0, 0, 3, 4, 5, 1, 2, 10, 100])
    # y_vals = np.array([1, 2, 3, 4, 5, 5, 4, 2, 1])

    x_vals = np.array([1, 2, 3, 4, 5, 1, 2, 4, 5])
    y_vals = np.array([1, 2, 3, 4, 5, 5, 4, 2, 1])
    sizes = np.array([1, 4, 6, 4, 3, 2, 6, 8, 11])
    labels = ['', 'aasdasdasasdasd', 'abc', 'def', 'abcde', 'andeag', '', 'a', 'abcdasdsegasw']

    save_bubble(x_vals, y_vals, labels=labels, sizes=sizes, label_on_point=True, size_on_point=False, size_legend=True,
                point_label_size=20, size_min=100, size_max=550,
                logx=True,
                logy=True,
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
                legend_title=None
                )

    # save_bubble(x_vals, y_vals, sizes=sizes, file_name='2')
    # save_bubble(x_vals, y_vals, sizes=sizes * 100, size_on_point=True, file_name='3')
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=True, labels=labels, file_name='4')
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=False, label_on_point=True, labels=labels, file_name='5')
    # save_bubble(x_vals, y_vals, size_on_point=True, label_on_point=True, file_name='6')
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=False, labels=labels, file_name='7')
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=False, labels=labels, file_name='8', legend_fontsize=10, point_label_size=10)
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=True, labels=labels,
    #             file_name='9', legend_fontsize=10, point_label_size=10)
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=True, labels=labels,
    #             file_name='10', legend_fontsize=10, point_label_size=10, xtick_size=0)
    # save_bubble(x_vals * 100000, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=True, labels=labels,
    #             file_name='11', legend_fontsize=10, point_label_size=10, logx=True)
    # save_bubble(x_vals * 100000, y_vals, sizes=sizes * 5e5, size_on_point=True, label_on_point=True, labels=labels,
    #             file_name='12', legend_fontsize=10, point_label_size=10, logx=False, xtick_size=10)
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=False, label_on_point=False, labels=labels, file_name='13', legend_fontsize=10, point_label_size=10)
    # save_bubble(x_vals, y_vals, sizes=sizes * 5e5, size_on_point=False, label_on_point=True, labels=labels, file_name='14', legend_fontsize=10, point_label_size=10)
