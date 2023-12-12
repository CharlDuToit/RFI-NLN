import numpy as np
import matplotlib.pyplot as plt
# import cv2

from .common import apply_plot_settings
import matplotlib.colors as colors



# ---------------------------
# x = uvd_sun.freq_array/1e6
# y = uvd_sun.get_lsts(antpairpol)
# X, Y = np.meshgrid(x, y)
# Z = uvd_sun.get_data(antpairpol)
# Zabs = np.abs(Z)
#
# figsize=(6, 3.5)
# dpi=200
# title=None
# fig, (ax1, ax2) = plt.subplots(
#         nrows=2,
#         ncols=1,
#         figsize=None,
#         dpi=dpi,
#         sharex=True,
#         sharey=True
#     )
# im1 = ax1.pcolormesh(X, Y, Zabs,norm=colors.LogNorm(vmin=Zabs.min()+0.1, vmax=Zabs.max()))
# fig.colorbar(im1,ax=ax1, orientation='vertical',label='abs(V(t,v))')
# ax1.set_title('contourf with levels')
# ax1.set_xlabel('lol')
#
#
# im2 = ax2.pcolormesh(X, Y, np.angle(Z),cmap='twilight')
# fig.colorbar(im2,ax=ax2, orientation='vertical',label='angle(V(t,v))')
# ax2.tick_params(axis='x', labelsize=6)
# ax2.set_xlabel('Frequency [MHz]')
# #ax2.tick_params(axis='x', labelsize='0')
# ax2.axes.yaxis.set_ticklabels([])
# ax2.axes.xaxis.set_ticklabels([])

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

# -----------------------------
def save_confusion_image(data, mask, mask_pred,
                         xlabel='Frequency Bins',
                         ylabel='Time Bins',
                         # hor_vals=None,
                         # ver_vals=None,
                         x_range=None,
                         y_range=None,
                         n_xticks=4,
                         n_yticks=4,
                         log=True,
                         integer_ticks=True,
                         black_TN=True,
                         show_ticks=True,
                         show_xtick_labels=True,
                         show_ytick_labels=True,

                         figsize=(10,20),
                         layout_rect=None,
                         dir_path='./',
                         file_name='test',
                         title=None,
                         title_fontsize=25,
                         axis_fontsize=25,
                         xtick_size=25,
                         ytick_size=25,
                         show=False,
                         show_legend=True,
                         legend_fontsize=25,
                         legendspacing=None,
                         legend_borderpad=None,
                         legend_bbox=None,
                         ):

    fig, ax = plt.subplots(figsize=figsize)
    # ---------------------------------------------------------------------
    # Calculate True Positive (TP), False Negative (FN), False Positive (FP), and True Negative (TN)
    TP = np.logical_and(mask, mask_pred)
    FN = np.logical_and(mask, np.logical_not(mask_pred))
    FP = np.logical_and(np.logical_not(mask), mask_pred)
    TN = np.logical_and(np.logical_not(mask), np.logical_not(mask_pred))

    recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))
    precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    F1 = 2 * recall * precision / (recall + precision)
    print(dict(recall=recall, FPR=FPR, precision=precision, F1=F1))


    # ---------------------------------------------------------------------
    # Create a color-coded image based on the original x
    cmap = plt.cm.jet
    cmap = plt.cm.viridis
    if log:
        norm = colors.LogNorm(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))
    x_color = cmap(norm(data))

    # ---------------------------------------------------------------------
    col=dict(
        TP=[0, 1, 0, 1],  # Green
        FN=[1, 0, 0, 1],  # Red
        FP=[0, 0, 1, 1],  # Blue
        # TN=[0, 0, 0, 1],  # Black
        TN=[0, 0, 0, 0],  # White

    )
    # Apply color coding for different categories
    x_color[TP] = col['TP']   # Green for TP
    x_color[FN] = col['FN']   # Red for FN
    x_color[FP] = col['FP']  # Blue for FP
    if black_TN:
        x_color[TN] = col['TN']
    else:
        x_color[TN] = cmap(norm(data[TN]))  # Original color for TN

    # ---------------------------------------------------------------------
    # Legend
    if show_legend:
        s = 30 * 3 * legend_fontsize/25
        color_legend_handles = []
        color_legend_handles.append( ax.scatter([], [], marker='s', s=s, color=col['TP'], label='TP') )
        color_legend_handles.append( ax.scatter([], [], marker='s', s=s, color=col['FN'], label='FN') )
        color_legend_handles.append( ax.scatter([], [], marker='s', s=s, color=col['FP'], label='FP') )
        if black_TN:
            color_legend_handles.append(ax.scatter([], [], marker='s', s=s, color=col['TN'], label='TN'))

        color_legend = ax.legend(handles=color_legend_handles,
                                 #title=tit,
                                 #loc='center left',
                                 bbox_to_anchor=legend_bbox,
                                 fontsize=legend_fontsize,
                                 labelspacing=legendspacing,
                                 title_fontsize=legend_fontsize,
                                 borderpad=legend_borderpad)
        ax.add_artist(color_legend)

    # ---------------------------------------------------------------------
    # Create the final image
    ax.imshow(x_color, norm=norm,
              # aspect='equal'
              )
    # plt.colorbar()

    # ---------------------------------------------------------------------
    # x and y ticks
    x_min = x_range[0] if x_range else 0
    x_max = x_range[1] if x_range else data.shape[0] - 1
    y_min = y_range[0] if y_range else 0
    y_max = y_range[1] if y_range else data.shape[1] - 1

    ax.set_xticks([])
    ax.set_yticks([])

    if show_ticks:
        xticks=np.arange(0, data.shape[0] - 1, int((data.shape[0] - 1)/n_xticks))
        xlabels=[ signif(x_min + i*(x_max-x_min)/(len(xticks)-1), 4) for i in range(len(xticks))]
        if integer_ticks:
            xlabels = [int(lab) for lab in xlabels]
        if show_xtick_labels:
            ax.set_xticks(ticks=xticks,
                          labels=xlabels
                          )
        else:
            ax.set_xticks(ticks=xticks,
                          labels=['' for x in xticks]
                          )


        yticks=np.arange(0, data.shape[1] - 1, int((data.shape[1] - 1)/n_yticks))
        ylabels=[ signif(y_min + i*(y_max-y_min)/(len(yticks)-1), 4) for i in range(len(yticks))]
        if integer_ticks:
            ylabels = [int(lab) for lab in ylabels]
        if show_ytick_labels:
            ax.set_yticks(ticks=yticks,
                          labels=ylabels
                          )
        else:
            ax.set_yticks(ticks=yticks,
                          labels=['' for y in yticks]
                          )


    # ---------------------------------------------------------------------
    apply_plot_settings(fig, ax,
                        dir_path=dir_path,
                        layout_rect=layout_rect,
                        file_name=file_name,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=title,
                        title_fontsize=title_fontsize,
                        axis_fontsize=axis_fontsize,
                        xtick_size=xtick_size,
                        ytick_size=ytick_size,
                        show=show)


def load_lofar_aof_test(index=0):
    from h5py import File
    hf = File('/home/ee487519/PycharmProjects/lofar_test_LTA/LOFAR_RFI_test_data_with_aoflags.h5', 'r')

    # hf['hand_labels']
    # hf['aoflags']

    return hf['test_data'][index, ..., 0], hf['hand_labels'][index, ..., 0], hf['aoflags'][index, ..., 0] > 31

if __name__ == '__main__':
    # Example usage

    # RFI-NLN: 18 highest, 47 lowest
    # self checking: 53 0.819, 44 0.0
    x, y_true, y_pred = load_lofar_aof_test(53)

    #cmap = plt.cm.jet
    # norm = colors.LogNorm(vmin=np.min(x), vmax=np.max(x))
    # plt.imshow(cmap(norm(x)))
    # plt.imshow(np.log(x))
    # plt.imshow(y_true)
    # plt.show()
    # exit()

    # x = np.random.rand(256, 256)
    # y_true = np.random.randint(0, 2, size=(256, 256))
    # y_pred = np.random.randint(0, 2, size=(256, 256))

    save_confusion_image(x, y_true, y_pred,
                         xlabel='Frequency Bins',
                         ylabel='Time Bins',
                         show=True,
                         log=True,
                         n_xticks=5,
                         n_yticks=5,
                         show_ticks=True,
                         show_ytick_labels=True,
                         black_TN=True,
                         legendspacing=0,
                         legend_borderpad=0,

                         legend_bbox=(1.2,0.5)
                         # x_range=(100,105),
                         # y_range=(-10,10)
                         )
