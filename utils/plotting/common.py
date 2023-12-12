import matplotlib.pyplot as plt
import os


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

    if dir_path is None:
        file = file_name
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if len(file_name) < 4 or file_name[-4:] != '.png':
            file_name += '.png'
        file = os.path.join(dir_path, f'{file_name}')

    fig.savefig(file)

    plt.close('all')
