def preprocessing_query():
    # does not include: train_with_test, limit, use_hyp_data, filters, train_f1, lr
    return ['patch_x == 64', 'scale_per_image == False', 'clip_per_image == False', 'rescale == True', 'log==True',
            'bn_first == False', 'perc_min == 0.2', 'perc_max == 99.8', 'clipper == "perc"', 'shuffle_patches == True',
            'patches == True', 'epochs > 49', 'lr_lin_decay == 1.0']


def common_gmm_options(fontsize=60, **plot_kwargs):
    plot_options=dict(
        hatch_ellipse=False,
        line_ellipse=False,
        means=True,
        points=True,
        legend_titles=None,
        show_legend=True,
        show=False,
        size=120,
        linewidth=3,
        title_fontsize=fontsize,
        legend_fontsize=fontsize,
        axis_fontsize=fontsize,
        xtick_size=fontsize,
        ytick_size=fontsize,
        figsize=(20, 20),
        color_legend_bbox=(1, 0.85),
        mean_size_factor=3,
        layout_rect=(0.06, 0.06, 0.75, 0.95)
    )
    return {**plot_options, **plot_kwargs}


def common_recall_prec_options(fontsize=60, **plot_kwargs):
    plot_options=dict(
        data_subset='test',
        gmm_thresholds=(0.5,),
        scatter_thresholds=True,
        scatter_gmm_means = True,
        scatter_gmm_points = True,
        hatch_ellipse=False,
        line_ellipse=True,
        legend_titles=None,
        show_legend=True,
        xlabel='Recall',
        ylabel='Precision',
        show=False,
        size=120,
        linewidth=3,
        title_fontsize=fontsize,
        legend_fontsize=fontsize,
        axis_fontsize=fontsize,
        xtick_size=fontsize,
        ytick_size=fontsize,
        figsize=(20, 20),
        color_legend_bbox=(1, 0.85),
        mean_size_factor=3,
        layout_rect=(0.06, 0.06, 0.75, 0.95)
    )
    return {**plot_options, **plot_kwargs}

def common_fpr_tpr_options(fontsize=60, **plot_kwargs):
    plot_options=dict(
        data_subset='test',
        gmm_thresholds=(0.5,),
        scatter_thresholds=True,
        scatter_gmm_means = True,
        scatter_gmm_points = True,
        hatch_ellipse=False,
        line_ellipse=True,
        legend_titles=None,
        show_legend=True,
        xlabel='False Positve Ratio',
        ylabel='True Positive Ratio',
        show=False,
        size=120,
        linewidth=3,
        title_fontsize=fontsize,
        legend_fontsize=fontsize,
        axis_fontsize=fontsize,
        xtick_size=fontsize,
        ytick_size=fontsize,
        figsize=(20, 20),
        color_legend_bbox=(1, 0.85),
        mean_size_factor=3,
        layout_rect=(0.06, 0.06, 0.75, 0.95)
    )
    return {**plot_options, **plot_kwargs}


def common_line_options(fontsize=30, **plot_kwargs):
    plot_options=dict(
        title_fontsize=fontsize,
        axis_fontsize=fontsize,
        xtick_size=fontsize,
        ytick_size=fontsize,
        legend_fontsize=fontsize,
        show=False,
        figsize=(20, 20),
        size=100,
        linewidth=5,
        scatter=True,
    )
    return {**plot_options, **plot_kwargs}

def common_bubble_options(fontsize=30, **plot_kwargs):
    plot_options=dict(
        label_on_point=True,
        size_on_point=False,
        size_legend=True,
        size_min=600,
        size_max=9000,
        legendspacing=2,
        show=False,
        size_legend_title='   Parameters   ',
        legend_size_labels=((800e3, '800k'), (350e3, '370k'), (290e3, '290k'), (32e3, '32k'), (20e3, '20k') ),
        point_label_size=fontsize,
        title_fontsize=fontsize,
        legend_fontsize=fontsize,
        axis_fontsize=fontsize,
        xtick_size=fontsize,
        ytick_size=fontsize,
        figsize=(20, 20)
    )
    return {**plot_options, **plot_kwargs}

def common_plot_options():
    return dict(
        # common to all
        xlabel=None,
        ylabel=None,
        ylim_top=None,
        ylim_bottom=None,
        xlim_top=None,
        xlim_bottom=None,
        logx=False,
        logy=False,
        title=None,
        title_fontsize=30,
        axis_fontsize=30,
        xtick_size=30,
        ytick_size=30,
        show=False,
        legend_fontsize=30,
        dir_path='./',
        file_name='test',
        figsize=(20, 10),
        # not applicable to scatter gmm or bubble
        show_legend=True,
        legend_title=None,
        # bubble plot
        label_on_point=True,
        size_on_point=True,
        size_legend=True,
        point_label_size=20,
        size_min=100,
        size_max=550,
        legendspacing=2,
        # Line plot
        scatter=True,
        # Line, scatter, scatter_gmm plot
        size=10,
        # scatter_gmm
        hatch_ellipse=False,
        line_ellipse=False,
        means=True,
        points=True,
        legend_titles=None,
        # line, scatter_gmm, epochs
        linewidth=10
    )
