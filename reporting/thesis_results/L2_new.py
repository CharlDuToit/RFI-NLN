from args_constructor import preprocessing_query, common_gmm_options, common_bubble_options, common_fpr_tpr_options, common_recall_prec_options
from utils import ResultsCollection


# ======================= GMM PLOT =======================================


def gmm(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_L2_kwargs()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L2_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= BUBBLE PLOT =======================================


def bubble(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_L2_kwargs()
    kwargs['params'] = True
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L2_bubble_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_bubble_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L2_kwargs(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'L2_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy',task='eval_test', **plot_kwargs):
    kwargs = common_L2_kwargs(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'L2_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= TABLES =======================================

def table(table_fields, query_strings=[], label_format='empty', std=True, save_name='dummy'):
    kwargs = common_L2_kwargs()
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'L2_table_{save_name}'
    kwargs['table_fields'] = table_fields
    # kwargs['label_fields'] = ['model_class']
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

def val_test_table(table_fields, query_strings=[], groupby=['short_model_class'], label_fields=None, label_format='empty',
                   std=True,
                   save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_L2_kwargs()
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'val_test_table'
    kwargs['std'] = std
    kwargs['save_name'] = f'L2_valtest_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def common_L2_kwargs(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        # groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        # groupby=['model_class'],
        groupby=['short_model_class'],
        datasets=['LOFAR'],
        query_strings=preprocessing_query() + [f'use_hyp_data == False', f'filters == 16', f'lr==0.0001', #  f'task=="{task}"',
                                               f'train_f1 > {min_f1}', f'train_with_test == False', f'limit == 1493',
                                               'loss == "dice"', 'kernel_regularizer == "l2"', 'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['short_model_class'],
        # label_fields=['model_class'],
        # label_fields=['model_class', 'loss'],
        label_format='empty',
        # output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L2_new',
    )
    return kwargs

# ======================= PREC/RECALL TPR/FPR =======================================

# # zoomed
# # each model, dice loss
#  # yes
# fpr_tpr(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             aof_score='LOFAR',
#             data_subset='test',
#         xlabel='Test FPR',
#             ylabel='Test Recall',
#             #gmm_thresholds=(0.5, 0.1),
#             gmm_thresholds=(0.5, 1e-5),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             # xlim_top=0.65, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.65, # without AOFlagger
#             # xlim_top=0.025, xlim_bottom=-0.0001, ylim_top=0.65, ylim_bottom=0.557, # with AOFlagger and 0.1 thre
#             xlim_top=0.025, xlim_bottom=-0.0001, ylim_top=0.66, ylim_bottom=0.557,  # with AOFlagger and 1e-5 thre
#
#         show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             # color_legend_bbox=(0.65, 0.7), # withouth aoflagger
#             color_legend_bbox=(0.686, 0.22955), # with aoflagger
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.133, 0.09, 0.95, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_zoomed')
# exit()


# zoomed
# each model dice loss
# yes
# recall_prec(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             aof_score='LOFAR',
#             f1_contours=(0.57, 0.62, 0.652),
#             data_subset='test',
#             val_and_test=False,  # False,
#             xlabel='Test Recall',
#             ylabel='Test Precision',
#             #gmm_thresholds=(0.5, 0.1),
#             gmm_thresholds=(0.5, 1e-5),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             # xlim_top=0.65, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.65, # without AOFlagger
#             # xlim_top=0.67, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.547, # with AOFlagger and 0.1 thr
#             xlim_top=0.67, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.525,  # with AOFlagger and 1e-5
#
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             # color_legend_bbox=(0.65, 0.7), # withouth aoflagger
#             color_legend_bbox=(0.686, 0.755), # with aoflagger
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_zoomed')
# exit()

# # # zoomed val and test
# # # each model, dice loss
# #  # yes, appendix
# fpr_tpr(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             aof_score='LOFAR',
#             data_subset='test',
#             val_and_test=True,
#             xlabel='FPR',
#             ylabel='Recall',
#             #gmm_thresholds=(0.5, 0.1),
#             gmm_thresholds=(0.5,),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             # xlim_top=0.65, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.65, # without AOFlagger
#             # xlim_top=0.025, xlim_bottom=-0.0001, ylim_top=0.65, ylim_bottom=0.557, # with AOFlagger and 0.1 thre
#             xlim_top=0.025, xlim_bottom=-0.0001, ylim_top=0.76, ylim_bottom=0.557,  # with AOFlagger and 1e-5 thre
#
#         show_legend=True,
#             legend_titles=(None,None,None),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             # color_legend_bbox=(0.65, 0.7), # withouth aoflagger
#             color_legend_bbox=(0.686, 0.22955), # with aoflagger
#             marker_legend_bbox=(1.686, 0.85),  # with aoflagger
#             line_legend_bbox=(0.71, 0.85),  # with aoflagger
#
#         legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.133, 0.09, 0.95, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='valtest_eachmodel_zoomed')
#
#
# # zoomed val and test
# # each model dice loss
# # yes, appen
recall_prec(groupby=['short_model_class'], incl_models=[],
            query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
            line_ellipse=False,
            aof_score='LOFAR',
            f1_contours=(0.57, 0.628, 0.652, 0.674, 0.755),
            # data_subset='test',
            val_and_test=True,  # False,
            xlabel='Recall', #  'Test Recall',
            ylabel='Precision', # 'Test Precision',
            #gmm_thresholds=(0.5, 0.1),
            gmm_thresholds=(0.5,),
            size=60,
            linewidth=2,
            scatter_thresholds=False,
            scatter_gmm_means=True,
            scatter_gmm_points=False,
            # xlim_top=0.65, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.65, # without AOFlagger
            # xlim_top=0.67, xlim_bottom=0.5, ylim_top=0.85, ylim_bottom=0.547, # with AOFlagger and 0.1 thr
            xlim_top=0.81, xlim_bottom=0.5, ylim_top=0.95, ylim_bottom=0.525,  # with AOFlagger and 1e-5

            show_legend=True,
            legend_titles=(None, None, None),
            #legend_titles=('Loss Function',),
            fontsize=55,
            contour_fontsize=40,
            figsize=(20, 15),
            # color_legend_bbox=(0.65, 0.7), # withouth aoflagger
            color_legend_bbox=(0.686, 0.756), # with aoflagger
            line_legend_bbox=(0.72, 0.45),
            marker_legend_bbox=(1.686, 0.25),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.14, 0.09, 0.97, 0.99),
            # xlim_top=1.2,
            mean_size_factor=9,
            save_name='valtest_eachmodel_zoomed')
exit()

# ======================= GMM PLOT =======================================

# MNRAS
# gmm('val_f1', 'test_f1', xlabel='Validation F1', ylabel='Test F1',
#                 groupby = ['short_model_class'],
#                 legend_titles=(None,),
#                 fontsize=55,
#                 line_ellipse=True,
#                 figsize=(20, 15),
#                 color_legend_bbox=(0.0, 0.5),
#                 legendspacing=0.0,
#                 legend_borderpad=0.0,
#                 layout_rect=(0.11, 0.09, 0.97, 0.98),
#                 #xlim_top=.251,
#     #xlim_bottom=-0.001,
#                 mean_size_factor=6,
#                 save_name='eachmodel')
# exit()

# ======================= BUBBLE PLOT =======================================

# # yes MNRAS
bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1',
       fontsize=55, figsize=(20, 15), size_legend=False,
       legend_bbox=(0.55, 0.8),
       layout_rect=(0.13, 0.07, 1.0, 1.0),
       legendspacing=0.5, xtick_size=50,
       ylim_bottom=0.616,
       ylim_top=0.84,
       adjustment_set=3,
       save_name='MNRAS'
       )
exit()

# -----------------------
# F1 AND FPR plots, in results chapter
# ---------------------------
# # yes
# bubble('test_fpr_new', 'test_f1', xlabel='Test FPR', ylabel='Test F1',
#        fontsize=55, figsize=(20, 15), size_legend=False,
#        legend_bbox=(0.55, 0.8),
#        layout_rect=(0.13, 0.08, 1.0, 1.0),
#        legendspacing=0.5, xtick_size=50,
#        # ylim_bottom=0.518
#        )
#
# # yes
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1',
#        fontsize=55, figsize=(20, 15), size_legend=False,
#        legend_bbox=(0.55, 0.8),
#        layout_rect=(0.13, 0.07, 1.0, 1.0),
#        legendspacing=0.5, xtick_size=50,
#        # ylim_bottom=0.518
#        )
# #---------------------------
# # yes
# bubble('val_f1', 'test_f1', xlabel='Validation F1', ylabel='Test F1',
#        fontsize=55,  size_legend=True, figsize=(20, 15),
#        legend_bbox=(0.55, 0.65),
#        legendspacing=0.5,
#        # ylim_bottom=0.518,
#        layout_rect=(0.13, 0.09, 1.0, 1.0))
#
#
#
# # ---------------------------
# # no
# bubble('val_fpr_new', 'test_fpr_new', xlabel='Validation FPR', ylabel='Test FPR',
#        fontsize=55,  size_legend=False, figsize=(20, 15),
#        # ylim_top=0.9,
#        #ylim_bottom=0.68,
#        layout_rect=(0.16, 0.09, 1.0, 1.0))
# exit()

# -----------------------
# AUPRC and AUROC plots, in appendix
# ---------------------------
# yes
# bubble('flops_image', 'test_auprc_new', xlabel='Floating Point Operations', ylabel='Test AUPRC',
#        fontsize=55, figsize=(20, 15), size_legend=True,
#        legend_bbox=(0.55, 0.4),
#        layout_rect=(0.13, 0.07, 1.0, 1.0),
#        legendspacing=0.5, xtick_size=50,
#        ylim_bottom=0.518
#        )
# #---------------------------
# yes
# bubble('val_auprc_new', 'test_auprc_new', xlabel='Validation AUPRC', ylabel='Test AUPRC',
#        fontsize=55,  size_legend=True, figsize=(20, 15),
#        legend_bbox=(0.55, 0.65),
#        legendspacing=0.5,
#        ylim_bottom=0.518,
#        layout_rect=(0.13, 0.09, 1.0, 1.0))


# ---------------------------
# yes
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (ms)',
#        fontsize=55, figsize=(20, 15), size_legend=True,
#        legendspacing=0.5,
#        legend_fontsize=52,
#        legend_borderpad=0,
#        legend_bbox=(0.53, 0.385),
#        xtick_size=50,
#        layout_rect=(0.13, -0.04, 1.0, 1.0))

# yes, MNRAS
bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (ms)',
       fontsize=42, figsize=(20, 15), size_legend=True,
       legendspacing=0.8,
       legend_fontsize=42,
       legend_borderpad=0,
       legend_bbox=(0.53, 0.385),
       xtick_size=42,
       layout_rect=(0.1, -0.04, 1.0, 1.0),
       save_name='fs42')
exit()



# ---------------------------
# yes
# bubble('val_auroc_new', 'test_auroc_new', xlabel='Validation AUROC', ylabel='Test AUROC',
#        fontsize=55,  size_legend=False, figsize=(20, 15),
#        ylim_top=0.9,
#        ylim_bottom=0.68,
#        layout_rect=(0.13, 0.09, 1.0, 1.0))

# maybe in appendix
# bubble('time_image', 'test_auprc', xlabel='Time per image (ms)', ylabel='Test AUPRC',
#        fontsize=55, figsize=(20, 15), size_legend=False,
#        legend_bbox=(0.55, 0.8),
#        layout_rect=(0.13, 0.09, 1.0, 1.0),
#        legendspacing=0.5, xtick_size=50, ylim_bottom=0.640)

# ======================= TABLES =======================================

# each model, val + test TN FP ....
# val_test_table(table_fields=['TP', 'TN', 'FP', 'FN', 'fpr_new', 'recall', 'precision', 'f1'], std=False, save_name='table_conf_mat_nostd_')


# each model
table(table_fields=['flops_image', 'params', 'time_image', 'test_f1', 'test_fpr_new'], std=True, save_name='table_comp_f1_fpr')



# each model
# table(table_fields=['flops_image', 'params', 'time_image'], std=True, save_name='table_all')
