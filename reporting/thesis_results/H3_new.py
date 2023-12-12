from args_constructor import preprocessing_query, common_gmm_options, common_bubble_options, common_fpr_tpr_options, \
    common_recall_prec_options, common_line_options
from utils import ResultsCollection


# ======================= GMM PLOT =======================================
def line_agg(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_H3_args()
    kwargs['groupby'] = ['model_class']
    kwargs['label_fields'] = ['model_class']
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'line_agg'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H3_line_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_line_options(**plot_kwargs)
    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GMM PLOT =======================================


def gmm(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], save_name='', **plot_kwargs):
    kwargs = common_H3_args()
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H3_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= BUBBLE PLOT =======================================


def bubble(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], save_name='', **plot_kwargs):
    kwargs = common_H3_args()
    kwargs['groupby'] = groupby
    kwargs['params'] = True
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'bubble_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H3_bubble_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_bubble_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H3_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'H3_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H3_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'H3_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= TABLES =======================================

def table(table_fields, query_strings=[], groupby=['short_model_class'], label_fields=None, label_format='empty',
          std=True, save_name=''):
    kwargs = common_H3_args()
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H3_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def val_test_table(table_fields, query_strings=[], groupby=['short_model_class'], label_fields=None,
                   label_format='empty',
                   std=True,
                   save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_H3_args()
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'val_test_table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H3_valtest_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def common_H3_args(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        # groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=[],
        datasets=['HERA_CHARL_AOF'],
        query_strings=preprocessing_query() + [f'filters == 16', f'lr==0.0001',  # f'task == "{task}"',
                                               f'train_f1 > {min_f1}', f'use_hyp_data == True',  # f'limit != "None"'
                                               'loss == "dice"', 'kernel_regularizer == "l2"', 'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['short_model_class'],
        # label_fields=['model_class', 'loss'],
        label_format='empty',
        # output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H3_new',
    )
    return kwargs


# each model, val + test TN FP ....
# yes, appendix
# val_test_table(table_fields=['TP', 'TN', 'FP', 'FN', 'fpr_new', 'recall', 'precision', 'f1'], std=False, save_name='table_conf_mat_nostd')
# exit()

# ======================= PREC/RECALL TPR/FPR =======================================
# # zoomed
# # each model
# # yes
# fpr_tpr(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             aof_score='HERA_CHARL',
#             line_ellipse=False,
#             data_subset='test',
#             val_and_test=True,
#             xlabel='FPR',
#             ylabel='Recall',
#             gmm_thresholds=(0.5,),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#            #  xlim_top=0.02, xlim_bottom=-0.0005, ylim_top=0.93, ylim_bottom=0.818, # incl AOF
#            xlim_top=0.02, xlim_bottom=-0.0005, ylim_top=0.93, ylim_bottom=0.48,
#         show_legend=False,
#             legend_titles=(None, None, None),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.6, 0.25),
#             line_legend_bbox=(0.6, 0.45),
#             marker_legend_bbox=(1.22, 0.15),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.13, 0.09, 0.95, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='valtest_eachmodel_dice_zoomed')
# #exit()
# #
# # # # zoomed
# # # # each model
# # # yes
recall_prec(groupby=['short_model_class'], incl_models=[],
            query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
            aof_score='HERA_CHARL',
            line_ellipse=False,
            f1_contours=(0.615, 0.651, 0.741, 0.824, 0.852),
            data_subset='test',
            val_and_test=True,  # False,
            xlabel='Recall', #  'Test Recall',
            ylabel='Precision', # 'Test Precision',
            gmm_thresholds=(0.5, ),
            size=60,
            linewidth=2,
            scatter_thresholds=False,
            scatter_gmm_means=True,
            scatter_gmm_points=False,
            # xlim_top=0.95, xlim_bottom=0.73, ylim_top=0.9, ylim_bottom=0.46, # incl AOF
            xlim_top=0.92, xlim_bottom=0.41, ylim_top=0.98, ylim_bottom=0.43, # incl AOF
            show_legend=True,
            legend_titles=(None, None, None),
            #legend_titles=('Loss Function',),
            fontsize=55,
            contour_fontsize=40,
            figsize=(20, 15),
            color_legend_bbox=(-0.02, 0.23),
            line_legend_bbox=(0.74, 0.93),
            marker_legend_bbox=(1.22, 0.25),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.13, 0.09, 0.96, 0.99),
            # xlim_top=1.2,
            mean_size_factor=9,
            save_name='valtest_eachmodel_dice_zoomed')
exit()

#
# # # zoomed
# # # each model
# # # yes
# fpr_tpr(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             aof_score='HERA_CHARL',
#             line_ellipse=False,
#             data_subset='test',
#             xlabel='Test FPR',
#             ylabel='Test Recall',
#             gmm_thresholds=(0.5, 4e-4),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.02, xlim_bottom=-0.0005, ylim_top=0.93, ylim_bottom=0.818, # incl AOF
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.62, 0.25),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.13, 0.09, 0.95, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_dice_zoomed')
# #exit()
# #
# # # # zoomed
# # # # each model
# # # yes
# recall_prec(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             aof_score='HERA_CHARL',
#             line_ellipse=False,
#             f1_contours=(0.62, 0.77, 0.83),
#             data_subset='test',
#             xlabel='Test Recall',
#             ylabel='Test Precision',
#             gmm_thresholds=(0.5, 4e-4),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.95, xlim_bottom=0.73, ylim_top=0.9, ylim_bottom=0.46, # incl AOF
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.02, 0.25),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.13, 0.09, 0.96, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_dice_zoomed')
# exit()

# ======================= LINE PLOT =======================================

# line_agg('limit', 'test_auprc', xlabel='Number of images', ylabel='Test AUPRC')
# line_agg('limit', 'test_f1', xlabel='Number of images', ylabel='Test F1')

# ======================= GMM PLOT =======================================
# yes in L1 vs H1 vs H3
# gmm('train_loss_over_val_loss', 'train_auprc_over_val_auprc_new',
#     xlabel='Loss Ratio', ylabel='AUPRC Ratio',
#                 groupby=['short_model_class'],
#                 # legend_titles=('Model',),
#                 legend_titles=(None,),
#                 show_legend=False,
#                 line_ellipse=True,
#                 fontsize=55,
#                 figsize=(20, 15),
#                 color_legend_bbox=(1.98, 0.6),
#                 layout_rect=(0.11, 0.09, 0.98, 0.98),
#     xlim_bottom=0.45, xlim_top=1.26, ylim_bottom=0.97, ylim_top=1.125,  # L1 vs H3 vs H1
#     # xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.968, ylim_top=1.08, # alone
#                  mean_size_factor=6,
#                 save_name='eachmodel_fits')
# exit()
# # no, but is in L2 vs H3 plot
# gmm('val_auprc_new', 'test_auprc_new', xlabel='Validation AUPRC', ylabel='Test AUPRC',
#                 groupby = ['short_model_class'],
#                 legend_titles=('Model',),
#                 line_ellipse=True,
#                 fontsize=55,
#                 figsize=(20, 15),
#                 color_legend_bbox=(-0.01, 0.285),
#                 legendspacing=0,
#                 layout_rect=(0.11, 0.09, 0.97, 0.98),
#                 #xlim_top=1.2,
#                 mean_size_factor=6,
#                 save_name='eachmodel')
# # no, but is in L2 vs H3 plot
# gmm('val_auroc_new', 'test_auroc_new', xlabel='Validation AUROC', ylabel='Test AUROC',
#                 groupby = ['short_model_class'],
#                 legend_titles=('Model',),
#                 line_ellipse=True,
#                 fontsize=55,
#                 figsize=(20, 15),
#                 color_legend_bbox=(1.98, 0.6),
#                 layout_rect=(0.11, 0.09, 0.98, 0.98),
#                 #xlim_top=1.2,
#                 mean_size_factor=6,
#                 save_name='eachmodel')

# not anymoire
# gmm('reg_distance_new', 'val_auprc_new', xlabel='Regularization Distance', ylabel='Validation AUPRC',
#                 groupby = ['short_model_class'],
#                 legend_titles=('Model',),
#                 fontsize=55,
#                 line_ellipse=True,
#                 figsize=(20, 15),
#                 color_legend_bbox=(1.98, 0.6),
#                 layout_rect=(0.11, 0.09, 0.98, 0.98),
#                 #xlim_top=1.2,
#                 mean_size_factor=6,
#                 save_name='eachmodel')



# # Yes, MNRAS
bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1',
       fontsize=55, size_legend=False, figsize=(20, 15),
       groupby=['short_model_class'],
       adjustment_set=3,
       layout_rect=(0.13, 0.09, 1.0, 0.99),
       xtick_size=50,
       save_name='MNRAS',
       ylim_bottom=0.616,
       ylim_top=0.84,
       )
exit()

# # Yes,
bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1',
       fontsize=55, size_legend=False, figsize=(20, 15),
       groupby=['short_model_class'],
       adjustment_set=2,
       layout_rect=(0.13, 0.09, 1.0, 0.99),
       xtick_size=50,
       # xlim_top=0.87,
       # ylim_bottom=0.9625, ylim_top=0.995,
       )
exit()

# MNRAS
gmm('reg_distance_new', 'val_f1', xlabel='Regularization Distance', ylabel='Validation F1',
    groupby=['short_model_class'],
    legend_titles=('Model',),
    fontsize=55,
    line_ellipse=True,
    figsize=(20, 15),
    color_legend_bbox=(1.98, 0.6),
    layout_rect=(0.11, 0.09, 0.97, 0.98),
    ylim_bottom=.64,
    ylim_top=.88,
    xlim_top=.58,
    xlim_bottom=-0.01,
    mean_size_factor=6,
    save_name='eachmodel_MNRAS')
exit()
# yes
gmm('reg_distance_new', 'val_f1', xlabel='Regularization Distance', ylabel='Validation F1',
    groupby=['short_model_class'],
    legend_titles=('Model',),
    fontsize=55,
    line_ellipse=True,
    figsize=(20, 15),
    color_legend_bbox=(1.98, 0.6),
    layout_rect=(0.11, 0.09, 0.97, 0.98),
    xlim_top=.251,
    xlim_bottom=-0.001,

    mean_size_factor=6,
    save_name='eachmodel')



gmm('val_f1', 'test_f1', xlabel='Validation F1', ylabel='Test F1',
    groupby=['short_model_class'],
    legend_titles=(None,),
    fontsize=55,
    line_ellipse=True,
    figsize=(20, 15),
    color_legend_bbox=(0.1, 0.5),
    legendspacing=0.0,
    legend_borderpad=0.0,
    layout_rect=(0.13, 0.09, 0.97, 0.98),
    # xlim_top=.251,
    # xlim_bottom=-0.001,
    mean_size_factor=6,
    save_name='eachmodel')
exit()

# no
# gmm('test_recall_new', 'test_precision_new', xlabel='Test Recall', ylabel='Test Precision',
#                 groupby=['short_model_class'],
#                 legend_titles=('Model',),
#                 line_ellipse=True,
#                 fontsize=55,
#                 figsize=(20, 15),
#                 color_legend_bbox=(1.98, 0.6),
#                 layout_rect=(0.11, 0.09, 0.98, 0.98),
#                 #xlim_top=1.2,
#                 mean_size_factor=6,
#                 save_name='eachmodel')


# ======================= BUBBLE PLOT =======================================

# bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC')
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1')
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (s)')
# bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')
# bubble('test_auroc', 'test_auprc', xlabel='Test AUROC', ylabel='Test AUPRC')
# bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC')


# ======================= TABLES =======================================

# eachmodel
# table(table_fields=['test_recall_new', 'test_precision_new', 'test_auroc_new', 'test_auprc_new',], std=True, save_name='table')
