from args_constructor import preprocessing_query,common_bar_options,  common_gmm_options, common_line_options, common_bubble_options, common_fpr_tpr_options, common_recall_prec_options
from utils import ResultsCollection


# ======================= GMM PLOT =======================================
def line_agg(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_H4_args()
    kwargs['groupby'] = ['model_class']
    kwargs['label_fields'] = ['model_class']
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'line_agg'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H4_line_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_line_options(**plot_kwargs)
    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GMM PLOT =======================================


def gmm(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], limit=False, freezee_top_layes=True, task=None, save_name='', **plot_kwargs):
    kwargs = common_H4_args(limit=limit, freeze_top_layers=freezee_top_layes, task=task)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H4_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H4_args(task=task, limit=False)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'H4_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H4_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'H4_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= BUBBLE PLOT =======================================


def bubble(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], save_name='', **plot_kwargs):
    kwargs = common_H4_args()
    kwargs['groupby'] = groupby
    kwargs['params'] = True
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'bubble_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H4_bubble_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_bubble_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= TABLES =======================================

def table(table_fields, query_strings=[], groupby=['model_class'], label_fields=None, label_format='empty',
          min_f1=0.5, task='train', freeze_top_layers=None, limit=True,
          std=True, save_name=''):
    kwargs = common_H4_args(min_f1=min_f1, task=task, freeze_top_layers=freeze_top_layers, limit=limit)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H4_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

def table_with_two_groups(table_fields, query_strings=[], groupby=['short_model_class', 'hera_trans_group'],
                          label_fields=None, label_format='empty', std=True,
                          save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_H4_args(task=task, freeze_top_layers=freeze_top_layers, limit=limit, **kwargs)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table_with_two_groups'
    kwargs['std'] = std
    kwargs['save_name'] = f'H4_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

def plot_bar( query_strings=[], limit=False, incl_models=[], save_name='dummy', **plot_kwargs):
    kwargs = common_H4_args(limit=limit)
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'bar'
    kwargs['save_name'] = f'H4_bar_{save_name}'

    plot_options = common_bar_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def common_H4_args(min_f1=0.5, task='train', freeze_top_layers=False, limit=True, **kwargs):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    limit_type: not_none,
    """
    H4_kwargs = dict(
        incl_models=[],
        excl_models=[],
        #groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=[],
        datasets=['HERA_CHARL', 'HERA_CHARL_AOF'],
        query_strings=preprocessing_query() + [f'filters == 16', f'lr==0.0001', # '(task=="train") or (task=="transfer_train")',
                                               f'test_f1 > {min_f1}', 'loss == "dice"', 'kernel_regularizer == "l2"', 'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['model_class'],
        # label_fields=['model_class', 'loss'],
        label_format='empty',
        # output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        # save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H4',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H4_new',

    )

    # only include runs with a limited number of images, otherwise all
    if limit:
        H4_kwargs['query_strings'] += [f'limit != "None"']

    # if task is None or task != 'eval_test':
    #     H4_kwargs['query_strings'] += ['(task=="train") or (task=="transfer_train")']
    # elif task is not None:
    #     H4_kwargs['query_strings'] += [f'task == "{task}"']
    # if task == 'transfer_train' and freeze_top_layers is not None:
    #     H4_kwargs['query_strings'] += [f'freeze_top_layers == {freeze_top_layers}']
    return H4_kwargs

# # zoomed
# yes, MNRAS
recall_prec(groupby=['hera_trans_group'], incl_models=[],
            query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"',
                           'hera_trans_group != "0"'],
            limit=False,
            line_ellipse=True,
            aof_score='HERA_CHARL',
            # f1_contours=(0.57, 0.62, 0.652),
            f1_contours=(0.615, 0.767, 0.896),
            data_subset='test',
            xlabel='Test Recall',
            ylabel='Test Precision',
            gmm_thresholds=(0.5, ),
            size=60,
            linewidth=2,
            scatter_thresholds=False,
            scatter_gmm_means=True,
            scatter_gmm_points=False,
            xlim_top=0.95, xlim_bottom=0.65, ylim_top=1.00, ylim_bottom=0.43, # incl AOF
            show_legend=True,
            legend_titles=(None,),
            #legend_titles=('Loss Function',),
            fontsize=55,
            contour_fontsize=40,
            figsize=(20, 15),
            color_legend_bbox=(0.02, 0.18),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.14, 0.09, 0.97, 0.99),
            mean_size_factor=9,
            save_name='test_eachgroup_zoomed')
exit()

# ======================= BARS =======================================

# FPR
# yes,these limits coincidentally the same as for L3
# plot_bar(column_name='test_fpr_new',
#          limit=False,
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          # group_3='data',
#          ylabel='Test FPR',
#          ylim_bottom=0.,
#          ylim_top=0.006,
#          color_legend_bbox=(0.7, 1.018),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-1,
#          layout_rect=(0.15, -0.0, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='fpr')
# --------------------------------------------------
# these limits are for when H4 by itself, not next to HL3
# F1
# yes, these limits are for when H4 by itself, not next to HL3
# plot_bar(column_name='test_f1',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          limit=False,
#          # group_3='data',
#          ylabel='Test F1',
#          ylim_bottom=0.7,
#          ylim_top=0.96,
#          color_legend_bbox=(0.7, 1.03),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-14,
#          layout_rect=(0.12, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='f1_alone')

# AUPRC
# yes, these limits are for when H4 by itself, not next to HL3
# plot_bar(column_name='test_auprc_new',
#          limit=False,
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          # group_3='data',
#          ylabel='Test AUPRC',
#          ylim_bottom=0.7,
#          ylim_top=0.96,
#          color_legend_bbox=(0.7, 1.03),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-14,
#          layout_rect=(0.11, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='auprc_alone')

# # AUROC
# # yes, these limits are for when H4 by itself, not next to HL3
# plot_bar(column_name='test_auroc_new',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          limit=False,
#          # group_3='data',
#          ylabel='Test AUROC',
#          ylim_bottom=0.9,
#          ylim_top=0.99,
#          color_legend_bbox=(0.4, 0.22),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-17.6,
#          layout_rect=(0.12, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='auroc_alone')
# exit()


# --------------------------------------------------
# these limits are for when L3 is plotted next to H4
# F1
# yes, these limits are for when L3 is plotted next to H4
# plot_bar(column_name='test_f1',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          limit=False,
#          # group_3='data',
#          ylabel='Test F1',
#          ylim_bottom=0.4,
#          ylim_top=1.0,
#          color_legend_bbox=(0.4, 0.22),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-8.6,
#          layout_rect=(0.12, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='f1')
# exit()

# AUPRC
# yes, these limits are for when L3 is plotted next to H4
# plot_bar(column_name='test_auprc_new',
#          limit=False,
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          # group_3='data',
#          ylabel='Test AUPRC',
#          ylim_bottom=0.4,
#          ylim_top=1.0,
#          color_legend_bbox=(0.4, 0.22),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-8.7,
#          layout_rect=(0.09, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='auprc')
#
# # AUROC
# # yes, these limits are for when L3 is plotted next to H4
# plot_bar(column_name='test_auroc_new',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"', 'trans_group != "new 28"'],
#          limit=False,
#          # group_3='data',
#          ylabel='Test AUROC',
#          ylim_bottom=0.6,
#          ylim_top=1.0,
#          color_legend_bbox=(0.4, 0.22),
#          hatch_legend_bbox=(0.8, 0.1),
#          label_1_ytext_factor=-12.6,
#          layout_rect=(0.12, -0.05, 0.98, 0.98),
#          fontsize=55,
#          figsize=(20,15),
#          save_name='auroc')
# exit()

# ======================= PREC/RECALL TPR/FPR =======================================
#
# fpr_tpr(groupby=['hera_trans_group'], incl_models=[], # query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#         line_ellipse=True,
#         data_subset='test',
#         gmm_thresholds=(0.5,),
#         size=30,
#         linewidth=2,
#         scatter_thresholds=True,
#         scatter_gmm_means=True,
#         scatter_gmm_points=False,
#         # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
#         show_legend=True,
#         legend_titles=(None,),
#         # legend_titles=('Loss Function',),
#         fontsize=55,
#         figsize=(20, 15),
#         color_legend_bbox=(0.06, 0.35),
#         legend_borderpad=0,
#         legendspacing=0,
#         layout_rect=(0.12, 0.09, 0.98, 0.99),
#         # xlim_top=1.2,
#         mean_size_factor=6,
#         save_name='test_allmodels_eachgroup')
#
# # eachmodel, best reg, mse
# recall_prec(groupby=['hera_trans_group'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=True,
#             # f1_contours=(0.65,),
#             data_subset='test',
#             gmm_thresholds=(0.5, ),
#             size=30,
#             linewidth=2,
#             scatter_thresholds=True,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.06, 0.35),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.98, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='test_allmodels_eachgroup')
# exit()

# ======================= LINE PLOT =======================================

# line_agg('limit', 'test_auprc', xlabel='Number of images', ylabel='Test AUPRC')
# line_agg('limit', 'test_f1', xlabel='Number of images', ylabel='Test F1')

# ======================= GMM PLOT =======================================

# only tune 14
# gmm('test_auprc', 'test_auroc',
#     xlabel='Test AUPRC', ylabel='Test AUROC',
#     groupby=['short_model_class'],
#     query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
#                    'hera_trans_group == "tune 14"',
#                    '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
#     incl_models=[],
#     line_ellipse=True,
#     min_f1=0.0,
#     task=None,
#     limit=False,
#     legend_titles=('Model',),
#     fontsize=55,
#     figsize=(20, 15),
#     color_legend_bbox=(-0.01, 0.824),
#     marker_legend_bbox=(1, 0.55),
#     layout_rect=(0.11, 0.10, 0.98, 0.98),
#     legendspacing=0.0,
#     xlim_top=0.95, xlim_bottom=0.89, ylim_top=0.97, ylim_bottom=0.91,
#     mean_size_factor=6,
#     save_name='eachmodel_tune'
#     )
# Yes
# make non-aof width smaller by factor 3.5 in scatter_gmm.py
gmm('test_fpr_new', 'test_f1',
    xlabel='Test FPR', ylabel='Test F1',
    groupby=['hera_trans_group'],
    query_strings=[# '(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                   'hera_trans_group != "0"',
                   'hera_trans_group != "new 56"',
                   'hera_trans_group != "new 112"',
                   # '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'
                   ],
    incl_models=[],
    line_ellipse=True,
    min_f1=0.0,
    task=None,
    limit=False,
    include_legend_titles=False,
    legend_titles=('Training Type',),
    fontsize=55,
    figsize=(20, 15),
    # color_legend_bbox=(-0.01, 0.824),
    color_legend_bbox=(0.65, 0.7),
    marker_legend_bbox=(1, 0.55),
    layout_rect=(0.11, 0.10, 0.96, 0.98),
    legendspacing=0.0,
    legend_borderpad=0.0,
    # xlim_top=0.98, xlim_bottom=0.68, ylim_top=0.995, ylim_bottom=0.91,
    mean_size_factor=6,
    save_name='transtype'
    )

# Yes
gmm('test_auprc_new', 'test_auroc_new',
    xlabel='Test AUPRC', ylabel='Test AUROC',
    groupby=['hera_trans_group'],
    query_strings=[# '(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                   'hera_trans_group != "0"',
                   'hera_trans_group != "new 56"',
                   'hera_trans_group != "new 112"',
                   # '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'
                   ],
    incl_models=[],
    line_ellipse=True,
    min_f1=0.0,
    task=None,
    limit=False,
    include_legend_titles=False,
    legend_titles=('Training Type',),
    fontsize=55,
    figsize=(20, 15),
    # color_legend_bbox=(-0.01, 0.824),
    color_legend_bbox=(0.65, 0.154),
    marker_legend_bbox=(1, 0.55),
    layout_rect=(0.11, 0.10, 0.98, 0.98),
    legendspacing=0.0,
    legend_borderpad=0.0,
    xlim_top=0.98, xlim_bottom=0.68, ylim_top=0.995, ylim_bottom=0.91,
    mean_size_factor=6,
    save_name='transtype'
    )


# ======================= BUBBLE PLOT =======================================

# bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC')
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1')
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (s)')
# bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')
# bubble('test_auroc', 'test_auprc', xlabel='Test AUROC', ylabel='Test AUPRC')
# bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC')


# ======================= TABLES =======================================

# limit count for train from scratch eachmodel
# table(table_fields=['test_auroc', 'test_f1', 'test_auprc', 'val_auprc'],
#     limit=True,
#     task='train',
#       freeze_top_layers=None,
#       min_f1=0.5,
#       groupby=['model_class', 'limit'],  std=True, save_name='table')


# table_with_two_groups(table_fields=['test_auprc', 'test_auroc'],
#                       groupby=['short_model_class', 'hera_trans_group'],
#                       query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
#                                      'hera_trans_group != "0"',
#                                      'hera_trans_group != "new 56"',
#                                      'hera_trans_group != "new 112"',
#                                      '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
#                       train_with_test=False,
#                       limit=False,
#                       task=None,
#                       freeze_top_layers=None,
#                       std=True,
#                       save_name='results',
#                       min_f1=0.0)


