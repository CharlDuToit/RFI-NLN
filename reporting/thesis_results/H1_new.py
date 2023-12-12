from args_constructor import preprocessing_query, common_gmm_options, common_recall_prec_options, common_fpr_tpr_options,  common_bar_options
from utils import ResultsCollection


def plot_bar( query_strings=[], incl_models=[], save_name='dummy', **plot_kwargs):
    kwargs = common_H1_args()
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'bar'
    kwargs['save_name'] = f'H1_bar_{save_name}'

    plot_options = common_bar_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= GENERIC GMM PLOTS =======================================
def generic_gmm(groupby, x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_H1_args()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'H1_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= TABLES =======================================


def table(groupby, table_fields, label_fields=None, query_strings=[], label_format='empty', std=True,
          save_name='dummy', min_f1=0.5):
    kwargs = common_H1_args(min_f1=min_f1)
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H1_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

def table_with_two_groups(table_fields, query_strings=[], groupby=['short_model_class', 'loss'],
                          label_fields=None, label_format='empty', std=True, min_f1=0.5,
                          save_name='dummy', **kwargs):
    kwargs = common_H1_args(min_f1=min_f1)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table_with_two_groups'
    kwargs['std'] = std
    kwargs['save_name'] = f'H1_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H1_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'H1_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H1_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'H1_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def common_H1_args(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        groupby=[],
        datasets=['HERA_CHARL'],
        query_strings=preprocessing_query() + [f'use_hyp_data == True', f'filters == 16', f'lr==0.0001', # f'task == "{task}"',
                                               f'train_f1 > {min_f1}', f'train_with_test == False', f'limit == "None"',
                                               'kernel_regularizer == "l2"', 'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=[],
        label_format=None,
        #output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H1_new',
    )
    return kwargs

# Yes
# plot_bar(column_name='train_auprc_over_val_auprc_new',
#          group_1='short_model_class',
#          group_2='loss',
#          query_strings=['val_auprc_new > 0.85', 'reg_distance_new < 0.7'],
#          # group_3='data',
#          ylabel='AUPRC Ratio',
#          ylim_bottom=0.4,
#          label_1_ytext_factor=-8.5,
#          fontsize=55,
#          figsize=(20,15),
#          layout_rect=(0.09, -0.02, 0.98, 0.98),
#          color_legend_bbox=(0.8, 0.3),
#          save_name='auprc_ratio')
# Yes
# plot_bar(column_name='train_loss_over_val_loss',
#          group_1='short_model_class',
#          group_2='loss',
#          query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#          # group_3='data',
#          ylabel='Loss Ratio',
#          ylim_bottom=0.4,
#          label_1_ytext_factor=-8.5,
#          fontsize=55,
#          figsize=(20,15),
#          layout_rect=(0.09, -0.02, 0.98, 0.98),
#          color_legend_bbox=(0.8, 1.0),
#          save_name='loss_ratio')

# ======================= PREC/RECALL TPR/FPR =======================================

# # zoomed
# # allmodels, best reg, each loss
# # yes
# fpr_tpr(groupby=['loss'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             data_subset='val',
#             xlabel='Validation FPR',
#             ylabel='Validation Recall',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.2, xlim_bottom=-0.002, ylim_top=0.98, ylim_bottom=0.8495,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.22, 0.18),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.96, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='val_allmodels_eachloss_zoomed')

#
# # # zoomed
# each model name, dice loss
# no
# recall_prec(groupby=['model_name'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             f1_contours=(0.92, 0.955),
#             data_subset='val',
#             xlabel='Validation Recall',
#             ylabel='Validation Precision',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=True,
#             scatter_gmm_means=False,
#             scatter_gmm_points=False,
#             xlim_top=1.0, xlim_bottom=0.8, ylim_top=1.0, ylim_bottom=0.8,
#             show_legend=False,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.02, 0.18),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='val_eachdice')
# exit()

# zoomed
# allmodels, each loss
# yes
# recall_prec(groupby=['loss'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             f1_contours=(0.93, 0.97),
#             data_subset='val',
#             xlabel='Validation Recall',
#             ylabel='Validation Precision',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=1.0, xlim_bottom=0.8, ylim_top=1.0, ylim_bottom=0.795,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.02, 0.18),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='val_allmodels_eachloss_zoomed')
# exit()

# unzoomed
# allmodels, best reg, each loss
# yes, appendix
# recall_prec(groupby=['loss'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             f1_contours=(0.93, 0.97),
#             data_subset='val',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=1.0, xlim_bottom=0.8,
#             # ylim_top=1.0, ylim_bottom=0.8,
#             ylim_bottom=-0.01,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.02, 0.18),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             mean_size_factor=9,
#             save_name='val_allmodels_eachloss_unzoomed')

# zoomed
# each model, best reg, dice loss
# no
# recall_prec(groupby=['short_model_class'], incl_models=[],
#             query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             f1_contours=(0.9, 0.95),
#             data_subset='val',
#             gmm_thresholds=(0.5, ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=True,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=1.0, xlim_bottom=0.8, ylim_top=1.0, ylim_bottom=0.8,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.02, 0.25),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='val_eachmodel_dice_zoomed')
# exit()

# # ======================= CUSTOM GMM PLOTS =======================================
le = True

# # ----------------------EACH MODEL, EACH LOSS (32)----------------------------------
# # Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['model_class', 'loss'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='eachmodel_loss',
#             xlim_bottom=0.5, xlim_top=2)

# # val auprc vs val auroc
# generic_gmm(groupby=['model_class', 'loss'], incl_models=[], query_strings=[], x_axis='val_auprc',
#             y_axis='val_auroc', xlabel='Validation AUPRC', ylabel='Validation AUROC', line_ellipse=le, save_name='eachmodel_each_loss',
#             xlim_bottom=0.85, ylim_bottom=0.92)


# # ----------------------ONE MODEL, EACH LOSSS (4)----------------------------------
# val auprc vs val auroc
# generic_gmm(groupby=['loss'], incl_models=['UNET'], query_strings=[], x_axis='val_auprc',
#             y_axis='val_auroc', xlabel='Validation AUPRC', ylabel='Validation AUROC', line_ellipse=le, save_name='UNET_eachloss',)
#
# # # Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['loss'], incl_models=['UNET'], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='UNET_eachloss',)


# # ----------------------EACH MODELS, DICE (8), without outliers----------------------------------
# 'val_auprc > 0.85', 'reg_distance < 0.7'

# Loss Ratio vs  AUPRC Ratio
# Yes ( at H3 experiment)
# generic_gmm(groupby=['short_model_class'], incl_models=[],
#             # query_strings=['val_auprc > 0.85', 'reg_distance < 0.7', 'loss=="dice"'],
#             query_strings=['loss=="dice"'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc_new', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             include_legend_titles=False,
#             legend_titles=('Model',),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             #color_legend_bbox=(0.1, 0.275), alone
#             color_legend_bbox=(0.1, 0.875),  # L1 vs H3 vs H1
#             marker_legend_bbox=(0.8, 0.8),
#             layout_rect=(0.15, 0.08, 0.975, 0.99),
#             xlim_bottom=0.45, xlim_top=1.26, ylim_bottom=0.97, ylim_top=1.125, # L1 vs H3 vs H1
#             # xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='eachmodel_dice_fits')
# exit()


# # ----------------------ALL MODELS, EACH LOSS (4), without outliers----------------------------------
# 'val_auprc > 0.85', 'reg_distance < 0.7'

# Loss Ratio vs  AUPRC Ratio
# replaced with bars
# generic_gmm(groupby=['loss'], incl_models=[],
#             query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Loss Function',),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.52, 0.82),
#             marker_legend_bbox=(0.8, 0.8),
#             layout_rect=(0.13, 0.08, 0.975, 0.99),
#             xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='allmodels_eachloss')



# Reg Distance vs  val AUPRC
# 'val_auprc > 0.85', 'reg_distance < 1.2'
# Yes
# generic_gmm(groupby=['loss'], incl_models=[],
#             #query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             #query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='reg_distance_new',
#             y_axis='val_auprc_new', xlabel='Regularization Distance', ylabel='Validation AUPRC', line_ellipse=le,
#             legend_titles=('Loss Function',),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.5, 0.3),
#             marker_legend_bbox=(1, 0.8),
#             layout_rect=(0.13, 0.09, 0.975, 0.99),
#             #xlim_top=2.2,
#             #xlim_bottom=0.5,
#             mean_size_factor=6,
#             save_name='allmodels_eachloss')


# # ----------------------EACH MODELS, DICE LOSS (8)----------------------------------
# Reg Distance vs  val AUPRC

# 'val_auprc > 0.85', 'reg_distance < 0.7'
# No longer
# generic_gmm(groupby=['short_model_class'], incl_models=[],
#             #query_strings=[ 'val_auprc > 0.85', 'reg_distance < 0.7'],
#             query_strings=['loss == "dice"', 'val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='reg_distance_new',
#             y_axis='val_auprc_new', xlabel='Regularization Distance', ylabel='Validation AUPRC', line_ellipse=le,
#             legend_titles=('Model',),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.66, 0.3),
#             marker_legend_bbox=(0.8, 0.8),
#             layout_rect=(0.13, 0.09, 0.99, 0.99),
#             #xlim_top=2.2,
#             xlim_bottom=0.0,
#             mean_size_factor=6,
#             save_name='eachmodel_dice')

# 'val_auprc > 0.85', 'reg_distance < 0.7'
# Yes, MNRAS
generic_gmm(groupby=['short_model_class'], incl_models=[],
            #query_strings=[ 'val_auprc > 0.85', 'reg_distance < 0.7'],
            query_strings=['loss == "dice"', 'val_auprc > 0.85', 'reg_distance < 0.7'],
            x_axis='reg_distance_new',
            y_axis='val_f1', xlabel='Regularization Distance', ylabel='Validation F1', line_ellipse=le,
            legend_titles=('Model',),
            legendspacing=0,
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.66, 0.6),
            marker_legend_bbox=(0.8, 0.8),
            layout_rect=(0.13, 0.09, 0.99, 0.99),
            xlim_top=0.58,
            xlim_bottom=-0.01,
            mean_size_factor=6,
            save_name='eachmodel_dice_MNRAS')
exit()

# 'val_auprc > 0.85', 'reg_distance < 0.7'
# Yes
generic_gmm(groupby=['short_model_class'], incl_models=[],
            #query_strings=[ 'val_auprc > 0.85', 'reg_distance < 0.7'],
            query_strings=['loss == "dice"', 'val_auprc > 0.85', 'reg_distance < 0.7'],
            x_axis='reg_distance_new',
            y_axis='val_f1', xlabel='Regularization Distance', ylabel='Validation F1', line_ellipse=le,
            legend_titles=('Model',),
            legendspacing=0,
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.66, 0.6),
            marker_legend_bbox=(0.8, 0.8),
            layout_rect=(0.13, 0.09, 0.99, 0.99),
            # xlim_top=0.58,
            xlim_bottom=0.0,
            mean_size_factor=6,
            save_name='eachmodel_dice')

# # ----------------------EACH LOSS, 3 MODELS, 3 MODELS, 2 MODELS, without outliers ----------------------------------
#                               # Loss Ratio vs  AUPRC Ratio
#                           'val_auprc > 0.85', 'reg_distance < 1.2'


# # Loss Ratio vs  AUPRC Ratio
# 'DSC_MONO_RESUNET', 'DSC_DUAL_RESUNET', 'RFI-NET'
# generic_gmm(groupby=['loss', 'short_model_class'],
#             incl_models=['DSC_MONO_RESUNET', 'DSC_DUAL_RESUNET', 'RFI_NET'],
#             query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Loss Function', 'Model'),
#             legendspacing=-1,
#             fontsize=55,
#             legend_fontsize=52,
#             figsize=(20, 15),
#             color_legend_bbox=(1.2, 0.17),
#             marker_legend_bbox=(-0.021, 0.17),
#             layout_rect=(0.13, 0.08, 0.98, 0.99),
#             xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='MONO_DUAL_RFI_eachloss')
# #
# # 'RNET5', 'RNET'
# generic_gmm(groupby=['loss', 'short_model_class'],
#             incl_models=['RNET5', 'RNET' ],
#             query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Loss Function', 'Model'),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(1.2, 0.17),
#             marker_legend_bbox=(0.67, 0.8),
#             layout_rect=(0.13, 0.08, 0.98, 0.99),
#             xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='R5_R7_eachloss')
#
# # 'UNET', 'AC_UNET', 'ASPP_UNET',
# generic_gmm(groupby=['loss', 'short_model_class'],
#             incl_models=['UNET', 'AC_UNET', 'ASPP_UNET', ],
#             query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Loss Function', 'Model'),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(1.2, 0.17),
#             marker_legend_bbox=(0.67, 0.8),
#             layout_rect=(0.13, 0.08, 0.98, 0.99),
#             xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='U_AC_ASPP_eachloss')

# # ----------------------FOUR MODEL (16)----------------------------------
# # without loss legend

# # Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['short_model_class', 'loss'], incl_models=['RNET5','RNET', 'DSC_MONO_RESUNET', 'DSC_DUAL_RESUNET'], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Model','Loss Function'),
#             fontsize=48,
#             figsize=(20, 15),
#             color_legend_bbox=(0.7, 0.758),
#             marker_legend_bbox=(1, 0.8),
#             layout_rect=(0.15, 0.1, 0.99, 0.95),
#             xlim_top=2.2,
#             xlim_bottom=0.5,
#             mean_size_factor=6,
#             save_name='R5_R7_MONO_DUAL_eachloss_nolosslegend')
#
# generic_gmm(groupby=['short_model_class', 'loss'], incl_models=['UNET', 'AC_UNET', 'ASPP_UNET', 'RFI_NET'], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Model','Loss Function'),
#             fontsize=48,
#             figsize=(20, 15),
#             color_legend_bbox=(0.0, 0.22),
#             marker_legend_bbox=(1, 0.8),
#             layout_rect=(0.15, 0.1, 0.99, 0.95),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='U_AC_ASPP_RFI_eachloss_nolosslegend')

# ======================= TABLES =======================================

# # eachmodel_bestreg_allloss
# table(groupby=['model_class'], table_fields=['reg_distance', 'val_auprc', 'val_auroc'], label_fields=None,
#       query_strings=['reg_label == "l2+0.1"'], label_format='empty', std=True, save_name='eachmodel_allloss')
#
# # ------
# # allmodels_bestreg_eachloss
# table(groupby=['loss'], table_fields=['reg_distance', 'val_auprc', 'val_auroc'], label_fields=None,
#       query_strings=['reg_label == "l2+0.1"'], label_format='empty', std=True, save_name='allmodels_eachloss')


# # run with no query strings to see how many are excluded
# table_with_two_groups(table_fields=['reg_distance', 'val_auprc'],
#                       groupby=['short_model_class', 'loss'],
#                       query_strings=['val_auprc > 0.85', 'reg_distance < 0.7'],
#                       std=True,
#                       save_name='results_eachmodel_eachloss',
#                       min_f1=0.0)

# Yes
table(table_fields=[ # 'train_auprc_over_val_auprc_new',
                     'train_loss_over_val_loss', 'reg_distance_new', 'val_auroc_new', 'val_auprc_new'],
                      groupby=[ 'loss'],
                      query_strings=['val_auprc_new > 0.85', 'reg_distance_new < 0.7'],
                      std=True,
                      save_name='results_allmodels_eachloss',
                      min_f1=0.0)
