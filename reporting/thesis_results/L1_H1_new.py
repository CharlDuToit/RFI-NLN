from args_constructor import preprocessing_query, common_gmm_options, common_recall_prec_options, common_fpr_tpr_options
from utils import ResultsCollection


# ======================= PER MODEL/LOSS/REG GMM PLOTS =======================================

# SELECT ONE MODEL, GROUPBY (REG, LOSS)


def model_gmm(model_class='UNET', x_axis='train_auprc', y_axis='val_auprc', **plot_kwargs):
    kwargs = common_L1_H1_kwargs()
    kwargs['incl_models'] = [model_class]
    kwargs['groupby'] = ['loss', 'reg_label']
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L1_gmm_{model_class}_{x_axis}_{y_axis}'
    kwargs['query_strings'] += ['dropout < 0.11']

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# SELECT ONE LOSS, GROUPBY (MODEL, REG)


def loss_gmm(loss='dice', x_axis='train_auprc', y_axis='val_auprc', **plot_kwargs):
    kwargs = common_L1_H1_kwargs()
    kwargs['query_strings'] += [f'loss == "{loss}"']
    kwargs['groupby'] = ['model_class', 'reg_label']
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L1_gmm_{loss}_{x_axis}_{y_axis}'
    kwargs['query_strings'] += ['dropout < 0.11']

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# SELECT ONE TEG, GROUPBY (MODEL, LOSS)


def reg_gmm(reg_label='l2+0.1', x_axis='train_auprc', y_axis='val_auprc', **plot_kwargs):
    kwargs = common_L1_H1_kwargs()
    kwargs['query_strings'] += [f'reg_label == "{reg_label}"']
    kwargs['groupby'] = ['model_class', 'loss']
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L1_gmm_{reg_label}_{x_axis}_{y_axis}'
    kwargs['query_strings'] += ['dropout < 0.11']

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GENERIC GMM PLOTS =======================================
def generic_gmm(groupby, x_axis, y_axis, query_strings=[], incl_models=[], save_name='dummy', **plot_kwargs):
    kwargs = common_L1_H1_kwargs()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L1_H1_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L1_H1_kwargs(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'L1_H1_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L1_H1_kwargs(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'L1_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= TABLES =======================================


def table(groupby, table_fields, label_fields=None, query_strings=[], label_format='empty', std=True,
          save_name='dummy'):
    kwargs = common_L1_H1_kwargs()
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'L1_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def table_with_two_groups(table_fields, query_strings=[], groupby=['short_model_class', 'lofar_trans_group'],
                          label_fields=None, label_format='empty', std=True,
                          save_name='dummy', **kwargs):
    kwargs = common_L1_H1_kwargs(**kwargs)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table_with_two_groups'
    kwargs['std'] = std
    kwargs['save_name'] = f'L1_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def common_L1_H1_kwargs(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        # groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=[],
        datasets=[],
        query_strings=preprocessing_query() + [f'use_hyp_data == True', f'filters == 16', f'lr==0.0001',
                                               # f'task=="{task}"',
                                               f'train_f1 > {min_f1}', f'train_with_test == False', f'limit == "None"'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=[],
        # label_fields=['model_class', 'loss'],
        label_format=None,
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results',
        # output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/temp/',
        #output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L1_H1_new',
    )
    return kwargs

# ======================= ROC / PREC RECALL PLOTS =======================================
#

# # # zoomed
# # # allmodels, best reg, each loss, lofar+hera
# MNRAS
recall_prec(groupby=['loss', 'data'], incl_models=[],
            query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
            line_ellipse=False,
            f1_contours=(0.66, 0.7, 0.92, 0.955),
            data_subset='val',
            xlabel='Validation Recall',
            ylabel='Validation Precision',
            gmm_thresholds=(0.5, 0.1 ),
            size=60,
            linewidth=2,
            scatter_thresholds=False,
            scatter_gmm_means=True,
            scatter_gmm_points=False,
            # xlim_top=0.8, xlim_bottom=0.4, ylim_top=1.0, ylim_bottom=0.49,
            xlim_top=1.0, xlim_bottom=0.4, ylim_top=1.0, ylim_bottom=0.49,
            show_legend=True,
            legend_titles=(None,None),
            #legend_titles=('Loss Function',),
            # fontsize=55,
            # contour_fontsize=40,
            fontsize=42,
            contour_fontsize=30,
            figsize=(20, 15),
            color_legend_bbox=(0.4, 0.68), # fontsize=55
            line_legend_bbox=(0.47, 0.9), # fontsize=55
            marker_legend_bbox=(1.5, 0.15), # fontsize=55
            legend_borderpad=0,
            legendspacing=0,
            # layout_rect=(0.13, 0.09, 0.97, 0.99), # fontsize=55
            layout_rect=(0.09, 0.06, 0.97, 0.99), # dontsize = 36
            # xlim_top=1.2,
            mean_size_factor=9,
            save_name='val_allmodels_eachloss_bestreg_zoomed')
# exit()

# # unzoomed
# # allmodels, best reg, each loss
# yes, appendox
# fpr_tpr(groupby=['loss'], incl_models=[],
#             query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=True,
#             data_subset='val',
#             xlabel='Validation FPR',
#             ylabel='Validation Recall',
#             gmm_thresholds=(0.99, 0.5, 0.01),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=True,
#             # xlim_top=0.2, xlim_bottom=-0.01, ylim_top=0.9, ylim_bottom=0.5,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.52, 0.16),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.96, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='val_allmodels_eachloss_bestreg_unzoomed')
# exit()

# # zoomed
# # allmodels, best reg, each loss
# yes
# fpr_tpr(groupby=['loss'], incl_models=[],
#             query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             # f1_contours=(0.68, 0.72),
#             data_subset='val',
#             xlabel='Validation FPR',
#             ylabel='Validation Recall',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.2, xlim_bottom=-0.01, ylim_top=0.9, ylim_bottom=0.5,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             contour_fontsize=40,
#             figsize=(20, 15),
#             color_legend_bbox=(0.52, 0.16),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.96, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='val_allmodels_eachloss_bestreg_zoomed')
# exit()

# # # zoomed
# # # allmodels, best reg, each loss
# yes
# recall_prec(groupby=['loss'], incl_models=[],
#             query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             f1_contours=(0.67, 0.7),
#             data_subset='val',
#             xlabel='Validation Recall',
#             ylabel='Validation Precision',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.8, xlim_bottom=0.4, ylim_top=1.0, ylim_bottom=0.49,
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
#             save_name='val_allmodels_eachloss_bestreg_zoomed')

# # # unzoomed
# # # allmodels, best reg, each loss
# yes, appendix
# recall_prec(groupby=['loss'], incl_models=[],
#             query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=True,
#             f1_contours=(0.0,),
#             data_subset='val',
#             xlabel='Validation Recall',
#             ylabel='Validation Precision',
#             gmm_thresholds=(0.99999, 0.5, 0.00001 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.8, xlim_bottom=0.22, ylim_top=1.0, ylim_bottom=0.2,
#             # ylim_bottom=-0.02,
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
#             save_name='val_allmodels_eachloss_bestreg_unzoomed')
# exit()


# ======================= PER MODEL/LOSS/REG GMM PLOTS =======================================
le = True
# yes: AC and R5, the rest in appendix
# for mc in ('UNET', 'AC_UNET', 'ASPP_UNET', 'RNET', 'RNET5', 'RFI_NET', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET'):
# #for mc in ('RNET5',):
# #         #model_gmm(model_class=mc, x_axis='train_auprc', y_axis='val_auprc', xlabel='Training AUPRC', ylabel='Validation AUPRC', line_ellipse=le)
# #         #model_gmm(model_class=mc, x_axis='train_loss', y_axis='val_loss', xlabel='Training Loss', ylabel='Validation Loss', line_ellipse=le)
# #
#     model_gmm(model_class=mc, x_axis='train_loss_over_val_loss', y_axis='train_auprc_over_val_auprc_new',
#               xlabel='Loss Ratio',
#               ylabel='AUPRC Ratio',
#               line_ellipse=True,
#               show_legend=True if mc in ('RNET', 'RNET5') else False,
#               color_legend_bbox=(0.06, 0.35),
#               marker_legend_bbox=(0.06, 0.75),
#               legendspacing=0,
#               legend_borderpad=0,
#               legend_titles=('Loss Function', 'Reg. Scheme'),
#               xlim_top=1.15, xlim_bottom=0.42, ylim_top=1.28, ylim_bottom=0.97,
#               size=120,
#               linewidth=3,
#               fontsize=55,
#               mean_size_factor=6,
#               layout_rect=(0.08, 0.10, 1.0, 1.0),
#               figsize=(20, 12))
#
# le = True
# for loss in ('dice', 'mse', 'bce', 'logcoshdice'):
#     loss_gmm(loss=loss, x_axis='train_auprc', y_axis='val_auprc', xlabel='Training AUPRC', ylabel='Validation AUPRC', line_ellipse=le)
#     loss_gmm(loss=loss, x_axis='train_loss', y_axis='val_loss', xlabel='Training Loss', ylabel='Validation Loss', line_ellipse=le)
# loss_gmm(loss=loss, x_axis='train_loss_over_val_loss', y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le)
#
# le = True
# for reg in ('l2+0.1', 'l2+0.0', 'None+0.0', 'None+0.1'):
#     reg_gmm(reg_label=reg, x_axis='train_auprc', y_axis='val_auprc', xlabel='Training AUPRC', ylabel='Validation AUPRC', line_ellipse=le)
#     reg_gmm(reg_label=reg, x_axis='train_loss', y_axis='val_loss', xlabel='Training Loss', ylabel='Validation Loss', line_ellipse=le)
#     reg_gmm(reg_label=reg, x_axis='train_loss_over_val_loss', y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le)

# # ======================= CUSTOM GMM PLOTS =======================================
# le = True
# =============================================================================================================
# --------------------------- TO SEE X, Y LIMITS ---------------------------------------------
# generic_gmm(groupby=['model_class', 'loss', 'reg_label'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='each_model_loss_reg',
#             xlim_top=1.3, xlim_bottom=0.42, ylim_top=1.28, ylim_bottom=0.97, show_legend=False)

#
# # -----------------------ALL MODELS, EACH REG, EACH LOSS (16)----------------------------------
# # THIS ONE - abit too busy
# # allmodels_eachreg_eachloss: Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['loss', 'reg_label'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='allmodels_eachreg_eachloss')

# # ----------------------ONE MODEL, BEST REG, EACH LOSS (4)-----------------------------------
# # UNET_bestreg_eachloss: Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['loss'], incl_models=['UNET'], query_strings=['reg_label == "l2+0.1"'], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='UNET_bestreg_eachloss')
#
# # -----------------------ONE MODEL, ALL REG, EACH LOSS (4)----------------------------------
# # RFI_NET_allreg_eachloss:  Loss Ratio vs  AUPRC Ratio
# generic_gmm(groupby=['loss'], incl_models=['RFI_NET'], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le, save_name='RFI_NET_allreg_eachloss')


# # ----------------------ALL MODELS, EACH REG, EACH LOSS (16)---------------------------------

# Loss Ratio vs  AUPRC Ratio
# No, too busy
# generic_gmm(groupby=['loss',], incl_models=[],
#             # query_strings=['val_auprc > 0.85', 'reg_distance < 0.7', 'loss=="dice"'],
#             query_strings=['reg_label=="l2+0.1"'],
#             x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc_new', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             include_legend_titles=False,
#             legend_titles=('Loss Function', 'Reg. Scheme'),
#             legendspacing=0,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.63, 0.99),
#             marker_legend_bbox=(0.8, 0.8),
#             layout_rect=(0.15, 0.08, 0.975, 0.99),
#             # xlim_top=1.15,
#             # xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.98, ylim_top=1.08, # From H3 vs L2
#             # xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
#             mean_size_factor=6,
#             save_name='allmodels_besthreg_eachloss')

# # ----------------------EACH LOSS, LOFAR AND HERA (4x2=8),---------------------------------

# MNRAS
generic_gmm(groupby=['loss', 'data'], incl_models=[],
            # query_strings=['val_auprc > 0.85', 'reg_distance < 0.7', 'loss=="dice"'],
            query_strings=['reg_label == "l2+0.1"'],
            x_axis='reg_distance_new',
            y_axis='val_f1', xlabel='Regularization Distance', ylabel='Validation F1', line_ellipse=le,
            include_legend_titles=False,
            legend_titles=('Loss',),
            legendspacing=0,
            legend_borderpad=0,
            fontsize=42,
            # fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.5, 0.85), # fontsize=55
            marker_legend_bbox=(0.57, 0.52), # fontsize=55
            line_legend_bbox=(1.57, 0.6),
            # layout_rect=(0.13, 0.1, 0.975, 0.99), # fontsize = 55
            layout_rect=(0.09, 0.06, 0.975, 0.99),  # fontsize = 36
            # xlim_bottom=0.45, xlim_top=1.26, ylim_bottom=0.97, ylim_top=1.125,  # L1 vs H3 vs H1
            # xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.98, ylim_top=1.08, # From H3 vs L2
            # xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
            mean_size_factor=6,
            save_name='eachloss_bestreg')

generic_gmm(groupby=['loss', 'data'], incl_models=[],
            # query_strings=['val_auprc > 0.85', 'reg_distance < 0.7', 'loss=="dice"'],
            query_strings=['reg_label == "l2+0.1"'],
            x_axis='reg_distance_new',
            y_axis='val_auprc_new', xlabel='Regularization Distance', ylabel='Validation AUPRC', line_ellipse=le,
            include_legend_titles=False,
            legend_titles=('Loss',),
            legendspacing=0,
            legend_borderpad=0,
            # fontsize=55,
            fontsize=42,
            figsize=(20, 15),
            color_legend_bbox=(0.5, 0.85), # fontsize=55
            marker_legend_bbox=(0.57, 0.52), # fontsize=55
            line_legend_bbox=(1.57, 0.6),
            # layout_rect=(0.13, 0.1, 0.975, 0.99), # fontsize = 55
            layout_rect=(0.09, 0.06, 0.975, 0.99),  # fontsize = 36
            # xlim_bottom=0.45, xlim_top=1.26, ylim_bottom=0.97, ylim_top=1.125,  # L1 vs H3 vs H1
            # xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.98, ylim_top=1.08, # From H3 vs L2
            # xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
            mean_size_factor=6,
            save_name='eachloss_bestreg')
exit()

# # ----------------------EACH MODELS, DICE (8),---------------------------------

# Loss Ratio vs  AUPRC Ratio
# Yes ( at H3 experiment), limits must match H3 plot limits
generic_gmm(groupby=['short_model_class'], incl_models=[],
            # query_strings=['val_auprc > 0.85', 'reg_distance < 0.7', 'loss=="dice"'],
            query_strings=['loss=="dice"', 'reg_label == "l2+0.1"'],
            x_axis='train_loss_over_val_loss',
            y_axis='train_auprc_over_val_auprc_new', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
            include_legend_titles=False,
            legend_titles=('Model',),
            legendspacing=0,
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.63, 0.99),
            marker_legend_bbox=(0.8, 0.8),
            layout_rect=(0.15, 0.08, 0.975, 0.99),
            xlim_bottom=0.45, xlim_top=1.26, ylim_bottom=0.97, ylim_top=1.125,  # L1 vs H3 vs H1
            # xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.98, ylim_top=1.08, # From H3 vs L2
            # xlim_top=1.6, xlim_bottom=0.4, ylim_top=1.015, ylim_bottom=0.9925,
            mean_size_factor=6,
            save_name='eachmodel_bestreg_dice_fits')
exit()

# =============================================================================================================
# # -----------------------ALL MODELS, BEST REG, EACH LOSS (4), TEST METRIC----------------------------------
#no
# generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'],
#             x_axis='reg_distance',
#             y_axis='test_auprc',
#             xlabel='Regularization distance',
#             ylabel='Test AUPRC', line_ellipse=le,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.50, 0.25),
#             legend_fontsize=55,
#             layout_rect=(0.13, 0.09, 0.98, 0.99),
#             legend_titles=('Loss Function',),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='allmodels_bestreg_eachloss')


# # -----------------------each MODELS, best REG, dice LOSS (8)----------------------------------
# no
# generic_gmm(groupby=['short_model_class'], incl_models=[],
#             query_strings=['reg_label=="l2+0.1"', 'loss=="dice"'], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc_new',
#             xlabel='Loss Ratio', ylabel='AUPRC Ratio',
#             line_ellipse=True,
#             include_legend_titles=False,
#             legend_titles=('Model',),
#             show_legend=True,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.01, 0.25),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.98, 0.99),
#             xlim_bottom=0.7, xlim_top=1.35, ylim_bottom=0.98, ylim_top=1.08,
#             mean_size_factor=6,
#             save_name='eachmodel_bestreg_dice')

# # -----------------------ALL MODELS, ALL REG, EACH LOSS (4)----------------------------------
# # allmodels_allreg_eachloss: Loss Ratio vs  AUPRC Ratio
# yes
# generic_gmm(groupby=['loss'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc_new', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Loss Function',),
#             show_legend=True,
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.56, 0.85),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.98, 0.99),
#             xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='allmodels_allreg_eachloss')

# # -----------------------ALL MODELS, EACH REG, ALL LOSS (4)----------------------------------
# Loss Ratio vs  AUPRC Ratio
# yes
# generic_gmm(groupby=['reg_label'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
#             y_axis='train_auprc_over_val_auprc_new', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
#             legend_titles=('Reg. Scheme',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.56, 0.85),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.12, 0.09, 0.98, 0.99),
#             xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='allmodels_eachreg_allloss')

# # -----------------------ALL MODELS, EACH REG, DICE LOSS (4)----------------------------------
# # NOT USED IN THESIS
# generic_gmm(groupby=['reg_label'], incl_models=[], query_strings=['loss == "dice"'], x_axis='reg_distance',
#             y_axis='val_auprc', xlabel='Regularization Distance', ylabel=' Validation AUPRC', line_ellipse=le,
#             legend_titles=('Reg. Scheme',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.55, 0.7),
#             layout_rect=(0.12, 0.1, 0.9, 0.95),
#             #xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='allmodels_eachreg_dice')


#
# # -----------------------ALL MODELS, BEST REG, EACH LOSS (4)----------------------------------
# yes
# generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'],
#             x_axis='reg_distance_new',
#             y_axis='val_auprc_new',
#             xlabel='Regularization distance',
#             ylabel='Validation AUPRC', line_ellipse=le,
#             fontsize=55,
#             figsize=(20, 15),
#             legendspacing=0.0,
#             color_legend_bbox=(0.50, 0.2),
#             legend_fontsize=55,
#             layout_rect=(0.13, 0.09, 0.98, 0.99),
#             legend_titles=('Loss Function',),
#             # xlim_top=1.2,
#             mean_size_factor=6,
#             save_name='allmodels_bestreg_eachloss')

# # -----------------------EACH MODEL, BEST REG, dice Loss (8)----------------------------------
# yes
# generic_gmm(groupby=['short_model_class'], incl_models=[], query_strings=['reg_label == "l2+0.1"', 'loss == "dice"'],
#             x_axis='reg_distance_new',
#             y_axis='val_auprc_new', xlabel='Regularization Distance', ylabel='Validation AUPRC', line_ellipse=le,
#             fontsize=55,
#             figsize=(20, 15),
#             legendspacing=0.0,
#             legend_borderpad=0,
#             # color_legend_bbox=(0.65, 0.29),
#             color_legend_bbox=(-0.02, 1.03),
#             include_legend_titles=False,
#             layout_rect=(0.12, 0.09, 0.97, 0.99),
#             legend_fontsize=55,
#             legend_titles=('Model',),
#             mean_size_factor=6,
#             save_name='eachmodels_bestreg_bestloss')

# yes, maybe
generic_gmm(groupby=['short_model_class'], incl_models=[], query_strings=['reg_label == "l2+0.1"', 'loss == "dice"'],
            x_axis='reg_distance_new',
            y_axis='val_f1', xlabel='Regularization Distance', ylabel='Validation F1', line_ellipse=le,
            fontsize=55,
            figsize=(20, 15),
            legendspacing=0.0,
            legend_borderpad=0,
            # color_legend_bbox=(0.65, 0.29),
            color_legend_bbox=(0.68, .5),
            include_legend_titles=False,
            layout_rect=(0.12, 0.09, 0.97, 0.99),
            legend_fontsize=55,
            legend_titles=('Model',),
            mean_size_factor=6,
            save_name='eachmodels_bestreg_bestloss')
exit()

# generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'], x_axis='val_auprc', legend_titles=('Loss Function',),
#             y_axis='val_auroc', xlabel='Validation AUPRC', ylabel='Validation AUROC', line_ellipse=le, save_name='allmodels_bestreg_eachloss')
#
#
# # allmodels_bestreg_eachloss:  val AUPRC  vs  val  AUROC
# generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'], x_axis='val_auprc',
#             y_axis='val_auroc', xlabel='Validation AUPRC', ylabel='Validation AUROC', line_ellipse=le, save_name='allmodels_bestreg_eachloss')
#
# # allmodels_bestreg_eachloss:  val AUPRC  vs  val  F1
# generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'], x_axis='val_f1',
#             y_axis='val_auroc', xlabel='Validation F1', ylabel='Validation AUROC', line_ellipse=le, save_name='allmodels_bestreg_eachloss')


# ======================= TABLES =======================================


# # eachmodel_worstreg_allloss
# table(groupby=['model_class'], table_fields=['reg_distance'], label_fields=None,
#       query_strings=['reg_label == "None+0.0"'], label_format='empty', std=True, save_name='eachmodel_worstreg_allloss')
#
# # eachmodel_bestreg_allloss
# table(groupby=['model_class'], table_fields=['reg_distance', 'val_auprc', 'val_auroc'], label_fields=None,
#       query_strings=['reg_label == "l2+0.1"'], label_format='empty', std=True, save_name='eachmodel_bestreg_allloss')
#
# # ------
# # allmodels_bestreg_eachloss
# table(groupby=['loss'], table_fields=['reg_distance', 'val_auprc', 'val_auroc'], label_fields=None,
#       query_strings=['reg_label == "l2+0.1"'], label_format='empty', std=True, save_name='allmodels_bestreg_eachloss')
#
# # allmodels_worstreg_eachloss
# table(groupby=['loss'], table_fields=['reg_distance'], label_fields=None,
#       query_strings=['reg_label == "None+0.0"'], label_format='empty', std=True, save_name='allmodels_worstreg_eachloss')
# # ------

# allmodels_eachreg_allloss
# table(groupby=['reg_label'], table_fields=['reg_distance', 'val_auprc', 'val_auroc'], label_fields=None,
#       query_strings=[], label_format='empty', std=True, save_name='allmodels_eachreg_allloss')

# allmodels_eachreg_eachloss
# table(groupby=['reg_label', 'loss'], table_fields=['val_auprc'], label_fields=None,
#       query_strings=[], label_format='empty', std=True, save_name='allmodels_eachreg_eachloss_v2')

# # ------

# allmodels_eachreg_eachloss
# yes
# table_with_two_groups(groupby=['reg_label', 'loss'],
#                       table_fields=['val_auprc_new', 'reg_distance_new'],
#                       label_fields=None,
#                       query_strings=[],
#                       label_format='empty',
#                       std=True,
#                       save_name='allmodels_eachreg_eachloss_v2')
