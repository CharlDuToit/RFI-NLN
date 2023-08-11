from args_constructor import preprocessing_query, common_gmm_options, common_recall_prec_options, common_fpr_tpr_options
from utils import ResultsCollection


# ======================= PER MODEL/LOSS/REG GMM PLOTS =======================================

# SELECT ONE MODEL, GROUPBY (REG, LOSS)


def model_gmm(model_class='UNET', x_axis='train_auprc', y_axis='val_auprc', **plot_kwargs):
    kwargs = common_L1_kwargs()
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
    kwargs = common_L1_kwargs()
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
    kwargs = common_L1_kwargs()
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
    kwargs = common_L1_kwargs()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L1_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L1_kwargs(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'L1_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L1_kwargs(task=task)
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
    kwargs = common_L1_kwargs()
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
    kwargs = common_L1_kwargs(**kwargs)
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


def common_L1_kwargs(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        # groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=[],
        datasets=['LOFAR'],
        query_strings=preprocessing_query() + [f'use_hyp_data == True', f'filters == 16', f'lr==0.0001',
                                               f'task=="{task}"',
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
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        #output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L1',
    )
    return kwargs

# ======================= ROC / PREC RECALL PLOTS =======================================
#
# recall_prec(groupby=['loss'], incl_models=[], query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#             task='eval_test',
#             line_ellipse=True,
#             data_subset='val',
#             gmm_thresholds=(0.5, 0.1),
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
#             save_name='valset_allmodels_bestreg_eachloss')


# fpr_tpr(groupby=['loss'], incl_models=[], query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#         line_ellipse=True,
#         data_subset='test',
#         gmm_thresholds=(0.5, 0.1),
#         size=30,
#         linewidth=2,
#         scatter_thresholds=True,
#         scatter_gmm_means=True,
#         scatter_gmm_points=False,
#         # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
#         show_legend=False,
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
#         save_name='allmodels_allreg_eachloss')

# eachmodel, best reg, mse
recall_prec(groupby=['short_model_class'], incl_models=[],
            query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
            line_ellipse=True,
            f1_contours=(0.7, 0.75),
            data_subset='val',
            gmm_thresholds=(0.5, 0.1, 0.01),
            size=30,
            linewidth=2,
            scatter_thresholds=True,
            scatter_gmm_means=True,
            scatter_gmm_points=False,
            # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
            show_legend=True,
            legend_titles=(None,),
            #legend_titles=('Loss Function',),
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.06, 0.35),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.12, 0.09, 0.98, 0.99),
            # xlim_top=1.2,
            mean_size_factor=6,
            save_name='valset_eachmodel_bestreg_mse')

exit()

# ======================= PER MODEL/LOSS/REG GMM PLOTS =======================================
le = True
# for mc in ('UNET', 'AC_UNET', 'ASPP_UNET', 'RNET', 'RNET5', 'RFI_NET', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET'):
# # #for mc in ('UNET',):
# #         #model_gmm(model_class=mc, x_axis='train_auprc', y_axis='val_auprc', xlabel='Training AUPRC', ylabel='Validation AUPRC', line_ellipse=le)
# #         #model_gmm(model_class=mc, x_axis='train_loss', y_axis='val_loss', xlabel='Training Loss', ylabel='Validation Loss', line_ellipse=le)
# #
#     model_gmm(model_class=mc, x_axis='train_loss_over_val_loss', y_axis='train_auprc_over_val_auprc',
#               xlabel='Loss Ratio',
#               ylabel='AUPRC Ratio',
#               line_ellipse=le,
#               show_legend=False,
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


# =============================================================================================================
# # -----------------------ALL MODELS, ALL REG, EACH LOSS (4)----------------------------------
# # allmodels_allreg_eachloss: Loss Ratio vs  AUPRC Ratio
generic_gmm(groupby=['loss'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
            y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
            legend_titles=('Loss Function',),
            show_legend=True,
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.56, 0.85),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.12, 0.09, 0.98, 0.99),
            xlim_top=1.2,
            mean_size_factor=6,
            save_name='allmodels_allreg_eachloss')

# # -----------------------ALL MODELS, EACH REG, ALL LOSS (4)----------------------------------
generic_gmm(groupby=['reg_label'], incl_models=[], query_strings=[], x_axis='train_loss_over_val_loss',
            y_axis='train_auprc_over_val_auprc', xlabel='Loss Ratio', ylabel='AUPRC Ratio', line_ellipse=le,
            legend_titles=('Reg. Scheme',),
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.56, 0.85),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.12, 0.09, 0.98, 0.99),
            xlim_top=1.2,
            mean_size_factor=6,
            save_name='allmodels_eachreg_allloss')

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
generic_gmm(groupby=['loss'], incl_models=[], query_strings=['reg_label == "l2+0.1"'],
            x_axis='reg_distance',
            y_axis='val_auprc',
            xlabel='Regularization distance',
            ylabel='Validation AUPRC', line_ellipse=le,
            fontsize=55,
            figsize=(20, 15),
            color_legend_bbox=(0.50, 0.25),
            legend_fontsize=55,
            layout_rect=(0.13, 0.09, 0.98, 0.99),
            legend_titles=('Loss Function',),
            # xlim_top=1.2,
            mean_size_factor=6,
            save_name='allmodels_bestreg_eachloss')

# # -----------------------EACH MODEL, BEST REG, dice Loss (8)----------------------------------
generic_gmm(groupby=['short_model_class'], incl_models=[], query_strings=['reg_label == "l2+0.1"', 'loss == "dice"'],
            x_axis='reg_distance',
            y_axis='val_auprc', xlabel='Regularization Distance', ylabel='Validation AUPRC', line_ellipse=le,
            fontsize=55,
            figsize=(20, 15),
            legendspacing=0.2,
            # legend_borderpad=0,
            color_legend_bbox=(0.65, 0.34),
            layout_rect=(0.12, 0.09, 0.98, 0.99),
            legend_fontsize=55,
            legend_titles=('Model',),
            mean_size_factor=6,
            save_name='eachmodels_bestreg_bestloss')

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
table_with_two_groups(groupby=['reg_label', 'loss'],
                      table_fields=['val_auprc', 'reg_distance'],
                      label_fields=None,
                      query_strings=[],
                      label_format='empty',
                      std=True,
                      save_name='allmodels_eachreg_eachloss_v2')
