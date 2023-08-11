from args_constructor import preprocessing_query, common_gmm_options, common_bubble_options, common_fpr_tpr_options, common_recall_prec_options, common_line_options
from utils import ResultsCollection


# ======================= GMM PLOT =======================================
def line_agg(x_axis, y_axis, query_strings=[], incl_models=[], save_name='dummy', task='train', freeze_top_layers=None,
             min_f1=0.5, **plot_kwargs):
    kwargs = common_L3_args(task=task, freeze_top_layers=freeze_top_layers, min_f1=min_f1)
    kwargs['groupby'] = ['model_class']
    kwargs['label_fields'] = ['model_class']
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'line_agg'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'L3_line_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_line_options(**plot_kwargs)
    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= GMM PLOT =======================================


def gmm(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], task='train', save_name='',
        freeze_top_layers=None, min_f1=0.5,
        train_with_test=True, limit=True, **plot_kwargs):
    kwargs = common_L3_args(task=task, freeze_top_layers=freeze_top_layers, min_f1=min_f1,
                            train_with_test=train_with_test, limit=limit)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'L3_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= BUBBLE PLOT =======================================


def bubble(x_axis, y_axis, query_strings=[], incl_models=[], groupby=['model_class'], save_name='', task='train',
           freeze_top_layers=False, **plot_kwargs):
    kwargs = common_L3_args(task=task, freeze_top_layers=freeze_top_layers)
    kwargs['groupby'] = groupby
    kwargs['params'] = True
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'bubble_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'L3_bubble_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_bubble_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


# ======================= TABLES =======================================

def table(table_fields, query_strings=[], groupby=['model_class'], label_fields=None, label_format='empty', std=True,
          save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_L3_args(task=task, freeze_top_layers=freeze_top_layers, limit=limit, **kwargs)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'L3_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()


def table_with_two_groups(table_fields, query_strings=[], groupby=['short_model_class', 'lofar_trans_group'],
                          label_fields=None, label_format='empty', std=True,
                          save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_L3_args(task=task, freeze_top_layers=freeze_top_layers, limit=limit, **kwargs)
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table_with_two_groups'
    kwargs['std'] = std
    kwargs['save_name'] = f'L3_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L3_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'L3_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_L3_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'L3_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def common_L3_args(min_f1=0.5, task='train', freeze_top_layers=False, train_with_test=True, limit=True, **kwargs):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    limit_type: not_none,
    """
    L3_kwargs = dict(
        incl_models=[],
        excl_models=[],
        # groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=[],
        datasets=['LOFAR'],
        query_strings=preprocessing_query() + [f'filters == 16', f'lr==0.0001', # '(task=="train") or (task=="transfer_train")',
                                               f'test_f1 > {min_f1}', 'loss == "dice"', 'kernel_regularizer == "l2"',
                                               'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['model_class'],
        # label_fields=['model_class', 'loss'],
        label_format='empty',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L3',
    )

    # only include runs with a limited number of images, otherwise all
    if limit:
        L3_kwargs['query_strings'] += [f'limit != "None"']

    # only include runs trained on test data, otherwise all
    if train_with_test:
        L3_kwargs['query_strings'] += [f'train_with_test == True']

    if task is None or task != 'eval_test':
        L3_kwargs['query_strings'] += ['(task=="train") or (task=="transfer_train")']
    elif task is not None:
        L3_kwargs['query_strings'] += [f'task == "{task}"']

    if task == 'transfer_train' and freeze_top_layers is not None:
        L3_kwargs['query_strings'] += [f'freeze_top_layers == {freeze_top_layers}']
    return L3_kwargs

# ======================= PREC/RECALL TPR/FPR =======================================

fpr_tpr(groupby=['lofar_trans_group'], incl_models=[], # query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
        line_ellipse=True,
        data_subset='test',
        gmm_thresholds=(0.5,),
        size=30,
        linewidth=2,
        scatter_thresholds=True,
        scatter_gmm_means=True,
        scatter_gmm_points=False,
        # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
        show_legend=True,
        legend_titles=(None,),
        # legend_titles=('Loss Function',),
        fontsize=55,
        figsize=(20, 15),
        color_legend_bbox=(0.06, 0.35),
        legend_borderpad=0,
        legendspacing=0,
        layout_rect=(0.12, 0.09, 0.98, 0.99),
        # xlim_top=1.2,
        mean_size_factor=6,
        save_name='test_allmodels_eachgroup')

# eachmodel, best reg, mse
recall_prec(groupby=['lofar_trans_group'], incl_models=[],
            # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
            line_ellipse=True,
            # f1_contours=(0.65,),
            data_subset='test',
            gmm_thresholds=(0.5, ),
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
            save_name='test_allmodels_eachgroup')
exit()

# ======================= LINE PLOT =======================================

# compare min f1 0.0 to 0.5
# line_agg('limit', 'test_auprc', task='train', xlabel='Number of images', ylabel='Test AUPRC', save_name='scratch', min_f1=0.5)

# line_agg('limit', 'test_auroc', task='transfer_train', xlabel='Number of images', ylabel='Test AUROC', save_name='trans', freeze_top_layers=True, min_f1=0.6)

# ======================= GMM PLOT =======================================
# 'UNET', 'AC_UNEET', 'ASPP_UNET'
gmm('test_auprc', 'test_auroc', xlabel='Test AUPRC', ylabel='Test AUROC', groupby=['lofar_trans_group', 'short_model_class'],
                query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                               'lofar_trans_group != "0"',
                               '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                incl_models=['UNET', 'AC_UNET', 'ASPP_UNET'],
                line_ellipse=True,
                min_f1=0.5,
                task=None,
                train_with_test=False,
                limit=False,
                legend_titles=('Training Type', 'Model'),
                legendspacing=0,
                fontsize=55,
                figsize=(20,15),
                # color_legend_bbox=(-0.02, 0.17),
                color_legend_bbox=(1.2, 0.17),
                marker_legend_bbox =(-0.02, 0.17),
                layout_rect=(0.12, 0.08, 0.99, 0.99),
                xlim_top=0.7118, xlim_bottom=0.519, ylim_top=0.913, ylim_bottom=0.6949,
                # layout_rect=(0.09, 0.08, 0.7, 0.95),
                # xlim_top=1.2,
                mean_size_factor=6,
                save_name='transtype_U_AC_ASPP'
    )

# 'DUAL', 'MONO', 'RFI'
gmm('test_auprc', 'test_auroc', xlabel='Test AUPRC', ylabel='Test AUROC', groupby=['lofar_trans_group', 'short_model_class'],
                query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                               'lofar_trans_group != "0"',
                               '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                incl_models=['DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET', 'RFI_NET'],
                line_ellipse=True,
                min_f1=0.5,
                task=None,
                train_with_test=False,
                limit=False,
                legend_titles=('Training Type', 'Model'),
                legendspacing=0,
                fontsize=55,
                figsize=(20,15),
                # color_legend_bbox=(-0.02, 0.17),
                color_legend_bbox=(1.2, 0.17),
                marker_legend_bbox =(-0.02, 0.19),
                layout_rect=(0.12, 0.08, 0.99, 0.99),
                xlim_top=0.7118, xlim_bottom=0.519, ylim_top=0.913, ylim_bottom=0.6949,
                # layout_rect=(0.09, 0.08, 0.7, 0.95),
                # xlim_top=1.2,
                mean_size_factor=6,
                save_name='transtype_DUAL_MONO_RFI'
    )

# 'R5', 'R7',
gmm('test_auprc', 'test_auroc', xlabel='Test AUPRC', ylabel='Test AUROC', groupby=['lofar_trans_group', 'short_model_class'],
                query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                               'lofar_trans_group != "0"',
                               '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                incl_models=['RNET', 'RNET5'],
                line_ellipse=True,
                min_f1=0.5,
                task=None,
                train_with_test=False,
                limit=False,
                legend_titles=('Training Type', 'Model'),
                legendspacing=0,
                fontsize=55,
                figsize=(20,15),
                # color_legend_bbox=(-0.02, 0.17),
                color_legend_bbox=(1.2, 0.17),
                marker_legend_bbox =(-0.02, 0.19),
                layout_rect=(0.12, 0.08, 0.99, 0.99),
                xlim_top=0.7118, xlim_bottom=0.519, ylim_top=0.913, ylim_bottom=0.6949,
                # layout_rect=(0.09, 0.08, 0.7, 0.95),
                # xlim_top=1.2,
                mean_size_factor=6,
                save_name='transtype_R5_R7'
    )

#

#
# ALL MODELS,  'trans_group'
gmm('test_auprc', 'test_auroc', xlabel='Test AUPRC', ylabel='Test AUROC', groupby=['lofar_trans_group'],
                query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                               'lofar_trans_group != "0"',
                               '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                incl_models=[],
                line_ellipse=True,
                min_f1=0.5,
                task=None,
                train_with_test=False,
                limit=False,
                legend_titles=('Training Type',),
                legendspacing=0,
                fontsize=55,
                figsize=(20,15),
                color_legend_bbox=(-0.02, 0.17),
                marker_legend_bbox =(1, 0.55),
                layout_rect=(0.12, 0.08, 0.99, 0.99),
                xlim_top=0.7118, xlim_bottom=0.519, ylim_top=0.913, ylim_bottom=0.6949,
                # layout_rect=(0.09, 0.08, 0.7, 0.95),
                # xlim_top=1.2,
                mean_size_factor=6,
                save_name='transtype'
    )

# EACH MODELS, 'short_model_class', only fine tuned data
gmm('test_auprc', 'test_auroc', xlabel='Test AUPRC', ylabel='Test AUROC', groupby=['short_model_class'],
                query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                               'lofar_trans_group == "tune 14"',
                               '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                incl_models=[],
                line_ellipse=True,
                min_f1=0.5,
                task=None,
                train_with_test=False,
                limit=False,
                legend_titles=('Training Type',),
                legendspacing=0,
                fontsize=55,
                figsize=(20, 15),
                color_legend_bbox=(-0.02, 0.17),
                marker_legend_bbox=(1, 0.55),
                layout_rect=(0.12, 0.08, 0.99, 0.99),
                xlim_top=0.7118, xlim_bottom=0.519, ylim_top=0.913, ylim_bottom=0.6949,
                # layout_rect=(0.09, 0.08, 0.7, 0.95),
                # xlim_top=1.2,
                mean_size_factor=6,
                save_name='eachmodel_tune'
    )
# ======================= BUBBLE PLOT =======================================

# bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC')
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1')
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (s)')
# bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')
# bubble('test_auroc', 'test_auprc', xlabel='Test AUROC', ylabel='Test AUPRC')
# bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC')


# ======================= TABLES =======================================

# # ======================= MODEL, TRANSFER COUNT =======================================
# eachmodel
# run with min_f1 =0 and 0.5 and look at counts to see how many runs are excluded
# table(table_fields=['test_auroc', 'test_f1', 'test_auprc',],
#       task='transfer_train',
#       freeze_top_layers=True,
#       std=True,
#       save_name='trans',
#       min_f1=0.5)
#
#
#
# # ======================= MODEL, LIMIT SCRATCH COUNT =======================================
#
# #
# run with min_f1 =0 and 0.5 and look at counts to see how many runs are excluded
# table(table_fields=['test_auprc'],
#       groupby=['model_class', 'limit'],
#       train_with_test=True,
#       limit=True,
#       task='train',
#       freeze_top_layers=None,
#       std=True,
#       save_name='scratch',
#       min_f1=0.5)

# ======================= EACH TRANSFER GROUP AND MODEL=======================================

#
# run with min_f1 =0 and 0.5 and look at counts to see how many runs are excluded
table_with_two_groups(table_fields=['test_auprc', 'test_auroc'],
                      groupby=['short_model_class', 'lofar_trans_group'],
                      query_strings=['(task == "transfer_train" and freeze_top_layers==True) or (task == "train")',
                                     'lofar_trans_group != "0"',
                                     '(task == "transfer_train" and test_f1 > 0.0) or (task == "train")'],
                      train_with_test=False,
                      limit=False,
                      task=None,
                      freeze_top_layers=None,
                      std=True,
                      save_name='results_2g_0.5f1',
                      min_f1=0.5)
