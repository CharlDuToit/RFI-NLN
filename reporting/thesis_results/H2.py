from args_constructor import preprocessing_query, common_gmm_options, common_bubble_options, common_fpr_tpr_options, common_recall_prec_options
from utils import ResultsCollection



# ======================= GMM PLOT =======================================


def gmm(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_H2_args()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'gmm_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H2_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

# ======================= BUBBLE PLOT =======================================


def bubble(x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_H2_args()
    kwargs['params'] = True
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    # kwargs['save_name'] = f'bubble_{save_name}_{x_axis}_{y_axis}'
    kwargs['save_name'] = f'H2_bubble_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_bubble_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()
# ======================= GENERIC RECALL-PREC/ ROC PLOTS =======================================
def recall_prec(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H2_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'recall_prec_curve'
    kwargs['save_name'] = f'H2_recall_prec_{save_name}'

    plot_options = common_recall_prec_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def fpr_tpr(groupby, query_strings=[], incl_models=[], save_name='dummy', task='eval_test', **plot_kwargs):
    kwargs = common_H2_args(task=task)
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'fpr_tpr_curve'
    kwargs['save_name'] = f'H2_fpr_tpr_{save_name}'

    plot_options = common_fpr_tpr_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()
# ======================= TABLES =======================================

def table(table_fields, query_strings=[], label_format='empty', std=True, save_name='dummy'):
    kwargs = common_H2_args()
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H2_table_{save_name}'
    kwargs['table_fields'] = table_fields
    # kwargs['label_fields'] = ['model_class']
    kwargs['label_format'] = label_format
    # kwargs['query_strings'] += ['dropout < 0.11'] want to include 0.2

    rc = ResultsCollection(plot_options=dict(), **kwargs)
    rc.perform_task()

def common_H2_args(min_f1=0.5, task='train'):
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        #groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        groupby=['short_model_class'],
        datasets=['HERA_CHARL'],
        query_strings=preprocessing_query() + [f'use_hyp_data == False', f'filters == 16', f'lr==0.0001', f'task == "{task}"',
                                               f'train_f1 > {min_f1}', f'train_with_test == False', f'limit == "None"',
                                               'loss == "dice"', 'kernel_regularizer == "l2"', 'dropout == 0.1'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['short_model_class'],
        label_format='empty',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H2',
    )
    return kwargs

# ======================= PREC/RECALL TPR/FPR =======================================

fpr_tpr(groupby=['short_model_class'], incl_models=[], # query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
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
        save_name='test_eachmodel')

# eachmodel, best reg, mse
recall_prec(groupby=['short_model_class'], incl_models=[],
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
            save_name='test_eachmodel')
exit()


# ======================= GMM PLOT =======================================

# gmm('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')

# ======================= BUBBLE PLOT =======================================

# bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC')
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1')
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (s)')
# bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')
# bubble('test_auroc', 'test_auprc', xlabel='Test AUROC', ylabel='Test AUPRC')
# bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC')

bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC',
       fontsize=55, figsize=(20, 15),
       adjustment_set=2,
       legend_bbox=(0.54, 0.6),
       layout_rect=(0.13, 0.07, 1.0, 0.99),
       legendspacing=0.5, xtick_size=50,
       ylim_bottom=0.97, ylim_top=1.0)

bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (ms)',
       fontsize=55, figsize=(20, 15), size_legend=False,
       adjustment_set=2,
       xtick_size=50,
       layout_rect=(0.13, 0.07, 1.0, 1.0))

bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC',
       fontsize=55,  size_legend=False, figsize=(20, 15),
       adjustment_set=2,
       layout_rect=(0.13, 0.09, 1.0, 1.0),
       ylim_top=0.996)

bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC',
       fontsize=55,  size_legend=False, figsize=(20, 15),
       adjustment_set=2,
       layout_rect=(0.13, 0.09, 1.0, 1.0))



# ======================= TABLES =======================================

# eachmodel
#table(table_fields=['test_auroc', 'test_f1', 'test_auprc', 'val_auprc'], std=True, save_name='table')

table(table_fields=['flops_image', 'params', 'time_image', 'test_auprc', 'test_auroc'], std=True, save_name='table_comp')


