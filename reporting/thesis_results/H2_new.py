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

def val_test_table(table_fields, query_strings=[], groupby=['short_model_class'], label_fields=None, label_format='empty',
                   std=True,
                   save_name='dummy', task='train', freeze_top_layers=None, limit=True, **kwargs):
    kwargs = common_H2_args()
    kwargs['groupby'] = groupby
    kwargs['query_strings'] += query_strings
    kwargs['task'] = 'val_test_table'
    kwargs['std'] = std
    kwargs['save_name'] = f'H2_valtest_table_{save_name}'
    kwargs['table_fields'] = table_fields
    kwargs['label_fields'] = label_fields if label_fields is not None else groupby
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
        query_strings=preprocessing_query() + [f'use_hyp_data == False', f'filters == 16', f'lr==0.0001',#  f'task == "{task}"',
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
        # output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        # save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H2',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/H2_new',
    )
    return kwargs



# each model, val + test TN FP ....
# yes, appendix
#val_test_table(table_fields=['TP', 'TN', 'FP', 'FN', 'fpr_new', 'recall', 'precision', 'f1'], std=False, save_name='table_conf_mat_nostd')
#exit()

# ======================= PREC/RECALL TPR/FPR =======================================

# zoomed val and test
# each model dice loss
# MNRAS
recall_prec(groupby=['short_model_class'], incl_models=[],
            query_strings=['loss=="dice"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
            line_ellipse=False,
            aof_score='',
            # f1_contours=(0.628, 0.652, 0.674, 0.755),
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
            f1_contours=( 0.97,0.99),
            # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
            xlim_top=1.0, xlim_bottom=0.93,
            ylim_top=1.0, ylim_bottom=0.949,

            show_legend=True,
            legend_titles=(None, None, None),
            #legend_titles=('Loss Function',),
            fontsize=55,
            contour_fontsize=40,
            figsize=(20, 15),
            # color_legend_bbox=(0.65, 0.7), # withouth aoflagger
            color_legend_bbox=(0.0, 0.366), # with aoflagger
            line_legend_bbox=(0.04, 0.06),
            marker_legend_bbox=(1.686, 0.25),
            legend_borderpad=0,
            legendspacing=0,
            layout_rect=(0.14, 0.09, 0.97, 0.99),
            # xlim_top=1.2,
            mean_size_factor=9,
            save_name='valtest_eachmodel_zoomed')
exit()

# # zoomed
# # each model
# # yes, appendix
# fpr_tpr(groupby=['short_model_class'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             data_subset='test',
#             xlabel='Test FPR',
#             ylabel='Test Recall',
#             gmm_thresholds=(0.5, 0.1 ),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             xlim_top=0.15, xlim_bottom=-0.002, ylim_top=1.0, ylim_bottom=0.939,
#             show_legend=True,
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.29, 0.23),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_zoomed')

#
# # zoomed
# # eachmodel,
# yes, appendix
# recall_prec(groupby=['short_model_class'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=False,
#             data_subset='test',
#             xlabel='Test Recall',
#             ylabel='Test Precision',
#             gmm_thresholds=(0.5, 0.1),
#             size=60,
#             linewidth=2,
#             scatter_thresholds=False,
#             scatter_gmm_means=True,
#             scatter_gmm_points=False,
#             f1_contours=( 0.97,0.99),
#             # xlim_top=0.7, xlim_bottom=0.4, ylim_top=0.8, ylim_bottom=0.4,
#             xlim_top=1.0, xlim_bottom=0.94,
#             ylim_top=1.0, ylim_bottom=0.949,
#             show_legend=False, # put plot next to fpr_tpr
#             legend_titles=(None,),
#             #legend_titles=('Loss Function',),
#             fontsize=55,
#             figsize=(20, 15),
#             color_legend_bbox=(0.01, 0.25),
#             legend_borderpad=0,
#             legendspacing=0,
#             layout_rect=(0.14, 0.09, 0.97, 0.99),
#             # xlim_top=1.2,
#             mean_size_factor=9,
#             save_name='test_eachmodel_zoomed')
# exit()


# ======================= GMM PLOT =======================================

# MNRAS
gmm('val_f1', 'test_f1', xlabel='Validation F1', ylabel='Test F1',
                groupby = ['short_model_class'],
                legend_titles=(None,),
                fontsize=55,
                line_ellipse=True,
                figsize=(20, 15),
                color_legend_bbox=(0.6, 0.5),
                legendspacing=0.0,
                legend_borderpad=0.0,
                layout_rect=(0.13, 0.09, 0.97, 0.98),
                #xlim_top=.251,
    #xlim_bottom=-0.001,
                mean_size_factor=6,
                save_name='eachmodel')
exit()

# ======================= BUBBLE PLOT =======================================

# bubble('flops_image', 'test_auprc', xlabel='Floating Point Operations', ylabel='Test AUPRC')
# bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1')
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (s)')
# bubble('val_auprc', 'test_auprc', xlabel='Validation AUPRC', ylabel='Test AUPRC')
# bubble('test_auroc', 'test_auprc', xlabel='Test AUROC', ylabel='Test AUPRC')
# bubble('val_auroc', 'test_auroc', xlabel='Validation AUROC', ylabel='Test AUROC')





# ------------ F1 and FPR plots

#
# # Yes,
bubble('val_f1', 'test_f1', xlabel='Validation F1', ylabel='Test F1',
       fontsize=55,  size_legend=False, figsize=(20, 15),
       adjustment_set=2,
       layout_rect=(0.13, 0.09, 1.0, 0.99),
       # ylim_bottom=0.9625, ylim_top=0.995,
       )

# yes,
bubble('val_fpr_new', 'test_fpr_new', xlabel='Validation FPR', ylabel='Test FPR',
       fontsize=55,  size_legend=False, figsize=(20, 15),
       adjustment_set=2,
       xlim_top=10.1e-5,
       ylim_top=10.1e-5,
       #xlim_bottom=0.994,
       # ylim_bottom=0.9938,
      # logx=True,
       # logy=True,
       layout_rect=(0.13, 0.09, 1.01, 0.99))

bubble('test_fpr_new', 'test_f1', xlabel='Test FPR', ylabel='Test F1',
       fontsize=55,  size_legend=False, figsize=(20, 15),
       adjustment_set=2,
       xlim_top=10.1e-5,
       # ylim_bottom=0.9938,
      # logx=True,
       # logy=True,
       layout_rect=(0.13, 0.09, 1.01, 0.99))


# Yes,
bubble('flops_image', 'test_f1', xlabel='Floating Point Operations', ylabel='Test F1',
       fontsize=55, figsize=(20, 15),
       adjustment_set=2,
       legend_bbox=(0.97, 0.6),
       layout_rect=(0.13, 0.07, 1.0, 0.99),
       legendspacing=0.5, xtick_size=50,
       # ylim_bottom=0.9625, ylim_top=0.995
       )
# ------------ AUPRC and AUROC plots

# # Yes, appendix
# bubble('flops_image', 'test_auprc_new', xlabel='Floating Point Operations', ylabel='Test AUPRC',
#        fontsize=55, figsize=(20, 15),
#        adjustment_set=2,
#        legend_bbox=(0.9, 0.6),
#        layout_rect=(0.13, 0.07, 1.0, 0.99),
#        legendspacing=0.5, xtick_size=50,
#        ylim_bottom=0.9625, ylim_top=0.995
#        )
#
#
# # Yes, appendix
# bubble('val_auprc_new', 'test_auprc_new', xlabel='Validation AUPRC', ylabel='Test AUPRC',
#        fontsize=55,  size_legend=False, figsize=(20, 15),
#        adjustment_set=2,
#        layout_rect=(0.13, 0.09, 1.0, 0.99),
#        ylim_bottom=0.9625, ylim_top=0.995,
#        )
#
# # yes, appendix
# # Mono excluded, too low = 0.985
# bubble('val_auroc_new', 'test_auroc_new', xlabel='Validation AUROC', ylabel='Test AUROC',
#        fontsize=55,  size_legend=False, figsize=(20, 15),
#        adjustment_set=2,
#        xlim_bottom=0.994,
#        ylim_bottom=0.9938,
#       # logx=True,
#        # logy=True,
#        layout_rect=(0.13, 0.09, 1.01, 0.99))

# # yes
# bubble('flops_image', 'time_image', xlabel='Floating Point Operations', ylabel='Time per image (ms)',
#        fontsize=55, figsize=(20, 15), size_legend=False,
#        adjustment_set=2,
#        xtick_size=50,
#        layout_rect=(0.13, 0.07, 1.0, 1.0))



# ======================= TABLES =======================================

# eachmodel
#table(table_fields=['test_auroc', 'test_f1', 'test_auprc', 'val_auprc'], std=True, save_name='table')

# yes
#table(table_fields=['flops_image', 'params', 'time_image', 'test_auprc_new', 'test_auroc_new'], std=True, save_name='table_comp')


