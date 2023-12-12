from args_constructor import preprocessing_query, common_gmm_options, common_recall_prec_options, common_fpr_tpr_options, common_bar_options
from utils import ResultsCollection


def plot_bar( query_strings=[], incl_models=[], save_name='dummy', **plot_kwargs):
    kwargs = common_args()
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'bar'
    kwargs['save_name'] = f'L2_L3_H3_H4_bar_{save_name}'

    plot_options = common_bar_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()


def common_args():
    """
    Fields to change after function returns kwargs:
    groupby, x_axis, y_axis, table_fields, label_fields, label_format, save_name, task, save_name
    """
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        groupby=[],
        datasets=['HERA_CHARL', 'HERA_CHARL_AOF', 'LOFAR'],
        # query_strings=preprocessing_query() + [ f'filters == 16', f'lr==0.0001',
        #                                        f'train_f1 > {min_f1}', f'train_with_test == False', f'limit == "None"',
        #                                        'kernel_regularizer == "l2"', 'dropout == 0.1'],
        query_strings= ['experiment=="L2" or experiment=="L3" or experiment=="H3" or experiment=="H4" ',
                        'trans_group != "new 28"'],
        task='bar',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=[],
        label_format=None,
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L2_L3_H3_H4',
    )
    return kwargs


plot_bar(column_name='test_auprc',
         group_1='short_model_class',
         group_2='trans_group',
         group_3='data',
         ylabel='Test AUPRC',
         fontsize=55,
         figsize=(20,15),
         save_name='auprc')

plot_bar(column_name='test_auroc',
         group_1='short_model_class',
         group_2='trans_group',
         group_3='data',
         ylabel='Test AUPRC',
         fontsize=55,
         figsize=(20,15),
         save_name='auroc')
# ======================= PREC/RECALL TPR/FPR =======================================

# fpr_tpr(groupby=['loss'], incl_models=[], # query_strings=['dropout==0.1', 'kernel_regularizer=="l2"'],
#         line_ellipse=True,
#         data_subset='val',
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
#         save_name='val_allmodels_eachloss')
#
# # eachmodel, best reg, mse
# recall_prec(groupby=['loss'], incl_models=[],
#             # query_strings=['loss=="mse"', 'dropout==0.1', 'kernel_regularizer=="l2"'],
#             line_ellipse=True,
#             # f1_contours=(0.65,),
#             data_subset='val',
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
#             save_name='val_allmodels_eachloss')
# exit()
