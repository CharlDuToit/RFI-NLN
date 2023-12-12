from args_constructor import preprocessing_query, common_gmm_options, common_recall_prec_options, common_fpr_tpr_options, common_bar_options
from utils import ResultsCollection


# ======================= GENERIC GMM PLOTS =======================================
def generic_gmm(groupby, x_axis, y_axis, query_strings=[], incl_models=[], save_name='', **plot_kwargs):
    kwargs = common_args()
    kwargs['query_strings'] += ['dropout < 0.11']
    kwargs['query_strings'] += query_strings
    kwargs['groupby'] = groupby
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'scatter_gmm'
    kwargs['x_axis'] = x_axis
    kwargs['y_axis'] = y_axis
    kwargs['save_name'] = f'L2_H3_gmm_{x_axis}_{y_axis}_{save_name}'

    plot_options = common_gmm_options(**plot_kwargs)

    rc = ResultsCollection(plot_options=plot_options, **kwargs)
    rc.perform_task()

def plot_bar( query_strings=[], incl_models=[], save_name='dummy', **plot_kwargs):
    kwargs = common_args()
    kwargs['query_strings'] += query_strings
    kwargs['incl_models'] = incl_models
    kwargs['task'] = 'bar'
    kwargs['save_name'] = f'L2_H3_{save_name}'

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
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/final_results/',
        save_name=f'dummy',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/L2_H3_new',
    )
    return kwargs

# ------------------------------
# L2, H3, test vs val gmms
# maybe
generic_gmm(groupby=['short_model_class', 'data'], incl_models=[],
            query_strings=['reg_label == "l2+0.1"', 'loss == "dice"', 'experiment=="L2" or experiment=="H3"'],
            x_axis='val_f1',
            y_axis='test_f1',
            xlabel='Validation F1', ylabel='Test F1',
            line_ellipse=True,
            show_legend=True,
            fontsize=55,
            figsize=(20, 15),
            legendspacing=0.0,
            legend_borderpad=0,
            color_legend_bbox=(.685, 0.8),
            marker_legend_bbox=(0.01, 0.51),
            line_legend_bbox=(1.21, 0.3),
            layout_rect=(0.14, 0.09, 0.96, 0.99),
            legend_fontsize=55,
            include_legend_titles=False,
            legend_titles=('Model', 'Data'),
            mean_size_factor=6,
            save_name='eachmodels_L2_H3')
exit()
# maybe
generic_gmm(groupby=['short_model_class', 'data'], incl_models=[],
            query_strings=['reg_label == "l2+0.1"', 'loss == "dice"', 'experiment=="L2" or experiment=="H3"'],
            x_axis='val_auroc_new',
            y_axis='test_auroc_new',
            xlabel='Validation AUROC', ylabel='Test AUROC',
            line_ellipse=True,
            show_legend=True,
            fontsize=55,
            figsize=(20, 15),
            legendspacing=0.0,
            legend_borderpad=0,
            color_legend_bbox=(.685, 0.50),
            marker_legend_bbox=(0.01, 0.51),
            line_legend_bbox=(1.21, 0.3),
            layout_rect=(0.14, 0.09, 0.97, 0.99),
            legend_fontsize=55,
            include_legend_titles=False,
            legend_titles=('Model', 'Data'),
            mean_size_factor=6,
            save_name='eachmodels_L2_H3')

# maybe
generic_gmm(groupby=['short_model_class', 'data'], incl_models=[],
            query_strings=['reg_label == "l2+0.1"', 'loss == "dice"', 'experiment=="L2" or experiment=="H3"'],
            x_axis='val_auprc_new',
            y_axis='test_auprc_new',
            xlabel='Validation AUPRC', ylabel='Test AUPRC',
            line_ellipse=True,
            show_legend=True,
            fontsize=55,
            figsize=(20, 15),
            legendspacing=0.0,
            legend_borderpad=0,
            color_legend_bbox=(.67, 0.50),
            marker_legend_bbox=(0.01, 0.4),
            line_legend_bbox=(1.21, 0.3),
            layout_rect=(0.12, 0.09, 0.97, 0.99),
            legend_fontsize=55,
            include_legend_titles=False,
            legend_titles=('Model', 'Data'),
            mean_size_factor=6,
            save_name='eachmodels_L2_H3')
exit()

# ------------------------------
# HERA
# AUPRC
# plot_bar(column_name='test_auprc',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "HERA"'],
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
#          save_name='auprc_hera')

# AUROC
plot_bar(column_name='test_auroc',
         group_1='short_model_class',
         group_2='trans_group',
         query_strings=['data == "HERA"'],
         # group_3='data',
         ylabel='Test AUROC',
         ylim_bottom=0.6,
         ylim_top=1.0,
         color_legend_bbox=(0.4, 0.22),
         hatch_legend_bbox=(0.8, 0.1),
         label_1_ytext_factor=-12.6,
         layout_rect=(0.12, -0.05, 0.98, 0.98),
         fontsize=55,
         figsize=(20,15),
         save_name='auroc_hera')
# ------------------------------
# LOFAR
# AUPRC
# plot_bar(column_name='test_auprc',
#          group_1='short_model_class',
#          group_2='trans_group',
#          query_strings=['data == "LOFAR"'],
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
#          save_name='auprc_lofar')

# AUROC
plot_bar(column_name='test_auroc',
         group_1='short_model_class',
         group_2='trans_group',
         query_strings=['data == "LOFAR"'],
         # group_3='data',
         ylabel='Test AUROC',
         ylim_bottom=0.6,
         ylim_top=1.0,
         color_legend_bbox=(0.4, 0.22),
         hatch_legend_bbox=(0.8, 0.1),
         label_1_ytext_factor=-12.6,
         layout_rect=(0.12, -0.05, 0.98, 0.98),
         fontsize=55,
         figsize=(20,15),
         save_name='auroc_lofar')


# ------------------------------
# HERA AND LOFAR
# --------------------------
# AUPRC
plot_bar(column_name='test_auprc',
         group_1='short_model_class',
         group_2='trans_group',
         group_3='data',
         ylabel='Test AUPRC',
         ylim_bottom=0.4,
         ylim_top=1.0,
         color_legend_bbox=(0.4, 0.22),
         hatch_legend_bbox=(0.8, 0.1),
         label_1_ytext_factor=-8.7,
         layout_rect=(0.09, -0.05, 0.98, 0.98),
         fontsize=55,
         figsize=(20,15),
         save_name='auprc')

# AUROC
plot_bar(column_name='test_auroc',
         group_1='short_model_class',
         group_2='trans_group',
         group_3='data',
         ylabel='Test AUROC',
         ylim_bottom=0.6,
         ylim_top=1.0,
         color_legend_bbox=(0.4, 0.22),
         hatch_legend_bbox=(0.8, 0.1),
         label_1_ytext_factor=-12.6,
         layout_rect=(0.12, -0.05, 0.98, 0.98),
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
