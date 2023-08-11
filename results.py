from utils import results_args, ResultsCollection
from utils import to_dict


def main(kwargs: dict):
    rc = ResultsCollection(**kwargs)
    rc.perform_task()


def counts_args(results_args_, dataset='LOFAR', loss='bce', filters=16, lr=0.0001, use_hyp_data=True):
    results_args_.incl_models = []
    results_args_.excl_models = []
    results_args_.groupby = ['model_class', 'loss', 'kernel_regularizer', 'dropout']
    results_args_.datasets = [dataset]
    epochs = 50 if loss == 'bce' else 100
    results_args_.query_strings = [f'use_hyp_data == {use_hyp_data}', f'filters == {filters}', 'patch_x == 64', f'lr=={lr}',
                                   f'loss=="{loss}"', 'scale_per_image == False', 'clip_per_image == False', f'epochs == {epochs}',
                                   'train_f1 > 0.5']
    results_args_.task = 'None'
    results_args_.table_fields = []
    results_args_.std = True
    results_args_.params = False
    results_args_.x_axis = 'test_auroc'
    results_args_.y_axis = 'test_auprc'
    results_args_.label_fields = ['model_class', 'loss']
    results_args_.label_format = 'empty'  # full empty or short
    results_args_.output_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/'

    results_args_.save_name = f'{dataset}_{loss}_counts'
    results_args_.save_path = f'/home/ee487519/PycharmProjects/RFI-NLN-HPC/plots and tables/{dataset}_counts/'
    results_args_.plot_options = dict()
    return results_args_


def all_reg_counts(results_args_):
    for dataset in ('LOFAR', 'HERA_CHARL'):
        for loss in ('bce', 'mse', 'dice', 'logcoshdice'):
            results_args_ = counts_args(results_args_, dataset, loss)
            main(to_dict(results_args.results_args))

def dummy_args(results_args_):
    # FIELDS IN excl_groupby WILL NOT BE IN GROUPED DATAFRAME

    #results_args_.incl_models = ['UNET']
    #results_args_.incl_models = ['UNET', 'RNET', 'RNET5', 'RFI_NET']
    results_args_.incl_models = []
    # results_args_.incl_models = ['UNET','RFI_NET']
    #results_args_.incl_models = ['RNET']
    results_args_.excl_models = []
    #results_args_.excl_groupby = ['dropout']
    #results_args_.groupby = ['model_class', 'loss', 'patch_x', 'batch_size', 'filters']
    #results_args_.groupby = ['model_class', 'batch_size', 'dropout']
    # results_args_.groupby = ['model_class', 'loss']
    #results_args_.groupby = ['model_class', 'loss', 'dropout', 'kernel_regularizer']
    #results_args_.groupby = ['model_class', 'loss', 'dropout', 'kernel_regularizer', 'filters', 'epochs']
    results_args_.groupby = ['model_class', 'loss']
    results_args_.groupby = ['model_class', 'perc_max', 'perc_min', 'rescale', 'log', 'bn_first', 'patch_x', 'kernel_regularizer']
    #results_args_.groupby = ['model_class', 'kernel_regularizer', 'dropout']
    # results_args_.groupby = ['model_class', 'dropout', 'kernel_regularizer']
    #results_args_.excl_groupby = []
    results_args_.datasets = ['LOFAR']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001', 'dropout == 0.1', 'batch_size == 64']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    #results_args_.query_strings = ['loss == "bce"']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ']
    #results_args_.query_strings = []
    #results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0']
    # results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "bce" ', 'patch_x == 64', 'lr==0.0001',] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']
    results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'patch_x == 64', 'lr==0.0001', 'loss=="mse"'] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']
    results_args_.query_strings = ['use_hyp_data == True', 'dropout == 0.1', 'kernel_regularizer=="l2"', 'filters == 16'] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']
    results_args_.query_strings = ['use_hyp_data == True', 'dropout == 0.1', 'filters == 16'] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']

    #results_args_.query_strings = ['use_hyp_data == True',  'loss=="logcoshdice"'] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']

    #results_args_.query_strings = ['use_hyp_data == True']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'patch_x == 64', 'lr==0.0001']

    #results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0', 'batch_size == 64', 'kernel_regularizer == "None"', 'lr == 0.00001']
    #results_args_.query_strings = []
    results_args_.task = 'table'
    results_args_.table_fields = ['val_auprc', 'train_auprc_over_val_auprc', 'train_loss_over_val_loss']
    results_args_.table_fields = ['val_auprc', 'train_auprc_over_val_auprc', 'train_loss_over_val_loss']
    results_args_.table_fields = ['val_auprc', 'val_auroc', 'val_f1']
    results_args_.table_fields = ['val_auprc', 'test_auprc']
    #results_args_.table_fields = ['val_auprc', 'train_auprc_over_val_auprc', 'train_loss_over_val_loss']


    #results_args_.table_fields = ['test_auroc', 'test_auprc', 'test_f1', 'val_auroc', 'val_auprc', 'val_f1']
    #results_args_.line_groups = ['model', 'batch_size', 'filters', 'loss']
    results_args_.std = True
    results_args_.params = False
    results_args_.x_axis = 'test_auroc'
    results_args_.y_axis = 'test_auprc'
    #results_args_.x_axis = 'dropout'
    #results_args_.y_axis = 'val_f1'
    # results_args_.label_fields = ['model_class', 'loss']
    results_args_.label_fields = ['kernel_regularizer', 'dropout']
    results_args_.label_fields = ['model_class', 'loss']
    results_args_.label_fields = ['model_class', 'rescale', 'log', 'perc_min', 'perc_max', 'bn_first', 'patch_x', 'kernel_regularizer']
    #results_args_.label_fields = ['model_class', 'kernel_regularizer', 'dropout']
    #results_args_.label_fields = ['model', 'batch_size', 'dropout']
    #results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    #results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    results_args_.label_format = 'empty' # full empty or short
    #results_args_.output_path = '/home/ee487519/final_results/all/'
    results_args_.output_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/'
    results_args_.output_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/outputs/26 June/'
    results_args_.save_name = 'preprocess'
    results_args_.save_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/plots and tables/hera_charl_loss_comp/'
    results_args_.save_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/outputs/26 June/results'

    return results_args_

if __name__ == '__main__':
    all_reg_counts(results_args.results_args)
    #results_args.results_args = dummy_args(results_args.results_args)
    #main(to_dict(results_args.results_args))
