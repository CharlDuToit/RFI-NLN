from utils import results_args, ResultsCollection


def main():
    rc = ResultsCollection(results_args.results_args)
    rc.perform_task()


def dummy_args(results_args_):
    # FIELDS IN excl_groupby WILL NOT BE IN GROUPED DATAFRAME

    results_args_.incl_models = ['AC_UNET']
    #results_args_.incl_models = ['RNET']
    results_args_.excl_models = []
    #results_args_.excl_groupby = ['dropout']
    #results_args_.groupby = ['model', 'loss', 'patch_x', 'batch_size', 'filters']
    #results_args_.groupby = ['model', 'batch_size', 'dropout']
    #results_args_.groupby = ['model']
    results_args_.groupby = ['model', 'dropout', 'kernel_regularizer', 'batch_size']
    #results_args_.excl_groupby = []
    results_args_.datasets = ['LOFAR']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001', 'dropout == 0.1', 'batch_size == 64']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    #results_args_.query_strings = ['loss == "bce"']
    #results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ']
    #results_args_.query_strings = []
    #results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0']
    results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'patch_x == 64', 'lr==0.0001',] #'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']

    # poster
    #results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0', 'batch_size == 64', 'kernel_regularizer == "None"', 'lr == 0.00001']
    #results_args_.query_strings = []
    results_args_.task = 'val_loss'
    results_args_.table_fields = ['test_auroc', 'test_auprc', 'test_f1', 'val_auroc', 'val_auprc', 'val_f1']
    #results_args_.line_groups = ['model', 'batch_size', 'filters', 'loss']
    results_args_.std = True
    results_args_.params = False
    results_args_.x_axis = 'dropout'
    results_args_.y_axis = 'test_f1'
    #results_args_.x_axis = 'dropout'
    #results_args_.y_axis = 'val_f1'
    #results_args_.label_fields = ['model']
    results_args_.label_fields = ['model', 'dropout', 'kernel_regularizer', 'batch_size']
    #results_args_.label_fields = ['model', 'batch_size', 'dropout']
    #results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    #results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    results_args_.label_format = 'short'
    results_args_.output_path = '/home/ee487519/final_results/all/'
    results_args_.save_name = 'stuff'
    results_args_.save_path = './outputs/test_v3'
    return results_args_

if __name__ == '__main__':
    results_args.results_args = dummy_args(results_args.results_args)
    main()