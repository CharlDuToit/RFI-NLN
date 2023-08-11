from utils import Namespace, ResultsCollection


def dummy_args(results_args_):
    # FIELDS IN excl_groupby WILL NOT BE IN GROUPED DATAFRAME

    results_args_.incl_models = ['UNET']
    # results_args_.incl_models = ['RNET']
    results_args_.excl_models = []
    # results_args_.excl_groupby = ['dropout']
    # results_args_.groupby = ['model', 'loss', 'patch_x', 'batch_size', 'filters']
    # results_args_.groupby = ['model', 'batch_size', 'dropout']
    # results_args_.groupby = ['model']
    results_args_.groupby = ['model', 'dropout', 'kernel_regularizer', 'batch_size']
    # results_args_.excl_groupby = []
    results_args_.datasets = ['LOFAR']
    # results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    # results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001', 'dropout == 0.1', 'batch_size == 64']
    # results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ', 'lr == 0.0001']
    # results_args_.query_strings = ['loss == "bce"']
    # results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "dice" ']
    # results_args_.query_strings = []
    # results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0']
    results_args_.query_strings = ['use_hyp_data == True', 'filters == 16', 'loss == "bce" ', 'patch_x == 64',
                                   'lr==0.0001', ]  # 'dropout==0.1', 'batch_size==1024']#'kernel_regularizer=="l1_l2"']

    # poster
    # results_args_.query_strings = ['use_hyp_data == False', 'filters == 64', 'loss == "bce" ', 'dropout == 0.0', 'batch_size == 64', 'kernel_regularizer == "None"', 'lr == 0.00001']
    # results_args_.query_strings = []
    results_args_.task = 'None'
    results_args_.table_fields = ['test_auroc', 'test_auprc', 'test_f1', 'val_auroc', 'val_auprc', 'val_f1']
    # results_args_.line_groups = ['model', 'batch_size', 'filters', 'loss']
    results_args_.std = True
    results_args_.params = False
    results_args_.x_axis = 'dropout'
    results_args_.y_axis = 'test_f1'
    # results_args_.x_axis = 'dropout'
    # results_args_.y_axis = 'val_f1'
    # results_args_.label_fields = ['model']
    results_args_.label_fields = ['model', 'dropout', 'kernel_regularizer', 'batch_size']
    # results_args_.label_fields = ['model', 'batch_size', 'dropout']
    # results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    # results_args_.label_fields = ['dropout', 'kernel_regularizer', 'lr', 'batch_size']
    results_args_.label_format = 'short'
    results_args_.output_path = '/home/ee487519/final_results/all/'
    results_args_.save_name = 'stuff'
    results_args_.save_path = './outputs/lofar_bce_unet_1e-4_csv'
    return results_args_


def main():
    output_path = '/home/ee487519/final_results/lofar_hera_new/'
    save_path = './mnemo_feb_2022/'
    # -----------------------------------------------------------
    # ns = Namespace(incl_models=['UNET'],
    #                excl_models=[],
    #                groupby=['model_class', 'dropout', 'clipper', 'scale_per_image',
    #                         'clip_per_image', 'shuffle_patches', 'final_activation', 'batch_size', 'patch_x', 'std_min',
    #                         'std_max', 'perc_min', 'perc_max', 'loss'],
    #                datasets=['LOFAR'],
    #                query_strings=['task == "train" ', 'use_hyp_data == True', 'filters == 16',
    #                               #'loss == "bce" ',
    #                               'patch_x == 64',
    #                               'lr==0.0001', ],
    #                task='scatter',
    #                x_axis='test_f1',
    #                y_axis='test_auprc',
    #                table_fields=[],
    #                std=True,
    #                params=False,
    #                label_fields=['model_class', 'dropout', 'scale_per_image', 'clip_per_image', 'shuffle_patches',
    #                              'loss', 'final_activation', 'batch_size', 'clipper', 'std_min', 'std_max', 'perc_min',
    #                              'perc_max', 'patch_x'],
    #                label_format='short',
    #                output_path=output_path,
    #                save_name='test',
    #                save_path=save_path + 'all')
    # rc = ResultsCollection(ns)
    # rc.perform_task()
    # -----------------------------------------------------------
    ns = Namespace(incl_models=['UNET'],
                   excl_models=[],
                   groupby=['model_class',
                            # 'dropout',
                            'clipper',
                            'scale_per_image',
                            'clip_per_image',
                            # 'shuffle_patches',
                            'final_activation',
                            'batch_size',
                            'patch_x',
                            'std_min',
                            'std_max',
                            'perc_min',
                            'perc_max',
                            'loss'],
                   datasets=['LOFAR'],
                   query_strings=['task == "train" ',
                                  'use_hyp_data == True',
                                  'filters == 16',
                                  # 'loss == "bce" ',
                                  'shuffle_patches == True ',
                                  'patch_x == 64',
                                  'lr==0.0001', ],
                   task='scatter',
                   x_axis='test_f1',
                   y_axis='test_auprc',
                   table_fields=[],
                   std=True,
                   params=False,
                   label_fields=[
                       # 'dropout',
                       'scale_per_image',
                       # 'shuffle_patches',
                       'loss',
                       'final_activation',
                       'clip_per_image',
                       'clipper',
                       'std_min',
                       'std_max',
                       'perc_min',
                       'perc_max'],
                   label_format='short',
                   output_path=output_path,
                   save_name='UNET_f1_auprc',
                   save_path=save_path + 'UNET')
    rc = ResultsCollection(ns)
    rc.perform_task()
    # -----------------------------------------------------------
    ns = Namespace(incl_models=['RNET5'],
                   excl_models=[],
                   groupby=['model_class',
                            # 'dropout',
                            'clipper',
                            'scale_per_image',
                            # 'clip_per_image',
                            # 'shuffle_patches',
                            'final_activation',
                            'batch_size',
                            'patch_x',
                            'std_min',
                            'std_max',
                            'perc_min',
                            'perc_max',
                            'loss'],
                   datasets=['LOFAR'],
                   query_strings=['task == "train" ',
                                  'use_hyp_data == True',
                                  'filters == 16',
                                  'clip_per_image == False',
                                  # 'loss == "bce" ',
                                  'shuffle_patches == True',
                                  'patch_x == 64',
                                  'lr==0.0001',
                                  'loss != "bbce" '],
                   task='scatter',
                   x_axis='test_f1',
                   y_axis='test_auprc',
                   table_fields=[],
                   std=True,
                   params=False,
                   label_fields=[
                       # 'dropout',
                       'scale_per_image',
                       # 'shuffle_patches',
                       'loss',
                       'final_activation',
                       # 'clip_per_image',
                       'clipper',
                       'std_min',
                       'std_max',
                       'perc_min',
                       'perc_max'],
                   label_format='short',
                   output_path=output_path,
                   save_name='RNET5_f1_auprc',
                   save_path=save_path + 'RNET5')
    rc = ResultsCollection(ns)
    rc.perform_task()
    # -----------------------------------------------------------
    ns = Namespace(incl_models=['RNET5'],
                   excl_models=[],
                   groupby=['model_class',
                            # 'dropout',
                            'clipper',
                            'scale_per_image',
                            # 'clip_per_image',
                            # 'shuffle_patches',
                            'final_activation',
                            'batch_size',
                            'patch_x',
                            'std_min',
                            'std_max',
                            'perc_min',
                            'perc_max',
                            'lr',
                            'loss',
                            'filters'],
                   datasets=['HERA_CHARL'],
                   query_strings=['task == "train" ',
                                  'use_hyp_data == True',
                                  #'filters == 16',
                                  'clip_per_image == False',
                                  # 'loss == "bce" ',
                                  'shuffle_patches == True',
                                  'patch_x == 64',
                                  # 'lr==0.0001',
                                  'loss != "bbce" ',
                                  ' ~(loss == "mse" and lr == 0.0001)'],
                   task='scatter',
                   x_axis='test_f1',
                   y_axis='test_auprc',
                   table_fields=[],
                   std=True,
                   params=False,
                   label_fields=[
                       # 'dropout',
                       # 'scale_per_image',
                       'lr',
                       'filters',
                       # 'shuffle_patches',
                       'loss',
                       'final_activation',
                       # 'clip_per_image',
                       'clipper',
                       'std_min',
                       'std_max',
                       'perc_min',
                       'perc_max',],
                   label_format='short',
                   output_path=output_path,
                   save_name='RNET5_f1_auprc',
                   save_path=save_path + 'RNET5_HERA')
    rc = ResultsCollection(ns)
    rc.perform_task()
    # -----------------------------------------------------------
    ns = Namespace(incl_models=['RNET5', 'UNET'],
                   excl_models=[],
                   groupby=['model_class',
                            # 'dropout',
                            'clipper',
                            'scale_per_image',
                            # 'clip_per_image',
                            # 'shuffle_patches',
                            'final_activation',
                            'batch_size',
                            'patch_x',
                            'std_min',
                            'std_max',
                            'perc_min',
                            'perc_max',
                            'lr',
                            'loss'],
                   datasets=['LOFAR'],
                   query_strings=['task == "train" ',
                                  'use_hyp_data == True',
                                  'filters == 16',
                                  'clip_per_image == False',
                                  'loss == "mse" ',
                                  'shuffle_patches == True',
                                  'patch_x == 64',
                                  # 'lr==0.0001',
                                  'loss != "bbce" ',
                                  ' ~(loss == "mse" and lr == 0.0001)'],
                   task='scatter',
                   x_axis='test_f1',
                   y_axis='test_auprc',
                   table_fields=[],
                   std=True,
                   params=False,
                   label_fields=[
                       'model_class',
                       # 'dropout',
                       # 'scale_per_image',
                       #'lr',
                       # 'shuffle_patches',
                       #'loss',
                       #'final_activation',
                       # 'clip_per_image',
                       'clipper',
                       'std_min',
                       'std_max',
                       'perc_min',
                       'perc_max'],
                   label_format='short',
                   output_path=output_path,
                   save_name='RNET5_UNET_f1_auprc_mse',
                   save_path=save_path + 'RNET5_UNET_mse')
    rc = ResultsCollection(ns)
    rc.perform_task()


if __name__ == '__main__':
    main()
