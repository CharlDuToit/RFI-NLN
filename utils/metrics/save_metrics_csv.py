import tensorflow as tf
import numpy as np
import os
import pandas as pd
#from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
#from math import isnan
#from inference import infer, get_error
#from utils.data import reconstruct


def save_metrics_csv(model_type,
                     train_data,
                     test_masks,
                     test_masks_orig,
                     alpha,
                     neighbour,
                     args,
                     **kwargs):
    """
        Either appends or saves a new .csv file with the top K 

        Parameters
        ----------
        model_type (str): type of model (vae,ae,..)
        args (Namespace):  arguments from utils.args
        ... (optional arguments)

        Returns
        -------
        nothing
    """
    if not os.path.exists('outputs/results_{}_{}.csv'.format(args.data,
                                                             args.seed)):
        df = pd.DataFrame(columns=['Model',
                                   'Name',
                                   'Latent_Dim',
                                   'Patch_Size',
                                   'Class',
                                   'Type',
                                   'Alpha',
                                   'Neighbour',
                                   'Percentage Anomaly',
                                   'N_Training_Samples',
                                   'RFI_Threshold',
                                   'OOD_RFI',

                                   'AUROC_AO',
                                   'AUROC_TRUE',
                                   'AUPRC_AO',
                                   'AUPRC_TRUE',
                                   'F1_AO',
                                   'F1_TRUE',

                                   'NLN_AUROC_AO',
                                   'NLN_AUROC_TRUE',
                                   'NLN_AUPRC_AO',
                                   'NLN_AUPRC_TRUE',
                                   'NLN_F1_AO',
                                   'NLN_F1_TRUE',

                                   'DISTS_AUROC_AO',
                                   'DISTS_AUROC_TRUE',
                                   'DISTS_AUPRC_AO',
                                   'DISTS_AUPRC_TRUE',
                                   'DISTS_F1_AO',
                                   'DISTS_F1_TRUE',

                                   'COMBINED_AUROC_AO',
                                   'COMBINED_AUROC_TRUE',
                                   'COMBINED_AUPRC_AO',
                                   'COMBINED_AUPRC_TRUE',
                                   'COMBINED_F1_AO',
                                   'COMBINED_F1_TRUE',

                                   'Flops',
                                   'Time',
                                   'Trainable_params'
                                   'Nontrainable_params'
                                   'Filters',
                                   'Levels',
                                   'Level_blocks',
                                   'Model_config'])
    else:
        df = pd.read_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                            args.seed))

    perc = round(((np.sum(test_masks) - np.sum(test_masks_orig)) / np.prod(test_masks_orig.shape)), 3)
    df = df.append({'Model': model_type,
                    'Name': args.model_name,
                    'Latent_Dim': args.latent_dim,
                    'Patch_Size': args.patch_x,
                    'Class': args.anomaly_class,
                    'Type': args.anomaly_type,
                    'Alpha': alpha,
                    'Neighbour': neighbour,
                    'Percentage Anomaly': perc,
                    'N_Training_Samples': len(train_data),
                    'RFI_Threshold': args.rfi_threshold,
                    'OOD_RFI': args.rfi,

                    'AUROC_AO': kwargs['ae_ao_auroc'],
                    'AUROC_TRUE': kwargs['ae_true_auroc'],
                    'AUPRC_AO': kwargs['ae_ao_auprc'],
                    'AUPRC_TRUE': kwargs['ae_true_auprc'],
                    'F1_AO': kwargs['ae_ao_f1'],
                    'F1_TRUE': kwargs['ae_true_f1'],

                    'NLN_AUROC_AO': kwargs['nln_ao_auroc'],
                    'NLN_AUROC_TRUE': kwargs['nln_true_auroc'],
                    'NLN_AUPRC_AO': kwargs['nln_ao_auprc'],
                    'NLN_AUPRC_TRUE': kwargs['nln_true_auprc'],
                    'NLN_F1_AO': kwargs['nln_ao_f1'],
                    'NLN_F1_TRUE': kwargs['nln_true_f1'],

                    'DISTS_AUROC_AO': kwargs['dists_ao_auroc'],
                    'DISTS_AUROC_TRUE': kwargs['dists_true_auroc'],
                    'DISTS_AUPRC_AO': kwargs['dists_ao_auprc'],
                    'DISTS_AUPRC_TRUE': kwargs['dists_true_auprc'],
                    'DISTS_F1_AO': kwargs['dists_ao_f1'],
                    'DISTS_F1_TRUE': kwargs['dists_true_f1'],

                    'COMBINED_AUROC_AO': kwargs['combined_ao_auroc'],
                    'COMBINED_AUROC_TRUE': kwargs['combined_true_auroc'],
                    'COMBINED_AUPRC_AO': kwargs['combined_ao_auprc'],
                    'COMBINED_AUPRC_TRUE': kwargs['combined_true_auprc'],
                    'COMBINED_F1_AO': kwargs['combined_ao_f1'],
                    'COMBINED_F1_TRUE': kwargs['combined_true_f1'],

                    'flops': kwargs['flops'],
                    'time': kwargs['tot_time'],
                    'trainable_params': kwargs['trainable_params'],
                    'nontrainable_params': kwargs['nontrainable_params'],
                    'filters': args.filters,
                    'level_blocks': args.level_blocks,
                    'model_config': args.model_config
                    }, ignore_index=True)

    df.to_csv('outputs/results_{}_{}.csv'.format(args.data,
                                                 args.seed), index=False)


def save_results_csv(data_name, seed, results_dict):
    none_dict = empty_dict()
    save_dict = {**none_dict, **results_dict}
    for k in save_dict.keys():
        save_dict[k] = str(save_dict[k] )
    #index = [i for i in range(len(save_dict))]
    save_df = pd.DataFrame([save_dict])
    dir_path = 'outputs/results_{}_{}.csv'.format(data_name, seed)
    if not os.path.exists(dir_path):
        save_df.to_csv(dir_path, index=False)
    else:
        df = pd.read_csv(dir_path)
        df = pd.concat([df, save_df])
        df.to_csv(dir_path, index=False)

    #perc = round(((np.sum(test_masks) - np.sum(test_masks_orig)) / np.prod(test_masks_orig.shape)), 3)


def empty_dict():
    return {
        'model': None,
        'name': None,
        'data': None,
        'anomaly_class': None,
        'anomaly_type': None,
        'rfi_threshold': None,
        'ood_rfi': None,
        'lofar_subset': None,
        'num_training': None,

        'raw_input_shape': None,
        'num_patches': None,
        'patch_x': None,
        'patch_y': None,
        'std_plus': None,
        'std_minus': None,
        'per_image': None,
        'combine_min_std_plus' : None,

        'latent_dim': None,
        'alpha': None,
        'neighbour': None,
        'algorithm': None,

        'model_config': None,
        'filters': None,
        'height': None,
        'level_blocks': None,

        'trainable_params': None,
        'nontrainable_params': None,

        'flops_patch': None,
        'time_patch': None,
        'flops_image': None,
        'time_image': None,

        'auroc': None,
        'auprc': None,
        'f1': None,

        'ae_auroc': None,
        'ae_auprc': None,
        'ae_f1': None,

        'nln_auroc': None,
        'nln_auprc': None,
        'nln_f1': None,

        'dists_auroc': None,
        'dists_auprc': None,
        'dists_f1': None,

        'combined_auroc': None,
        'combined_auprc': None,
        'combined_f1': None,


    }
