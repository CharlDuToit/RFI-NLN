import pandas as pd
import glob
import os


def save_csv(data_name, seed, results_dict, output_path='./outputs', **kwargs):

    save_dict = results_dict
    for k in save_dict.keys():
        save_dict[k] = str(save_dict[k])
    save_df = pd.DataFrame([save_dict])

    # save_df = pd.DataFrame([results_dict])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file = os.path.join(output_path, 'results_{}_{}.csv'.format(data_name, seed))
    if not os.path.exists(file):
        save_df.to_csv(file, index=False)
    else:
        df = pd.read_csv(file)
        df = pd.concat([df, save_df])
        df.to_csv(file, index=False)


def load_csv(output_path='./outputs', **kwargs):
    """Loads every csv. Can filter it afterwards"""
    glob_str = '{}/results_*_*.csv'.format(output_path)
    files = glob.glob(glob_str)
    df_all = None
    for f in files:
        if df_all is None:
            df_all = pd.read_csv(f)
        else:
            df = pd.read_csv(f)
            df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all



def empty_dict():
    return {
        'model': None,
        'name': None,
        'parent_name': None, # new
        'data': None,
        'anomaly_class': None,
        'anomaly_type': None,
        'rfi_threshold': None,
        'ood_rfi': None,
        'lofar_subset': None,
        'limit': None,
        'num_train': None,
        'num_val': None,
        'split_seed': None, # new
        'use_hyp_data': None,
        'epochs': None,
        'batch_size': None,
        'dropout': None,
        'kernel_regularizer': None,  # new
        'final_activation': None, # new

        'raw_input_shape': None,
        'input_shape': None,
        'num_patches': None,
        'patch_x': None,
        'patch_y': None,

        'lr': None,
        'loss': None,

        'std_plus': None,
        'std_minus': None,
        'scale_per_image': None,
        'clip_per_image': None,
        'combine_min_std_plus': None,

        'latent_dim': None,
        'alpha': None,
        'neighbour': None,
        'algorithm': None,

        'model_config': None,
        'filters': None,
        'height': None,
        'level_blocks': None,
        'dilation_rate': None,

        'early_stop': None, # new
        #'first_epoch': None,
        #'best_epoch': None,

        'trainable_params': None,
        'nontrainable_params': None,

        'last_epoch': None,
        'epoch_time': None,

        'time_image': None,
        'time_patch': None,

        'flops_image': None,
        'flops_patch': None,

        'train_loss': None,
        'val_loss': None,
        #'opt_thresh':  None,

        'val_auroc': None,
        'val_auprc': None,
        'val_f1': None,

        'test_auroc': None,
        'test_auprc': None,
        'test_f1': None,

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

if __name__ == '__main__':
    # Test
    save_csv('test', '44', {})
    save_csv('test', '45', {})
    df_ = load_csv()
    df_.to_csv('./outputs/both.csv', index=False)
