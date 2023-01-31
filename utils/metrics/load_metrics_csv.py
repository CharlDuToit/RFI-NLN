import pandas as pd
import glob


def load_csv(data_name, dir_path=None):
    if dir_path is None:
        dir_path = './outputs'
    dir_path = '{}/results_{}_*.csv'.format(dir_path, data_name)
    files = glob.glob(dir_path)
    df_all = None
    for f in files:
        if df_all is None:
            df_all = pd.read_csv(f)
        else:
            df = pd.read_csv(f)
            df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all


def extract_results(df: pd.DataFrame, use_hyp_data=False):
    """Dataframe as loaded from load_results_csv"""

    groupby_cols = [
        'data', 'model', 'anomaly_class', 'anomaly_type', 'rfi_threshold', 'ood_rfi', 'limit', 'use_hyp_data', 'epochs',
        'batch_size', 'lr', 'std_plus', 'std_minus', 'per_image', 'filters', 'height', 'dropout', 'kernel_regularizer',
        'level_blocks', 'dilation_rate', 'patch_x', 'patch_y', 'loss', 'lofar_subset', 'latent_dim', 'alpha',
        'neighbour', 'algorithm', 'combine_min_std_plus'
    ]
    # dfg = df.groupby(groupby_cols).agg({'f1': ['mean', 'std']}).reset_index(drop=True) # this one works
    # dfg = df.groupby(groupby_cols, as_index=False).agg({'f1': ['mean', 'std']}).reset_index(drop=True)
    # dfg = df.groupby(groupby_cols, as_index=False).agg({'f1': ['mean', 'std']}).reset_index() #this one too
    dfg = df.groupby(groupby_cols, as_index=False).agg(
        {
            'f1': ['mean', 'std'],  # test f1
            'auroc': ['mean', 'std'],
            'auprc': ['mean', 'std'],
            'time_image': ['mean', 'std'],
            'flops_image': ['mean', 'std'],  # trainable params, # last epoch
        }
    )
    label_list = []
    f1_list = []
    auroc_list = []
    auprc_list = []
    time_image_list = []
    flops_image_list = []
    for i in range(dfg.shape[0]):
        label = get_label_old(dfg.iloc[i], use_hyp_data)
        if label is not None:
            label_list.append(label)
            f1_list.append(float(dfg.iloc[i]['f1']['mean']))
            auroc_list.append(float(dfg.iloc[i]['auroc']['mean']))
            auprc_list.append(float(dfg.iloc[i]['auprc']['mean']))
            time_image_list.append(float(dfg.iloc[i]['time_image']['mean']))
            flops_image_list.append(int(dfg.iloc[i]['flops_image']['mean']))

    return label_list, f1_list, auroc_list, auprc_list, time_image_list, flops_image_list


# perhaps have many params and then call get_label with **kwargs ?
# params for how label msut be built
def get_label_old(row: pd.core.series.Series, use_hyp_data=False):
    """Get labels for a row, can return None. The labels of the groups will not contain these columns, as they are
    assumed to all have the same value as specified below. E.g change epochs to 50 to save all rows with epochs equal
    to 50. When an if statement is removed, then it is recommended to add it to the label

    Assume same std_plus, std_minus for all rows, and patch_x == patch_y
    """

    if row['anomaly_class'][0] != 'rfi':
        return None
    if row['limit'][0] != 'None':
        return None
    if row['epochs'][0] != 150:
        return None
    if row['per_image'][0] != False:
        return None
    if row['level_blocks'][0] != 1:
        return None
    if row['ood_rfi'][0] != 'None':
        return None
    if row['lofar_subset'][0] != 'full':
        return None
    if row['rfi_threshold'][0] != 'None':
        return None
    if row['use_hyp_data'][0] != use_hyp_data:
        return None

    label = '{} f-{} h-{} p-{} b-{} lr-{} lo-{}'.format(
        row['model'][0],
        row['filters'][0],
        row['height'][0],
        row['patch_x'][0],
        row['batch_size'][0],
        row['lr'][0],
        row['loss'][0],
    )
    if row['model'][0] in ('AC_UNET', 'ASPP_UNET'):
        label += ' dr-{}'.format(row['dilation_rate'][0])

    return label


def all_true(row: pd.core.series.Series, grouped_df, filter_dict ):
    """Returns True if all (key,value) pairs in filter_dict match with (column,value) pairs in row.
    Value of None will be ignored. A non-None value with a key that is not in row columns will throw error.
    Set filter_dict[key] = 'None' to search for None values stored in .csv file.
    Keys in filter_dict must be in groupby_cols else an error will be thrown"""
    for k in filter_dict.keys():
        if filter_dict[k] is not None:
            if grouped_df:
                if filter_dict[k] != row[k][0]:
                    return False
            else:
                if filter_dict[k] != row[k]:
                    return False
    return True


def get_label(row: pd.core.series.Series, grouped_df, model, filters, height, patch, batch_size, lr, loss, dropout, kernel_regularizer,
              level_blocks, dilation_rate, params):
    """If two different rows yields the same label, then not enough params are True"""
    label = ''
    if grouped_df:
        if model:
            label += '{} '.format(row['model'][0])
        if filters:
            label += 'f-{} '.format(row['filters'][0])
        if height:
            label += 'h-{} '.format(row['height'][0])
        if patch:
            label += 'p-{} '.format(row['patch_x'][0])
        if batch_size:
            label += 'b-{} '.format(row['batch_size'][0])
        if lr:
            label += 'lr-{} '.format(row['lr'][0])
        if loss:
            label += 'lo-{} '.format(row['loss'][0])
        if dropout:
            label += 'd-{} '.format(row['dropout'][0])
        if kernel_regularizer:
            label += 'k-{} '.format(row['kernel_regularizer'][0])
        if level_blocks:
            label += 'bl-{} '.format(row['level_blocks'][0])
        if dilation_rate:
            if row['model'][0] in ('AC_UNET', 'ASPP_UNET'):
                label += 'dr-{} '.format(row['dilation_rate'][0])
        if params:
            label += 'pa-{} '.format(round(row['trainable_params'][0]/1e6),2)
    else:
        if model:
            label += '{} '.format(row['model'])
        if filters:
            label += 'f-{} '.format(row['filters'])
        if height:
            label += 'h-{} '.format(row['height'])
        if patch:
            label += 'p-{} '.format(row['patch_x'])
        if batch_size:
            label += 'b-{} '.format(row['batch_size'])
        if lr:
            label += 'lr-{} '.format(row['lr'])
        if loss:
            label += 'lo-{} '.format(row['loss'])
        if dropout:
            label += 'd-{} '.format(row['dropout'])
        if kernel_regularizer:
            label += 'k-{} '.format(row['kernel_regularizer'])
        if level_blocks:
            label += 'bl-{} '.format(row['level_blocks'])
        if dilation_rate:
            if row['model'] in ('AC_UNET', 'ASPP_UNET'):
                label += 'dr-{} '.format(row['dilation_rate'])
        if params:
            label += 'pa-{} '.format(round(row['trainable_params']/1e6),2)
    return label


def empty_label_dict():
    """Pass this result to get_label(**dict) to only have model in the label"""
    return {
        'model': False,
        'filters': False,
        'height': False,
        'patch': False,
        'batch_size': False,
        'lr': False,
        'loss': False,
        'dropout': False,
        'kernel_regularizer': False,
        'level_blocks': False,
        'dilation_rate': False,
        'params': False,
    }

