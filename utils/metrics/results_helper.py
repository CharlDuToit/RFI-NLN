import pandas as pd
import copy
import os
import numpy as np

# last epoch
AGG_FIELDS = (
    'test_f1', 'test_auroc', 'test_auprc', 'val_f1', 'val_auroc', 'val_auprc', 'time_image', 'flops_image',
    'train_loss', 'val_loss', 'last_epoch', 'trainable_params'
)

# early_stop, final_activation
GROUPBY_FIELDS = (
    'data', 'model', 'anomaly_class', 'anomaly_type', 'rfi_threshold', 'ood_rfi', 'limit', 'use_hyp_data', 'epochs',
    'batch_size', 'lr', 'std_plus', 'std_minus', 'per_image', 'filters', 'height', 'dropout', 'kernel_regularizer',
    'level_blocks', 'dilation_rate', 'patch_x', 'patch_y', 'loss', 'lofar_subset', 'latent_dim', 'alpha',
    'neighbour', 'algorithm', 'combine_min_std_plus', # early_stop, final_activation
)


def groupby_and_agg(df: pd.DataFrame, groupby_fields: list[str] | str, agg_fields: list[str] | str = None ):

    # fields in excl_groupby will not be a column in grouped dataframe
    #excl_groupby = [] if excl_groupby is None else excl_groupby
    #groupby_fields = [f for f in GROUPBY_FIELDS if (f not in excl_groupby)]

    if agg_fields is None:
        agg_fields = ['test_f1']
    agg_flds = copy.deepcopy(agg_fields)
    if not isinstance(agg_flds, list):
        agg_flds = [agg_flds]
    for f in AGG_FIELDS:
        if f not in agg_flds:
            agg_flds.append(f)

    grpby_flds = copy.deepcopy(groupby_fields)
    if not isinstance(grpby_flds, list):
        grpby_flds = [grpby_flds]

    agg_dict = {}
    for field in agg_flds:
        agg_dict[field] = ['mean', 'std']  # count ?

    dfg = df.groupby(grpby_flds, as_index=False).agg(agg_dict)

    return dfg


def query_df(df: pd.DataFrame, query_strings: list[str], incl_models=None, excl_models=None, datasets=None):
    """ands all the query strings"""

    qs = copy.deepcopy(query_strings)
    if not isinstance(qs, list):
        qs = [qs]

    if incl_models is not None and len(incl_models) > 0:
        incl_models = ['( model == "' + m + '" )' for m in incl_models]
        incl_models_expr = ' or '.join(incl_models)
        qs.append(incl_models_expr)

    if excl_models is not None and len(excl_models) > 0:
        excl_models = ['( model != "' + m + '" )' for m in excl_models]
        excl_models_expr = ' and '.join(excl_models)
        qs.append(excl_models_expr)

    if datasets is not None and len(datasets) > 0:
        datasets = ['( data == "' + d + '" )' for d in datasets]
        datasets_expr = ' or '.join(datasets)
        qs.append(datasets_expr)

    qs = ['( ' + s + ' )' for s in qs]
    expr = ' and '.join(qs)
    #return expr
    return df.query(expr)


def to_query(field, val):
    if isinstance(val, str):
        return f'{field} == "{val}"'
    else:
        return f'{field} == {val}'


def query_df_with_row_vals(df, row, fields: list[str] | str ):
    #excl_groupby = [] if excl_groupby is None else excl_groupby
    #groupby_fields = [f for f in GROUPBY_FIELDS if (f not in excl_groupby)]

    flds = copy.deepcopy(fields)
    if not isinstance(flds, list):
        flds = [flds]
    query_strings = [to_query(f, get_val(row, f)) for f in flds]
    return query_df(df, query_strings)


def get_val(row, field: str, agg=None):
    val = None
    try:
        if agg is None:
            if isinstance(row[field], (str, float, int, bool)):
                val = row[field]
            else:
                val = row[field][0]
        else:
            val = row[field][agg]
    except:
        if isinstance(row[field], (str, float, int, bool)):
            val = row[field]
        else:
            val = row[field][0]

    if not isinstance(val, (str, int, float, bool)):
        try:
            if val.dtype == float:
                val = float(val)
            if val.dtype == str:
                val = str(val)
            if val.dtype == int:
                val = int(val)
            if val.dtype == bool:
                val = bool(val)
        except:
            val = val

    if str(val) == 'nan':
        val = 0
    return val

    #try:
    #    return float(val)
    #except:
    #    return val


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def signif_row(row, p):
    for i in range(len(row)):
        if isinstance(row[i], (float, int)):
            row[i] = signif(row[i], p)
    return row


def row_to_latex_str(row, table_fields: list[str], label_fields: list[str], label_format: str, std=False):

    row_vals = [get_val(row, f) for f in table_fields]
    row_vals = signif_row(row_vals, 4)
    row_vals = [str(val) for val in row_vals]

    if std:
        std_vals = [get_val(row, f, 'std') for f in table_fields]
        std_vals = signif_row(std_vals, 4)
        std_vals = [str(val) for val in std_vals]
        row_vals = [f'{mean}\\pm{std}' for mean, std in zip(row_vals, std_vals)]

    label = get_label(row, label_fields, label_format)
    row_vals = [label] + row_vals
    row_str = ' & '.join(row_vals)
    row_str += ' \\\\'
    return row_str


def df_to_latex_table(df, table_fields: list[str], label_fields: list[str], label_format: str, std: bool):
    top_row = ['label'] + table_fields
    table_str = ' & '.join(top_row) + ' \\\\ \n'
    for i in range(df.shape[0]):
        table_str += row_to_latex_str(df.iloc[i], table_fields, label_fields, label_format, std) + ' \n'
    return table_str


def get_vals(df, field: str, agg=None):
    vals = []
    for i in range(df.shape[0]):
        vals.append( get_val(df.iloc[i], field, agg) )
    return vals


def get_label(row, label_fields: list[str], label_format: str):
    """
    row from a pd.dataframe
    label_fields is a list of fields in row
    label_format must be one of 'full', 'empty', 'short' """

    if label_format == 'full':
        return ' '.join([f'{lf}-{get_val(row,lf)}' for lf in label_fields])
    elif label_format == 'empty':
        return ' '.join([f'{get_val(row, lf)}' for lf in label_fields])
    elif label_format == 'short':
        return ' '.join([f'{get_tag(lf)}{get_val(row, lf)}' for lf in label_fields])
    else:
        raise ValueError('label_format must be full empty or short')


def get_labels(df: pd.DataFrame, label_fields: list[str], label_format: str):
    labels = []
    for i in range(df.shape[0]):
        labels.append( get_label(df.iloc[i], label_fields, label_format) )
    return labels


def get_model_losses(dir_path, model, anomaly_class, model_name, prefix='val'):
    file = os.path.join(dir_path, model, anomaly_class, model_name, 'losses', f'{prefix}_epoch_losses.txt')
    with open(file, 'r') as f:
        losses_list = [float(line.rstrip()) for line in f]
    return losses_list


def get_all_losses(df, dir_path, prefix='val'):
    list_of_list_of_losses = []
    for i in range(df.shape[0]):
        model = get_val(df.iloc[i], 'model')
        anomaly_class = get_val(df.iloc[i], 'anomaly_class')
        model_name = get_val(df.iloc[i], 'name')
        list_of_list_of_losses.append(get_model_losses(dir_path, model, anomaly_class, model_name, prefix))
    return list_of_list_of_losses


def get_means_stds_from_grouped_df(dir_path, full_df, grouped_df, groupby_fields, prefix='val'):
    means = []
    stds = []
    for i in range(grouped_df.shape[0]):
        queried_df = query_df_with_row_vals(full_df, grouped_df.iloc[i], groupby_fields)
        list_of_list_of_losses_for_group = get_all_losses(queried_df, dir_path, prefix)
        group_means, group_stds = calc_losses_means_std(list_of_list_of_losses_for_group)
        means.append(group_means)
        stds.append(group_stds)
    return means, stds


def calc_losses_means_std(list_of_list_of_losses):
    # list_of_list_of_losses may have different lenghts

    if len(list_of_list_of_losses) == 1:
        return list_of_list_of_losses[0], [0]*len(list_of_list_of_losses[0])

    lengths = []

    for list_of_losses in list_of_list_of_losses:
        if len(list_of_losses) not in lengths:
            lengths.append(len(list_of_losses))
    lengths.sort()

    if len(lengths) == 1:
        temp = np.array(list_of_list_of_losses)
        means = np.mean(temp, axis=0)
        stds = np.std(temp, axis=0)
        return means, stds

    means = np.empty(lengths[-1], dtype='float32')
    stds = np.empty(lengths[-1], dtype='float32')
    lo = 0
    for i in range(len(lengths)):
        hi = lengths[i]
        temp = []
        for list_of_losses in list_of_list_of_losses:
            if len(list_of_losses) >= hi:
                temp.append(list_of_losses[lo:hi])
        temp = np.array(temp)
        temp_means = np.mean(temp, axis=0)
        temp_stds = np.std(temp, axis=0)
        means[lo:hi] = temp_means
        stds[lo:hi] = temp_stds
        lo = hi

    return means, stds


def get_tag(field):
    if field == 'model':  # no tag required
        return ''
    if field == 'data':  # no tag required
        return ''
    elif field == 'filters':
        return 'f-'
    elif field == 'height':
        return 'h-'
    elif field == 'filters':
        return 'f-'
    elif field == 'patch_x':
        return 'p-'
    elif field == 'patch_y':
        return 'p-'
    elif field == 'batch_size':
        return 'b-'
    elif field == 'lr':
        return 'lr-'
    elif field == 'loss':  # no tag required
        return ''
    elif field == 'dropout':
        return 'd-'
    elif field == 'kernel_regularizer':
        return 'k-'
    elif field == 'level_blocks':
        return 'bl-'
    elif field == 'dilation_rate':
        return 'dr-'
    elif field == 'trainable_params':
        return 'pa-'
    else:
        return ''


if __name__ == '__main__':
    #  Testing
    if True:
        print(query_df(None, ['filters > 8', 'height == 4'], ['UNET', 'RNET'], ['RFI-NET'], ['LOFAR']))
        print(query_df(None, ['filters > 8', 'height == 4'], ['UNET', 'RNET'], ['RFI-NET'], []))
        print(query_df(None, [], ['UNET', 'RNET'], ['RFI-NET'], ['LOFAR', 'HERA']))

    if True:
        a = [
            [0],
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 10, 11],
            [12,13,14,15]
        ]
        #   4.4   6.5   7.5   10.33   11
        #   4.58  4.5   4.5     3.68   0
        print(calc_losses_means_std(a))
        print(calc_losses_means_std([[1, 2, 3, 4, 5]]))
        print(calc_losses_means_std([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))

