import pandas as pd
import copy

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


def groupby(df: pd.DataFrame, excl_groupby=None):

    groupby_fields = [f for f in GROUPBY_FIELDS if (f not in excl_groupby)]

    agg_dict = {}
    for field in AGG_FIELDS:
        agg_dict[field] = ['mean', 'std']

    dfg = df.groupby(groupby_fields, as_index=False).agg(agg_dict)

    return dfg


def query_df(df: pd.DataFrame, query_strings: list[str], incl_models=None, excl_models=None, datasets=None):
    """ands all the query strings"""

    qs = copy.deepcopy(query_strings)

    if incl_models is not None and len(incl_models) > 0:
        incl_models = ['( model == ' + m + ' )' for m in incl_models]
        incl_models_expr = ' or '.join(incl_models)
        qs.append(incl_models_expr)

    if excl_models is not None and len(excl_models) > 0:
        excl_models = ['( model != ' + m + ' )' for m in excl_models]
        excl_models_expr = ' and '.join(excl_models)
        qs.append(excl_models_expr)

    if datasets is not None and len(datasets) > 0:
        datasets = ['( data == ' + d + ' )' for d in datasets]
        datasets_expr = ' or '.join(datasets)
        qs.append(datasets_expr)

    qs = ['( ' + s + ' )' for s in qs]
    expr = ' and '.join(qs)
    #return expr
    return df.query(expr)


def get_val(row, field, agg=None):
    val = None
    try:
        if agg is None:
            val = row[field][0]
        else:
            val = row[field][agg]
    except:
        val = row[field]
    # return val
    try:
        return float(val)
    except:
        return val


def get_val_(row, field, agg=None):
    val = None
    try:
        if agg is None:
            val = row[field]
        else:
            val = row[field][agg]
    except:
        try:
            val = row[field][0]
        except:
            try:
                val = row[field]
            except:
                val = row[field]['mean']
    # return val
    try:
        return float(val)
    except:
        return val


def get_vals(df, field, agg=None):
    vals = []
    for i in range(df.shape[0]):
        vals.append( get_val(df.iloc[i], field, agg) )
    return vals


def get_label(row, label_fields, label_format):
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


def get_labels(df: pd.DataFrame, label_fields, label_format):
    labels = []
    for i in range(df.shape[0]):
        labels.append( get_label(df.iloc[i], label_fields, label_format) )
    return labels


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
