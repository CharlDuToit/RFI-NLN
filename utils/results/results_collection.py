#import copy
#import pandas as pd
import numpy as np
import pandas as pd

from .load_save_csv import load_csv
from .results_helper import (query_df,
                             groupby_and_agg,
                             get_vals,
                            get_val,
                            signif,
                             get_labels,
                             get_label,
                             get_means_stds_from_grouped_df,
                             to_query,
                             query_df_with_row_vals,
                             df_to_latex_table)
from utils.plotting import save_bubble, save_epochs_curve, save_lines, save_scatter_gmm, save_recall_prec_curve, save_fpr_tpr_curve
import os
from copy import deepcopy


class ResultsCollection:

    def __init__(self,
                 incl_models,
                 excl_models,
                 groupby,
                 datasets,
                 query_strings,
                 task,
                 table_fields,
                 std,
                 params,
                 x_axis,
                 y_axis,
                 label_fields,
                 label_format,
                 output_path,
                 save_name,
                 save_path,
                 plot_options: dict = dict(),
                 **kwargs):

        # Copy args
        self.incl_models = incl_models
        self.excl_models = excl_models
        self.groupby = groupby
        self.datasets = datasets
        self.query_strings = query_strings
        self.task = task
        self.table_fields = table_fields
        self.std = std
        self.params = params
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.label_fields = label_fields
        self.label_format = label_format
        self.output_path = output_path
        self.save_name = save_name
        self.save_path = save_path

        # Load dataframe
        df = load_csv(self.output_path)

        # Fix columns
        incl_values_columns = (self.task == 'recall_prec_curve' or self.task == 'fpr_tpr_curve')
        df = fix_columns(df, incl_values_columns=incl_values_columns)

        # self.reg_effect = None
        # if 'reg_label' not in groupby and task == 'table' and 'reg_label == "l2+0.1"' in query_strings:
        #     qs = deepcopy(query_strings)
        #     qs.remove('reg_label == "l2+0.1"' )
        #     qs += ['reg_label == "None+0.0"']
        #     df_new = query_df(df, qs, self.incl_models, self.excl_models, self.datasets)
        #     df_orig = query_df(df, query_strings, self.incl_models, self.excl_models, self.datasets)
        #     df_new_grp = df_new.groupby(self.groupby).agg({'reg_distance': ['mean', 'std']})
        #     df_orig_grp = df_orig.groupby(self.groupby).agg({'reg_distance': ['mean', 'std']})
        #     df['reg_effect'] = 0
        #
        #     self.reg_effect = df_new_grp['reg_distance']['mean'] - df_orig_grp['reg_distance']['mean']
        #
        #     cond = [df['model_class'] == mdl for mdl in df['model_class'].unique()]

        # query df
        df = query_df(df, self.query_strings, self.incl_models, self.excl_models, self.datasets)
        self.queried_df = df

        # L3 dfq = df.query('limit < 30')
        # H4 dfq = df.query('data_name=="HERA_CHARL"')

        # names = ' '.join([a for a in df['model_name']])

        # dfq = df.query('loss == "mse" and dropout==0.1 and kernel_regularizer=="l2"')
        # np.std(dfq['test_recall']) 0.014,
        # np.std(dfq['val_recall']) 0.037

        # dfq = df.query('loss == "dice" and dropout==0.1 and kernel_regularizer=="l2"')
        # np.std(dfq['test_recall']) 0.011,
        # np.std(dfq['val_recall']) 0.028

        # dfq = df.query('dropout==0.1 and kernel_regularizer=="l2" and model_class == "DSC_DUAL_RESUNET"')
        # names = ' '.join([a for a in dfq['model_name']])

        # debug here to copy string
        transfer_strings = [mdl_class +','+mdl_name for mdl_class, mdl_name in zip(df['model_class'], df['model_name'])]
        transfer_strings = ' '.join(transfer_strings) + ';'

        # Create path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Save queried dataframe
        file = os.path.join(self.save_path, f'{self.save_name}.csv')
        df.to_csv(file, index=False)

        # Save counts of grouped dataframe
        try:
            dfg = df.groupby(self.groupby)['model_class'].count()
            file = os.path.join(self.save_path, f'{self.save_name}_counts.csv')
            dfg.to_csv(file, index=True)
        except Exception:
            pass

        # Populate plot options
        if 'xlabel' not in plot_options.keys():
            plot_options['xlabel'] = self.x_axis
        if 'ylabel' not in plot_options.keys():
            plot_options['ylabel'] = self.y_axis
        if 'x_axis' not in plot_options.keys():
            plot_options['x_axis'] = self.x_axis
        if 'y_axis' not in plot_options.keys():
            plot_options['y_axis'] = self.y_axis
        if 'file_name' not in plot_options.keys():
            plot_options['file_name'] = self.save_name
        if 'dir_path' not in plot_options.keys():
            plot_options['dir_path'] = self.save_path
        if 'groupby' not in plot_options.keys():
            plot_options['groupby'] = self.groupby  # for scatter_gmm
        if 'logx' not in plot_options.keys() and self.x_axis == 'flops_image':
            plot_options['logx'] = True
        if 'logy' not in plot_options.keys() and self.y_axis == 'flops_image':
            plot_options['logy'] = True

        self.plot_options = plot_options

        # Save config
        self.save_config()

    def save_config(self):
        file = os.path.join(self.save_path, f'{self.save_name}.txt')
        with open(file, 'w') as fp:
            fp.write(f'incl_models: {self.incl_models} \n')
            fp.write(f'excl_models: {self.excl_models} \n')
            fp.write(f'groupby: {self.groupby} \n')
            fp.write(f'datasets: {self.datasets} \n')
            fp.write(f'query_strings: {self.query_strings} \n')
            fp.write(f'task: {self.task} \n')
            fp.write(f'table_fields: {self.table_fields} \n')

    def perform_task(self):
        task = self.task

        if task == 'scatter':
            self.queried_df['time_image'] = self.queried_df['time_image'] * 1000  # ms

            dfg = groupby_and_agg(self.queried_df, self.groupby, self.y_axis)

            x_vals = np.array(get_vals(dfg, self.x_axis))
            y_vals = get_vals(dfg, self.y_axis)
            labels = get_labels(dfg, self.label_fields, self.label_format)
            params = np.array(get_vals(dfg, 'params')) if self.params else None

            save_bubble(x_vals, y_vals, labels=labels, sizes=params, **self.plot_options)

        elif task in ('train_loss', 'val_loss', 'train_val_loss'):
            dfg = groupby_and_agg(self.queried_df, self.groupby)
            means, stds, labels = [], [], []
            if task == 'train_loss':
                means, stds = get_means_stds_from_grouped_df(self.output_path, self.queried_df, dfg, self.groupby, prefix='train')
                labels = get_labels(dfg, self.label_fields, self.label_format)
            elif task == 'val_loss':
                means, stds = get_means_stds_from_grouped_df(self.output_path, self.queried_df, dfg, self.groupby, prefix='val')
                labels = get_labels(dfg, self.label_fields, self.label_format)
            elif task == 'train_val_loss':
                train_means, train_stds = get_means_stds_from_grouped_df(self.output_path, self.queried_df, dfg, self.groupby, prefix='train')
                val_means, val_stds = get_means_stds_from_grouped_df(self.output_path, self.queried_df, dfg, self.groupby, prefix='val')
                means = list(train_means) + list(val_means)
                stds = list(train_stds) + list(val_stds)
                labels = get_labels(dfg, self.label_fields, self.label_format)
                labels = [f'train {lbl}' for lbl in labels] + [f'val {lbl}' for lbl in labels]
            if not self.std:
                stds = None
            # save_epochs_curve(self.save_path, means, labels, stds, file_name=self.save_name, ylim_top=0.01, ylim_bottom=0.003)
            save_epochs_curve(metrics=means, labels=labels, metrics_std=stds, **self.plot_options)

        elif task == 'line':
            # Makes a line plot of individual rows, no aggregration in plotted data
            dfg = groupby_and_agg(self.queried_df, self.groupby, self.y_axis)
            x_vals = []
            y_vals = []
            labels = get_labels(dfg, self.label_fields, self.label_format)
            for i in range(dfg.shape[0]):
                row = dfg.iloc[i]
                df = query_df_with_row_vals(self.queried_df, row, self.groupby)
                x_vals.append(get_vals(df, self.x_axis))
                y_vals.append(get_vals(df, self.y_axis))

            save_lines(x_vals, y_vals, labels=labels, **self.plot_options)

        # elif task == 'line_marker_size':
        #     # One of the fields in groupby must be numerical (limit) so that the size of markers on a spesific line
        #     # indicates this field (limit)
        #     # self.line_marker_size_field
        #     # x_axis and y_axis must not be in grouby
        #
        #     dfg = groupby_and_agg(self.queried_df, self.groupby, [self.x_axis, self.y_axis])
        #     x_vals = []
        #     y_vals = []
        #     labels = get_labels(dfg, self.label_fields, self.label_format)
        #     for i in range(dfg.shape[0]):
        #         row = dfg.iloc[i]
        #         df = query_df_with_row_vals(self.queried_df, row, self.groupby)
        #         x_vals.append(get_vals(df, self.x_axis))
        #         y_vals.append(get_vals(df, self.y_axis))
        #
        #     # list of list of x, list of list of y, list of list of sizes, list of labels,
        #     save_lines(x_vals, y_vals, labels=labels, **self.plot_options)
        #     # legend for labels, legend for sizes (or draw on point)

        elif task == 'line_agg':
            # For each group in dfg there is a parallel list of x and y
            # for each unique x, find the average of y
            # each group gets its own line and label
            dfg = self.queried_df.groupby(self.groupby)
            x_values = []
            y_values = []
            labels = []
            for i, (group_keys, group) in enumerate(dfg):
                # label = get_label(group, self.label_fields, self.label_format)
                # labels.append(label)
                labels.append(group_keys)
                x_unique = np.unique(group[self.x_axis])
                counts = [0] * len(x_unique)
                y_vals = [0] * len(x_unique)
                for x, y in zip(group[self.x_axis], group[self.y_axis]):
                    for j in range(len(x_unique)):
                        if x_unique[j] == x:
                            y_vals[j] += y
                            counts[j] += 1
                for j in range(len(x_unique)):
                    y_vals[j] /= counts[j]

                x_values.append(list(x_unique))
                y_values.append(y_vals)

            save_lines(x_values, y_values, labels=labels, **self.plot_options)

        elif task == 'text':
            pass
        elif task == 'table':
            self.queried_df['flops_image'] = self.queried_df['flops_image']/1e9 # GFLOPS
            self.queried_df['time_image'] = self.queried_df['time_image']* 1000 # ms
            self.queried_df['params'] = self.queried_df['params']/1000  # k params
            dfg = groupby_and_agg(self.queried_df, self.groupby, self.table_fields)

            # if self.reg_effect:
            #     dfg[('reg_effect', 'mean')] = self.reg_effect
            #     self.table_fields += ['reg_effect']

            table_str = df_to_latex_table(dfg, self.table_fields, self.label_fields, self.label_format, self.std)
            file = os.path.join(self.save_path, f'{self.save_name}.table')
            with open(file, 'w') as f:
                f.write(table_str)
        elif task == 'table_with_two_groups':
            if len(self.groupby) != 2:
                raise('Need two groupby fields for task: table_with_two_groups')
            # dfg = groupby_and_agg(self.queried_df, self.groupby, self.table_fields)
            g1_values = pd.unique(self.queried_df[self.groupby[0]])
            g2_values = pd.unique(self.queried_df[self.groupby[1]])

            top_row = ['g1', 'field'] + list(g2_values)
            table_str = ' & '.join(top_row) + ' \\\\ \n'

            for g1 in g1_values:
                row = [g1]
                for f in self.table_fields:
                    if row:
                        row += [f]
                    else:
                        row += [' ', f]
                    for g2 in g2_values:
                        df_q = self.queried_df.query(f'{self.groupby[0]} == "{g1}" and {self.groupby[1]} == "{g2}"')
                        dfg = groupby_and_agg(df_q, self.groupby, f)
                        val = ' '
                        if len(dfg) > 0:
                            val = str(signif(get_val(dfg.iloc[0], f, 'mean'), 3))
                            if self.std:
                                val += '$\\pm$' + str(signif(get_val(dfg.iloc[0], f, 'std'), 1))
                        row += [val]
                    table_str += ' & '.join(row) + ' \\\\ \n'
                    row = []
            table_str = table_str.replace('_', '\\_')
            file = os.path.join(self.save_path, f'{self.save_name}.table')
            with open(file, 'w') as f:
                f.write(table_str)

        elif task == 'scatter_gmm':
            save_scatter_gmm(self.queried_df, **self.plot_options)
        elif task == 'recall_prec_curve':
            save_recall_prec_curve(self.queried_df, **self.plot_options)
        elif task == 'fpr_tpr_curve':
            save_fpr_tpr_curve(self.queried_df, **self.plot_options)
        else:
            print(f'Task {task} not defined')


def fix_columns(df, incl_values_columns=False):
    # Add/ fix columns:

    # if 'flops_image' not in df.columns:
    #     df['flops_image'] = 0
    # if 'flops_patch' not in df.columns:
    #     df['flops_patch'] = 0
    # if 'trainable_params' not in df.columns:
    #     df['trainable_params'] = 0
    # if 'nontrainable_params' not in df.columns:
    #     df['nontrainable_params'] = 0
    # if 'train_with_test' not in df.columns:
    #     df['train_with_test'] = False
    # if 'rescale' not in df.columns:
    #     df['rescale'] = True
    # if 'bn_first' not in df.columns:
    #     df['bn_first'] = False
    # if 'lr_lin_decay' not in df.columns:
    #     df['lr_lin_decay'] = 1.0
    # if 'calc_train_val_auc' not in df.columns:
    #     df['calc_train_val_auc'] = True
    # if 'freeze_top_layers' not in df.columns:
    #     df['freeze_top_layers'] = False
    # if 'n_splits' not in df.columns:
    #     df['n_splits'] = 5

    if incl_values_columns:
        for subset in ('val', 'test'):
            df = fix_column(df, f'{subset}_TP', None)
            df = fix_column(df, f'{subset}_TN', None)
            df = fix_column(df, f'{subset}_FP', None)
            df = fix_column(df, f'{subset}_FN', None)

            df = fix_column(df, f'{subset}_fpr_vals', None)
            df = fix_column(df, f'{subset}_tpr_vals', None)
            df = fix_column(df, f'{subset}_fpr_tpr_thr_vals', None)
            df = fix_column(df, f'{subset}_prec_vals', None)
            df = fix_column(df, f'{subset}_recall_vals', None)
            df = fix_column(df, f'{subset}_prec_recall_thr_vals', None)

            df = df_str_to_list(df, f'{subset}_fpr_vals')
            df = df_str_to_list(df, f'{subset}_tpr_vals')
            df = df_str_to_list(df, f'{subset}_fpr_tpr_thr_vals')
            df = df_str_to_list(df, f'{subset}_prec_vals')
            df = df_str_to_list(df, f'{subset}_recall_vals')
            df = df_str_to_list(df, f'{subset}_prec_recall_thr_vals')



    df = fix_column(df, 'flops_image', 0)
    df = fix_column(df, 'flops_patch', 0)
    df = fix_column(df, 'nontrainable_params', 0)
    df = fix_column(df, 'trainable_params', 0)

    df = fix_column(df, 'task', 'train')
    df = fix_column(df, 'n_splits', 5)
    df = fix_column(df, 'freeze_top_layers', False)
    df = fix_column(df, 'train_with_test', False)
    df = fix_column(df, 'calc_train_val_auc', True)
    df = fix_column(df, 'lr_lin_decay', 1.0)
    df = fix_column(df, 'rescale', True)
    df = fix_column(df, 'bn_first', False)

    # df.loc[df['task'].isnull(), 'task'] = 'train'
    # df.loc[df['n_splits'].isnull(), 'n_splits'] = 5
    # df.loc[df['freeze_top_layers'].isnull(), 'freeze_top_layers'] = False
    # df.loc[df['train_with_test'].isnull(), 'train_with_test'] = False
    # df.loc[df['calc_train_val_auc'].isnull(), 'calc_train_val_auc'] = True
    # df.loc[df['lr_lin_decay'].isnull(), 'lr_lin_decay'] = 1.0
    # df.loc[df['rescale'].isnull(), 'rescale'] = True
    # df.loc[df['bn_first'].isnull(), 'bn_first'] = False

    cond = [np.logical_and(df['model_class'] == 'UNET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'AC_UNET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'ASPP_UNET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'RNET5',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'RNET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'RFI_NET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'DSC_MONO_RESUNET',  df['filters'] == 16),
            np.logical_and(df['model_class'] == 'DSC_DUAL_RESUNET',  df['filters'] == 16),
            ]

    # LOFAR
    lofar_cond_trans = [
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 1493), df['data_name'] == 'LOFAR'),
        # np.logical_and(df['task'] == 'train', df['limit'] == 'None'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 14), df['data_name'] == 'LOFAR'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 28), df['data_name'] == 'LOFAR'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 56), df['data_name'] == 'LOFAR'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 112), df['data_name'] == 'LOFAR'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 224), df['data_name'] == 'LOFAR'),
        np.logical_and(np.logical_and(df['task'] == 'transfer_train', df['limit'] == 14), df['data_name'] == 'LOFAR'),
    ]
    trans_groups = ['aof', 'new 14','new 28','new 56','new 112','new 224', 'tune 14']

    df['lofar_trans_group'] = np.select(lofar_cond_trans, trans_groups)

    # HERA
    hera_cond_trans = [
        np.logical_and(np.logical_and(df['task'] == 'train', df['use_hyp_data'] == True), df['data_name'] == 'HERA_CHARL_AOF'),
        # np.logical_and(df['task'] == 'train', df['limit'] == 'None'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 14), df['data_name'] == 'HERA_CHARL'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 28), df['data_name'] == 'HERA_CHARL'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 56), df['data_name'] == 'HERA_CHARL'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 112), df['data_name'] == 'HERA_CHARL'),
        np.logical_and(np.logical_and(df['task'] == 'train', df['limit'] == 224), df['data_name'] == 'HERA_CHARL'),
        np.logical_and(np.logical_and(df['task'] == 'transfer_train', df['limit'] == 14), df['data_name'] == 'HERA_CHARL'),
    ]
    trans_groups = ['aof', 'new 14','new 28','new 56','new 112','new 224', 'tune 14']

    df['hera_trans_group'] = np.select(hera_cond_trans, trans_groups)
    # train_params = [73241, 73241, 73241, 5105, 8353, 202673, 185226, 370494]
    # nontrain_params = [352, 352, 352, 64, 64, 1760, 2112, 4224]
    # flops_image = [1389363200, 1554016256, 1471689728, 1317142528, 2158100480, 3642621952, 2815819776, 5642518528]

    train_params = [291633, 291633, 291633, 19809, 32705, 805473, 185226, 370494]
    nontrain_params = [704, 704, 704, 128, 192, 3520, 2112, 4224]
    flops_image = [5496504320, 6152720384, 5824612352, 5150736384, 8510373888, 14482538496, 2815819776, 5642518528]
    short_model_class = ['U', 'AC', 'ASPP', 'R5', 'R7', 'RFI', 'MONO', 'DUAL']

    df['short_model_class'] = np.select(cond, short_model_class)

    df['trainable_params'] = np.select(cond, train_params)
    df['nontrainable_params'] = np.select(cond, nontrain_params)
    df['params'] = df['trainable_params'] + df['nontrainable_params']
    df['flops_image'] = np.select(cond, flops_image)
    df['flops_patch'] = df['flops_image'] / 64

    df['train_auprc_over_val_auprc'] = df['train_auprc'] / df['val_auprc']
    df['train_loss_over_val_loss'] = df['train_loss'] / df['val_loss']

    df['reg_distance'] = np.sqrt((df['train_auprc_over_val_auprc'] - 1)**2 + (df['train_loss_over_val_loss'] - 1 )**2)

    # reg label
    # df['reg_label'] = df['kernel_regularizer'] + df['dropout']
    df['reg_label'] = [df.iloc[i]['kernel_regularizer'] + '+' + str(df.iloc[i]['dropout']) for i in range(df.shape[0])]

    df = df.sort_values(by=['model_class', 'loss', 'reg_label'])
    # df1 = df.sort_values(by=['model_class', 'loss', 'reg_label'])
    # df1['model_class']
    return df

def fix_column(df, column_name, default_value):
    if column_name not in df.columns:
        df[column_name] = default_value
        return df

    df.loc[df[column_name].isnull(), column_name] = default_value
    df.loc[df[column_name] == 'nan', column_name] = default_value
    df.loc[df[column_name] == '', column_name] = default_value
    df.loc[df[column_name] == ' ', column_name] = default_value
    df.loc[df[column_name] == '[]', column_name] = default_value

    return df

def str_to_list(str_list):
    if str_list == [] or str_list is None or str_list == '' or str_list == '[]' or str_list=='nan':
        return None
    if '...' in str_list: return None
    str_list = str_list.replace(',', '').replace('[', '').replace(']', '').replace('\n', ' ').split(' ')
    while '' in str_list: str_list.remove('')
    list_ = [float(a) for a in str_list]
    # if len(list_) > 200: return None
    return list_

def df_str_to_list(df, column_name):
    for i in range(len(df)):
       # df.iloc[i][column_name] = str_to_list(df.iloc[i][column_name])
        df[column_name][i] = str_to_list(df.iloc[i][column_name])
    return df

