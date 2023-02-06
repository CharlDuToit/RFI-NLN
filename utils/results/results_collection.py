#import copy
#import pandas as pd
import numpy as np
from .load_save_csv import load_csv
from .results_helper import (query_df,
                             groupby_and_agg,
                             get_vals,
                             get_labels,
                             get_means_stds_from_grouped_df,
                             to_query,
                             query_df_with_row_vals,
                             df_to_latex_table)
from utils.plotting import save_scatter, save_epochs_curve, save_lines
import os


class ResultsCollection:

    def __init__(self, results_args):

        # Copy args
        self.incl_models = results_args.incl_models
        self.excl_models = results_args.excl_models
        self.groupby = results_args.groupby
        self.datasets = results_args.datasets
        self.query_strings = results_args.query_strings
        self.task = results_args.task
        self.table_fields = results_args.table_fields
        self.std = results_args.std
        self.params = results_args.params
        self.x_axis = results_args.x_axis
        self.y_axis = results_args.y_axis
        self.label_fields = results_args.label_fields
        self.label_format = results_args.label_format
        self.output_path = results_args.output_path
        self.save_name = results_args.save_name
        self.save_path = results_args.save_path

        # Load and query dataframe
        df = load_csv(self.output_path)
        df = query_df(df, self.query_strings, self.incl_models, self.excl_models, self.datasets)
        self.queried_df = df

        # Create path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Save queried dataframe
        file = os.path.join(self.save_path, f'{self.save_name}.csv')
        df.to_csv(file, index=False)

        # Save config
        self.save_config()

    def save_config(self):
        file = os.path.join(self.save_path, f'{self.save_name}.config')
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

        logx = True if self.x_axis == 'flops_image' else False
        logy = True if self.y_axis == 'flops_image' else False

        if task == 'scatter':
            dfg = groupby_and_agg(self.queried_df, self.groupby, self.y_axis)

            x_vals = np.array(get_vals(dfg, self.x_axis))
            y_vals = get_vals(dfg, self.y_axis)
            labels = get_labels(dfg, self.label_fields, self.label_format)
            params = np.array(get_vals(dfg, 'trainable_params')) if self.params else None

            #  Tweak save_scatter font sizes to your liking
            save_scatter(x_vals, y_vals, labels=labels, sizes=params, xlabel=self.x_axis, ylabel=self.y_axis,
                         file_name=self.save_name, dir_path=self.save_path, logx=logx, logy=logy, label_on_point=False)

            # gass v1
            # save_scatter(x_vals, y_vals, labels=labels, sizes=params, xlabel='FLOPs', ylabel='F1-score',
            #              file_name=self.save_name, dir_path=self.save_path, logx=logx, logy=logy, label_on_point=False,
            #              point_label_size=25, ytick_size=25, xtick_size=35
            #              )

            # gass v2
            # save_scatter(x_vals, y_vals, labels=labels, sizes=params, xlabel='Floating point operations', ylabel='F1-score',
            #              file_name=self.save_name, dir_path=self.save_path, logx=logx, logy=logy, label_on_point=True,
            #              point_label_size=25, ytick_size=25, xtick_size=25, axis_fontsize=25, size_on_point=False, size_legend=True,
            #              size_max=9000, legend_title='Parameters', legend_size=25
            #              )

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
            save_epochs_curve(self.save_path, means, labels, stds, file_name=self.save_name)

        elif task == 'line':
            dfg = groupby_and_agg(self.queried_df, self.groupby, self.y_axis)
            x_vals = []
            y_vals = []
            labels = get_labels(dfg, self.label_fields, self.label_format)
            for i in range(dfg.shape[0]):
                row = dfg.iloc[i]
                df = query_df_with_row_vals(self.queried_df, row, self.groupby)
                x_vals.append(get_vals(df, self.x_axis))
                y_vals.append(get_vals(df, self.y_axis))

            #  labels = None
            #  Tweak save_lines font sizes to your liking
            save_lines(x_vals, y_vals, scatter=True, labels=labels, xlabel=self.x_axis, ylabel=self.y_axis,
                       file_name=self.save_name, dir_path=self.save_path, logx=logx, logy=logy)

        elif task == 'text':
            pass
        elif task == 'table':
            dfg = groupby_and_agg(self.queried_df, self.groupby, self.table_fields)
            table_str = df_to_latex_table(dfg, self.table_fields, self.label_fields, self.label_format, self.std)
            file = os.path.join(self.save_path, f'{self.save_name}.table')
            with open(file, 'w') as f:
                f.write(table_str)
        else:
            print(f'Task {task} not defined')


