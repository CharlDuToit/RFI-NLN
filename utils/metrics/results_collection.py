import pandas as pd
from .load_save_csv import load_csv
from .dataframe_helper import query_df, groupby, get_vals, get_labels
from utils.plotting import save_scatter


class ResultsCollection:

    def __init__(self, results_args):

        self.incl_models = results_args.incl_models
        self.excl_models = results_args.excl_models
        self.excl_groupby = results_args.excl_groupby
        self.outputs_path = results_args.outputs_path
        self.datasets = results_args.datasets
        self.query_strings = results_args.query_strings
        self.task = results_args.task
        self.std = results_args.std
        self.params = results_args.params
        self.scatter_x = results_args.scatter_x
        self.scatter_y = results_args.scatter_y
        self.label_field = results_args.label_fields
        self.label_format = results_args.label_format
        self.outputs_path = results_args.outputs_path
        self.save_name = results_args.save_name
        self.save_path = results_args.save_path

        df = load_csv(self.outputs_path)
        df = query_df(df, self.query_strings, self.incl_models, self.excl_models, self.datasets)
        self.queried_df = df


        self.labels = []
        self.val_f1_means = []
        self.val_f1_stds = []
        self.val_auroc_means = []
        self.val_auroc_stds = []
        self.val_auprc_means = []
        self.val_auprc_stds = []
        self.test_f1_means = []
        self.test_f1_stds = []
        self.test_auroc_means = []
        self.test_auroc_stds = []
        self.test_auprc_means = []
        self.test_auprc_stds = []
        self.train_loss_means = []
        self.train_loss_stds = []
        self.val_loss_means = []
        self.val_loss_stds = []
        self.time_image_means = []
        self.time_image_stds = []
        self.flops_image_means = []
        self.trainable_params_means = []
        self.nontrainable_params_means = []
        self.last_epoch_means = []
        self.last_epoch_stds = []

    def perform_task(self):

        task = self.task

        if task == 'scatter':
            dfg = groupby(self.queried_df, self.excl_groupby)
            x_vals = get_vals(dfg, self.scatter_x)
            y_vals = get_vals(dfg, self.scatter_y)
            labels = get_labels(dfg, self.label_field, self.label_format)
            params = get_vals(dfg, 'trainable_params') if self.params else None
            save_scatter(x_vals, y_vals, labels=labels, sizes=params, xlabel=self.scatter_x, ylabel=self.scatter_y,
                         file_name=self.save_name, dir_path=self.save_path)
        elif task == 'loss':
            pass
        elif task == 'text':
            pass
        elif task == 'table':
            pass
        else:
            print(f'Task {task} not defined')


