import pandas as pd
from .load_metrics_csv import load_csv, empty_label_dict, all_true, get_label
from utils.plotting import save_training_metrics, save_flops_metric


class ResultsCollection:

    def __init__(self, data_name, group_filter_dict, group_label_list=('model', 'lr', 'loss'), dir_path=None):
        self.data_name = data_name
        self.group_filter_dict = group_filter_dict
        # self.label_list = label_list
        label_dict = empty_label_dict()
        for label in group_label_list:
            label_dict[label] = True
        self.group_label_dict = label_dict
        if dir_path is None:
            dir_path = './outputs'
        self.dir_path = dir_path
        self.df = load_csv(self.data_name, self.dir_path)

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

    def load_groups(self):
        groupby_cols = [
            'data', 'model', 'anomaly_class', 'anomaly_type', 'rfi_threshold', 'ood_rfi', 'limit', 'use_hyp_data',
            'epochs', 'batch_size', 'lr', 'std_plus', 'std_minus', 'per_image', 'filters', 'height', 'dropout',
            'kernel_regularizer', 'level_blocks', 'dilation_rate', 'patch_x', 'patch_y', 'loss', 'lofar_subset',
            'latent_dim', 'alpha', 'neighbour', 'algorithm', 'combine_min_std_plus'
        ]
        # perhaps a list for the counts
        dfg = self.df.groupby(groupby_cols, as_index=False).agg(
            {
                'val_f1': ['mean', 'std'],
                'val_auroc': ['mean', 'std'],
                'val_auprc': ['mean', 'std'],
                'test_f1': ['mean', 'std'],
                'test_auroc': ['mean', 'std'],
                'test_auprc': ['mean', 'std'],
                'train_loss': ['mean', 'std'],
                'val_loss': ['mean', 'std'],
                'time_image': ['mean', 'std'],
                'flops_image': ['mean', 'std'],  # trainable params, # last epoch
                'trainable_params': ['mean', 'std'],
                'nontrainable_params': ['mean', 'std'],
                'last_epoch': ['mean', 'std'],
            }
        )

        for i in range(dfg.shape[0]):
            if all_true(dfg.iloc[i], True, self.group_filter_dict):
                label = get_label(dfg.iloc[i], True, **self.group_label_dict)
                self.labels.append(label)

                self.val_f1_means.append(float(dfg.iloc[i]['val_f1']['mean']))
                self.val_f1_stds.append(float(dfg.iloc[i]['val_f1']['std']))
                self.val_auroc_means.append(float(dfg.iloc[i]['val_auroc']['mean']))
                self.val_auroc_stds.append(float(dfg.iloc[i]['val_auroc']['std']))
                self.val_auprc_means.append(float(dfg.iloc[i]['val_auprc']['mean']))
                self.val_auprc_stds.append(float(dfg.iloc[i]['val_auprc']['std']))

                self.test_f1_means.append(float(dfg.iloc[i]['test_f1']['mean']))
                self.test_f1_stds.append(float(dfg.iloc[i]['test_f1']['std']))
                self.test_auroc_means.append(float(dfg.iloc[i]['test_auroc']['mean']))
                self.test_auroc_stds.append(float(dfg.iloc[i]['test_auroc']['std']))
                self.test_auprc_means.append(float(dfg.iloc[i]['test_auprc']['mean']))
                self.test_auprc_stds.append(float(dfg.iloc[i]['test_auprc']['std']))

                self.train_loss_means.append(float(dfg.iloc[i]['train_loss']['mean']))
                self.train_loss_stds.append(float(dfg.iloc[i]['train_loss']['std']))

                self.val_loss_means.append(float(dfg.iloc[i]['val_loss']['mean']))
                self.val_loss_stds.append(float(dfg.iloc[i]['val_loss']['std']))

                self.time_image_means.append(float(dfg.iloc[i]['time_image']['mean']))
                self.time_image_stds.append(float(dfg.iloc[i]['time_image']['std']))

                self.flops_image_means.append(float(dfg.iloc[i]['flops_image']['mean']))
                self.trainable_params_means.append(float(dfg.iloc[i]['trainable_params']['mean']))
                self.nontrainable_params_means.append(float(dfg.iloc[i]['nontrainable_params']['mean']))

                self.last_epoch_means.append(float(dfg.iloc[i]['last_epoch']['mean']))
                self.last_epoch_stds.append(float(dfg.iloc[i]['last_epoch']['std']))

    def save_flops_f1(self, file_name='flops_f1', ylabel='f1 score'):
        save_flops_metric(self.dir_path, self.flops_image_means, self.test_f1_means, self.labels,
                          self.trainable_params_means, file_name=file_name, ylabel=ylabel)

    def save_flops_auroc(self, file_name='flops_auroc', ylabel='auroc'):
        save_flops_metric(self.dir_path, self.flops_image_means, self.test_auroc_means, self.labels,
                          self.trainable_params_means, file_name=file_name, ylabel=ylabel)

    def save_flops_auprc(self, file_name='flops_auprc', ylabel='auprc'):
        save_flops_metric(self.dir_path, self.flops_image_means, self.test_auprc_means, self.labels,
                          self.trainable_params_means, file_name=file_name, ylabel=ylabel)

    def save_training_metrics(self, model, loss, filter_dict={}, label_list=['lr'], plot_train=False):
        """One metrics curve for each (model, loss) pair. Additional filters are contained in filter_dict
        If exactly the same model solution is repeated, then labels will be repeated and the curves should be similar"""
        label_dict = empty_label_dict()
        for label in label_list:
            label_dict[label] = True
        filter_dict['model'] = model
        filter_dict['loss'] = loss

        labels = []
        losses_list = [] # One list for each label, list lengths may differ if early stopping occurred

        for i in range(self.df.shape[0]):
            if all_true(self.df.iloc[i], False, filter_dict):
                label = get_label(self.df.iloc[i], False,  **label_dict)
                labels.append(label)

                # Assume models are in same dir as self.dir_path
                path = '{}/{}/{}/{}/losses'.format(self.dir_path, model, self.df.iloc[i]['anomaly_class'], self.df.iloc[i]['name'])

                if plot_train:
                    with open(path+'/val_epoch_losses.txt', 'r') as f:
                        losses_list.append([float(line.rstrip()) for line in f])
                else:
                    with open(path+'/train_epoch_losses.txt', 'r') as f:
                        losses_list.append([float(line.rstrip()) for line in f])

        # Only plot val_losses for now
        if len(labels) > 0:
            file_name = '{}_{}_training'.format(model, loss)
            save_training_metrics(self.dir_path, losses_list, labels, file_name, ylabel=loss)
