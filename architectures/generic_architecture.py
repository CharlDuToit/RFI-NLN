from utils import DiceLoss, f1, auroc, auprc
from utils import save_csv
from utils import num_trainable_params, num_non_trainable_params, get_flops

from utils import print_epoch, save_checkpoint_to_path
# from model_config import *
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from scipy.io import savemat

from utils import save_epochs_curve, save_data_masks_inferred, save_data_inferred
from data_collection import DataCollection

# from utils.data import patches
import time
import os
import numpy as np
# from matplotlib import pyplot as plt
import random
import tensorflow as tf


# from sklearn.metrics import (roc_curve,
#                              auc,
#                              f1_score,
#                              accuracy_score,
#                              average_precision_score,
#                              jaccard_score,
#                              roc_auc_score,
#                              precision_recall_curve)


class GenericArchitecture:
    def __init__(self, model, args, checkpoint='None'):
        """checkpoint must be in ['None', 'self', 'parent']"""

        self.model = model

        self.loss = args.loss
        if self.loss == 'bce':
            self.loss_func = losses.BinaryCrossentropy()
        elif self.loss == 'mse':
            self.loss_func = losses.MeanSquaredError()
        elif self.loss == 'dice':
            self.loss_func = DiceLoss()
        else:
            raise ValueError(f'Loss function: {self.loss} is not supported')
        self.lr = args.lr
        self.optimizer = optimizers.Adam(lr=self.lr)

        self.early_stop = args.early_stop
        self.final_activation = args.final_activation

        self.val_split = args.val_split
        self.split_seed = args.split_seed
        self.num_samples = args.images_per_epoch  # for saving images, must be smaller than batch size
        self.epoch_image_interval = args.epoch_image_interval
        self.save_dataset = args.save_dataset

        self.use_hyp_data = args.use_hyp_data
        self.algorithm = args.algorithm
        self.model_name = args.model_name
        self.parent_model_name = args.parent_model_name
        self.anomaly_class = args.anomaly_class

        self.epochs = args.epochs
        self.model_type = args.model
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.dir_path = '{}/{}/{}/{}'.format(args.output_path, self.model_type, self.anomaly_class, self.model_name)
        #self.dir_path = 'outputs/{}/{}/{}'.format(self.model_type, self.anomaly_class, self.model_name)

        self.latent_dim = args.latent_dim
        self.neighbours = args.neighbours
        self.alphas = args.alphas

        # These have nothing to do with training, only how the model was built
        self.dilation_rate = args.dilation_rate
        self.dropout = args.dropout
        self.kernel_regularizer = args.kernel_regularizer
        self.height = args.height
        self.filters = args.filters
        self.model_config = args.model_config
        self.level_blocks = args.level_blocks

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if not os.path.exists(self.dir_path + '/epochs'):
            os.makedirs(self.dir_path + '/epochs')
        if not os.path.exists(self.dir_path + '/losses'):
            os.makedirs(self.dir_path + '/losses')
        if not os.path.exists(self.dir_path + '/training_checkpoints'):
            os.makedirs(self.dir_path + '/training_checkpoints')

        if checkpoint == 'self':
            self.load_checkpoint()

        #self.parent_loss = None
        if checkpoint == 'parent':
            if self.parent_model_name is None:
                raise ValueError('Cant load parent checkpoint if parent name is None')
            parent_args = EmptyArchArgs(self.parent_model_name, self.anomaly_class, self.model_type)
            parent_arch = GenericArchitecture(self.model, parent_args, checkpoint='None')
            parent_arch.load_checkpoint()
            self.model = parent_arch.model
            self.save_checkpoint()
            #self.first_epoch = parent_arch.get_best_epoch() + 1

            # try:
            #     with open(self.dir_path + '/losses/val_epoch_losses.txt', 'w') as f:
            #         parent_val_losses = parent_arch.get_losses('val')
            #         self.parent_loss = np.min(parent_val_losses)
            #         for i in range(0, self.first_epoch):
            #             f.write(f'{parent_val_losses[i]}\n')
            # except:
            #     print('Parent validation loss text file error')
            #
            # with open(self.dir_path + '/losses/train_epoch_losses.txt', 'w') as f:
            #     parent_train_losses = parent_arch.get_losses('train')
            #     if self.parent_loss is not None:
            #         self.parent_loss = np.min(parent_train_losses)
            #     for i in range(0, self.first_epoch):
            #         f.write(f'{parent_train_losses[i]}\n')

        # non arg dependant members
        self.val_loss = None
        self.train_loss = None
        self.val_auroc = None
        self.val_auprc = None
        self.val_f1 = None
        self.num_train = 0
        self.num_val = 0
        self.last_epoch = 0
        #self.best_epoch = 0
        #self.first_epoch = 0
        self.epoch_time = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    @tf.function
    def train_step(self, x, y):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            x_hat = self.model(x, training=True)
            loss = self.loss_func(x_hat, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    # ------------------------------------------------------------------------------------------------------------------
    def train(self, data_collection: DataCollection):
        # Create dataset

        (train_data_dataset,
         val_data_dataset,
         train_mask_dataset,
         val_mask_dataset,
         self.num_train,
         self.num_val,
         self.split_seed) = data_collection.get_datasets(self.val_split,
                                                         self.use_hyp_data,
                                                         False,
                                                         self.buffer_size,
                                                         self.batch_size,
                                                         self.split_seed)
        use_val_data = (val_data_dataset is not None)
        print('Created tf batched datasets')

        # Perform training
        train_losses = []
        val_losses = []
        train_start = time.time()
        for epoch in range(self.epochs):
            start = time.time()

            # Train model
            for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
                loss = self.train_step(image_batch, mask_batch)
                # 3 GB mem incrase for 256,64,64,1 batch with model height=4, filters=64

            # Infer data after training to compare to validation loss
            train_loss = 0.0
            save_image_batch = None
            save_mask_batch = None
            for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
                x_hat = self.model(image_batch, training=False)
                loss = self.loss_func(x_hat, mask_batch)
                train_loss += loss * len(image_batch)
                if save_image_batch is None:  # get the first batch
                    save_image_batch = image_batch
                    save_mask_batch = mask_batch
            train_loss = (train_loss / self.num_train).numpy()  # divide by number of observations
            train_losses.append(train_loss)

            with open(self.dir_path + '/losses/train_epoch_losses.txt', 'a+') as f:
                f.write(f'{train_loss}\n')

            # Calculate validation loss
            val_loss = None
            if use_val_data:
                val_loss = 0.0
                for image_batch, mask_batch in zip(val_data_dataset, val_mask_dataset):
                    x_hat = self.model(image_batch, training=False)
                    loss = self.loss_func(x_hat, mask_batch)
                    val_loss += loss * len(image_batch)
                val_loss = (val_loss / self.num_val).numpy()  # divide by number of observations
                val_losses.append(val_loss)

                with open(self.dir_path + '/losses/val_epoch_losses.txt', 'a+') as f:
                    f.write(f'{val_loss}\n')

            self.print_epoch(epoch, time.time() - start, [train_loss, val_loss],
                             ['train loss', 'val loss'])

            # Save inferred images
            if self.epoch_image_interval > 0 and self.num_samples > 0 and epoch % self.epoch_image_interval == 0:
                data_save = save_image_batch[:self.num_samples].numpy()
                masks_save = save_mask_batch[:self.num_samples].numpy()
                masks_inferred = self.infer(data_save)
                self.save_data_masks_images(data_save,
                                            masks_save,
                                            masks_inferred,
                                            epoch)

            # Save checkpoint if lowest loss thus far
            losses = train_losses
            if use_val_data:
                losses = val_losses
            # if self.parent_loss is not None:
            #     losses = [self.parent_loss] + losses
            self.save_checkpoint(epoch, self.model_type, losses)
            if np.argmin(losses) + self.early_stop < len(losses):  # No improvement for 20 epochs
                print(f'No improvement for {self.early_stop} epochs, stopping training')
                break

        # END for epoch in range(self.epochs):
        train_time = time.time() - train_start
        self.last_epoch = epoch #+ self.first_epoch
        self.epoch_time = train_time / (epoch + 1)

        print('__________________________________')
        print(f'Total training time: {train_time // 60} min')

        # loads the best checkpoint
        self.load_checkpoint()

        # Save losses
        self.train_loss = np.min(train_losses)
        if use_val_data:
            self.val_loss = np.min(val_losses)
            self.train_loss = train_losses[np.argmin(val_losses)]  # same epoch
        self.save_training_metrics_image([train_losses, val_losses], ['train loss', 'val loss'])

        # Validation metrics
        val_masks_inferred = np.empty([self.num_val] + self.model.outputs[0].shape[1:], dtype=np.float32)
        val_masks = np.empty([self.num_val] + self.model.outputs[0].shape[1:], dtype=np.float32)
        strt, fnnsh = 0, 0
        for image_batch, mask_batch in zip(val_data_dataset, val_mask_dataset):
            fnnsh += len(image_batch)
            val_masks_inferred[strt:fnnsh, ...] = self.model(image_batch, training=False).numpy()
            val_masks[strt:fnnsh, ...] = mask_batch.numpy()
            strt = fnnsh
        val_masks_inferred[val_masks_inferred == np.inf] = np.finfo(val_masks_inferred.dtype).max
        self.val_auroc, self.val_auprc, self.val_f1 = self.get_metrics(val_masks, val_masks_inferred)

    # ------------------------------------------------------------------------------------------------------------------
    def infer(self, data):
        # data is a numpy ndarray or 'TensorSliceDataset'
        # output is np.ndarray
        # assume self.model is not a list type e.g. (ae, disc) and that the model has only one output
        # i.e. len(model.outputs) == 1

        if str(type(data)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>":
            dataset = data
            n_obs = 0
            for batch in dataset:
                n_obs += len(batch)
        elif isinstance(data, np.ndarray):
            n_obs = len(data)
            dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        else:
            raise ValueError('data must be np.ndarray or BatchDataset')

        output = np.empty([n_obs] + self.model.outputs[0].shape[1:], dtype=np.float32)
        strt, fnnsh = 0, 0
        for batch in dataset:
            fnnsh += len(batch)
            output[strt:fnnsh, ...] = self.model(batch, training=False).numpy()  # .astype(np.float32)
            strt = fnnsh

        output[output == np.inf] = np.finfo(output.dtype).max
        return output

    # ------------------------------------------------------------------------------------------------------------------
    def evaluate_and_save(self, dc: DataCollection):

        # Infer data
        print("__________________________________ \nEvaluating data")
        start = time.time()

        test_masks_inferred = self.infer(dc.test_data)
        infer_time = time.time() - start
        time_patch = infer_time / dc.test_data.shape[0]  # per patch
        time_image = time_patch * dc.patches_per_image()
        print('  Inference time : {:.2f} sec'.format(infer_time))

        # Generate data to be saved to image
        test_masks_inferred_recon = dc.reconstruct(test_masks_inferred)
        test_data_recon = dc.reconstruct(dc.test_data)
        test_masks_recon = dc.reconstruct(dc.test_masks)

        inds = random.sample(range(len(test_data_recon)), self.num_samples)  # 10 random images
        save_data = test_data_recon[inds]
        save_masks = test_masks_recon[inds]
        save_masks_inferred = test_masks_inferred_recon[inds]

        self.save_data_masks_images(save_data, save_masks, save_masks_inferred)
        self.save_data_masks_images(save_data, save_masks, save_masks_inferred, thresh=0.5, figsize=(40,80))

        # Generate and save metrics
        test_auroc, test_auprc, test_f1 = self.get_metrics(dc.test_masks, test_masks_inferred)  # this takes some time
        test_metrics_dict = {'test_auroc': test_auroc, 'test_auprc': test_auprc, 'test_f1': test_f1}

        arch_dict = self.to_dict()
        data_dict = dc.to_dict()
        flops_patch = get_flops(self.model)
        flops_image = flops_patch * dc.patches_per_image()
        results = {**test_metrics_dict, **arch_dict, **data_dict,
                   'time_patch': time_patch,
                   'time_image': time_image,
                   'flops_patch': flops_patch,
                   'flops_image': flops_image,
                   }
        save_csv(dc.data_name, dc.seed, results)

        # Save solution and model info
        self.save_summary()
        self.save_solution_config({**arch_dict, **data_dict})

        # Save inferred dataset
        if self.save_dataset:
            path = os.path.join(self.dir_path, 'inferred', dc.data_name)
            masks_dict = {'masks_inferred': test_masks_recon}
            mat_file = os.path.join(path, 'masks_inferred.mat')
            savemat(mat_file, masks_dict)

        print('Data Evaluation time : {:.2f} sec'.format(time.time() - start))

    # ------------------------------------------------------------------------------------------------------------------
    def infer_and_save_train_data(self, dc: DataCollection):
        """
        infers the training data and saves it
        """

        masks_inferred = self.infer(dc.train_data)

        masks_inferred_recon = dc.reconstruct(masks_inferred)
        data_recon = dc.reconstruct(dc.train_data)

        if self.num_samples < len(data_recon):
            inds = random.sample(range(len(data_recon)), self.num_samples)
        else:
            inds = range(len(data_recon))

        save_data = data_recon[inds]
        save_masks_inferred = masks_inferred_recon[inds]

        path = os.path.join(self.dir_path, 'inferred', dc.data_name)
        save_data_inferred(path, save_data, save_masks_inferred, thresh=-1)
        save_data_inferred(path, save_data, (save_masks_inferred > 0.5).astype(np.float32), thresh=0.5, figsize=(80,40))

        if self.save_dataset:
            masks_dict = {'masks_inferred': masks_inferred_recon}
            mat_file = os.path.join(path, 'masks_inferred.mat')
            savemat(mat_file, masks_dict)

        # =============================================HELPERS==============================================================

    def save_summary(self):
        with open(self.dir_path + '/summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f'GFLOPS patch: {get_flops(self.model) / 1e9}')

    def print_epoch(self, epoch, _time, metrics, metric_labels):
        print_epoch(self.model_type, epoch, _time, metrics, metric_labels)

    def save_training_metrics_image(self, metrics, metric_labels):
        save_epochs_curve(self.dir_path, metrics, metric_labels)


    def save_checkpoint(self, epoch=-1, model_subtype=None, losses=None):
        if model_subtype is None:
            model_subtype = self.model_type
        save_checkpoint_to_path(self.dir_path, self.model, model_subtype, epoch, losses)

    def save_data_masks_images(self, data, masks, masks_inferred, epoch=-1, thresh=-1.0, figsize=(10,20)):
        if 0.0 < thresh < 1.0:
            masks_inferred = (masks_inferred > thresh).astype(np.float32)
        save_data_masks_inferred(self.dir_path, data, masks, masks_inferred, epoch, thresh, figsize)

    def save_data_images(self, data, masks_inferred, thresh=-1.0):
        if 0.0 < thresh < 1.0:
            masks_inferred = (masks_inferred > thresh).astype(np.float32)
        save_data_inferred(self.dir_path, data, masks_inferred, thresh)

    def save_solution_config(self, sol_dict):
        with open('{}/solution.txt'.format(self.dir_path), 'w') as fp:
            for k in sol_dict.keys():
                fp.write(f'{k}: {sol_dict[k]}\n')

    def get_metrics(self, masks, masks_inferred):
        return auroc(masks, masks_inferred), auprc(masks, masks_inferred), f1(masks, masks_inferred)

    def load_checkpoint(self):
        # Note that the correct name must be in args when initializing Architecture
        # path = f'{self.dir_path}/training_checkpoints/checkpoint_full_model_{self.model_type}'
        path = f'{self.dir_path}/training_checkpoints/checkpoint_{self.model_type}'
        self.model.load_weights(path)

    def get_losses(self, prefix='val'):
        """ prefix must be in ['val', 'train']. Child classes can have more options"""
        losses_list = []
        with open(f'{self.dir_path}/{prefix}_epoch_losses.txt', 'r') as f:
            losses_list.append([float(line.rstrip()) for line in f])
        return losses_list

    # def get_best_epoch(self):
    #     try:
    #         best_epoch = np.argmin(self.get_losses('val'))
    #         return best_epoch
    #     except:
    #         print('Failed to load validation losses from text file, trying training loss')
    #     try:
    #         best_epoch = np.argmin(self.get_losses('train'))
    #         return best_epoch
    #     except:
    #         raise FileNotFoundError('Failed to load losses')

    def to_dict(self):
        return {'model': self.model_type,
                'name': self.model_name,
                'parent_name': self.parent_model_name,
                'height': self.height,
                'filters': self.filters,
                'level_blocks': self.level_blocks,
                'model_config': self.model_config,
                'latent_dim': self.latent_dim,
                'trainable_params': num_trainable_params(self.model),
                'nontrainable_params': num_non_trainable_params(self.model),
                'algorithm': self.algorithm,
                'use_hyp_data': self.use_hyp_data,
                'num_train': self.num_train,
                'num_val': self.num_val,
                'split_seed': self.split_seed,
                'epoch_time': self.epoch_time,
                'lr': self.lr,
                'loss': self.loss,
                'epochs': self.epochs,
                #'first_epoch': self.first_epoch,
                #'best_epoch': self.get_best_epoch(),
                'last_epoch': self.last_epoch,
                'dilation_rate': self.dilation_rate,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'kernel_regularizer': self.kernel_regularizer,
                'final_activation': self.final_activation,
                'early_stop': self.early_stop,
                'val_loss': self.val_loss,
                'train_loss': self.train_loss,
                'val_auprc': self.val_auprc,
                'val_auroc': self.val_auroc,
                'val_f1': self.val_f1,

                # 'flops_patch': get_flops(self.model) / 1e9
                }


class EmptyArchArgs:
    def __init__(self, model_name, anomaly_class, model_type):

        self.model_name = model_name
        self.anomaly_class = anomaly_class
        self.model = model_type

        self.height = 0
        self.filters = 0
        self.level_blocks = 0
        self.parent_model_name = None
        self.model_config = None
        self.use_hyp_data = None
        self.loss = 'bce'
        self.lr = 0.001
        self.early_stop = 0
        self.images_per_epoch = 0
        self.epoch_image_interval = 0
        self.final_activation = None
        self.val_split = 0
        self.split_seed = 0
        self.algorithm = None
        self.dilation_rate = 0
        self.dropout = 0
        self.kernel_regularizer = None
        self.epochs = 0
        self.batch_size = 0
        self.buffer_size = 0
        self.latent_dim = 0
        self.neighbours = 0
        self.alphas = 0
