from utils.metrics import get_nln_metrics, save_metrics_csv, evaluate_performance, get_metrics, save_results_csv
from utils.profiling import num_trainable_params, num_non_trainable_params, get_flops

from utils.training import print_epoch, save_checkpoint
# from model_config import *
from tensorflow.keras import optimizers
from tensorflow.keras import losses
# bce = tf.keras.losses.BinaryCrossentropy()
# mse = tf.keras.losses.MeanSquaredError()
# mae = tf.keras.losses.MeanAbsoluteError()
from utils.plotting import save_training_metrics, save_data_masks_inferred
from data_collection import DataCollection

from utils.data import patches
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

from sklearn.metrics import (roc_curve,
                             auc,
                             f1_score,
                             accuracy_score,
                             average_precision_score,
                             jaccard_score,
                             roc_auc_score,
                             precision_recall_curve)


class GenericArchitecture:
    def __init__(self, model, args):
        random.seed(42)
        self.algorithm = args.algorithm
        self.num_samples = 10  # for saving images
        self.model = model
        self.model_name = args.model_name
        self.anomaly_class = args.anomaly_class
        # self.loss_func = loss_func
        self.loss_func = losses.BinaryCrossentropy()  # can always change after initialization
        # self.optimizer = optimizer
        self.optimizer = optimizers.Adam()  # can always change after initialization
        self.epochs = args.epochs
        self.model_type = args.model
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.dir_path = 'outputs/{}/{}/{}'.format(self.model_type, self.anomaly_class, self.model_name)
        self.latent_dim = args.latent_dim
        self.neighbours = args.neighbours
        self.alphas = args.alphas
        self.height = args.height
        self.filters = args.filters
        self.model_config = args.model_config
        self.level_blocks = args.level_blocks
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if not os.path.exists(self.dir_path + '/epochs'):
            os.makedirs(self.dir_path + '/epochs')
        if not os.path.exists(self.dir_path + '/training_checkpoints'):
            os.makedirs(self.dir_path + '/training_checkpoints')

    def save_summary(self):
        with open(self.dir_path + '/model.summary', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f'GFLOPS: {get_flops(self.model) / 1e9}')

    @tf.function
    def train_step(self, x, y):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            x_hat = self.model(x, training=True)
            # For AE
            # TypeError: outer_factory.<locals>.inner_factory.<locals>.tf____call__() got an unexpected keyword argument 'training'
            loss = self.loss_func(x_hat, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, data_collection: DataCollection):
        if not data_collection.all_not_none(['train_data', 'train_masks']):
            raise ValueError('data_collection is missing train_data or train_masks')

        epoch_losses = []
        train_mask_dataset = tf.data.Dataset.from_tensor_slices(
            data_collection.train_masks.astype('float32')).shuffle(self.buffer_size, seed=42).batch(self.batch_size)
        train_data_dataset = tf.data.Dataset.from_tensor_slices(
            data_collection.train_data).shuffle(self.buffer_size, seed=42).batch(self.batch_size)
        train_start = time.time()
        for epoch in range(self.epochs):
            start = time.time()
            epoch_loss = 0.0
            for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
                loss = self.train_step(image_batch, mask_batch)
                epoch_loss += loss

            epoch_loss /= len(train_data_dataset)  # divide by number of batches
            epoch_losses.append(epoch_loss)
            with open(self.dir_path + '/epoch_losses.txt', 'a+') as f:
                f.write(f'{epoch_loss}\n')

            inds = random.sample(range(len(data_collection.train_data)), self.num_samples)  # 10 random images
            # slice indices must be integers or None or have an __index__ method
            train_masks_inferred = self.infer(data_collection.train_data[inds])
            self.save_data_images(data_collection.train_data[inds],
                                  data_collection.train_masks[inds],
                                  train_masks_inferred,
                                  epoch)
            self.save_checkpoint(epoch)
            self.print_epoch(epoch, time.time() - start, epoch_loss, 'loss')

        print(f'Total training time: {(time.time() - train_start) // 60} min')
        self.save_checkpoint()
        self.save_training_metrics_image(epoch_losses, f'{self.model_type} loss')
        # self.save_summary()

    def print_epoch(self, epoch, _time, metrics, metric_labels):
        print_epoch(self.model_type, epoch, _time, metrics, metric_labels)

    def save_training_metrics_image(self, metrics, metric_labels):
        save_training_metrics(self.dir_path, metrics, metric_labels)

    def save_checkpoint(self, epoch=-1, model_subtype=None):
        if model_subtype is None:
            model_subtype = self.model_type
        save_checkpoint(self.dir_path, self.model, model_subtype, epoch)

    def save_data_images(self, data, masks, masks_inferred, epoch=-1):
        save_data_masks_inferred(self.dir_path, data, masks, masks_inferred, epoch)

    def infer(self, data):
        # assume self.model is not a list type e.g. (ae, disc) and that the model has only one output
        # i.e. len(model.outputs) == 1
        # data can be in patches or not

        data_tensor = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        output = np.empty([len(data)] + self.model.outputs[0].shape[1:], dtype=np.float32)
        strt, fnnsh = 0, 0
        for batch in data_tensor:
            fnnsh += len(batch)
            output[strt:fnnsh, ...] = self.model(batch, training=False).numpy()  # .astype(np.float32)
            strt = fnnsh

        output[output == np.inf] = np.finfo(output.dtype).max
        return output

    def evaluate_and_save(self, dc: DataCollection):

        # Infer data
        start = time.time()
        test_masks_inferred = self.infer(dc.test_data)
        time_patch = (time.time() - start) / dc.test_data.shape[0]  # per patch
        time_image = time_patch * dc.patches_per_image()

        # Generate data to be saved to image
        test_masks_inferred_recon = dc.reconstruct(test_masks_inferred)
        test_data_recon = dc.reconstruct(dc.test_data)
        test_masks_recon = dc.reconstruct(dc.test_masks)

        inds = random.sample(range(len(test_data_recon)), self.num_samples)  # 10 random images
        save_data = test_data_recon[inds]
        save_masks = test_masks_recon[inds]
        save_masks_inferred = test_masks_inferred_recon[inds]

        self.save_data_images(save_data, save_masks, save_masks_inferred)

        # Generate and save metrics
        auroc, auprc, f1 = self.get_metrics(dc.test_masks, test_masks_inferred)
        metrics_dict = {'auroc': auroc, 'auprc': auprc, 'f1': f1}
        arch_dict = self.to_dict()
        data_dict = dc.to_dict()
        flops_patch = get_flops(self.model)
        flops_image = flops_patch * dc.patches_per_image()
        results = {**metrics_dict, **arch_dict, **data_dict,
                   'time_patch': time_patch,
                   'time_image': time_image,
                   'flops_patch': flops_patch,
                   'flops_image': flops_image,
                   }
        save_results_csv(dc.data_name, dc.seed, results)

        # Save solution and model info
        self.save_summary()
        self.save_solution_config({**arch_dict, **data_dict})

    def save_solution_config(self, sol_dict):
        with open('{}/solution.config'.format(self.dir_path), 'w') as fp:
            for k in sol_dict.keys():
                fp.write(f'{k}: {sol_dict[k]}\n')

    def get_metrics(self, masks, masks_inferred):
        # _, auroc,_, auprc, _, f1 = get_metrics(None, masks, masks_inferred)
        fpr, tpr, thr = roc_curve(masks.flatten() > 0, masks_inferred.flatten())
        auroc = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(masks.flatten() > 0,
                                                               masks_inferred.flatten())
        auprc = auc(recall, precision)

        f1_scores = 2 * recall * precision / (recall + precision)
        f1 = np.nanmax(f1_scores)

        return auroc, auprc, f1

    def load_checkpoint(self):
        # Note that the correct name must be in args when initializing Architecture
        path = f'{self.dir_path}/training_checkpoints/checkpoint_full_model_{self.model_type}'
        self.model.load_weights(path)

    def to_dict(self):
        return {'model': self.model_type,
                'name': self.model_name,
                'height': self.height,
                'filters': self.filters,
                'level_blocks': self.level_blocks,
                'model_config': self.model_config,
                'latent_dim': self.latent_dim,
                'trainable_params': num_trainable_params(self.model),
                'nontrainable_params': num_non_trainable_params(self.model),
                'algorithm': self.algorithm
                # 'flops_patch': get_flops(self.model) / 1e9
                }
