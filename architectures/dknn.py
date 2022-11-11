from .generic_architecture import GenericArchitecture

from utils.metrics import (get_nln_metrics,
                           save_metrics_csv,
                           evaluate_performance,
                           save_results_csv,
                           nln,
                           get_nln_errors,
                           get_dists)
from utils.profiling import (num_trainable_params,
                             num_non_trainable_params,
                             get_flops)

from utils.training import print_epoch, save_checkpoint
from utils.plotting import save_training_metrics, save_data_masks_inferred, save_data_masks_dknn
from data_collection import DataCollection
from utils.data import patches
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

"""
DKNN Uses ResNet weights, therefore may not be retrained.
Create a new architecture and/or model if you want to retrain it
"""


class DKNNArchitecture(GenericArchitecture):

    #def __init__
    #def save_summary

    @tf.function
    def train_step(self, x, y):
        pass

    def train(self, data_collection: DataCollection):
        print('Training failed: DKNN may not be trained, as it uses weights of RESNET')

    # output = np.empty([len(data), 2048], np.float32)
    # should still work
    # output = np.empty([len(data)] + self.model.outputs[0].shape[1:], dtype=np.float32)
    #def infer(self, data):

    #def print_epoch(self, epoch, _time, metrics, metric_labels):

    #def save_checkpoint(self, epoch, model_subtype=None):

    # Reimplement?
    def save_data_masks_images_dknn(self, data_recon, masks_recon, dists_recon):
        save_data_masks_dknn(self.dir_path, data_recon, masks_recon, dists_recon)

    def evaluate_and_save(self, dc: DataCollection):
        z_train = self.infer(dc.train_data)

        # Infer and get dists
        start = time.time()  # z_train can be stored on disk
        flops_patch = get_flops(self.model)
        z_test = self.infer(dc.test_data)

        # Nearest neighbours to each patch
        flops_patch += 0  # TODO, dependance on train_data size
        neighbours_dist, _, _, _ = nln(z_train, z_test, None, 'knn', 2, -1)

        flops_patch += 0  # TODO, should be linear
        dists_recon = get_dists(neighbours_dist,
                                dc.raw_input_shape,
                                dc.patch_x,
                                dc.patch_y)
        test_masks_recon = dc.reconstruct(dc.test_masks)

        # Timing and flops
        time_patch = (time.time() - start) / dc.test_data.shape[0]
        time_image = time_patch * dc.patches_per_image()
        flops_image = flops_patch * dc.patches_per_image()

        # Save images
        n_patches = self.num_samples * dc.patches_per_image()
        save_data = dc.reconstruct(dc.test_data[:n_patches, ...])
        save_masks = test_masks_recon[:self.num_samples, ...]
        save_dists_recon = dists_recon[:self.num_samples, ...]
        self.save_data_masks_images_dknn(save_data, save_masks, save_dists_recon)

        # Dictionaries to save
        #metrics_dict = self.get_metrics(test_masks_recon, dists_recon, True)
        auroc, auprc, f1 = self.get_metrics(test_masks_recon, dists_recon)
        metrics_dict = {'auroc': auroc, 'auprc': auprc, 'f1': f1 }
        arch_dict = self.to_dict()
        data_dict = dc.to_dict()
        results = {**metrics_dict, **arch_dict, **data_dict,
                   'time_patch': time_patch,
                   'time_image': time_image,
                   'flops_patch': flops_patch,
                   'flops_image': flops_image,
                   }
        save_results_csv(dc.data_name, dc.seed, results)

    #def get_metrics(self, masks, masks_inferred, ret_dict=False):

    #def load_checkpoint(self):

    #def to_dict(self):
