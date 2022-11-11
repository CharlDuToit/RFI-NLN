#import os
#import tensorflow as tf
import numpy as np


def save_checkpoint(dir_path, model, model_subtype, epoch=-1, losses=None):
    """
        Saves model weights at given checkpoint 

        model (tf.keras.Model): the model 
        epoch (int): current epoch of training
        model_subtype (str): the part of the model (ae, disc,...)
        losses: (list float)
    """

    #if ((epoch + 1) % 10 == 0) and epoch > -1:
    # if epoch > -1:
    #     if losses is None:
    #         model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path, model_subtype))
    #     else:
    #         if np.argmin(losses) == len(losses) -1: # only save if lowest loss so far
    #             model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path, model_subtype))
    # else:
    #     model.save_weights('{}/training_checkpoints/checkpoint_full_model_{}'.format( dir_path, model_subtype))
    #     print(f'Successfully Saved Model: {model_subtype}')
    if losses is None:
        model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path, model_subtype))
    else:
        if np.argmin(losses) == len(losses) -1:  # only save if lowest loss so far
            model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path, model_subtype))
