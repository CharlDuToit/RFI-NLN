import os
import tensorflow as tf


def save_checkpoint(dir_path, model, model_subtype, epoch=-1):
    """
        Saves model weights at given checkpoint 

        model (tf.keras.Model): the model 
        epoch (int): current epoch of training
        args (Namespace): the arguments from cmd_args
        model_type (str): the type of model (AE,VAE,...)
        model_subtype (str): the part of the model (ae, disc,...) 
    """
    # dir_path = 'outputs/{}/{}/{}'.format(model_type,
    #                                      args.anomaly_class,
    #                                      args.model_name)
    if ((epoch + 1) % 10 == 0) and epoch > -1:
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)

        model.save_weights('{}/training_checkpoints/checkpoint_{}'.format(dir_path, model_subtype))

    if epoch < 0:
        model.save_weights('{}/training_checkpoints/checkpoint_full_model_{}'.format( dir_path, model_subtype))

        # TODO: This is a really ugly quick fix to write the config
        #with open('{}/model.config'.format(dir_path), 'w') as fp:
        #    for arg in args.__dict__:
        #        fp.write('{}: {}\n'.format(arg, args.__dict__[arg]))

        print(f'Successfully Saved Model: {model_subtype}')
