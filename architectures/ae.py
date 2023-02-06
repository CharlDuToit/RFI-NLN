import tensorflow as tf
import numpy as np
from sklearn import neighbors
from matplotlib import pyplot as plt
import time
from models import Autoencoder

from utils  import  (generate_and_save_images,
                              save_epochs_curve)

from utils import print_epoch,save_checkpoint_to_path
from model_config import mse
from .helper import end_routine
from inference import infer

optimizer = tf.keras.optimizers.Adam(1e-4)

def l2_loss(x,x_hat):
    return mse(x,x_hat)

@tf.function
def train_step(model, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x)
        loss = l2_loss(x,x_hat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(ae,train_dataset,train_images, test_images,test_labels,args,verbose=True,save=True):
    ae_loss= []
    dir_path = 'outputs/{}/{}/{}'.format(args.model_class, args.anomaly_class, args.model_name)
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
            auto_loss = train_step(ae,image_batch)

        generate_and_save_images(ae,
                                 epoch + 1,
                                 image_batch[:25,...],
                                 'AE',
                                 args)
        save_checkpoint_to_path(dir_path, ae, 'AE', epoch)

        ae_loss.append(auto_loss)

        #print_epoch('AE',epoch,time.time()-start,{'AE Loss':auto_loss.numpy()},None)
        print_epoch('AE', epoch, time.time() - start, auto_loss.numpy(), 'loss')

    save_checkpoint_to_path(dir_path, ae, 'AE')
    save_epochs_curve(dir_path, ae_loss, 'AE loss')
    generate_and_save_images(ae,epoch,image_batch[:25,...],'AE',args)

    return ae

def main(train_dataset,train_images,train_labels,test_images,test_labels, test_masks,test_masks_orig,args):
    if args.data_name == 'MVTEC':
        ae = Autoencoder_MVTEC(args)
    else:
        ae = Autoencoder(args)

    ae = train(ae,train_dataset, train_images,test_images,test_labels,args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [ae], 'AE', args)

    
if __name__  == '__main__':
    main()
