from utils.common import loss_file
import tensorflow as tf
# from tensorflow.keras import losses
from tensorflow import losses


def get_losses(output_path, model_class, anomaly_class, model_name, model_type=None, data='val', **kwargs):
    """ prefix must be in ['val', 'train']. Child classes can have more options"""
    losses_list = []
    file = loss_file(output_path, model_class, anomaly_class, model_name, model_type, data, **kwargs)
    with open(file, 'r') as f:
        losses_list.append([float(line.rstrip()) for line in f])
    return losses_list


def save_loss(loss, output_path, model_class, anomaly_class, model_name, model_type=None, data='val', **kwargs):
    file = loss_file(output_path, model_class, anomaly_class, model_name, model_type, data, **kwargs)
    with open(file, 'a+') as f:
        f.write(f'{loss}\n')


#------------------------- loss functions -0--------------------------


class DiceLoss(tf.keras.losses.Loss):
    # https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o
    def __init__(self, smooth=1e-6, gamma=1):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gamma) + tf.reduce_sum(y_true ** self.gamma) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


def ssim_loss(x, x_hat):
    return 1 / 2 - tf.reduce_mean(tf.image.ssim(x, x_hat, max_val=1.0)) / 2


# ---------------------------------------------------------

def get_loss_func(loss, **kwargs):

    if loss == 'bce':
        loss_func = losses.BinaryCrossentropy()
    elif loss == 'mse':
        loss_func = losses.MeanSquaredError()
    elif loss == 'dice':
        loss_func = DiceLoss()
    elif loss == 'se-ssim' or loss == 'ssim':
        return ssim_loss
    else:
        raise ValueError(f'Loss function: {loss} is not supported')

    return loss_func


