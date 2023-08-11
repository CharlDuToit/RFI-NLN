import numpy as np

from utils.common import loss_file
import tensorflow as tf
# from tensorflow.keras import losses
from tensorflow import losses
from sklearn.utils.class_weight import compute_class_weight
import keras.backend as K


def get_losses(output_path, model_class, anomaly_class, model_name, model_type=None, data='val', **kwargs):
    """ prefix must be in ['val', 'train']. Child classes can have more options"""
    losses_list = []
    file = loss_file(output_path, model_class, anomaly_class, model_name, model_type, data, **kwargs)
    with open(file, 'r') as f:
        losses_list = [float(line.rstrip()) for line in f]
    return losses_list


def get_loss_metrics(output_path, model_class, anomaly_class, model_name, model_type=None, **kwargs):
    """ prefix must be in ['val', 'train']. Child classes can have more options"""
    val_losses = get_losses(output_path, model_class, anomaly_class, model_name, model_type, data='val', **kwargs)
    train_losses = get_losses(output_path, model_class, anomaly_class, model_name, model_type, data='train', **kwargs)
    val_loss = np.min(val_losses)
    train_loss = train_losses[np.argmin(val_losses)]
    train_loss_over_val_loss = train_loss / val_loss
    return dict(train_loss=train_loss, val_loss=val_loss, train_loss_over_val_loss=train_loss_over_val_loss)

def save_loss(loss, output_path, model_class, anomaly_class, model_name, model_type=None, data_subset='val', **kwargs):
    file = loss_file(output_path, model_class, anomaly_class, model_name, model_type, data_subset, **kwargs)
    with open(file, 'a+') as f:
        f.write(f'{loss}\n')


# ------------------------- loss functions -0--------------------------

class DiceLoss(tf.keras.losses.Loss):
    # https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o
    def __init__(self, smooth=1, gamma=1):
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


class LogCoshDiceLoss(tf.keras.losses.Loss):
    # https: // github.com / shruti - jadon / Semantic - Segmentation - Loss - Functions / blob / master / loss_functions.py
    def __init__(self, smooth=1):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

    def generalized_dice_coefficient(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + self.smooth) / (
                K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss


# ---------------------------------------------------------


def ssim_loss(x, x_hat):
    return 1 / 2 - tf.reduce_mean(tf.image.ssim(x, x_hat, max_val=1.0)) / 2


# ---------------------------------------------------------


class BoundingBoxLoss(tf.keras.losses.Loss):

    def __init__(self, num_anchors, **kwargs):
        super(BoundingBoxLoss, self).__init__()
        self.num_anchors = num_anchors

    def call(self, y_true, y_pred):
        # Reshape the inputs to separate anchor boxes
        true_class = tf.reshape(y_true[..., :self.num_anchors], [-1])
        true_bbox = tf.reshape(y_true[..., self.num_anchors:], [-1, 4])
        pred_class = tf.reshape(y_pred[..., :self.num_anchors], [-1])
        pred_bbox = tf.reshape(y_pred[..., self.num_anchors:], [-1, 4])

        # Compute the binary cross-entropy loss for the class predictions
        # class_loss = tf.keras.losses.mean_squared_error(true_class, pred_class)

        pos_class_loss = tf.keras.losses.mean_squared_error(true_class, pred_class * true_class)
        neg_class_loss = tf.keras.losses.mean_squared_error(true_class * (1-true_class), pred_class * (1 - true_class))

        # Mask the bounding box loss for negative examples
        # bbox_mask = tf.tile(tf.expand_dims(true_class, axis=-1), [1, 4])

        # Compute the mean squared error loss for the bounding box coordinates
        bbox_loss = tf.keras.losses.mean_squared_error(true_bbox, pred_bbox)

        # Apply the mask to ignore negative examples
        # bbox_loss = tf.where(bbox_mask, bbox_loss, tf.zeros_like(bbox_loss))
        bbox_loss = bbox_loss * true_class

        # Compute the total loss for each anchor box
        total_loss = 10 * pos_class_loss + neg_class_loss + 5 * tf.reduce_sum(bbox_loss, axis=-1)

        # Compute the mean loss across all anchor boxes
        # num_positives = tf.reduce_sum(true_class)
        # mean_loss = tf.reduce_sum(total_loss) / tf.maximum(num_positives, 1)

        return total_loss



class ChatGPTTorchConversion_CharlOneBox(tf.keras.losses.Loss):
    """
    Calculate the loss for YOLOv1 model
    """

    def __init__(self, S=8, B=1, C=1):
        super(ChatGPTTorchConversion_CharlOneBox, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def call(self, y_true, y_pred):
        #  E.g. C = 20 and B = 2 will have a vector of 30 values in each cell
        # The first 20 values are the classes and the last 10 represent the bounding boxes
        y_pred = tf.reshape(y_pred, [-1, self.S, self.S, self.C + self.B * 5])

        # Let's assume that B=1 and C = 1

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # iou_b1 = intersection_over_union(y_pred[..., 21:25], y_true[..., 21:25])
        # iou_b2 = intersection_over_union(y_pred[..., 26:30], y_true[..., 21:25])
        # ious = tf.concat([tf.expand_dims(iou_b1, axis=0), tf.expand_dims(iou_b2, axis=0)], axis=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # iou_maxes, bestbox = tf.math.reduce_max(ious, axis=0)
        # exists_box = y_true[..., 20:21]  # in paper this is Iobj_i
        exists_box = y_true[..., self.C:self.C+1]  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Let's assume that B=1 and C = 1

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        # box_predictions = exists_box * (
        #     (
        #         bestbox * y_pred[..., 26:30]
        #         + (1 - bestbox) * y_pred[..., 21:25]
        #     )
        # )
        box_predictions = exists_box * y_pred[..., self.C+1:self.C+5]

        # Original implementation assumes the true coordinates are always at 21:25
        # box_targets = exists_box * y_true[..., 21:25]
        box_targets = exists_box * y_true[..., self.C+1:self.C+5]

        # Take sqrt of width, height of boxes to ensure that
        # ChatGPT implementation concats x,y to width,height
        # Original does it separately

        # box_predictions[..., 2:4] = tf.sign(box_predictions[..., 2:4]) * tf.sqrt(
        #     tf.abs(box_predictions[..., 2:4] + 1e-6)
        # )
        # box_targets[..., 2:4] = tf.sqrt(box_targets[..., 2:4])

        #  COMMENT TO DISABLE SQRT OF WIDTH and HEIGHT
        box_predictions = tf.concat(
            [box_predictions[..., :2],
             tf.sign(box_predictions[..., 2:4]) * tf.sqrt(tf.abs(box_predictions[..., 2:4] + 1e-6)),
             ], axis=-1)
        box_targets = tf.concat([box_targets[..., :2], tf.sqrt(box_targets[..., 2:4])], axis=-1)

        box_loss = tf.keras.backend.mean(tf.keras.backend.square(box_predictions - box_targets), axis=[1, 2, 3])

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # Let's assume that B=1 and C = 1

        # pred_box is the confidence score for the bbox with highest IoU
        # pred_box = (
        #     bestbox * y_pred[..., 25:26] + (1 - bestbox) * y_pred[..., 20:21]
        # )
        pred_box = y_pred[..., self.C:self.C+1]

        object_loss = tf.keras.backend.mean(tf.keras.backend.square(exists_box * (pred_box - y_true[..., self.C:self.C+1])), axis=[1, 2, 3])

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # Let's assume that B=1 and C = 1

        no_object_loss = tf.keras.backend.mean(tf.keras.backend.square((1 - exists_box) * y_pred[..., self.C:self.C+1]),
                                               axis=[1, 2, 3])
        # no_object_loss = tf.keras.backend.mean(tf.keras.backend.square((1 - exists_box) * y_pred[..., 20:21]), axis=[1, 2, 3, 4])
        # no_object_loss += tf.keras.backend.mean(tf.keras.backend.square((1 - exists_box) * y_pred[..., 25:26


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = tf.keras.backend.mean(
            tf.keras.backend.square(
                exists_box * y_pred[..., 0:self.C] - exists_box * y_true[..., 0:self.C]
            ),
            axis=[1, 2, 3]
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss



class ChatGPTTorchConversion_CharlOneBox_Noclass(tf.keras.losses.Loss):
    """
    Calculate the loss for YOLOv1 model
    """

    def __init__(self, S=8):
        super(ChatGPTTorchConversion_CharlOneBox_Noclass, self).__init__()
        self.S = S
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, self.S, self.S, 5])

        # Let's assume that B=1 and C = 1

        exists_box = y_true[..., 0:1]  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = exists_box * y_pred[..., 1:5]

        box_targets = exists_box * y_true[..., 1:5]

        box_predictions = tf.concat(
            [box_predictions[..., :2],
             tf.sign(box_predictions[..., 2:4]) * tf.sqrt(tf.abs(box_predictions[..., 2:4] + 1e-6)),
             ], axis=-1)
        box_targets = tf.concat([box_targets[..., :2], tf.sqrt(box_targets[..., 2:4])], axis=-1)

        box_loss = tf.keras.backend.mean(tf.keras.backend.square(box_predictions - box_targets), axis=[1, 2, 3])

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        pred_box = y_pred[..., 0:1]

        object_loss = tf.keras.backend.mean(tf.keras.backend.square(exists_box * (pred_box - y_true[..., 0:1])), axis=[1, 2, 3])

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = tf.keras.backend.mean(tf.keras.backend.square((1 - exists_box) * y_pred[..., 0:1]),
                                               axis=[1, 2, 3])

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + 10 * object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
        )

        return loss
# ---------------------------------------------------------

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, pos_weight=1, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        neg_term = (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
        pos_term = self.pos_weight * y_true * tf.math.log(y_pred + 1e-6)
        result = -tf.reduce_mean(neg_term + pos_term)
        return result


# ---------------------------------------------------------

class BalancedBinaryCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, neg_weight=1, pos_weight=1, y_true=None, **kwargs):
        super(BalancedBinaryCrossEntropy, self).__init__()
        if y_true is not None:
            neg_weight, pos_weight = compute_class_weight('balanced', classes=[0, 1], y=y_true.flatten())
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        neg_term = self.neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
        pos_term = self.pos_weight * y_true * tf.math.log(y_pred + 1e-6)
        result = -tf.reduce_mean(neg_term + pos_term)
        return result


# ---------------------------------------------------------

def get_loss_func(loss, **kwargs):
    if loss == 'bbce':
        loss_func = BalancedBinaryCrossEntropy(**kwargs)
    elif loss == 'wbce':
        loss_func = WeightedBinaryCrossEntropy(**kwargs)
    elif loss == 'bce':
        loss_func = losses.BinaryCrossentropy()
    elif loss == 'mse':
        loss_func = losses.MeanSquaredError()
    elif loss == 'dice':
        loss_func = DiceLoss()
    elif loss == 'logcoshdice':
        loss_func = LogCoshDiceLoss()
    elif loss == 'se-ssim' or loss == 'ssim':
        loss_func = ssim_loss
    elif loss == 'bb':
        # loss_func = BoundingBoxLoss(**kwargs)
        # loss_func = ChatGPTTorchConversion_CharlOneBox(S=8, B=1, C=1)
        loss_func = ChatGPTTorchConversion_CharlOneBox_Noclass(S=8)
    else:
        raise ValueError(f'Loss function: {loss} is not supported')

    return loss_func


def main2():
    bce = losses.BinaryCrossentropy()
    wbce = WeightedBinaryCrossEntropy()

    y_true = [1, 1, 0, 0, 1]
    y_pred = [0.9, 0.1, 0.9, 0.1, 1]

    y_true = [1, 1]
    y_pred = [1.0, 1.0]

    # y_true = (np.random.random((16,16,1)) > 0.5).astype(int)
    # y_pred = np.random.random((16,16,1))

    print(bce(y_true, y_pred))
    print(wbce(y_true, y_pred))


def main():
    n_anchors = 3
    loss_func = BoundingBoxLoss(n_anchors)
    # loss_func = DiceLoss()

    y_true = np.zeros((2, 2, 5 * n_anchors))

    y_true[0, 0, 0] = 1
    y_true[0, 0, n_anchors:n_anchors + 4] = [0.25, 0.5, 1, 1]

    y_true[0, 0, 1] = 1
    y_true[0, 0, n_anchors + 4:n_anchors + 8] = [0.25, 0.5, 1, 1]

    y_true[0, 0, 2] = 0
    y_true[0, 0, n_anchors + 4:n_anchors + 8] = [0.25, 0.5, 1, 1]

    y_pred = np.zeros((2, 2, 5 * n_anchors))

    y_pred[0, 0, 0] = 0.9
    y_pred[0, 0, n_anchors:n_anchors + 4] = [1, 1, 1, 1]

    y_pred[0, 0, 1] = 0.5
    y_pred[0, 0, n_anchors + 4:n_anchors + 8] = [0.25, 0.5, 1, 1]

    y_pred[0, 0, 2] = 0.8
    y_pred[0, 0, n_anchors + 8:n_anchors + 12] = [1, 1, 1, 1]

    loss = loss_func(y_true, y_pred)
    print(loss)


def main3():
    loss_func = ChatGPTTorchConversion_CharlOneBox(S=2, B=1, C=1)

    y_true = np.zeros((1, 2, 2, 1 + 5*1))
    y_true[0, 0, 0, :] = [1, 1, 0.5, 0.5, 0.5, 0.5]
    y_true[0, 0, 1, :] = [1, 1, 0.5, 0.5, 0.5, 0.5]
    y_true[0, 1, 0, :] = [0, 0, 0, 0, 0, 0]
    y_true[0, 1, 1, :] = [0, 0, 0, 0, 0, 0]

    y_pred = np.zeros((1, 2, 2, 1 + 5*1))
    y_pred[0, 0, 0, :] = [1, 1, 0.5, 0.5, 0.5, 0.5]
    y_pred[0, 0, 1, :] = [0.9, 0.9, 0.4, 0.4, 0.4, 0.4]
    y_pred[0, 1, 0, :] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    y_pred[0, 1, 1, :] = [0, 0, 0, 0, 0, 0]

    # box_loss = ( 0 + 2 * 0.1**2 + 2*(sqrt(0.5) - sqrt(0.4))**2  + 0 + 0) / (1*2*2*4) = 1.946e-3
    # object loss = 0 + 0.1**2 + 0 + 0 / 4 = 2.5e-3
    # no-object loss = 0 + 0 + 0.2**2 + 0  /4 = 0.01
    # class loss = 0 + 0.1**2 + 0 + 0 / 4 = 2.5e-3

    loss = loss_func(y_true, y_pred)
    print(loss)


if __name__ == '__main__':
    main3()

# y_true = tf.constant([0, 0, 0, 0, 0, 0, 1])
# y_pred = tf.constant([0.1, 0.1, 0.05, 0.01, 0.02, 0.1, 0.9])

# loss_func = BalancedBinaryCrossEntropy(1, 1)
# print(loss_func(y_true, y_pred))

# loss_func = BalancedBinaryCrossEntropy(y_true=y_true.numpy())
# print(compute_class_weight('balanced', classes=[0, 1], y=y_true.numpy()))
# print(loss_func(y_true, y_pred))
