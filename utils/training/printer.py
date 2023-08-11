def print_epoch(model_class, epoch, time, metrics, metric_labels, **kwargs):
    """
        Messages to print while training

        model_type (str): type of model_type
        epoch (int): The current epoch
        time (int): The time elapsed per Epoch
        metrics (list of scalar): the metrics  of the model
        metric_labels (list of str): AUROC score of the model

    """
    # if kwargs['train_with_test']:
    #     if epoch % 5 == 0:
    #         test_f1(kwargs['model_'], kwargs['test_data'], kwargs['test_masks'])
            #if kwargs['task'] == 'transfer_train':
             #   print(kwargs['model_'].layers[1].kernel[0, 0, 0 , 0])

    if not isinstance(metrics, list):
        metrics = [metrics]
    if not isinstance(metric_labels, list):
        metric_labels = [metric_labels]
    for epochs_metric, label in zip(metrics, metric_labels):
        if epochs_metric is None:  # 0.0 loss is possible for certain loss functions
            metrics.remove(epochs_metric)
            metric_labels.remove(label)
    print('__________________')
    print('{} at epoch {}, time {:.2f} sec'.format(model_class, epoch, time))
    for mtrc, label in zip(metrics, metric_labels):
        print(f'{label}: {mtrc}')


#------------------------------------


import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
def test_f1(model, test_data, test_masks, batch_size=64):
    masks_inferred = infer_fcn(model, test_data, batch_size=batch_size)
    masks_inferred = np.clip(masks_inferred, 0.0, 1.0)
    _f1 = f1_score(test_masks.flatten(), masks_inferred.flatten() > 0.5)
    print('test f1: ', _f1)


def batch(batch_size, *args):
    # lens = [len(a) for a in args]
    # for le in lens:
    #     if le != lens[0]:
    #         raise ValueError('Passed args do not have the same lenghts')

    ret_args = []
    for a in args:
        if a is None:
            ret_args.append(None)
        else:
            ret_args.append(tf.data.Dataset.from_tensor_slices(a).batch(batch_size))
    if len(ret_args) > 1:
        return tuple(ret_args)
    else:
        return ret_args[0]


def infer_fcn(model, data, batch_size=64, **kwargs):
    # data is a numpy ndarray or 'TensorSliceDataset'
    # output is np.ndarray
    # assume self.model is not a list type e.g. (ae, disc) and that the model has only one output
    # i.e. len(model.outputs) == 1

    input_data_is_batches = False
    if str(type(data)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>":
        input_data_is_batches = True
        dataset = data
        n_obs = 0
        for btch in dataset:
            n_obs += len(btch)
    elif isinstance(data, np.ndarray):
        input_data_is_batches = False
        n_obs = len(data)
        # dataset = batch(batch_size, data)
    else:
        raise ValueError('data must be np.ndarray or BatchDataset')

    output = np.empty([n_obs] + model.outputs[0].shape[1:], dtype=np.float32)
    strt, fnnsh = 0, 0

    dataset = batch(batch_size, data)
    for btch in dataset:
        fnnsh += len(btch)
        output[strt:fnnsh, ...] = model(btch, training=False).numpy()  # .astype(np.float32)
        strt = fnnsh

    output[output == np.inf] = np.finfo(output.dtype).max
    return output
