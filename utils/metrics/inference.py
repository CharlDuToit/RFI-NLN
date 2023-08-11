import numpy as np
from utils.data import batch


def infer_fcn(model, data, batch_size=64, n_splits=5, **kwargs):
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

    if input_data_is_batches == False:
        # n_splits = 5
        n_data = len(data)
        interval_length = int((n_data // (n_splits * batch_size)) * batch_size)
        interval_length = np.maximum(interval_length, batch_size)
        interval_length = np.minimum(interval_length, n_splits)
        n_splits = int(np.ceil(n_splits / interval_length))
        intervals = []
        for i in range(n_splits):
            if i == n_splits - 1:
                intervals.append((interval_length * i, n_data))
            else:
                intervals.append((interval_length * i, interval_length * (i + 1)))

        for inter in intervals:
            dataset = batch(batch_size, data[inter[0]:inter[1], ...])
            for btch in dataset:
                fnnsh += len(btch)
                output[strt:fnnsh, ...] = model(btch, training=False).numpy()  # .astype(np.float32)
                strt = fnnsh
    else:
        for btch in dataset:
            fnnsh += len(btch)
            output[strt:fnnsh, ...] = model(btch, training=False).numpy()  # .astype(np.float32)
            strt = fnnsh

    # output[output == np.inf] = np.finfo(output.dtype).max
    return output

