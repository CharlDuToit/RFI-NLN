import numpy as np
from utils.data import batch


def infer(model, data, batch_size=64, **kwargs):
    # data is a numpy ndarray or 'TensorSliceDataset'
    # output is np.ndarray
    # assume self.model is not a list type e.g. (ae, disc) and that the model has only one output
    # i.e. len(model.outputs) == 1

    if str(type(data)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>":
        dataset = data
        n_obs = 0
        for btch in dataset:
            n_obs += len(btch)
    elif isinstance(data, np.ndarray):
        n_obs = len(data)
        dataset = batch(batch_size, data)
    else:
        raise ValueError('data must be np.ndarray or BatchDataset')

    output = np.empty([n_obs] + model.outputs[0].shape[1:], dtype=np.float32)
    strt, fnnsh = 0, 0
    for btch in dataset:
        fnnsh += len(btch)
        output[strt:fnnsh, ...] = model(btch, training=False).numpy()  # .astype(np.float32)
        strt = fnnsh

    output[output == np.inf] = np.finfo(output.dtype).max
    return output
