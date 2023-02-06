import tensorflow as tf


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
