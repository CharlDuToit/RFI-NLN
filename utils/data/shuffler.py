import numpy as np


def shuffle(seed, *args):
    """
    Shuffles args in place

    Parameters
    ----------
    seed: seed to use. None will generate random seed
    args: arrays of all same length to shuffle

    Returns
    -------
    seed used
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 -1)

    lens = [len(a) for a in args if a is not None]
    for le in lens:
        if le != lens[0]:
            raise ValueError('Passed args do not have the same lenghts')

    for a in args:
        if a is not None:
            np.random.RandomState(seed=seed).shuffle(a)

    return seed


def unshuffle(seed, *args):
    """
    Un-shuffles args in place

    Parameters
    ----------
    seed: seed to use. None will generate random seed
    args: arrays of all same length to shuffle

    Returns
    -------
    unshuffled args
    """
    if seed is None:
        raise ValueError('Requires a seed to unshuffle')

    lens = [len(a) for a in args if a is not None]
    for le in lens:
        if le != lens[0]:
            raise ValueError('Passed args do not have the same lenghts')

    ret_args = []
    for a in args:
        if a is None:
            ret_args.append(None)
        if a is not None:
            indexes = np.arange(len(a))
            np.random.RandomState(seed=seed).shuffle(indexes)
            indexes = np.argsort(indexes)
            ret_args.append(a[indexes])

    if len(ret_args) > 1:
        return tuple(ret_args)
    else:
        return ret_args[0]
