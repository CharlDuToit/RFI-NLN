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

