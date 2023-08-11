import numpy as np


def clip_std(data: np.ndarray, std_min: float, std_max: float, clip_per_image: bool):
    """
    Assumes data has only 1 channel. If you want to clip other channels then pass data with only that channel
    using slicing and expand_dim

    Parameters
    ----------
    data: shape = (N, x, y, 1)
    std_min
    std_max
    clip_per_image

    Returns
    -------

    """
    if clip_per_image:
        result = np.empty(data.shape, dtype=data.dtype)
        for i, image in enumerate(data):
            mean = np.mean(image)
            std = np.std(image)
            _min = mean + std * std_min
            _max = mean + std * std_max

            _max = np.minimum(np.max(image), _max)
            _min = np.maximum(np.min(image), _min)

            _min = np.maximum(_min, _max / 1e7)  # for very small and negative cases
            result[i, ...] = np.clip(image, _min, _max)
    else:
        mean = np.mean(data)
        std = np.std(data)
        _min = mean + std * std_min
        _max = mean + std * std_max
        result = np.clip(data, _min, _max)
    return result


def clip_known(data: np.ndarray, masks: np.ndarray, clip_per_image: bool):
    """

    Parameters
    ----------
    data
    masks
    clip_per_image

    Returns
    -------

    """
    if masks is None:
        raise ValueError('Cant clip based on known rfi if masks is None')
    if clip_per_image:
        result = np.empty(data.shape, dtype=data.dtype)
        for i, d_m in enumerate(zip(data, masks)):
            image, mask = d_m
            nonrfi_max = np.max(image[~mask])
            try:
                rfi_min = np.min(image[mask])
            except:
                rfi_min = 0

            if nonrfi_max > rfi_min:
                rfi_min = np.maximum(rfi_min, nonrfi_max/1e5)
                result[i] = np.clip(image, rfi_min, nonrfi_max)
            else:
                # Can get perfect answer with thresholding
                result[i] = np.clip(image, nonrfi_max, rfi_min)
            if np.any(np.isnan(result[i])):
                print(f'nan at index {i}')
    else:
        nonrfi_max = np.max(data[~masks])
        rfi_min = np.min(data[masks])
        if nonrfi_max > rfi_min:
            result = np.clip(data, rfi_min, nonrfi_max)
        else:
            result = np.clip(data, rfi_min, nonrfi_max)
    return result


def clip_dyn_std(data: np.ndarray, data_name):
    """
    always per image, function is hardcoded for dataset

    Parameters
    ----------
    data
    data_name

    Returns
    -------

    """
    def f(x, a, b, c):
        return c + 1/(a*x**2 + b*x)

    if data_name == 'LOFAR':
        a_mi, b_mi, c_mi = -0.2, -0.9, -0.03
        a_ma, b_ma, c_ma = 0.02, 0.01, 0.2
    else:
        a_mi, b_mi, c_mi = -0.2, -0.9, -0.03
        a_ma, b_ma, c_ma = 0.02, 0.01, 0.2

    result = np.empty(data.shape, dtype=data.dtype)
    for i, image in enumerate(data):
        mean = np.mean(image)
        std = np.std(image)
        std_over_mean = std/mean

        _min_std = f(std_over_mean, a_mi, b_mi, c_mi)
        _max_std = f(std_over_mean, a_ma, b_ma, c_ma)

        _min = mean + std * _min_std
        _max = mean + std * _max_std

        _max = np.minimum(np.max(image), _max)
        _min = np.maximum(np.min(image), _min)

        _min = np.maximum(_min, _max / 1e6)
        result[i, ...] = np.clip(image, _min, _max)
    return result


def clip_perc(data, perc_min, perc_max, clip_per_image: bool):
    if clip_per_image:
        result = np.empty(data.shape, dtype=data.dtype)
        for i, image in enumerate(data):
            _min = np.percentile(image, perc_min)
            _max = np.percentile(image, perc_max)

            result[i, ...] = np.clip(image, _min, _max)
    else:
        # print(data.shape)
        _min = np.percentile(data, perc_min)
        _max = np.percentile(data, perc_max)
        result = np.clip(data, _min, _max)
    return result


def clip(data: np.ndarray,
         masks: np.ndarray,
         std_min: float,
         std_max: float,
         perc_min: float,
         perc_max: float,
         clip_per_image: bool,
         data_name: str,
         clipper: str,
         **kwargs):
    if clipper is None or clipper == 'None':
        return data
    elif clipper == 'std':
        return clip_std(data, std_min, std_max, clip_per_image)
    elif clipper == 'dyn_std':
        return clip_dyn_std(data, data_name)
    elif clipper == 'known':
        return clip_known(data, masks, clip_per_image)
    elif clipper == 'perc':
        return clip_perc(data, perc_min, perc_max, clip_per_image)


