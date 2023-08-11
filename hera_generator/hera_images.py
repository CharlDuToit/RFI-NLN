from pyuvsim.simsetup import initialize_uvdata_from_params, _complete_uvdata
from hera_sim import Simulator
from pathlib import Path
from pyuvdata import UVData
from hera_sim.visibilities import VisibilitySimulation, ModelData


import numpy as np

# (lsts, fqs)
def uvd_images(data_class: Simulator | UVData | VisibilitySimulation | ModelData, pols=('xx', 'yx', 'xy', 'yy'), autos=True, dtype='float32') -> np.ndarray:

    get_data_instance = None
    if isinstance(data_class, Simulator):
        get_data_instance = data_class
    elif isinstance(data_class, UVData):
        get_data_instance = data_class
    elif isinstance(data_class, VisibilitySimulation):
        get_data_instance = data_class.data_model.uvdata
    elif isinstance(data_class, ModelData):
        get_data_instance = data_class.uvdata
    else:
        raise TypeError('data_class is wrong type')

    images = []
    antpairpols = get_data_instance.get_antpairpols()
    antpairpols = [pair for pair in antpairpols if pair[2] in pols]
    if not autos:
        antpairpols = [pair for pair in antpairpols if pair[0] != pair[1]]
    for pair in antpairpols:
        images.append(get_data_instance.get_data(pair))

    return np.expand_dims(np.array(np.abs(images), dtype=dtype), axis=-1)


def sim_images(sim: Simulator, components, antpairpols=None, dtype='float32') -> np.ndarray:

    if antpairpols is None:
        antpairpols = sim.get_antpairpols()

    if components is None:
        raise TypeError('names may not be None')
    if isinstance(components, str):
        components = [components]
    if not isinstance(components, (list, tuple)):
        raise TypeError('names must be in a list or tuple')

    images = []
    for pair in antpairpols:
        im = None
        for name in components:
            if im is None:
                im = sim.get(name, pair)
            else:
                im += sim.get(name, pair)
        images.append(im)

    return np.expand_dims(np.array(np.abs(images), dtype=dtype), axis=-1)


def sim_masks(sim: Simulator, rfi_components, antpairpols=None) -> np.ndarray:

    if antpairpols is None:
        antpairpols = sim.get_antpairpols()

    if rfi_components is None:
        raise TypeError('names may not be None')
    if isinstance(rfi_components, str):
        rfi_components = [rfi_components]
    if not isinstance(rfi_components, (list, tuple)):
        raise TypeError('names must be in a list or tuple')

    masks = []
    shape = sim.get_data(antpairpols[0]).shape
    for pair in antpairpols:
        mask = np.zeros(shape, dtype=bool)
        for name in rfi_components:
            im = np.abs(sim.get(name, pair))
            mask = np.logical_or(mask, im > 0)
        masks.append(mask)

    return np.expand_dims(np.array(masks), axis=-1)


def sim_images_masks(sim: Simulator, rfi_components, antpairpols=None, incl_phase=True, dtype='float32') -> (np.ndarray, np.ndarray):

    if antpairpols is None:
        antpairpols = sim.get_antpairpols()

    if rfi_components is None:
        raise TypeError('names may not be None')
    if isinstance(rfi_components, str):
        rfi_components = [rfi_components]
    if not isinstance(rfi_components, (list, tuple)):
        raise TypeError('names must be in a list or tuple')

    masks = []
    images = []
    shape = sim.get_data(antpairpols[0]).shape  # 2 dims: Nlsts, Nfreqs
    for pair in antpairpols:
        mask = np.zeros(shape, dtype=bool)
        for name in rfi_components:
            im = np.abs(sim.get(name, pair))
            mask = np.logical_or(mask, im > 0)
        masks.append(mask)
        images.append(sim.get_data(pair))

    if incl_phase:
        return np.concatenate(
            [np.expand_dims(np.abs(images), axis=-1),
             np.expand_dims(np.angle(images), axis=-1)],
            axis=3, dtype=dtype
        ), np.expand_dims(np.array(masks), axis=-1)
    else:
        return np.expand_dims(np.array(np.abs(images), dtype=dtype), axis=-1), np.expand_dims(np.array(masks), axis=-1)

