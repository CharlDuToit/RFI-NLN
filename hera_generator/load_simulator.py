from pyuvsim.simsetup import initialize_uvdata_from_params, _complete_uvdata
from hera_sim import Simulator
from pathlib import Path
from pyuvdata import UVData
from hera_sim.visibilities import VisibilitySimulation, ModelData

import numpy as np


def load_sim(sim_config: dict | str | Path | UVData | VisibilitySimulation | ModelData, remove_vis: bool = False) -> Simulator:
    """Creates a new simulator with empty data_array and returns it"""
    if isinstance(sim_config, dict):
        sim = Simulator(**sim_config)
        remove_vis = False
    elif isinstance(sim_config, (str, Path)):
        uvdata, beam_list, beam_ids = initialize_uvdata_from_params(sim_config)
        _complete_uvdata(uvdata, inplace=True)
        sim = Simulator(data=uvdata)
        remove_vis = False
    elif isinstance(sim_config, UVData):
        sim = Simulator(data=sim_config.copy())
    elif isinstance(sim_config, VisibilitySimulation):
        sim = Simulator(data=sim_config.data_model.uvdata.copy())
    elif isinstance(sim_config, ModelData):
        sim = Simulator(data=sim_config.uvdata.copy())
    else:
        raise TypeError('sim_config is wrong type')

    if remove_vis:
        shape = sim.data.data_array.shape
        sim.data.data_array = np.zeros(shape, dtype='complex128')
        sim.data.flag_array = np.zeros(shape, dtype=bool).astype(bool)

    return sim
