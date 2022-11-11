import tempfile
import time
import pickle
import datetime
from tqdm import tqdm
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from astropy import units

import hera_sim
from hera_sim import Simulator, DATA_PATH, utils
from uvtools.plot import labeled_waterfall

station_models = ['HERA_H1C_RFI_STATIONS.npy',
                  'HERA_H2C_RFI_STATIONS.npy']

start_time = 2457458.1738949567  # JD
integration_time = 3.512  # 4.68
Ntimes = int(30 * units.min.to("s") / integration_time)  # 10 minute observation

# Define the frequency parameters.
Nfreqs = 2 ** 9
bandwidth = 88e6  # 8e7# 100 MHz
start_freq = 107e6  # start at 100MHz

array_layout = hera_sim.antpos.hex_array(2, split_core=False, outriggers=0)

sim_params = dict(
    Nfreqs=Nfreqs,
    start_freq=start_freq,
    bandwidth=bandwidth,
    Ntimes=Ntimes,
    start_time=start_time,
    integration_time=integration_time,
    array_layout=array_layout,
)
sim = Simulator(**sim_params)


def waterfall(sim, antpairpol=(0, 1, "xx"), figsize=(6, 3.5), dpi=200, title=None):
    """Convenient plotting function to show amp/phase."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        dpi=dpi,
    )
    fig, ax1 = labeled_waterfall(
        sim.data,
        antpairpol=antpairpol,
        mode="log",
        ax=ax1,
        set_title=title,
    )
    ax1.set_xlabel(None)
    ax1.set_xticklabels(['' for tick in ax1.get_xticks()])
    fig, ax2 = labeled_waterfall(
        sim.data,
        antpairpol=antpairpol,
        mode="phs",
        ax=ax2,
        set_title=False,
    )
    #plt.savefig('/tmp/temp/{}_{}_{}'.format(*antpairpol), dpi=300)
    plt.savefig('./temp/{}_{}_{}'.format(*antpairpol), dpi=300)


def simulate(id_rfis):
    """
        Adds parameters to the hera sim simulator class and generates the data

        Parameters
        ----------
        id_rfis (tuple/list)  the in distribution (ID) RFI

        Returns
        -------
        Simulator 

    """
    sim.refresh()
    hera_sim.defaults.set("h1c")

    #########################################################################################

    sim.add(
        "diffuse_foreground",
        component_name="diffuse_foreground",
        seed="once",
    )

    #########################################################################################

    # if np.random.random(1)[0] >0.5:
    if 'rfi_stations' in id_rfis:
        # /home/ee487519/hera_sim-main/hera_sim/data/
        sim.add(
            "rfi_stations",
            stations="/home/ee487519/hera_sim-main/hera_sim/data/{}".format(station_models[np.random.randint(0, 2)]),
            component_name="rfi_stations",
            seed="once",
        )

    if 'rfi_dtv' in id_rfis:
        sim.add(
            "rfi_dtv",
            dtv_band=(0.174, 0.214),
            dtv_channel_width=0.08,
            dtv_chance=0.025,
            dtv_strength=20000,
            dtv_std=200.0,
            component_name="rfi_dtv",
            seed="once",
        )

    if 'rfi_impulse' in id_rfis:
        sim.add(
            "rfi_impulse",
            impulse_chance=0.005,  # A lot of sources
            impulse_strength=20000.00,
            component_name="rfi_impulse",
            seed="once",
        )

    if 'rfi_scatter' in id_rfis:
        sim.add(
            "rfi_scatter",
            scatter_chance=0.0008,  # A lot of sources
            scatter_strength=20000.00,
            scatter_std=200.0,
            component_name="rfi_scatter",
            seed="once",
        )

    #########################################################################################
    sim.add("thermal_noise",
            seed="initial",
            Trx=0,
            component_name="noisy_ant")

    sim.add("whitenoisecrosstalk", amplitude=1.0, seed="once")
    sim.add("bandpass", gain_spread=0.1, dly_rng=(-20, 20))

    #########################################################################################
    return sim


def extract_data(sim, subset):
    """
        Extracts the visibilities at each randomly sampled baseline

        Parameters
        ----------
        sim (Simulator) the previously instantiated simulator with all effects and features 
        baselines (int) number of baselines to be sampled
        subset (tuple) the list of rfi waveforms to be extracted from the simulator 

        Returns
        -------
        np.array, np.array, np.array

    """
    amps, phases, masks = [], [], []

    for app in sim.get_antpairpols():
        data_temp = np.absolute(sim.get_data(app)).astype('float16')
        phase_temp = np.angle(sim.get_data(app)).astype('float16')
        amps.append(data_temp)
        phases.append(phase_temp)

        mask_temp = np.zeros(data_temp.shape, dtype='bool')
        for rfi in subset:
            try:
                mask_temp = np.logical_or(mask_temp,
                                          np.absolute(sim.get(rfi, app)) > 0)
            except Exception as e:
                continue
        masks.append(mask_temp)

    amps = np.expand_dims(np.array(amps), axis=-1)
    phases = np.expand_dims(np.array(phases), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    return amps, phases, masks


def plot(sim):
    """
        Save waterfall plots of the magnitude and phase of the simulated data 

        Parameters
        ----------
        sim (Simulator) the previously instantiated simulator with all effects and features 

        Returns
        -------
        None

    """
    for i in sim.get_antpairpols():
        fig1 = waterfall(sim, antpairpol=i, title=' '.join(str(e) for e in i))
        plt.close('all')


def main():
    """
        Runs the simulator with different subsets of RFI and saves them as pickles

        Parameters
        ----------
        None

        Returns
        -------
        None
    """
    n = 80 # 20
    rfis = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']

    test_split = 0.2

    amps = np.empty([n * 28, 2 ** 9, 2 ** 9, 1], dtype='float16') #560, 512, 512, 1
    phases = np.empty([n * 28, 2 ** 9, 2 ** 9, 1], dtype='float16')
    masks = np.empty([n * 28, 2 ** 9, 2 ** 9, 1], dtype='bool')
    st, en = 0, 28
    for i in tqdm(range(n)):
        _sim = simulate(rfis)
        _amps, _phases, _masks = extract_data(_sim, rfis)
        amps[st:en, ...], phases[st:en, ...], masks[st:en] = _amps, _phases, _masks
        st = en
        en += 28

    data = np.concatenate([amps, phases], axis=-1)
    n_test = int(data.shape[0] * test_split)
    test_data = data[:n_test]
    train_data = data[n_test:]
    test_masks = masks[:n_test]
    train_masks = masks[n_test:]

    f_name = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/HERA_{}_all.pkl'.format(
        datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([train_data, train_masks, test_data, test_masks], open(f_name, 'wb'), protocol=4)
    print('{} saved!'.format(f_name))


if __name__ == '__main__':
    main()
