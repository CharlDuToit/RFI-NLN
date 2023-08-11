import hera_sim
from hera_sim import Simulator
import numpy as np
import datetime
import pickle

from utils import save_image_batches, save_image_masks_batches, get_multichannel_patches

from load_simulator import load_sim
from hera_images import sim_images, uvd_images, sim_images_masks
from stats import DataStats, DataMasksStats


def simulate(sim: Simulator):

    sim.refresh()
    if np.random.uniform(0) < 0.5:
        hera_sim.defaults.set('h1c')
    else:
        hera_sim.defaults.set('h2c')

    nonrfi_components = []

    # ########################################  FOREGROUNDS  ################################################

    sim.add(
        'diffuse_foreground',
        component_name='diffuse_foreground',
        seed='redundant',
    )
    nonrfi_components.append('diffuse_foreground')

    sim.add(
        'pntsrc_foreground',
        component_name='pntsrc_foreground',
        seed='redundant',
        nsrcs=1000,  # 1000
        Smin=0.3,  # 0.3,
        Smax=1000,  # 1000,
        beta=-1.5,  # -1.5,
        spectral_index_mean=-1,  # -1,
        spectral_index_std=0.5,  # 0.5
        reference_freq=np.random.uniform(0.35, 0.55)  # 0.15
    )
    nonrfi_components.append('pntsrc_foreground')

    # ########################################  RFI  ################################################

    f_min = sim.freqs.min()
    f_max = sim.freqs.max()
    ch_width = (f_max - f_min) / sim.Nfreqs

    rfi_components = []

    # ------------------------------ RFI DTV ------------------------------

    n_dtv = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_dtv):
        component_name = f'rfi_dtv_{i}'
        sim.add(
            "rfi_dtv",
            dtv_band=(f_min, f_max),
            dtv_channel_width=np.random.uniform(5*ch_width, 64*ch_width),
            dtv_chance=np.abs(np.random.normal(0.0001, 0.002)),
            dtv_strength=np.random.uniform(200, 20000),
            dtv_std=np.random.uniform(100, 30000),
            component_name=component_name,
            seed="redundant",
        )
        rfi_components.append(component_name)
    # ------------------------------ RFI IMPULSE ------------------------------

    n_impulse = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_impulse):
        component_name = f'rfi_impulse_{i}'
        sim.add(
            "rfi_impulse",
            impulse_chance=np.abs(np.random.normal(0.0001, 0.002)),
            impulse_strength=np.random.uniform(100, 40000),
            component_name=component_name,
            seed="redundant",
        )
        rfi_components.append(component_name)

    # ------------------------------ RFI SCATTER ------------------------------
    # pixel blips
    n_scatter = int(np.random.uniform(0, 1) * 6) + 0
    for i in range(n_scatter):
        component_name = f'rfi_scatter_{i}'
        sim.add(
            "rfi_scatter",
            component_name=component_name,
            seed="redundant",
            scatter_chance=np.abs(np.random.normal(0.0001, 0.002)),
            scatter_strength=np.random.uniform(200, 20000),
            scatter_std=np.random.uniform(100, 30000),
        )
        rfi_components.append(component_name)

    # ------------------------------ RFI STATIONS ------------------------------

    # + 2 * 0.09 / 512
    n_stations = int(np.random.uniform(0, 1) * 6) + 0
    stations = []
    for i in range(n_stations):
        st = hera_sim.rfi.RfiStation(f0=np.random.uniform(f_min, f_max),
                                     duty_cycle=np.random.uniform(0.1, 0.9),
                                     std=np.random.uniform(100, 30000),
                                     strength=np.random.uniform(200, 20000),
                                     timescale=np.random.uniform(30, 300))
        stations.append(st)
    if stations:
        sim.add('stations', stations=stations, seed='redundant', component_name='rfi_stations')
        rfi_components.append('rfi_stations')

    # ########################################  ADDITIONAL EFFECTS  ################################################

    sim.add("thermal_noise",
            seed="initial",
            Trx=60,
            component_name="thermal_noise")
    nonrfi_components.append('thermal_noise')

    sim.add("whitenoisecrosstalk", amplitude=20, seed="redundant", component_name='whitenoisecrosstalk')
    nonrfi_components.append('whitenoisecrosstalk')

    sim.add("bandpass", gain_spread=0.1, dly_rng=(-20, 20))

    return sim, nonrfi_components, rfi_components


def stats_and_images(sim, components):
    if isinstance(components, str):
        components = [components]

    for comp in components:
        images = sim_images(sim, comp)
        if comp in ('rfi_scatter', 'rfi_impulse'):
            stats = DataStats(images[np.abs(images) > 0], dir_path='./test', name=comp)
        else:
            stats = DataStats(images, dir_path='./test', name=comp)
        stats.main()
        save_image_batches(f'./test/{comp}', np.log(images))

    images = uvd_images(sim)
    stats = DataStats(images, dir_path='./test', name='all')
    stats.main()
    save_image_batches(f'./test/all', np.log(images))


def stats_and_image_masks(sim, rfi_components):

    images, masks = sim_images_masks(sim, rfi_components)
    stats = DataMasksStats(images, masks, dir_path='./test_masks2', name='all')
    stats.main()
    save_image_masks_batches(f'./test_masks2/all', np.log(images), masks)


def simulate_and_save(sim, sample_ratio, n_sims, incl_phase, test_split=0.2 ):

    all_antpairpols = sim.get_antpairpols()
    n_baselines = int(len(all_antpairpols) * sample_ratio)
    if incl_phase:
        images_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 2)
    else:
        images_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 1)
    masks_shape = (n_baselines*n_sims, sim.Ntimes, sim.Nfreqs, 1)
    images = np.empty(images_shape, dtype='float32')
    masks = np.empty(masks_shape, dtype=bool)
    for i in range(n_sims):
        lo = i * n_baselines
        hi = (i + 1) * n_baselines

        # Randomly sample baselines
        indexes = np.random.choice(np.arange(len(all_antpairpols)), n_baselines, replace=False)
        antpairpols = [all_antpairpols[i] for i in indexes]

        # Simulate and extract images and masks
        sim, nonrfi_components, rfi_components = simulate(sim)
        images[lo:hi, ...], masks[lo:hi, ...] = sim_images_masks(sim, rfi_components, antpairpols, incl_phase)
        print(f'Sim: {i}')

    # Save stats and images
    dir_path = './hera_images_and_stats'
    stats = DataMasksStats(images, masks, dir_path=dir_path, name='all_components')
    stats.main()
    save_image_masks_batches(f'{dir_path}/image_masks', np.log(images[..., 0:1]), masks)
    if incl_phase:
        save_image_masks_batches(f'{dir_path}/phase_masks', images[..., 1:2], masks)

    # Save patches
    # image_patches = get_multichannel_patches(images, 64,64,64,64)
    # mask_patches = get_multichannel_patches(masks, 64,64,64,64)
    # stats = DataMasksStats(image_patches, mask_patches.astype(bool), dir_path='./test12_many_sims', name='all_components_64')
    # stats.main()
    # save_image_masks_batches(f'./test12_many_sims/image_masks_64', np.log(image_patches), mask_patches)

    # Split data
    print(images.shape, masks.shape)
    n_test = int(images.shape[0] * test_split)
    np.random.RandomState(seed=42).shuffle(images)
    np.random.RandomState(seed=42).shuffle(masks)
    test_data = images[:n_test]
    train_data = images[n_test:]
    test_masks = masks[:n_test]
    train_masks = masks[n_test:]

    # Save to pickle
    f_name = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/HERA_{}_all.pkl'.format(
        datetime.datetime.now().strftime("%d-%m-%Y"))
    pickle.dump([train_data, train_masks, test_data, test_masks], open(f_name, 'wb'), protocol=4)
    print('{} saved!'.format(f_name))


def main():
    sim_params = dict(
        Nfreqs=512,
        start_freq=100e6,
        bandwidth=90e6,
        Ntimes=512,
        start_time=2457458.1738949567,
        integration_time=3.512,
        array_layout=hera_sim.antpos.hex_array(2, split_core=False, outriggers=0),
    )
    # obsparam_file = './config/obsparam_hex_nosky.yaml'
    # components = ('diffuse_foreground', 'pntsrc_foreground', 'rfi_scatter', 'rfi_impulse', 'rfi_dtv', 'rfi_stations')
    # components = ( 'pntsrc_foreground', 'rfi_scatter', 'rfi_impulse', 'rfi_dtv')
    # sim = load_sim(obsparam_file)

    sim = load_sim(sim_params)
    simulate_and_save(sim, 0.5, 100, incl_phase=True, test_split=0.2)

    #sim, nonrfi_components, rfi_components = simulate(sim)
    #stats_and_image_masks(sim, rfi_components)


def get_hera_charl_data(data_path):
    (train_data, train_masks,
    test_data, test_masks) = np.load(f'{data_path}/HERA_28-03-2023_all.pkl', allow_pickle=True)

    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max

    return train_data.astype('float32'), train_masks, test_data.astype('float32'), test_masks


def get_hera_charl_stats():
    train_data, train_masks, test_data, test_masks = get_hera_charl_data('/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/')
    stats = DataMasksStats(train_data[..., 0:1], train_masks, dir_path='./hera_charl_stats_july', name='all_components')
    stats.main()


def load_and_generate_images():
    train_data, train_masks, test_data, test_masks = get_hera_charl_data('/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/')
    dir_path = './test1_reload'
    save_image_masks_batches(f'.{dir_path}/image_masks', np.log(train_data[..., 0:1]), train_masks)
    save_image_masks_batches(f'.{dir_path}/phase_masks', train_data[..., 1:2], train_masks)

if __name__ == '__main__':
    #load_and_generate_images()
    # main()
    get_hera_charl_stats()
