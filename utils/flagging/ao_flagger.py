# def flag_data(data, data_name, rfi_threshold, **kwargs):
#     pass
#
import aoflagger as aof
import numpy as np
from tqdm import tqdm

def flag_data(data, data_name, rfi_threshold, **kwargs):
    """
        Applies AOFlagger to simulated HERA visibilities


        Parameters
        ----------
        data (np.array) array of hera simluated data
        args (Namespace) utils.cmd_arguments

        Returns
        -------
        np.array, np.array
    """
    # pass

    if data is None:
        return None
    mask = np.empty(data[..., 0].shape, dtype=np.bool)

    aoflagger = aof.AOFlagger()
    if data_name in ('HERA', 'HERA_CHARL', 'HERA_CHARL_AOF'):
        strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/hera_{}.lua'.format(rfi_threshold))
    elif data_name == 'LOFAR':
        strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/lofar-default-{}.lua'.format(rfi_threshold))

    #strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/bighorns-{}.lua'.format(rfi_threshold))

    # LOAD data into AOFlagger structure
    for indx in tqdm(range(len(data))):
        _data = aoflagger.make_image_set(data.shape[1], data.shape[2], 1)
        _data.set_image_buffer(0, data[indx,...,0]) # Amplitude values

        flags = strategy.run(_data)
        flag_mask = flags.get_buffer()
        mask[indx,...] = flag_mask

    return np.expand_dims(mask.astype('bool'), axis=-1)

