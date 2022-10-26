import aoflagger as aof
import numpy as np 
from tqdm import tqdm

def flag_data(data, data_name, rfi_threshold):
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
    mask = np.empty(data[...,0].shape, dtype=np.bool)

    aoflagger = aof.AOFlagger()
    if data_name == 'HERA':
        strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/hera_{}.lua'.format(rfi_threshold))
    elif data_name == 'LOFAR':
        strategy = aoflagger.load_strategy_file('utils/flagging/stratergies/lofar-default-{}.lua'.format(rfi_threshold))

    # LOAD data into AOFlagger structure
    for indx in tqdm(range(len(data))):
        _data = aoflagger.make_image_set(data.shape[1], data.shape[2], 1)
        _data.set_image_buffer(0, data[indx,...,0]) # Real values 

        flags = strategy.run(_data)
        flag_mask = flags.get_buffer()
        mask[indx,...] = flag_mask
    
    return mask.astype('bool')

