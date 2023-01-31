from utils import args
from data_collection import get_data_collection_from_args
#from utils.flagging import flag_data
import time


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def main():
    """
        Reads data and cmd arguments and trains models
    """
    # print(args.args)
    # return
    args.args.data = 'LOFAR'
    args.args.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    start = time.time()
    print('===============================================')
    print("__________________________________ \nFetching and preprocessing data: {}".format(args.args.data))
    data_collection = get_data_collection_from_args(args.args)
    data_collection.load_raw_data()

    aof_masks = data_collection.flag_data(data_collection.test_data, args.args.data, '10')

    print('Total time : {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')



if __name__ == '__main__':
    main()
