from utils import main_args
from data_collection import load_data_collection
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
    main_args.args.data_name = 'LOFAR'
    main_args.args.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    start = time.time()
    print('===============================================')
    print("__________________________________ \nFetching and preprocessing data: {}".format(main_args.args.data_name))
    data_collection = load_data_collection(main_args.args)
    data_collection.load_raw_data()

    aof_masks = data_collection.flag_data(data_collection.test_data, main_args.args.data_name, '10')

    print('Total time : {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')



if __name__ == '__main__':
    main()
