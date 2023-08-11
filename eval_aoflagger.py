from utils import main_args, load_raw_data, to_dict, flag_data, aof_recall_precision_f1_fpr, save_csv, Namespace, save_image_masks_batches
import time
import datetime, pickle


def main():

    ns = Namespace(data_name='HERA_CHARL_AOF',
                   data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
                   rfi_threshold=None,
                   seed='AOF_HERA_CHARL_20_july',
                   output_path='./outputs'
                   )
    kwargs = to_dict(ns)

    # '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    # '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'

    train_data, train_masks, test_data, test_masks = load_raw_data(**kwargs)



    subset = 'train'
    if subset == 'test':
        data = test_data
        masks = test_masks
    else:
        data = train_data
        masks = train_masks

    for rfi_threshold in (10,):
        print('rfi_threshold : ', rfi_threshold)

        kwargs['rfi_threshold'] = rfi_threshold

        start = time.time()
        masks_aof = flag_data(data, **kwargs)
        infer_time = time.time() - start

        image_time = infer_time / data.shape[0]
        recall, precision, f1, fpr = aof_recall_precision_f1_fpr(masks, masks_aof)
        print('    f1: ', f1)

        results = dict(subset=subset,
                       image_time=image_time,
                       f1=f1,
                       precision=precision,
                       recall=recall,
                       fpr=fpr)
        csv_kwargs = {**kwargs, **results}
        save_csv(results_dict=csv_kwargs, **kwargs)

        # save_image_masks_batches(dir_path='./outputs/AOF/'+kwargs['data_name']+f'/{rfi_threshold}',
        #                          data=test_data,
        #                          masks=masks_aof,
        #                          batch_size=28)

    # Save to pickle
    # f_name = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/HERA_AOF_{}_all.pkl'.format(
    #     datetime.datetime.now().strftime("%d-%m-%Y"))
    # pickle.dump([data, masks_aof, test_data, test_masks], open(f_name, 'wb'), protocol=4)
    # print('{} saved!'.format(f_name))


if __name__ == '__main__':
    main()
