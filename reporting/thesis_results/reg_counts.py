from args_constructor import preprocessing_query
from utils import ResultsCollection


# run with min_f1 = 0 and 0.5
def all_reg_counts():
    for dataset in ('LOFAR', 'HERA_CHARL'):
        for loss in ('bce', 'mse', 'dice', 'logcoshdice'):
            kwargs = reg_counts_args(dataset, loss, min_f1=0.5)
            rc = ResultsCollection(**kwargs)
            rc.perform_task()


def reg_counts_args(dataset='LOFAR',
                    loss='bce',
                    filters=16,
                    lr=0.0001,
                    use_hyp_data=True,
                    min_f1=0.0,
                    limit=None,
                    train_with_test=False):

    epochs = 50 if loss == 'bce' else 100
    lim = '"None"' if limit is None else limit
    kwargs = dict(
        incl_models=[],
        excl_models=[],
        groupby=['model_class', 'loss', 'kernel_regularizer', 'dropout'],
        datasets=[dataset],
        query_strings=preprocessing_query() + [f'use_hyp_data == {use_hyp_data}', f'filters == {filters}', f'lr=={lr}',
                                               f'loss=="{loss}"', f'epochs == {epochs}', f'train_f1 > {min_f1}',
                                               f'train_with_test == {train_with_test}', f'limit == {lim}'],
        task='None',
        table_fields=[],
        std=True,
        params=False,
        x_axis=None,
        y_axis=None,
        label_fields=['model_class', 'loss'],
        label_format=None,
        output_path='/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/',
        save_name=f'{dataset}_{loss}',
        save_path=f'/home/ee487519/PycharmProjects/RFI-NLN/reporting/thesis_results/{dataset}_counts/',
        plot_options=dict()
    )
    return kwargs


all_reg_counts()
