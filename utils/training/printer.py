def print_epoch(model_type, epoch, time, metrics, metric_labels):
    """
        Messages to print while training

        model_type (str): type of model_type
        epoch (int): The current epoch
        time (int): The time elapsed per Epoch
        metrics (list of scalar): the metrics  of the model
        metric_labels (list of str): AUROC score of the model

    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    if not isinstance(metric_labels, list):
        metric_labels = [metric_labels]
    for epochs_metric, label in zip(metrics, metric_labels):
        if epochs_metric is None:  # 0.0 loss is possible for certain loss functions
            metrics.remove(epochs_metric)
            metric_labels.remove(label)
    print('__________________')
    print('{} at epoch {}, time {:.2f} sec'.format(model_type, epoch, time))
    for mtrc, label in zip(metrics, metric_labels):
        print(f'{label}: {mtrc}')

