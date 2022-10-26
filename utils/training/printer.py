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
    #print ('__________________')
    #print('Epoch {} at {} sec \n{} losses: {} \nAUC = {}'.format(epoch,
    #                                                             time,
    #                                                             model_type,
    #                                                             metrics,
    #                                                             metric_labels))
    print('__________________')
    print(f'{model_type} at epoch {epoch}, time {time} sec')
    for mtrc, label in zip(metrics, metric_labels):
        print(f'{label}: {mtrc}')

