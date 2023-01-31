from utils.metrics import get_nln_metrics, save_metrics_csv, evaluate_performance  # , save_results_csv

def end_routine(train_data,
                test_data,
                test_labels,
                test_masks,
                test_masks_orig,
                model,
                model_type,
                args):
    for neighbour in args.neighbours:
        (ae_ao_auroc, ae_true_auroc,
         ae_ao_auprc, ae_true_auprc,
         ae_ao_f1, ae_true_f1,
         nln_ao_auroc, nln_true_auroc,
         nln_ao_auprc, nln_true_auprc,
         nln_ao_f1, nln_true_f1,
         dists_ao_auroc, dists_true_auroc,
         dists_ao_auprc, dists_true_auprc,
         dists_ao_f1, dists_true_f1,
         combined_ao_aurocs, combined_true_aurocs,
         combined_ao_auprcs, combined_true_auprcs,
         combined_ao_f1s, combined_true_f1s,
         flops, tot_time) = evaluate_performance(model,
                                                 # rather return kwargs? can pre-populate with invalid values it and make a inheritable evaluate_performance
                                                 train_data,
                                                 test_data,
                                                 test_labels,
                                                 test_masks,
                                                 test_masks_orig,
                                                 model_type,
                                                 neighbour,
                                                 args)

        for i, alpha in enumerate(args.alphas):
            save_metrics_csv(model_type,
                             train_data,
                             test_masks,
                             test_masks_orig,
                             alpha,
                             neighbour,
                             args,

                             # just pass **kwargs,
                             # pre-populate kwargs in save_metrics

                             flops=flops,  # per patch
                             tot_time=tot_time,  # per patch,
                             trainable_params=0,  #num_trainable_params(model), # model might be a tuple
                             nontrainable_params=0, #num_non_trainable_params(model),

                             ae_ao_auroc=ae_ao_auroc,
                             ae_true_auroc=ae_true_auroc,
                             ae_ao_auprc=ae_ao_auprc,
                             ae_true_auprc=ae_true_auprc,
                             ae_ao_f1=ae_ao_f1,
                             ae_true_f1=ae_true_f1,

                             nln_ao_auroc=nln_ao_auroc,
                             nln_true_auroc=nln_true_auroc,
                             nln_ao_auprc=nln_ao_auprc,
                             nln_true_auprc=nln_true_auprc,
                             nln_ao_f1=nln_ao_f1,
                             nln_true_f1=nln_true_f1,

                             dists_ao_auroc=dists_ao_auroc,
                             dists_true_auroc=dists_true_auroc,
                             dists_ao_auprc=dists_ao_auprc,
                             dists_true_auprc=dists_true_auprc,
                             dists_ao_f1=dists_ao_f1,
                             dists_true_f1=dists_true_f1,

                             combined_ao_auroc=combined_ao_aurocs[i],
                             combined_true_auroc=combined_true_aurocs[i],
                             combined_ao_auprc=combined_ao_auprcs[i],
                             combined_true_auprc=combined_true_auprcs[i],
                             combined_ao_f1=combined_ao_f1s[i],
                             combined_true_f1=combined_true_f1s[i])

            if model_type == 'UNET' or model_type == 'DKNN' or model_type == 'RNET' or model_type == 'RFI_NET' or 'CNN_RFI_SUN':
                return