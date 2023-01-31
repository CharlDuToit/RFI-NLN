import argparse
import os

"""
    Arguments to create results files
"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Extract information from results saved in csv')

parser.add_argument('-incl_models', metavar='-im', type=str, nargs='+', default=[],
                    help='Which models to include')
parser.add_argument('-excl_models', metavar='-em', type=str, nargs='+', default=[],
                    help='All models will be included except those specified by excl_models')
parser.add_argument('-groupby', metavar='-gb', type=str, nargs='+',
                    default=['model', 'data', 'loss', 'patch_x', 'batch_size', 'filters', 'lr', 'dropout', 'kernel_regularizer'],
                    help='Fields to group by')
parser.add_argument('-datasets', metavar='-d', type=str, nargs='+', default=[],
                    help='Which datasets to extract from csv. Default is all datasets')
parser.add_argument('-query_strings', metavar='-qs', type=str, nargs='+', default=['filters > 0', 'height > 0'],
                    help='Queries which are anded and then applied to df')
parser.add_argument('-task', metavar='-tsk', type=str, default='scatter',
                    choices={'train_loss' 'val_loss', 'train_val_loss', 'scatter', 'bubble', 'text', 'table', 'line'},
                    help='Task to perform')
parser.add_argument('-table_fields', metavar='-tf', type=str, nargs='+',
                    default=['test_auroc', 'test_auprc', 'test_f1', 'val_auroc', 'val_auprc', 'val_f1'],
                    help='Columns to save in LaTex table')
# parser.add_argument('-line_groups', metavar='-lg', type=str, nargs='+', default=['model'],
#                     help='Groups to add to line plot. Note that these may not be in excl_groupby')
parser.add_argument('-std', metavar='-std', type=str2bool, default=False,
                    help='Include std in loss plot?')
parser.add_argument('-params', metavar='-pa', type=str2bool, default=False,
                    help='Size of points in scatter determined by trainable parameters?')
parser.add_argument('-x_axis', metavar='-sx', type=str, default='flops_image',
                    help='Value to plot on x-axis of scatter or line plot')
parser.add_argument('-y_axis', metavar='-sy', type=str, default='test_f1',
                    #choices={'full', 'empty', 'short'},
                    help='Value to plot on x-axis of scatter or line plot or bubble plot'),
parser.add_argument('-label_fields', metavar='-l', type=str, nargs='+', default=['model'],
                    help='Which fields to add to the labels of scatter plot and loss plot')
parser.add_argument('-label_format', metavar='-lf', type=str, default='short',
                    choices={'full', 'empty', 'short'},
                    help='How to format the labels')
parser.add_argument('-output_path', metavar='-op', type=str, default='./outputs',
                    help='Path where models and csvs are saved')
parser.add_argument('-save_name', metavar='-sf', type=str, default='task_result',
                    help='Name of file to write results. Default uses generic name. Do not add extention e.g .png')
parser.add_argument('-save_path', metavar='-sp', type=str, default='None',
                    help='Path to write results')

results_args = parser.parse_args()

if len(results_args.incl_models) > 0 and len(results_args.excl_models) > 0:
    raise ValueError('Included and excluded models may not both be specified')

if results_args.save_path == 'None':
    results_args.save_path = results_args.output_path

if results_args.x_axis == results_args.y_axis:
    raise ValueError('x-axis and y-axis must be different')

for label_field in results_args.label_fields:
    if label_field not in results_args.groupby:
        raise ValueError(f'Cant label by field {label_field} if data is not grouped by it')

if results_args.task == 'scatter':
    AGG_FIELDS = (
        'test_f1', 'test_auroc', 'test_auprc', 'val_f1', 'val_auroc', 'val_auprc', 'time_image', 'flops_image',
        'train_loss', 'val_loss', 'last_epoch', 'trainable_params'
    )
    if results_args.x_axis not in AGG_FIELDS and results_args.x_axis not in results_args.groupby:
        raise ValueError(f'Cant extract x-axis field {results_args.x_axis} if data is not grouped or aggregated by it')

    if results_args.y_axis in results_args.groupby:
        raise ValueError(f'Cant aggregate y-axis {results_args.y_axis} if data is grouped it')






