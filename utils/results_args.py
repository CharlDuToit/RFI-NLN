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
                    help='Which models to exclude')
parser.add_argument('-excl_groupby', metavar='-eg', type=str, nargs='+', default=[],
                    help='Field to not group by. See ... for default field for groupby')
parser.add_argument('-data', metavar='-d', type=str, nargs='+', default=[],
                    help='Which datasets to extract from csv. Default is all datasets')
parser.add_argument('-query_strings', metavar='-qs', type=str, nargs='+', default=['filters > 0', 'height > 0'],
                    help='Queries which are anded and then applied to df')
parser.add_argument('-task', metavar='-tsk', type=str, default='loss',
                    choices={'loss', 'scatter', 'text', 'table'},
                    help='Task to perform')
parser.add_argument('-scatter_x', metavar='-sx', type=str, default='f1',
                    choices={'f1', 'auroc', 'auprc', 'time', 'flops'},
                    help='Value to plot on x-axis of scatter plot')
parser.add_argument('-scatter_y', metavar='-sy', type=str, default='flops',
                    choices={'f1', 'auroc', 'auprc', 'time', 'flops'},
                    help='Value to plot on x-axis of scatter plot')
parser.add_argument('-labels', metavar='-l', type=str, nargs='+', default=['model', 'filters'],
                    help='Which fields to add to the labels')
parser.add_argument('-tag', metavar='-tg', type=str, default='short',
                    choices={'full', 'empty', 'short'},
                    help='How to format the labels')
parser.add_argument('-output_path', metavar='-op', type=str, default='./outputs',
                    help='Path where models and csvs are saved')
parser.add_argument('-save_file', metavar='-sf', type=str, default='None',
                    help='Name of file to write results. Default uses generic name. Do not add extention e.. .png')
parser.add_argument('-save_path', metavar='-sp', type=str, default='None',
                    help='Path to write results')

metrics_args = parser.parse_args()

if len(metrics_args.incl_models) > 0 and len(metrics_args.excl_models) > 0:
    raise ValueError('Included and excluded models may not both be specified')

if metrics_args.out_file == 'None':
    metrics_args.out_file = None

if metrics_args.save_path == 'None':
    metrics_args.save_path = metrics_args.output_path


