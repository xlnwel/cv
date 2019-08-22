import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from pathlib import Path
import tensorflow as tf

from model import SAGAN
from utility.utils import set_global_seed
from utility.yaml_op import load_args
from run.grid_search import GridSearch


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--saved_model', '-m',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args

def main(args, saved_model=None):
    # you may need this code to train multiple instances on a single GPU
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth=True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # remember to pass sess_config to Model

    model = SAGAN('model', args, log_tensorboard=True, save=not saved_model, device='/gpu:0')
    if saved_model:
        model.restore(saved_model)
    model.train()

if __name__ == '__main__':
    cmd_args = parse_cmd_args()

    set_global_seed()
    args_file = 'args.yaml'

    if cmd_args.saved_model:
        args = load_args(args_file)
        main(args, cmd_args.saved_model)

    gs = GridSearch(args_file, main, dir_prefix=cmd_args.prefix)

    gs()
