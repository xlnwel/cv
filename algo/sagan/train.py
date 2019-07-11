import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from pathlib import Path
import tensorflow as tf

from model import SAGAN
from utility.utils import set_global_seed
from run.grid_search import GridSearch


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    args = parser.parse_args()

    return args

def main(args):
    # you may need this code to train multiple instances
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth=True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # remember to pass sess_config to Model

    model = SAGAN('model', args, log_tensorboard=True, save=True, device='/gpu:0')
    # model.restore('logs/0711-0741/saved_models/baseline')
    model.train()

if __name__ == '__main__':
    cmd_args = parse_cmd_args()

    set_global_seed()
    args_file = 'args.yaml'

    gs = GridSearch(args_file, main, dir_prefix=cmd_args.prefix)

    gs()