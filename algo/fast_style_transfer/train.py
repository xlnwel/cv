import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
import tensorflow as tf

from model import RTSTSRModel
from utility.utils import set_global_seed
from run.grid_search import GridSearch


def main(args):
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth=True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45

    model = RTSTSRModel('model', args, log_tensorboard=True, save=True, device='/gpu:0')
    model.train()

if __name__ == '__main__':
    set_global_seed()
    args_file = 'args.yaml'

    gs = GridSearch(args_file, main)

    gs(style_weights=[100], content_weight=[2], tv_weight=[200])
    # gs(style_transfer=dict(padding=['same', 'reflect']))
