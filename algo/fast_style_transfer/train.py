import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
import tensorflow as tf

from model import StyleTransferModel
from utility.utils import set_global_seed
from utility.grid_search import GridSearch


def main(args):
    # you may need this code to train multiple instances on a single GPU
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth=True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # remember to pass sess_config to Model

    style_image_path, _ = os.path.splitext(args['style_image_path'])
    _, style_image = os.path.split(style_image_path)

    args['model_name'] = style_image

    model = StyleTransferModel('model', args, log_tensorboard=False, save=False, device='/gpu:0')
    model.train()

if __name__ == '__main__':
    set_global_seed()
    args_file = 'args.yaml'

    gs = GridSearch(args_file, main)

    gs()
