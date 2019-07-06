import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
import tensorflow as tf

from model import RTSTSRModel
from run.grid_search import GridSearch


def main(args):
    # data_root_orig = tf.keras.utils.get_file('flower_photos',
    #                                      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    #                                      untar=True)
    # data_root = Path(data_root_orig)
    # args['train_dir'] = data_root
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    model = RTSTSRModel('model', args, sess_config=sess_config, log_tensorboard=True, log_params=True, save=True)
    model.train()

if __name__ == '__main__':
    args_file = 'args.yaml'

    gs = GridSearch(args_file, main)

    gs()
    # gs(style_transfer=dict(padding=['same', 'reflect']))
    