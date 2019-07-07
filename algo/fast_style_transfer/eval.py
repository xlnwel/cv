import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from pathlib import Path
import tensorflow as tf

from model import StyleTransferModel
from utility.yaml_op import load_args
from utility.utils import set_global_seed
from run.grid_search import GridSearch

def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str)
    parser.add_argument('--checkpoint', '-c', type=str)
    args = parser.parse_args()

    return args

def main():
    set_global_seed()
    args_file = 'args.yaml'
    args = load_args(args_file)
    cmd_args = parse_cmd_args()

    args['eval_image_path'] = cmd_args.image

    model = StyleTransferModel('model', args, log_tensorboard=False, device='/gpu:0')
    model.restore(cmd_args.checkpoint)
    model.eval()

if __name__ == '__main__':
    main()