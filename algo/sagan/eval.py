from math import ceil
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
from pathlib import Path
from skimage.data import imread
import tensorflow as tf

from model import SAGAN
from utility.yaml_op import load_args
from utility.utils import set_global_seed


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=None)
    parser.add_argument('--iterations', '-i', type=int, default=1)
    parser.add_argument('--checkpoint', '-c', type=str)
    args = parser.parse_args()

    return args

def main():
    set_global_seed()
    args_file = 'args.yaml'
    args = load_args(args_file)
    cmd_args = parse_cmd_args()

    if cmd_args.batch_size:
        args['eval_batch_size'] = cmd_args

    model = SAGAN('model', args, device='/cpu:0')
    model.restore(cmd_args.checkpoint)
    model.evaluate(n_iterations=cmd_args.iterations)

if __name__ == '__main__':
    main()
