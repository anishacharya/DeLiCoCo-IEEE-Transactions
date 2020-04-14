import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--i', type=str, default='data/epsilon_normalized.pickle',
                        help='Pass input file path')

    args = parser.parse_args()
    return args


def get_data():
    raise NotImplementedError


def run_experiment():
    raise NotImplementedError


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    print("load data-set")
    data_file = args.i
    with open(data_file, 'rb') as f:
        A, y = pickle.load(f)
        print("A.shape= {}".format(A.shape))
