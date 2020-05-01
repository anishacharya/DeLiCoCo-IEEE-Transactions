import argparse
import os
from dec_opt.utils import pickle_it
from dec_opt.experiment import run_exp
import numpy as np
import multiprocessing as mp
from functools import partial
from dec_opt.logistic_regression import LogisticRegression

"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""

curr_dir = os.path.dirname(__file__)


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--d', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--r', type=str, default=os.path.join(curr_dir, './data/'),
                        help='Pass data root')
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default='ours')

    parser.add_argument('--n_cores', type=int, default=9)

    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--Q', type=int, default=2)
    parser.add_argument('--consensus_lr', type=float, default=0.3)

    parser.add_argument('--quantization_function', type=str, default='full')
    parser.add_argument('--num_bits', type=int, default=2)
    parser.add_argument('--fraction_coordinates', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr_type', type=str, default='constant')
    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--epoch_decay_lr', type=float, default=0.9)
    parser.add_argument('--regularizer', type=float, default=0)

    parser.add_argument('--estimate', type=str, default='final')
    parser.add_argument('--n_proc', type=int, default=3)
    parser.add_argument('--n_repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args


# TODO:
#  For Each Data-set in [MNIST, CIFAR10]
#  Vary Compression : *Pruning  *Quantization: Vary n_cores
#  Vary Topology : Vary n_cores
#  Vary Q values : Vary n_cores


if __name__ == '__main__':
    arg = _parse_args()
    directory = "results/" + arg.d + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    result_file = arg.algorithm + "." + str(arg.n_cores) + '.' + \
        arg.topology + '.' + str(arg.Q) + '.' + arg.quantization_function

    model = LogisticRegression(params=arg)
    args = []
    results = []
    for random_seed in np.arange(1, arg.n_repeat + 1):
        arg.seed = random_seed
        results.append(run_exp(model=model, args=arg))
        # args += [arg]
    # with mp.Pool(arg.n_proc) as pool:
    #     results = pool.map(partial(run_exp, model=model), args)

    # Dumps the results in appropriate files
    pickle_it(arg, 'parameters.' + result_file, directory)
    pickle_it(results, result_file, directory)
    print('results saved in "{}"'.format(directory))
