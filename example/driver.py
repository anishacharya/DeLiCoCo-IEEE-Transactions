import argparse
import os
from dec_opt.utils import pickle_it
from dec_opt.experiment import run_exp
import numpy as np
from dec_opt.logistic_regression import LogisticRegression

"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""

curr_dir = os.path.dirname(__file__)


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--d', type=str, default='breast_cancer',
                        help='Pass data-set')
    parser.add_argument('--r', type=str, default=os.path.join(curr_dir, './data/'),
                        help='Pass data root')
    parser.add_argument('--o', type=str, default=None, help='Pass op location')

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


if __name__ == '__main__':
    arg = _parse_args()
    print(arg)
    if not arg.o:
        directory = "results/" + arg.d + "/"
    else:
        directory = "results/" + arg.o + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    result_file = 'a_' + arg.algorithm + '.n_' + str(arg.n_cores) + '.t_' + \
        arg.topology + '.q_' + str(arg.Q) + '.lr_' + str(arg.consensus_lr) +\
        '.c_' + str(arg.quantization_function) + '.f_' + str(arg.fraction_coordinates) +\
        '.p_' + str(arg.dropout_p) + '.b_' + str(arg.num_bits)

    model = LogisticRegression(params=arg)
    args = []
    results = []
    for random_seed in np.arange(1, arg.n_repeat + 1):
        arg.seed = random_seed
        results.append(run_exp(model=model, args=arg))

    # Dumps the results in appropriate files
    pickle_it(arg, 'parameters.' + result_file, directory)
    pickle_it(results, result_file, directory)
    print('results saved in "{}"'.format(directory))
