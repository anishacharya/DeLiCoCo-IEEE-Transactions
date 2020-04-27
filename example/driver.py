import argparse
import os
import time
from dec_opt.data_reader import DataReader
from dec_opt.dec_gd import DecGD
from dec_opt.logistic_regression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

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

    parser.add_argument('--n_cores', type=int, default=100)
    parser.add_argument('--n_proc', type=int, default=10)

    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--Q', type=int, default=3)
    parser.add_argument('--consensus_lr', type=float, default=0.2)

    parser.add_argument('--quantization_function', type=str, default='top')
    parser.add_argument('--num_levels', type=int, default=10)
    parser.add_argument('--coordinates_to_keep', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_type', type=str, default='constant')
    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--epoch_decay_lr', type=float, default=0.9)
    parser.add_argument('--regularizer', type=float, default=0.1)

    parser.add_argument('--estimate', type=str, default='final')
    args = parser.parse_args()
    return args

# TODO:
#  For Each Data-set in [MNIST, CIFAR10]
#  Vary Compression : *Pruning  *Quantization
#  Vary Topology
#  Vary Q values
#  Vary n_cores


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    data_set = args.d
    root = args.r

    """ 
    Get Data specifically,  AX = Y , 
    A [samples x Feature] 
    X parameters to estimate 
    Y Target
    """
    print("loading data-set")
    t0 = time.time()
    data_reader = DataReader(root=root, data_set=data_set, download=True)
    print('Time taken to load Data {} sec'.format(time.time() - t0))

    """ Run Experiment """
    model = LogisticRegression(params=args)
    dec_gd = DecGD(feature=data_reader.A_train,
                   target=data_reader.y_train,
                   hyper_param=args,
                   model=model)
    print("Now we can plot losses")
    # fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    # line, = ax.plot(dec_gd.epoch_losses, color='blue', lw=2)
    # ax.set_yscale('log')
    plt.plot(np.log(dec_gd.epoch_losses))
    plt.show()
    # pylab.show()





