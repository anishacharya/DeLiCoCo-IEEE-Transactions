import argparse
import os
import time

from dec_opt.data_reader import DataReader
from dec_opt.dec_gd import DecGD
from dec_opt.logistic_regression import LogisticRegression

curr_dir = os.path.dirname(__file__)


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--d', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--r', type=str, default=os.path.join(curr_dir, './data/'),
                        help='Pass data root')
    parser.add_argument('--stochastic', type=bool, default=True)

    parser.add_argument('--n_proc', type=int, default=5)
    parser.add_argument('--n_cores', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_type', type=str, default='constant')
    parser.add_argument('--initial_lr', type=float, default=0.1)
    parser.add_argument('--regularizer', type=float, default=0.1)

    args = parser.parse_args()
    return args


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
    dec_gd = DecGD(feature=data_reader.A_train[0:10, :],
                   target=data_reader.y_train[0:10, :],
                   hyper_param=args,
                   model=model)




