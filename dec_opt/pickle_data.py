import pickle
import argparse
from sklearn.datasets import load_svmlight_file
import time


def _parse_args():
    parser = argparse.ArgumentParser(description='pickle_data.py')
    parser.add_argument('--i', type=str, default='data/rcv1_test.binary.bz2',
                        help='Pass data-set path')
    parser.add_argument('--o', type=str, default='data/rcv1_test.pickle',
                        help='Pass output file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    input_file = args.i
    output_file = args.o
    t0 = time.time()
    A, y = load_svmlight_file(f=input_file)
    print("Loading SVMLight File took {} sec.".format(time.time()-t0))
    print("The file is huge hence caching into a pickle file for future use")
    A = A.toarray()
    with open(output_file, 'wb') as pickle_file:
        pickle.dump((A, y), pickle_file, protocol=4)
