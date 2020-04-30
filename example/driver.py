import argparse
import os
from dec_opt.utils import pickle_it
from dec_opt.experiment import run_exp

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
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default='ours')

    parser.add_argument('--n_cores', type=int, default=9)

    parser.add_argument('--topology', type=str, default='ring')
    parser.add_argument('--Q', type=int, default=2)
    parser.add_argument('--consensus_lr', type=float, default=0.3)

    parser.add_argument('--quantization_function', type=str, default='full')
    parser.add_argument('--num_levels', type=int, default=10)
    parser.add_argument('--fraction_coordinates', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_type', type=str, default='constant')
    parser.add_argument('--initial_lr', type=float, default=0.01)
    parser.add_argument('--epoch_decay_lr', type=float, default=0.9)
    parser.add_argument('--regularizer', type=float, default=0)

    parser.add_argument('--estimate', type=str, default='final')
    args = parser.parse_args()
    return args

# TODO:
#  For Each Data-set in [MNIST, CIFAR10]
#  Vary Compression : *Pruning  *Quantization
#  Vary Topology : Vary n_cores
#  Vary Q values : Vary n_cores


if __name__ == '__main__':
    # define experiment directory
    # Load default arguments
    arg = _parse_args()
    # Define Directory and result file name
    directory = "results/" + arg.d + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # DO Experiments here :
    # change arguments as needed for the experiment
    arg.topology = 'torus'
    res = run_exp(args=arg)  # res: {[train_loss], [test_loss]}
    # Dumps the results in appropriate files
    result_file = arg.algorithm + "." + str(arg.n_cores) + arg.topology + str(arg.Q) + arg.quantization_function
    pickle_it(arg, 'parameters.'+result_file, directory)
    pickle_it(res, result_file, directory)
    print('results saved in "{}"'.format(directory))
    # Loss_Plots = []
    # for quantization_function in ['full', 'rand', 'top']:
    #     args.quantization_function = quantization_function
    #     print(args)
    #     train_loss, test_loss = run_exp(args=args)
    #     Loss_Plots.append(train_loss)
    # print("Now we can plot losses")
    # for train_losses in Loss_Plots:
    #     plt.plot(train_losses)
    # # plt.plot(dec_gd.test_losses)
    # plt.show()





