from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import itertools
import pickle


def plot_results(repeats, label, plot='train', optimal=0.0):
    scores = []
    for result in repeats:
        loss_val = result[0] if plot == 'train' else result[1]
        # Get Sub Optimal Loss
        loss_val = loss_val - optimal
        scores += [np.log(loss_val)]

    # scores = np.array(scores)
    # scores[scores == 0] = 1e20
    # scores[np.isinf(scores)] = 1e20
    # scores[np.isnan(scores)] = 1e20

    mean = np.mean(scores, axis=0)
    x = np.arange(mean.shape[0])
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)

    plt.plot(x, mean, label=label)
    plt.fill_between(x, LB, UB, alpha=0.2)


def plot_loop(data_set: str, algorithm: List[str], n_cores: List[int],
              topology: List[str], Q: List[int], consensus_lr: List[float],
              quantization_func: List[str],
              label: List[str]):

    # Get Optimal Values
    baselines = {'mnist': 0.35247397975085026}
    optimal = baselines[data_set]

    # Load Hyper Parameters
    all_hyper_param = list(itertools.product(algorithm, n_cores, topology,
                                             Q, consensus_lr, quantization_func))
    # Load Data
    data = unpickle_dir(d='./results/' + data_set)
    # Generate Plots
    i = 0
    for hyper_param in all_hyper_param:
        result_file = hyper_param[0] + '.' + str(hyper_param[1]) + '.' + hyper_param[2] + \
                      '.' + str(hyper_param[3]) + '.' + str(hyper_param[4]) + '.' + hyper_param[5]
        plot_results(repeats=data[result_file], label=label[i], optimal=optimal)
        i += 1


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()

    # Specify what result runs you want to plot together
    # this is what you need to modify
    labels = []

    """ 
    Understand Effects of Varying Q
    """
    Q_var = [0, 1, 2, 3]
    for q in Q_var:
        labels.append('Q=' + str(q) + ', CLR=0.3')

    # Now run to get plots
    plot_loop(data_set='mnist', n_cores=[9], algorithm=['ours'], topology=['ring'],
              Q=Q_var, consensus_lr=[0.3], label=labels, quantization_func=['top'])

    plt.xlabel('epochs')
    plt.ylabel('Sub-Optimality (log)')
    plt.grid(axis='both')
    plt.legend()
    plt.show()


