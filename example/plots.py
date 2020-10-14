from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import itertools

# Get Optimal Values
baselines = {'mnist': 0.35247397975085026,
             'mnist_partial': 0.07954815167630427}


def plot_results(repeats, label, plot='train',
                 optima=0.0, line_style=None, line_width=5, marker=None, scale=1):
    scores = []
    for result in repeats:
        loss_val = result[0] if plot == 'train' else result[1]
        loss_val = loss_val - optima
        scores += [loss_val]

    scores = np.array(scores)
    mean = np.mean(scores, axis=0)
    x = np.arange(mean.shape[0]) * scale
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)

    plt.plot(x, mean, label=label, linewidth=line_width, linestyle=line_style, marker=marker)
    plt.fill_between(x, LB, UB, alpha=0.2, linewidth=1)


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()
    data_set = 'syn2/'
    # optimal_baseline = baselines[data_set]

    # baseline - Gradient Descent Vanilla Centralized
    # baselines = unpickle_dir(d='./results/baselines')
    # repeats_baseline = baselines[data_set + '_gd']
    # no communication
    # repeats_disconnected = baselines[data_set + '_dis']

    # Specify what result runs you want to plot together
    # this is what you need to modify

    # MNIST
    results_dir = '.'
    data = unpickle_dir(d='./results/' + data_set + results_dir)
    plt.title('Non-Convex Objective(SYN-2): DeLiCoCo vs Choco-GD', fontsize=14)

    # plt.subplot(1, 2, 2)
    plot_results(repeats=data['q_1.b_8'],
                 label='8 bit,Q=1 (Choco-GD)', line_width=3, scale=8)
    plot_results(repeats=data['q_2.b_4'],
                 label='4 bit,Q=2 (DeLiCoCo)', line_width=3, scale=8)
    plot_results(repeats=data['q_4.b_2'],
                 label='2 bit,Q=4 (DeLiCoCo)', line_width=3, scale=8)
    plot_results(repeats=data['q_3.b_2'],
                 label='2 bit,Q=3 (DeLiCoCo)', line_width=3, scale=6)
    # plot_results(repeats=data['q_1.b_8'],
    #              label='8 bit,Q=1, decayed lr', line_width=3, scale=8, line_style='--')

    plt.legend(fontsize=11)
    plt.yscale("log")
    plt.ylim(bottom=5e-3, top=1)
    plt.xlim(left=0, right=4000)
    plt.xlabel('Total Bits Communicated', fontsize=14)
    plt.ylabel('$f - f^*$', fontsize=14)
    plt.grid(axis='both')
    plt.tick_params(labelsize=12)

    plt.show()
