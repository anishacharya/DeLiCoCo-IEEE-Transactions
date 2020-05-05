from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import itertools

# Get Optimal Values
baselines = {'mnist': 0.35247397975085026,
             'mnist_partial': 0.07954815167630427}  # 4000: 0.08184391665417677 5000: 0.07954815167630427}


def plot_results(repeats, label, plot='train', optima=0.0, line_style=None):
    scores = []
    for result in repeats:
        loss_val = result[0] if plot == 'train' else result[1]
        # Get Sub Optimal Loss
        # Get Optimal Values

        loss_val = loss_val - optima
        scores += [loss_val]

    scores = np.array(scores)
    # scores[scores <= 1e-2] = 1e-2
    # scores[np.isinf(scores)] = 1e20
    # scores[np.isnan(scores)] = 1e20

    mean = np.mean(scores, axis=0)
    x = np.arange(mean.shape[0])
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)

    plt.plot(x, mean, label=label, linewidth=5, linestyle=line_style)
    plt.fill_between(x, LB, UB, alpha=0.2, linewidth=1)


def plot_loop(data, algorithm: List[str], n_cores: List[int],
              topology: List[str], Q: List[int], consensus_lr: List[float],
              quantization_func: List[str],
              label: List[str], optima: float):
    # Load Hyper Parameters
    all_hyper_param = list(itertools.product(algorithm, n_cores, topology,
                                             Q, consensus_lr, quantization_func))

    # Generate Plots
    i = 0
    for hyper_param in all_hyper_param:
        result_file = hyper_param[0] + '.' + str(hyper_param[1]) + '.' + hyper_param[2] + \
                      '.' + str(hyper_param[3]) + '.' + str(hyper_param[4]) + '.' + hyper_param[5]
        plot_results(repeats=data[result_file], label=label[i], optima=optima)
        i += 1


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()
    data_set = 'mnist_partial'
    optimal_baseline = baselines[data_set]

    # plot baseline
    baselines = unpickle_dir(d='./results/baselines')
    repeats_baseline = baselines[data_set + '_gd']

    # plot no communication
    repeats_disconnected = baselines[data_set + '_dis']

    plt.xlabel('Number of gradient steps')
    plt.ylabel('training suboptimality')
    plt.grid(axis='both')

    """ 
    Understand Effects of Varying Q
    """
    # Specify what result runs you want to plot together
    # this is what you need to modify
    labels = []
    clr_var = [0.01, 0.1, 0.3, 1.0]
    for clr in clr_var:
        labels.append('consensus=' + str(clr))

    labels = []
    q_var = [1, 5, 10]
    for q in q_var:
        labels.append('Q=' + str(q))

    # Now run to get plots
    plot_results(repeats=repeats_disconnected, label='Disconnected',
                 optima=optimal_baseline)

    results_dir = '/paper/Q_clr/'  # For Q vs consensus plots
    data = unpickle_dir(d='./results/' + data_set + results_dir)
    plot_loop(data=data, n_cores=[9], algorithm=['ours'], topology=['ring'],
              Q=q_var, consensus_lr=[0.9], label=labels, quantization_func=['top'],
              optima=optimal_baseline)

    plot_results(repeats=repeats_baseline, label='Centralized',
                 optima=optimal_baseline, line_style='dashed')
    plt.legend()
    plt.yscale("log")
    plt.ylim(bottom=1e-3, top=2)
    plt.xlim(left=0, right=5000)
    plt.title('Consensus learning rate = 0.9')
    plt.show()


