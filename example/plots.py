from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt


def plot_results(repeats, label, plot='train'):
    scores = []
    for result in repeats:
        loss_val = result[0] if plot == 'train' else result[1]
        scores += [loss_val]
    mean = np.mean(scores, axis=0)
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)
    x = np.arange(mean.shape[0])
    plt.plot(x, mean, label=label)
    plt.fill_between(x, UB, LB)


if __name__ == '__main__':
    # Example plot generation :
    # Follow this template to generate your own combination of plots
    # remember the naming convention of the results
    # Load all results of a particular data-set
    data = unpickle_dir(d='./results/breast_cancer')
    print('Loaded Data')

    # Now Lets
    plt.figure()
    fig = plt.gcf()

    # Specify what result runs you want to plot together
    plot_results(repeats=data['ours.9.centralized.2.full'], label='centralized')
    plot_results(repeats=data['ours.9.torus.2.full'], label='torus')
    plot_results(repeats=data['ours.9.ring.2.full'], label='ring')
    plot_results(repeats=data['ours.9.disconnected.2.full'], label='disconnected')

    plt.legend()
    plt.show()
