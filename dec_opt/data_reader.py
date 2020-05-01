import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import numpy as np
import time
from sklearn import preprocessing

"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""


class DataReader:
    def __init__(self, root: str, data_set: str, download=True, test_split: float = 0.2):
        self.data_set = data_set
        self.root = root
        self.download = download
        if data_set == 'mnist':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_mnist()
        elif data_set == 'cifar10':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_cifar10()
        elif data_set == 'breast_cancer':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_breast_cancer(test_split=test_split)
        else:
            raise NotImplementedError

    def _get_mnist(self, do_sorting=True):
        mnist_train = datasets.MNIST(root=self.root, download=self.download, train=True)
        mnist_test = datasets.MNIST(root=self.root, download=self.download, train=False)

        x_train = mnist_train.train_data.numpy()/255.0
        x_test = mnist_test.test_data.numpy()/255.0
        y_train = mnist_train.train_labels.numpy()
        y_test = mnist_test.test_labels.numpy()

        # flatten the images
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

        # convert to binary labels
        y_train[y_train < 5] = 0
        y_train[y_train >= 5] = 1
        y_test[y_test < 5] = 0
        y_test[y_test >= 5] = 1

        # # for 4 vs. 9
        # idx_train = np.argwhere(y_train % 5 == 4)
        # idx_test = np.argwhere(y_test % 5 == 4)
        #
        # # for 2 vs. 7
        # # idx_train = np.argwhere(y_train % 5 == 2)
        # # idx_test = np.argwhere(y_test % 5 == 2)
        #
        # x_train = x_train[idx_train[:, 0], :]
        # x_test = x_test[idx_test[:, 0], :]
        #
        # y_train = y_train[idx_train[:, 0]]
        # y_train[y_train == 4] = 0
        # y_train[y_train == 9] = 1
        # y_test = y_test[idx_test[:, 0]]
        # y_test[y_test == 4] = 0
        # y_test[y_test == 9] = 1
        
        if do_sorting:
            y_sorted_ix = np.argsort(y_train)
            x_train = x_train[y_sorted_ix]
            y_train = y_train[y_sorted_ix]

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # Now add the Bias term - Add a Fake dim of all 1s to the parameters
        x_train_aug = np.ones((x_train.shape[0], x_train.shape[1] + 1))
        x_train_aug[:, 0:x_train.shape[1]] = x_train

        x_test_aug = np.ones((x_test.shape[0], x_test.shape[1] + 1))
        x_test_aug[:, 0:x_test.shape[1]] = x_test

        return x_train_aug, y_train, x_test_aug, y_test
    
    def _get_cifar10(self, do_sorting=True):
        cifar10_train = datasets.CIFAR10(root='./data', download=self.download, train=True)
        cifar10_test = datasets.CIFAR10(root='./data', download=self.download, train=False)

        x_train = cifar10_train.data/255.0
        x_test = cifar10_test.data/255.0

        y_train = np.asarray(cifar10_train.targets)
        y_test = np.asarray(cifar10_test.targets)

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # convert to binary labels
        y_train[y_train < 5] = 0
        y_train[y_train >= 5] = 1
        y_test[y_test < 5] = 0
        y_test[y_test >= 5] = 1

        # Convert to gray-scale
        x_train = (x_train[:, :, :, 0] + x_train[:, :, :, 1] + x_train[:, :, :, 2])/3.0
        x_test = (x_test[:, :, :, 0] + x_test[:, :, :, 1] + x_test[:, :, :, 2])/3.0
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

        if do_sorting:
            y_sorted_ix = np.argsort(y_train)
            x_train = x_train[y_sorted_ix]
            y_train = y_train[y_sorted_ix]

        # Now add the Bias term - Add a Fake dim of all 1s to the parameters
        x_train_aug = np.ones((x_train.shape[0], x_train.shape[1] + 1))
        x_train_aug[:, 0:x_train.shape[1]] = x_train

        x_test_aug = np.ones((x_test.shape[0], x_test.shape[1] + 1))
        x_test_aug[:, 0:x_test.shape[1]] = x_test

        return x_train_aug, y_train, x_test_aug, y_test
        
    @staticmethod
    def _get_breast_cancer(test_split, do_sorting=True):
        print('Reading Breast Cancer Data')
        t0 = time.time()
        data_bunch = load_breast_cancer()
        x = data_bunch.data
        x = preprocessing.scale(x)
        y = data_bunch.target
        x, y = shuffle(x, y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

        if do_sorting:
            y_sorted_ix = np.argsort(y_train)
            x_train = x_train[y_sorted_ix]
            y_train = y_train[y_sorted_ix]

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # Now add the Bias term - Add a Fake dim of all 1s to the parameters
        x_train_aug = np.ones((x_train.shape[0], x_train.shape[1] + 1))
        x_train_aug[:, 0:x_train.shape[1]] = x_train

        x_test_aug = np.ones((x_test.shape[0], x_test.shape[1] + 1))
        x_test_aug[:, 0:x_test.shape[1]] = x_test
        print('Time to read Breast Cancer Data = {}s'.format(time.time() - t0))
        return x_train_aug, y_train, x_test_aug, y_test
