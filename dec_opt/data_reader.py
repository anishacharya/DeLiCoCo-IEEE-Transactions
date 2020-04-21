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
    def __init__(self, root: str, data_set: str = 'mnist', download=True, test_split: float = 0.2):
        self.data_set = data_set
        self.root = root
        self.download = download
        if data_set == 'mnist':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_mnist()
        elif data_set == 'breast_cancer':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_breast_cancer(test_split=test_split)
        else:
            raise NotImplementedError

    def _get_mnist(self):
        mnist_train = datasets.MNIST(root=self.root, download=self.download, train=True)
        mnist_test = datasets.MNIST(root=self.root, download=self.download, train=False)
        x_train = mnist_train.train_data.numpy().reshape(60000, 784).astype(np.float32)
        y_train = mnist_train.train_labels.numpy().reshape(60000, 1).astype(np.float32)

        x_test = mnist_test.train_data.numpy().reshape(10000, 784).astype(np.float32)
        y_test = mnist_test.train_labels.numpy().reshape(10000, 1).astype(np.float32)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def _get_breast_cancer(test_split):
        print('Reading Breast Cancer Data')
        t0 = time.time()
        data_bunch = load_breast_cancer()
        x = data_bunch.data
        x = preprocessing.scale(x)
        y = data_bunch.target
        x, y = shuffle(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        print('Time to read Breast Cancer Data = {}s'.format(time.time() - t0))

        return x_train, y_train, x_test, y_test
