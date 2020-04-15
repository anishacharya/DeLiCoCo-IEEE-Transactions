import torchvision.datasets as datasets
import numpy as np


class DataReader:
    def __init__(self, data_set: str = 'mnist'):
        self.data_set = data_set
        if data_set == 'mnist':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_mnist()
        else:
            raise NotImplementedError

    @staticmethod
    def _get_mnist():
        mnist_train = datasets.MNIST(root='../', download=True, train=True)
        mnist_test = datasets.MNIST(root='./Data', download=True, train=False)
        x_train = mnist_train.train_data.numpy().reshape(60000, 784).astype(np.float32)
        y_train = mnist_train.train_labels.numpy().reshape(60000, 1).astype(np.float32)

        x_test = mnist_test.train_data.numpy().reshape(10000, 784).astype(np.float32)
        y_test = mnist_test.train_labels.numpy().reshape(10000, 1).astype(np.float32)

        return x_train,y_train,x_test,y_test
