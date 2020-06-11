import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import numpy as np
import time
from sklearn import preprocessing


class DataReader:
    def __init__(self, root: str, data_set: str, download=True, test_split: float = 0.2):
        self.data_set = data_set
        self.root = root
        self.download = download
        if data_set == 'mnist':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_mnist()
        elif data_set == 'mnist_partial':
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_mnist_partial()
        elif data_set == 'syn1':
            gen = True #set this to False if SYN1 is already generated once
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_syn1(gen)
        elif data_set == 'syn2':
            gen = True #set this to False if SYN2 is already generated once
            self.A_train, self.y_train, self.A_test, self.y_test = self._get_syn2(gen)
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

    def _get_mnist_partial(self, do_sorting=True):
        mnist_train = datasets.MNIST(root=self.root, download=self.download, train=True)
        mnist_test = datasets.MNIST(root=self.root, download=self.download, train=False)

        x_train = mnist_train.train_data.numpy() / 255.0
        x_test = mnist_test.test_data.numpy() / 255.0
        y_train = mnist_train.train_labels.numpy()
        y_test = mnist_test.test_labels.numpy()

        # flatten the images
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        # Keep only labels 4 and 9
        idx_train = np.argwhere(y_train % 5 == 4)
        idx_test = np.argwhere(y_test % 5 == 4)
        x_train = x_train[idx_train[:, 0], :]
        x_test = x_test[idx_test[:, 0], :]

        y_train = y_train[idx_train[:, 0]]
        y_train[y_train == 4] = 0
        y_train[y_train == 9] = 1

        y_test = y_test[idx_test[:, 0]]
        y_test[y_test == 4] = 0
        y_test[y_test == 9] = 1

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

    def _get_syn1(self, generate = False):
        if generate:
            train_exs = 10000
            y_dim = 1000
            x_train = np.random.normal(0.0,1.0,(2*y_dim,train_exs))
            
            A = np.random.normal(1.0,1.0,(1,2*y_dim))/np.sqrt(y_dim)
            noise = np.random.normal(0.0,0.05,(1,train_exs))
            
            y_train = np.matmul(A,x_train) + noise
            
            # Training set
            np.save('x_train_SYN1.npy',x_train)
            np.save('y_train_SYN1.npy',y_train)        
            
            test_exs = 5000
            x_test = np.random.normal(0.0,1.0,(2*y_dim,test_exs))
            noise = np.random.normal(0.0,0.05,(1,test_exs))
            
            y_test = np.matmul(A,x_test) + noise
            
            # Test set
            np.save('x_test_SYN1.npy',x_test)
            np.save('y_test_SYN1.npy',y_test)
            
            print("Generated!")
            
        else:
            x_train = np.load('x_train_SYN1.npy')
            y_train = np.load('y_train_SYN1.npy')
            x_test = np.load('x_test_SYN1.npy')
            y_test = np.load('y_test_SYN1.npy')
        
        return np.transpose(x_train), np.transpose(y_train), np.transpose(x_test), np.transpose(y_test)
    
    
    def _get_syn2(self, generate = False):
        if generate:
            train_exs = 10000
            y_dim = 1000
            x_train = np.random.normal(0.0,1.0,(2*y_dim,train_exs))
            
            A = np.random.normal(1.0,1.0,(1,2*y_dim))/np.sqrt(y_dim)
            noise = np.random.normal(0.0,0.05,(1,train_exs))
            
            y_train = np.maximum(np.matmul(A,x_train), 0) + noise
            
            # Training set
            np.save('x_train_SYN2.npy',x_train)
            np.save('y_train_SYN2.npy',y_train)        
            
            test_exs = 5000
            x_test = np.random.normal(0.0,1.0,(2*y_dim,test_exs))
            noise = np.random.normal(0.0,0.05,(1,test_exs))
            
            y_test = np.maximum(np.matmul(A,x_test), 0) + noise
            
            # Test set
            np.save('x_test_SYN2.npy',x_test)
            np.save('y_test_SYN2.npy',y_test)
            
            print("Generated!")
            
        else:
            x_train = np.load('x_train_SYN2.npy')
            y_train = np.load('y_train_SYN2.npy')
            x_test = np.load('x_test_SYN2.npy')
            y_test = np.load('y_test_SYN2.npy')
        
        return np.transpose(x_train), np.transpose(y_train), np.transpose(x_test), np.transpose(y_test)
    
        
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
