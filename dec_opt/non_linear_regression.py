import numpy as np
from typing import Dict


class NonLinearRegression:
    def __init__(self, params):
        self.params = params

        self.x_estimate = None
        self.Z = None
        self.S = None

    def loss(self, A, y):
        x = np.copy(self.x_estimate)
        x = np.mean(x, axis=1)
        predictions = self.predict(A=A)

        predictions = predictions.reshape(predictions.shape[0], 1)
        
        loss = ((y - predictions)* (y - predictions)).sum() / (2*A.shape[0])
        if self.params.regularizer:
            loss += self.params.regularizer * np.square(x).sum() / 2       
            
        return loss

    def predict(self, A, machine=None):
        """
          Returns 1D array of probabilities
          that the class label == 1
        """
        if machine is None:
            x = np.copy(self.x_estimate)
            x = np.mean(x, axis=1)
        else:
            x = self.x_estimate[:, machine]
        
        logits = A @ x
        pred = self.relu(logits)
        return pred
    
    @staticmethod
    def decision_boundary(prob):
        return 1 if prob >= .5 else 0

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    def classify(self, predictions):
        """
        input  - N element array of predictions between 0 and 1
        output - N element array of 0s (False) and 1s (True)
        """
        decision_boundary = np.vectorize(self.decision_boundary)
        return decision_boundary(predictions).flatten()

    @staticmethod
    def accuracy(predicted_labels, actual_labels):
        predicted_labels = predicted_labels.reshape(predicted_labels.shape[0], 1)
        diff = predicted_labels - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def get_grad(self, machine: int, A, y, stochastic: bool, indices: Dict):
        x = self.x_estimate[:, machine]
        
        # compute full gradient
        data_ix = indices[machine]
        sliced_A = A[data_ix, :]
        sliced_y = y[data_ix]
        N = sliced_A.shape[0]

        # Get Predictions
        predictions = self.predict(A=sliced_A, machine=machine)
        predictions = predictions.reshape(predictions.shape[0], 1)
        
        # for ReLU---
        gradient = np.dot(sliced_A.T, (np.sign(predictions) * (predictions - sliced_y)))
        
        gradient /= N
        minus_grad = - gradient

        if self.params.regularizer:
            minus_grad -= self.params.regularizer * (x.reshape(x.shape[0], 1))

        return np.squeeze(minus_grad)


    def lr(self, epoch, iteration, num_samples):
        if self.params.lr_type == 'constant':
            return self.params.initial_lr
        elif self.params.lr_type == 'epoch_decay':
            return self.params.initial_lr * (self.params.epoch_decay_lr ** epoch)



