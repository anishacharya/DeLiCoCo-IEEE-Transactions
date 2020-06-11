import numpy as np
from scipy.sparse import isspmatrix
from typing import Dict


class LogisticRegression:
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

        # Take the error when label=1
        class1_cost = -y * np.log(predictions)
        # Take the error when label=0
        class2_cost = (1 - y) * np.log(1 - predictions)
        loss = class1_cost - class2_cost
        loss = loss.sum() / A.shape[0]

        # loss = np.sum(np.log(1 + np.exp(-y * (A @ x)))) / A.shape[0]
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
        pred = self.sigmoid(logits)
        return pred
    
    @staticmethod
    def decision_boundary(prob):
        return 1 if prob >= .5 else 0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

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
        if stochastic:
            # compute stochastic gradient
            sample_idx = np.random.choice(indices[machine])
            sliced_A = A[sample_idx]
            minus_grad = y[sample_idx] * sliced_A * self.sigmoid(-y[sample_idx] * sliced_A.dot(x).squeeze())
            if isspmatrix(sliced_A):
                minus_grad = minus_grad.toarray().squeeze(0)

        else:
            # compute full gradient
            data_ix = indices[machine]
            sliced_A = A[data_ix, :]
            sliced_y = y[data_ix]
            N = sliced_A.shape[0]

            # Get Predictions
            predictions = self.predict(A=sliced_A, machine=machine)
            predictions = predictions.reshape(predictions.shape[0], 1)
            gradient = sliced_A.T @ (predictions - sliced_y)
            gradient /= N
            minus_grad = - gradient

        if self.params.regularizer:
            minus_grad -= self.params.regularizer * (x.reshape(x.shape[0], 1))

        return np.squeeze(minus_grad)

    # def update_estimate(self, t):
    #     t = int(t)  # to avoid overflow with np.int32
    #     p = self.params
    #     if p.estimate == 'final':
    #         self.x_estimate = self.x
    #     elif p.estimate == 'mean':
    #         rho = 1 / (t + 1)
    #         self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
    #     elif p.estimate == 't+tau':
    #         rho = 2 * (t + p.tau) / ((1 + t) * (t + 2 * p.tau))
    #         self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
    #     elif p.estimate == '(t+tau)^2':
    #         rho = 6 * ((t + p.tau) ** 2) / ((1 + t) * (6 * (p.tau ** 2) + t + 6 * p.tau * t + 2 * (t ** 2)))
    #         self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho

    def lr(self, epoch, iteration, num_samples):
        if self.params.lr_type == 'constant':
            return self.params.initial_lr
        elif self.params.lr_type == 'epoch_decay':
            return self.params.initial_lr * (self.params.epoch_decay_lr ** epoch)



