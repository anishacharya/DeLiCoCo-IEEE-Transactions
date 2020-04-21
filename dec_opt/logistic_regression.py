import numpy as np
from scipy.sparse import isspmatrix
from typing import Dict
import matplotlib as plt
"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""


class LogisticRegression:
    def __init__(self, params):
        self.params = params
        self.x_estimate = None

    def loss(self, A, y):
        x = np.copy(self.x_estimate)
        x = np.mean(x, axis=1)
        predictions = self.predict(A=A)

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

    def predict(self, A):
        """
          Returns 1D array of probabilities
          that the class label == 1
        """
        x = np.copy(self.x_estimate)
        x = np.mean(x, axis=1)

        logits = A @ x
        z = np.dot(A, x)
        pred = self.sigmoid(logits)
        # pred = 1 * (logits >= 0.)
        return pred

    def score(self, A, y):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc

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
        diff = predicted_labels - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def get_grad(self, A, y, stochastic: bool, indices: Dict, machine: int):
        x = self.x_estimate[:, machine]
        if stochastic:
            # compute stochastic gradient
            sample_idx = np.random.choice(indices[machine])
            a = A[sample_idx]
            minus_grad = y[sample_idx] * a * self.sigmoid(-y[sample_idx] * a.dot(x).squeeze())
            if isspmatrix(a):
                minus_grad = minus_grad.toarray().squeeze(0)

        else:
            # compute full gradient
            N = A.shape[1]
            # Get Predictions
            predictions = self.predict(A=A)
            gradient = np.dot(A.T, predictions - y)
            gradient /= N
            minus_grad = - gradient

        if self.params.regularizer:
            minus_grad -= self.params.regularizer * x

        return minus_grad

    @staticmethod
    def plot_decision_boundary(trues, falses):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        no_of_preds = len(trues) + len(falses)

        ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
        ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

        plt.legend(loc='upper right');
        ax.set_title("Decision Boundary")
        ax.set_xlabel('N/2')
        ax.set_ylabel('Predicted Probability')
        plt.axhline(.5, color='black')
        plt.show()
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

    def lr(self, epoch, iteration, num_samples, tau):
        t = epoch * num_samples + iteration
        if self.params.lr_type == 'constant':
            return self.params.initial_lr
        if self.params.lr_type == 'epoch-decay':
            return self.params.initial_lr * (self.params.epoch_decay_lr ** epoch)
        if self.params.lr_type == 'decay':
            return self.params.initial_lr / (self.params.regularizer * (t + tau))
        if self.params.lr_type == 'bottou':
            return self.params.initial_lr / (1 + self.params.initial_lr * self.params.regularizer * t)


