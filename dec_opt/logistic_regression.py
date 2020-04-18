import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import isspmatrix
from typing import Dict
"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""


class LogisticRegression:
    def __init__(self, params):
        self.params = params
        self.x_estimate = None
        self.x = None

    def loss(self, A, y):
        if self.x is None:
            raise Exception
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = x.copy().mean(axis=1)
        loss = np.sum(np.log(1 + np.exp(-y * (A @ x)))) / A.shape[0]
        if self.params.regularizer:
            loss += self.params.regularizer * np.square(x).sum() / 2
        return loss

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

    def predict(self, A):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = x.mean(axis=1)
        logits = A @ x
        return sigmoid(logits)

    def score(self, A, y):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc

    def get_grad(self, A, y, stochastic: bool, indices: Dict, machine: int):
        x = self.x[:, machine]
        if stochastic:
            sample_idx = np.random.choice(indices[machine])
            a = A[sample_idx]
            minus_grad = y[sample_idx] * a * sigmoid(-y[sample_idx] * a.dot(x).squeeze())
            if isspmatrix(a):
                minus_grad = minus_grad.toarray().squeeze(0)
            if self.params.regularizer:
                minus_grad -= self.params.regularizer * x
        else:
            raise NotImplementedError

        return minus_grad

    def update_estimate(self, t):
        t = int(t)  # to avoid overflow with np.int32
        p = self.params
        if p.estimate == 'final':
            self.x_estimate = self.x
        elif p.estimate == 'mean':
            rho = 1 / (t + 1)
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == 't+tau':
            rho = 2 * (t + p.tau) / ((1 + t) * (t + 2 * p.tau))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == '(t+tau)^2':
            rho = 6 * ((t + p.tau) ** 2) / ((1 + t) * (6 * (p.tau ** 2) + t + 6 * p.tau * t + 2 * (t ** 2)))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.params)

    def __repr__(self):
        return str(self)
