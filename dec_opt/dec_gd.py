import numpy as np

INIT_WEIGHT_STD = 0.01
LOSS_PER_EPOCH = 10


class DecGD:
    def __init__(self, feature, target, hyper_param):
        self.A = feature
        self.y = target
        self.param = hyper_param

    def _estimation(self):

        # initialize x_hat, x_estimate  Ax = y is the problem we are solving
        # -------------------------------------------------------------------
        losses = np.zeros(self.param.epochs + 1)
        num_samples, num_features = self.A.shape

        self.x = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
        self.x = np.tile(self.x, (self.param.n_cores, 1)).T

        self.x_estimate = np.copy(self.x)
        self.x_hat = np.copy(self.x)

        # Now Distribute the Data among machines
        # ----------------------------------------
