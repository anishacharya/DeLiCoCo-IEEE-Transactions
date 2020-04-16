import numpy as np

INIT_WEIGHT_STD = 0.01
LOSS_PER_EPOCH = 10
np.random.seed(1)


class DecGD:
    def __init__(self, feature, target, hyper_param):
        self.A = feature
        self.y = target
        self.param = hyper_param

        # initialize x_hat, x_estimate  Ax = y is the problem we are solving
        # -------------------------------------------------------------------
        self.losses = np.zeros(self.param.epochs + 1)
        self.num_samples, self.num_features = self.A.shape

        self.x = np.random.normal(0, INIT_WEIGHT_STD, size=(self.num_features,))
        self.x = np.tile(self.x, (self.param.n_cores, 1)).T

        self.x_estimate = np.copy(self.x)
        self.x_hat = np.copy(self.x)

        # Now Distribute the Data among machines
        # ----------------------------------------
        self.data_partition_ix = self._distribute_data()

    def _distribute_data(self):
        data_partition_ix = []
        num_samples_per_machine = self.num_samples // self.param.n_cores
        all_indexes = np.arange(self.num_samples)
        np.random.shuffle(all_indexes)

        for machine in range(0, self.param.n_cores - 1):
            data_partition_ix += [all_indexes[num_samples_per_machine * machine: num_samples_per_machine * (machine + 1)]]
        # put the rest in the last machine
        data_partition_ix += [all_indexes[num_samples_per_machine * (self.param.n_cores - 1):]]
        print("All but last machine has {} data points".format(num_samples_per_machine))
        print("length of last machine indices:", len(data_partition_ix[-1]))
        return data_partition_ix



