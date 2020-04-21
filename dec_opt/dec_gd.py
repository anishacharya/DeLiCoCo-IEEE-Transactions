import numpy as np
import time

from dec_opt.gossip_matrix import GossipMatrix
from dec_opt.compression import Compression

"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""

INIT_WEIGHT_STD = 0.01
LOSS_PER_EPOCH = 10
np.random.seed(1)


class DecGD:
    def __init__(self, feature, target, hyper_param, model):
        self.A = feature
        self.y = target
        self.param = hyper_param
        self.model = model
        self.W = GossipMatrix(topology=self.param.topology, n_cores=self.param.n_cores).W
        self.Q = Compression(num_levels=self.param.num_levels,
                             quantization_function=self.param.quantization_function,
                             coordinates_to_keep=self.param.coordinates_to_keep)
        # initialize x_hat, x_estimate  Ax = y is the problem we are solving
        # -------------------------------------------------------------------
        self.losses = np.zeros(self.param.epochs + 1)
        self.num_samples, self.num_features = self.A.shape

        self.model.x_estimate = np.random.normal(0, INIT_WEIGHT_STD, size=(self.num_features,))
        self.model.x_estimate = np.tile(self.model.x_estimate, (self.param.n_cores, 1)).T
        # self.model.x_estimate = np.copy(self.model.x)
        # self.model.x_hat = np.copy(self.model.x)

        print("Number of different labels:", len(np.unique(self.y)))

        # Now Distribute the Data among machines
        # ----------------------------------------
        self.data_partition_ix, self.num_samples_per_machine = self._distribute_data()

        # Decentralized Training
        # --------------------------
        self.epoch_losses = self._dec_train()

    def _distribute_data(self):
        data_partition_ix = []
        num_samples_per_machine = self.num_samples // self.param.n_cores
        all_indexes = np.arange(self.num_samples)
        np.random.shuffle(all_indexes)

        for machine in range(0, self.param.n_cores - 1):
            data_partition_ix += [
                all_indexes[num_samples_per_machine * machine: num_samples_per_machine * (machine + 1)]]
        # put the rest in the last machine
        data_partition_ix += [all_indexes[num_samples_per_machine * (self.param.n_cores - 1):]]
        print("All but last machine has {} data points".format(num_samples_per_machine))
        print("length of last machine indices:", len(data_partition_ix[-1]))
        return data_partition_ix, num_samples_per_machine

    def _dec_train(self):
        losses = np.zeros(self.param.epochs + 1)
        losses[0] = self.model.loss(self.A, self.y)

        compute_loss_every = int(self.num_samples_per_machine / LOSS_PER_EPOCH) + 1
        all_losses = np.zeros(int(self.num_samples_per_machine * self.param.epochs / compute_loss_every) + 1)
        train_start = time.time()

        for epoch in np.arange(self.param.epochs):
            loss = self.model.loss(self.A, self.y)
            if np.isinf(loss) or np.isnan(loss):
                print("training exit - diverging")
                break
            lr = self.model.lr(epoch=epoch,
                               iteration=epoch,
                               num_samples=self.num_samples_per_machine,
                               tau=self.num_features)

            # Gradient step
            x_plus = np.zeros_like(self.model.x_estimate)
            #  for t in 0...T − 1 do in parallel for all workers i ∈[n]
            for machine in range(0, self.param.n_cores):
                # Compute neg. Gradient (or stochastic gradient) based on algorithm
                minus_grad = self.model.get_grad(A=self.A,
                                                 y=self.y,
                                                 stochastic=self.param.stochastic,
                                                 indices=self.data_partition_ix,
                                                 machine=machine)
                x_plus[:, machine] = lr * minus_grad

            # Communication step
            if self.param.algorithm == 'vanilla':
                self.model.x_estimate = (self.model.x_estimate + x_plus).dot(self.W)

            elif self.param.algorithm == 'choco':
                x_plus += self.model.x
                self.model.x = x_plus + self.param.consensus_lr * \
                    self.model.x_hat.dot(self.W - np.eye(self.param.n_cores))

                quantized = self.Q.quantize(self.model.x - self.model.x_hat)
                self.model.x_hat += quantized
            else:
                raise NotImplementedError

            # self.model.update_estimate(t)
            losses[epoch + 1] = self.model.loss(self.A, self.y)
            pred = self.model.predict(A=self.A)
            pred_labels = self.model.classify(predictions=pred)
            acc = self.model.accuracy(pred_labels, self.y)
            print("epoch : {}; loss: {}; accuracy : {}".format(epoch, losses[epoch + 1], acc))

            if np.isinf(losses[epoch + 1]) or np.isnan(losses[epoch + 1]):
                print("Break training - Diverged")
                break

        print("Training took: {}s".format(time.time() - train_start))
        return losses
