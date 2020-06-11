import numpy as np
import time
from dec_opt.gossip_matrix import GossipMatrix
from dec_opt.compression import Compression


class DecGD:
    def __init__(self, data_reader, hyper_param, model):
        self.A_train = data_reader.A_train
        self.y_train = data_reader.y_train
        self.A_test = data_reader.A_test
        self.y_test = data_reader.y_test

        self.param = hyper_param
        self.model = model

        self.W = GossipMatrix(topology=self.param.topology, n_cores=self.param.n_cores).W
        self.C = Compression(num_bits=self.param.num_bits,
                             quantization_function=self.param.quantization_function,
                             dropout_p=self.param.dropout_p,
                             fraction_coordinates=self.param.fraction_coordinates)

        # initialize parameters for each node. Ax = y is the problem we are solving
        # ----------------------------------------------------------------------------------
        self.losses = np.zeros(self.param.epochs + 1)
        self.num_samples, self.num_features = self.A_train.shape
        INIT_WEIGHT_STD = 1 / np.sqrt(self.num_features)

        np.random.seed(self.param.seed)
        self.model.x_estimate = np.random.normal(0, INIT_WEIGHT_STD, size=(self.num_features, self.param.n_cores))
        # self.model.x_estimate = np.tile(self.model.x_estimate, (self.param.n_cores, 1)).T

        self.model.Z = np.zeros(self.model.x_estimate.shape)
        self.model.S = np.zeros(self.model.x_estimate.shape)

        # Now Distribute the Data among machines
        # ----------------------------------------
        self.data_partition_ix, self.num_samples_per_machine = self._distribute_data()

        # Decentralized Training
        # --------------------------
        self.train_losses, self.test_losses = self._dec_train()

    def _distribute_data(self):
        data_partition_ix = []
        num_samples_per_machine = self.num_samples // self.param.n_cores
        all_indexes = np.arange(self.num_samples)
        # np.random.shuffle(all_indexes)

        for machine in range(0, self.param.n_cores - 1):
            data_partition_ix += [
                all_indexes[num_samples_per_machine * machine: num_samples_per_machine * (machine + 1)]]
        # put the rest in the last machine
        data_partition_ix += [all_indexes[num_samples_per_machine * (self.param.n_cores - 1):]]
        print("All but last machine has {} data points".format(num_samples_per_machine))
        print("length of last machine indices:", len(data_partition_ix[-1]))
        return data_partition_ix, num_samples_per_machine

    def _dec_train(self):
        train_losses = np.zeros(self.param.epochs + 1)
        test_losses = np.zeros(self.param.epochs + 1)
        train_losses[0] = self.model.loss(self.A_train, self.y_train)
        test_losses[0] = self.model.loss(self.A_test, self.y_test)
        train_start = time.time()
        for epoch in np.arange(self.param.epochs):
            loss = self.model.loss(self.A_train, self.y_train)
            if np.isinf(loss) or np.isnan(loss):
                print("training exit - diverging")
                break
            lr = self.model.lr(epoch=epoch,
                               iteration=epoch,
                               num_samples=self.num_samples_per_machine)

            # Gradient step
            # --------------------------
            x_plus = np.zeros_like(self.model.x_estimate)
            # for t in 0...T − 1 do in parallel for all workers i ∈[n]
            for machine in range(0, self.param.n_cores):
                # Compute neg. Gradient (or stochastic gradient) based on algorithm
                minus_grad = self.model.get_grad(A=self.A_train,
                                                 y=self.y_train,
                                                 stochastic=self.param.stochastic,
                                                 indices=self.data_partition_ix,
                                                 machine=machine)
                x_plus[:, machine] = lr * minus_grad
            # machine = range(0, self.param.n_cores)
            # pool = Pool(processes=self.param.n_proc)
            # get_grad_multi = partial(self.model.get_grad,
            #                          A=self.A,
            #                          y=self.y,
            #                          stochastic=self.param.stochastic,
            #                          indices=self.data_partition_ix)
            # minus_grad_list = pool.map(get_grad_multi, machine)
            # pool.close()
            # for i in range(0, self.param.n_cores):
            #     x_plus[:, i] = lr * minus_grad_list[i]

            # x_(t+1/2) = x_(t) - lr * grad - Do GD Update
            x_cap = self.model.x_estimate + x_plus

            # Communication step
            # -----------------------------------------
            if self.param.algorithm == 'exact_comm':
                # Xiao, Boyd; Fast Linear Iterations for Distributed Averaging
                self.model.x_estimate = x_cap @ self.W
            elif self.param.algorithm == 'ours':
                # local gradient update i.e. X_t0
                self.model.x_estimate = x_cap
                # now iterate and update the estimate to X_tQ
                for i in range(0, self.param.Q):
                    # Exchanging messages
                    message_exchange = self.C.quantize(self.model.x_estimate - self.model.Z)
                    self.model.S = self.model.S + message_exchange @ self.W
                    # Compression error feedback
                    # error_feedback = self.C.quantize(self.model.x_estimate - self.model.Z)
                    error_feedback = message_exchange
                    self.model.Z = self.model.Z + error_feedback
                    # Local gossip update
                    gossip_update = (self.model.S - self.model.Z)
                    self.model.x_estimate = self.model.x_estimate + \
                        self.param.consensus_lr * gossip_update
            elif self.param.algorithm == 'choco-sgd':
                # Koloskove,Stich,Jaggi; Decentralized Stochastic
                # Optimization and Gossip Algorithms with Compressed Communication
                # x_(t+1) = x_(t+1/2) + \gamma W.dot.(x^_j(t+1) - x^_i(t+1))
                self.model.x_estimate = x_cap + \
                        self.param.consensus_lr * self.model.x_hat.dot(self.W - np.eye(self.param.n_cores))
                pass
            else:
                print('Running Plain GD. n_cores = 1 else convergence results not guaranteed')
                # do nothing just plain GD
                self.model.x_estimate = x_cap

            train_losses[epoch + 1] = self.model.loss(self.A_train, self.y_train)
            test_losses[epoch + 1] = self.model.loss(self.A_test, self.y_test)

            #train_acc = compute_accuracy(model=self.model, feature=self.A_train, target=self.y_train)
            #test_acc = compute_accuracy(model=self.model, feature=self.A_test, target=self.y_test)
            print("epoch : {}; loss: {}".
                  format(epoch, train_losses[epoch + 1]))
            # print("epoch : {}; loss: {}; Test accuracy : {}".format(epoch, test_losses[epoch + 1], test_acc))
            if np.isinf(train_losses[epoch + 1]) or np.isnan(train_losses[epoch + 1]):
                print("Break training - Diverged")
                break
        print("Training took: {}s".format(time.time() - train_start))
        return train_losses, test_losses

'''
def compute_accuracy(model, feature, target):
    pred = model.predict(A=feature)
    pred_labels = model.classify(predictions=pred)
    acc = model.accuracy(pred_labels, target)
    return acc
'''
