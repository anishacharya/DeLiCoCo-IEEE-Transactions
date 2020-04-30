import numpy as np

"""
Author: Anish Acharya
Contact: anishacharya@utexas.edu
"""


class Compression:
    def __init__(self, num_bits: int,
                 quantization_function: str,
                 dropout_p: float,
                 fraction_coordinates: float):
        self.quantization_function = quantization_function
        self.num_bits = num_bits
        self.fraction_coordinates = fraction_coordinates
        self.dropout_p = dropout_p

    def quantize(self, x):
        # quantize according to quantization function
        # x: shape(num_features, n_cores)
        if self.quantization_function in ['qsgd-biased', 'qsgd-unbiased']:
            is_biased = (self.quantization_function == 'qsgd-biased')
            assert self.num_bits
            q = np.zeros_like(x)
            for i in range(0, q.shape[1]):
                q[:, i] = self.qsgd_quantize(x[:, i], self.num_bits, is_biased)
            return q
        if self.quantization_function == 'full':
            return x
        if self.quantization_function == 'top':
            q = np.zeros_like(x)

            k = round(self.fraction_coordinates * q.shape[0])
            for i in range(0, q.shape[1]):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q
        if self.quantization_function == 'rand':
            q = np.zeros_like(x)
            k = round(self.fraction_coordinates * q.shape[0])
            for i in range(0, q.shape[1]):
                perm_i = np.random.permutation(q.shape[0])
                q[perm_i[0:k], i] = x[perm_i[0:k], i]
            return q
        # NEW--
        if self.quantization_function == 'dropout-biased':
            q = np.zeros_like(x)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = x[:, i] * bin_i
            return q
        # NEW--
        if self.quantization_function == 'dropout-unbiased':
            q = np.zeros_like(x)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = x[:, i] * bin_i
            return q / p
        # NEW--
        if self.quantization_function == 'qsgd-abol':
            q = np.zeros_like(x)
            bits = self.num_bits
            s = 2 ** bits
            tau = 1 + min((np.sqrt(q.shape[0])/s), (q.shape[0]/(s**2)))
            for i in range(0, q.shape[1]):
                unif_i = np.random.rand(q.shape[0],)
                x_i = x[:, i]
                q[:, i] = ((np.sign(x_i) * np.linalg.norm(x_i))/(s*tau)) * np.floor((s*np.abs(x_i)/np.linalg.norm(x_i)) + unif_i)
            return q

        '''
        assert self.quantization_function in ['random-biased', 'random-unbiased']
        Q = np.zeros_like(x)
        k = self.coordinates_to_keep
        for i in range(0, Q.shape[1]):
            indexes = np.random.choice(np.arange(Q.shape[0]), k, replace=False)
            Q[indexes[:k], i] = x[indexes[:k], i]
        if self.quantization_function == 'random-unbiased':
            return x.shape[0] / k * Q
        return Q
        '''
        
    @staticmethod
    def qsgd_quantize(x, d, is_biased):
        norm = np.sqrt(np.sum(np.square(x)))
        level_float = d * np.abs(x) / norm
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        scale = 1
        if is_biased:
            n = len(x)
            scale = 1. / (np.minimum(n / d ** 2, np.sqrt(n) / d) + 1.)
        return scale * np.sign(x) * norm * new_level / d
