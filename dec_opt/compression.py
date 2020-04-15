import numpy as np


class Compression:
    def __init__(self, x,
                 num_levels: int = 1,
                 quantization_function: str = 'top',
                 coordinates_to_keep: str = 1):
        self.x = x
        self.quantization_function = quantization_function
        self.num_levels = num_levels
        self.coordinates_to_keep = coordinates_to_keep
        self.Q = self.__quantize(x=x)

    def __quantize(self, x):
        # quantize according to quantization function
        # x: shape(num_features, n_cores)
        if self.quantization_function in ['qsgd-biased', 'qsgd-unbiased']:
            is_biased = (self.quantization_function == 'qsgd-biased')
            assert self.num_levels
            q = np.zeros_like(x)
            for i in range(0, q.shape[1]):
                q[:, i] = self.qsgd_quantize(x[:, i], self.num_levels, is_biased)
            return q
        if self.quantization_function == 'full':
            return x
        if self.quantization_function == 'top':
            q = np.zeros_like(x)
            k = self.coordinates_to_keep
            for i in range(0, q.shape[1]):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q

        assert self.quantization_function in ['random-biased', 'random-unbiased']
        Q = np.zeros_like(x)
        k = self.coordinates_to_keep
        for i in range(0, Q.shape[1]):
            indexes = np.random.choice(np.arange(Q.shape[0]), k, replace=False)
            Q[indexes[:k], i] = x[indexes[:k], i]
        if self.quantization_function == 'random-unbiased':
            return x.shape[0] / k * Q
        return Q

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
