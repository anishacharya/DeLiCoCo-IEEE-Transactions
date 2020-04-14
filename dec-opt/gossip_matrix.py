import numpy as np
import networkx


class GossipMatrix:
    def __init__(self, topology, n_cores):
        self.n_cores = n_cores
        self.topology = topology

    # noinspection PyPep8Naming
    @staticmethod
    def _get_gossip_matrix(topology, n_cores):
        if topology == 'ring':
            W = np.zeros(shape=(n_cores, n_cores))
            value = 1. / 3 if n_cores >= 3 else 1. / 2
            np.fill_diagonal(W, value)
            np.fill_diagonal(W[1:], value, wrap=False)
            np.fill_diagonal(W[:, 1:], value, wrap=False)
            W[0, n_cores - 1] = value
            W[n_cores - 1, 0] = value
            return W
        elif topology == 'centralized':
            W = np.ones((n_cores, n_cores), dtype=np.float64) / n_cores
            return W
        elif topology == 'disconnected':
            W = np.eye(n_cores)
            return W
        elif topology == 'torus':
            print('torus topology!')
            assert topology == 'torus'
            assert int(np.sqrt(n_cores)) ** 2 == n_cores
            G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(n_cores)),
                                                          int(np.sqrt(n_cores)), periodic=True)
            W = networkx.adjacency_matrix(G).toarray()
            for i in range(0, W.shape[0]):
                W[i][i] = 1
            W = W / 5
            return W
        else:
            raise NotImplementedError
