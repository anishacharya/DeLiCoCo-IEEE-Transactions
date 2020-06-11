import numpy as np
import networkx


class GossipMatrix:
    def __init__(self, topology, n_cores):
        self.nodes = n_cores
        self.topology = topology
        self.W = self._get_gossip_matrix(topology=self.topology, nodes=self.nodes)

    # noinspection PyPep8Naming
    @staticmethod
    def _get_gossip_matrix(topology, nodes):
        if topology == 'ring':
            W = np.zeros(shape=(nodes, nodes))
            value = 1. / 3 if nodes >= 3 else 1. / 2
            np.fill_diagonal(W, value)
            np.fill_diagonal(W[1:], value, wrap=False)
            np.fill_diagonal(W[:, 1:], value, wrap=False)
            W[0, nodes - 1] = value
            W[nodes - 1, 0] = value
            return W
        elif topology == 'fully_connected':
            W = np.ones((nodes, nodes), dtype=np.float64) / nodes
            return W
        elif topology == 'disconnected':
            W = np.eye(nodes)
            return W
        elif topology == 'torus':
            assert int(np.sqrt(nodes)) ** 2 == nodes
            G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(nodes)),
                                                          int(np.sqrt(nodes)), periodic=True)
            W = networkx.adjacency_matrix(G).toarray()
            for i in range(0, W.shape[0]):
                W[i][i] = 1
            W = W / 5
            return W
        else:
            raise NotImplementedError


if __name__ == '__main__':
    gossip_matrix_1 = GossipMatrix(topology='ring', n_cores=10).W
    gossip_matrix_2 = GossipMatrix(topology='centralized', n_cores=10).W
