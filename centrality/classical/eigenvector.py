from centrality.base import BaseCentrality
import numpy as np

class EigenvectorCentrality(BaseCentrality):


    def __init__(self, graph, max_iter=1000, tol=1e-6):
        super().__init__(graph)
        self.max_iter = max_iter
        self.tol = tol

    def compute(self) -> dict:
        nodes = list(self.graph.nodes())
        n = len(nodes)
        if n == 0:
            self.scores = {}
            return self.scores

        node_index = {node: i for i, node in enumerate(nodes)}

        x = np.ones(n) / np.sqrt(n)
        for _ in range(self.max_iter):
            # A + I shift helps avoid oscillation on bipartite graphs.
            x_new = x.copy()
            if self.graph.is_multigraph():
                for u, v, _ in self.graph.edges(keys=True):
                    i, j = node_index[u], node_index[v]
                    x_new[i] += x[j]
                    x_new[j] += x[i]
            else:
                for u, v in self.graph.edges():
                    i, j = node_index[u], node_index[v]
                    x_new[i] += x[j]
                    x_new[j] += x[i]

            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                break
            x = x_new

        self.scores = {nodes[i]: float(x[i]) for i in range(n)}
        return self.scores
