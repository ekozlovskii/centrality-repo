import numpy as np
from centrality.base import BaseCentrality


class WeightedEigenvectorCentrality(BaseCentrality):

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
            x_new = x.copy()

            if self.graph.is_multigraph():
                for u, v, _, data in self.graph.edges(keys=True, data=True):
                    weight = data.get("weight", 1.0)
                    x_new[node_index[u]] += weight * x[node_index[v]]
                    x_new[node_index[v]] += weight * x[node_index[u]]
            else:
                for u, v, data in self.graph.edges(data=True):
                    weight = data.get("weight", 1.0)
                    x_new[node_index[u]] += weight * x[node_index[v]]
                    x_new[node_index[v]] += weight * x[node_index[u]]

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
