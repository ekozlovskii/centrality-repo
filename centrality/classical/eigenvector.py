from centrality.base import BaseCentrality
import networkx as nx
import numpy as np

class EigenvectorCentrality(BaseCentrality):


    def __init__(self, graph, max_iter=1000, tol=1e-6):
        super().__init__(graph)
        self.max_iter = max_iter
        self.tol = tol

    def compute(self) -> dict:
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        # Строим матрицу смежности
        A = np.zeros((n, n))
        for u, v in self.graph.edges():
            i, j = node_index[u], node_index[v]
            A[i][j] = 1
            A[j][i] = 1

        # Степенная итерация (power iteration)
        x = np.ones(n) / n
        for _ in range(self.max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new

        self.scores = {nodes[i]: float(x[i]) for i in range(n)}
        return self.scores