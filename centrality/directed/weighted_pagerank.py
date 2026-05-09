import numpy as np
from centrality.base import BaseCentrality


class WeightedPageRankCentrality(BaseCentrality):

    def __init__(self, graph, damping=0.85, max_iter=100, tol=1e-6):
        super().__init__(graph)
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def _out_weight_sum(self, node):
        if self.graph.is_multigraph():
            return sum(
                data.get("weight", 1.0)
                for _, _, _, data in self.graph.out_edges(node, keys=True, data=True)
            )
        return sum(
            data.get("weight", 1.0)
            for _, _, data in self.graph.out_edges(node, data=True)
        )

    def _incoming_edges(self, node):
        if self.graph.is_multigraph():
            for predecessor, _, _, data in self.graph.in_edges(node, keys=True, data=True):
                yield predecessor, data.get("weight", 1.0)
        else:
            for predecessor, _, data in self.graph.in_edges(node, data=True):
                yield predecessor, data.get("weight", 1.0)

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        nodes = list(self.graph.nodes())
        n = len(nodes)
        if n == 0:
            self.scores = {}
            return self.scores

        pr = np.ones(n) / n
        node_index = {node: i for i, node in enumerate(nodes)}
        out_weight = {node: self._out_weight_sum(node) for node in nodes}

        for _ in range(self.max_iter):
            dangling_rank = sum(
                pr[node_index[node]]
                for node in nodes
                if out_weight[node] == 0
            )
            pr_new = np.ones(n) * ((1 - self.damping) / n)
            pr_new += self.damping * dangling_rank / n

            for node in nodes:
                i = node_index[node]
                for predecessor, weight in self._incoming_edges(node):
                    total = out_weight[predecessor]
                    if total > 0:
                        j = node_index[predecessor]
                        pr_new[i] += self.damping * pr[j] * weight / total

            if np.linalg.norm(pr_new - pr, ord=1) < self.tol:
                pr = pr_new
                break
            pr = pr_new

        self.scores = {nodes[i]: float(pr[i]) for i in range(n)}
        return self.scores
