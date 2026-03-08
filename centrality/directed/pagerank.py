from centrality.base import BaseCentrality
import numpy as np

class PageRankCentrality(BaseCentrality):
    """
    PageRank Centrality — probability that a random walker
    following edges will land on this node.
    
    Formula: PR(v) = (1-d)/N + d * sum( PR(u)/out_degree(u) )
    where d is damping factor (usually 0.85),
    u are all nodes pointing to v.
    
    Idea: you are important if important nodes point to you.
    This is the original Google search ranking algorithm.
    """

    def __init__(self, graph, damping=0.85, max_iter=100, tol=1e-6):
        super().__init__(graph)
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        # Начальный PageRank — равномерно
        pr = np.ones(n) / n

        for _ in range(self.max_iter):
            pr_new = np.ones(n) * (1 - self.damping) / n

            for node in nodes:
                i = node_index[node]
                # Все входящие рёбра
                for predecessor in self.graph.predecessors(node):
                    j = node_index[predecessor]
                    out_deg = self.graph.out_degree(predecessor)
                    if out_deg > 0:
                        pr_new[i] += self.damping * pr[j] / out_deg

            if np.linalg.norm(pr_new - pr) < self.tol:
                break
            pr = pr_new

        self.scores = {nodes[i]: float(pr[i]) for i in range(n)}
        return self.scores