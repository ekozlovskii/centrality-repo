from centrality.base import BaseCentrality
import networkx as nx

class WeightedBetweennessCentrality(BaseCentrality):
    """
    Weighted Betweenness Centrality — betweenness that uses weighted
    shortest paths instead of hop-count shortest paths.

    Formula: C(v) = sum( sigma_w(s,t|v) / sigma_w(s,t) ) for all s != v != t
    where sigma_w(s,t) is the number of weighted shortest paths between s and t,
    sigma_w(s,t|v) is those passing through v.

    Note: uses 1/weight as distance so stronger edges = shorter paths.
    """

    def compute(self) -> dict:
        n = self.graph.number_of_nodes()
        self.scores = {node: 0.0 for node in self.graph.nodes()}

        # Копия графа с distance = 1/weight
        G_dist = self.graph.copy()
        for u, v, data in G_dist.edges(data=True):
            w = data.get('weight', 1.0)
            G_dist[u][v]['distance'] = 1.0 / w if w != 0 else float('inf')

        for s in self.graph.nodes():
            # Все кратчайшие взвешенные пути от s
            paths = nx.single_source_dijkstra_path(
                G_dist, s, weight='distance'
            )

            for t in self.graph.nodes():
                if s == t or t not in paths:
                    continue

                path = paths[t]
                for v in path[1:-1]:
                    self.scores[v] += 1

        # Нормализация
        norm = (n - 1) * (n - 2)
        if norm > 0:
            self.scores = {k: v / norm for k, v in self.scores.items()}

        return self.scores