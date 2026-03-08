from centrality.base import BaseCentrality
import networkx as nx

class WeightedClosenessCentrality(BaseCentrality):
    """
    Weighted Closeness Centrality — closeness centrality that accounts
    for edge weights as distances between nodes.
    
    Formula: C(v) = (N-1) / sum( d_w(v, u) ) for all u != v
    where d_w(v, u) is the shortest weighted path between v and u.
    
    Note: higher weight = stronger connection = shorter distance.
    We use 1/weight as the actual distance.
    """

    def compute(self) -> dict:
        self.scores = {}
        n = self.graph.number_of_nodes()

        # Создаём копию графа где вес = 1/weight (сильная связь = короткое расстояние)
        G_dist = self.graph.copy()
        for u, v, data in G_dist.edges(data=True):
            w = data.get('weight', 1.0)
            G_dist[u][v]['distance'] = 1.0 / w if w != 0 else float('inf')

        for node in self.graph.nodes():
            lengths = nx.single_source_dijkstra_path_length(
                G_dist, node, weight='distance'
            )
            
            total = sum(lengths.values())
            reachable = len(lengths) - 1

            if reachable == 0 or total == 0:
                self.scores[node] = 0.0
            else:
                self.scores[node] = (reachable / (n - 1)) * (reachable / total)

        return self.scores