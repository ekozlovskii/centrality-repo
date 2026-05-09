from centrality.base import BaseCentrality
import networkx as nx


def _distance_from_weight(_, __, data):
    if "weight" in data:
        weight = data.get("weight", 1.0)
    else:
        weights = [edge_data.get("weight", 1.0) for edge_data in data.values()]
        weight = max(weights) if weights else 1.0
    return 1.0 / weight if weight > 0 else float("inf")


class WeightedBetweennessCentrality(BaseCentrality):


    def compute(self) -> dict:
        n = self.graph.number_of_nodes()
        self.scores = {node: 0.0 for node in self.graph.nodes()}

        for s in self.graph.nodes():
            # Все кратчайшие взвешенные пути от s
            paths = nx.single_source_dijkstra_path(
                self.graph, s, weight=_distance_from_weight
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
