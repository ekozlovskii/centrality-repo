from centrality.base import BaseCentrality
import networkx as nx


def _distance_from_weight(_, __, data):
    if "weight" in data:
        weight = data.get("weight", 1.0)
    else:
        weights = [edge_data.get("weight", 1.0) for edge_data in data.values()]
        weight = max(weights) if weights else 1.0
    return 1.0 / weight if weight > 0 else float("inf")


class WeightedClosenessCentrality(BaseCentrality):


    def compute(self) -> dict:
        self.scores = {}
        n = self.graph.number_of_nodes()
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

        for node in self.graph.nodes():
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, node, weight=_distance_from_weight
            )
            
            total = sum(lengths.values())
            reachable = len(lengths) - 1

            if reachable == 0 or total == 0:
                self.scores[node] = 0.0
            else:
                self.scores[node] = (reachable / (n - 1)) * (reachable / total)

        return self.scores
