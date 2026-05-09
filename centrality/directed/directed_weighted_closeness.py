import networkx as nx
from centrality.base import BaseCentrality


def _distance_from_weight(_, __, data):
    weight = data.get("weight", 1.0)
    return 1.0 / weight if weight > 0 else float("inf")


class DirectedWeightedClosenessCentrality(BaseCentrality):

    def __init__(self, graph, mode="in"):
        super().__init__(graph)
        if mode not in {"in", "out"}:
            raise ValueError("mode must be either 'in' or 'out'")
        self.mode = mode

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        n = self.graph.number_of_nodes()
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

        graph_for_paths = self.graph.reverse(copy=False) if self.mode == "in" else self.graph
        self.scores = {}

        for node in self.graph.nodes():
            lengths = nx.single_source_dijkstra_path_length(
                graph_for_paths,
                node,
                weight=_distance_from_weight,
            )

            total_distance = sum(lengths.values())
            reachable = len(lengths) - 1

            if reachable == 0 or total_distance == 0:
                self.scores[node] = 0.0
            else:
                self.scores[node] = (reachable / (n - 1)) * (reachable / total_distance)

        return self.scores
