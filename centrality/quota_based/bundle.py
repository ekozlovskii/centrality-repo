from itertools import combinations

import networkx as nx

from centrality.base import BaseCentrality


def _as_weighted_digraph(graph):
    if graph.is_directed() and not graph.is_multigraph():
        digraph = graph.copy()
    else:
        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes(data=True))

        if graph.is_multigraph():
            edges = graph.edges(keys=True, data=True)
            for u, v, _, data in edges:
                weight = float(data.get("weight", 1.0))
                digraph.add_edge(u, v, weight=digraph.get_edge_data(u, v, {}).get("weight", 0.0) + weight)
                if not graph.is_directed():
                    digraph.add_edge(v, u, weight=digraph.get_edge_data(v, u, {}).get("weight", 0.0) + weight)
        else:
            for u, v, data in graph.edges(data=True):
                weight = float(data.get("weight", 1.0))
                digraph.add_edge(u, v, weight=digraph.get_edge_data(u, v, {}).get("weight", 0.0) + weight)
                if not graph.is_directed():
                    digraph.add_edge(v, u, weight=digraph.get_edge_data(v, u, {}).get("weight", 0.0) + weight)

    for u, v, data in digraph.edges(data=True):
        data["weight"] = float(data.get("weight", 1.0))
    return digraph


class _QuotaBasedCentrality(BaseCentrality):
    def __init__(self, graph, quotas, max_group_size=3):
        super().__init__(graph)
        self.quotas = quotas or {}
        self.max_group_size = int(max_group_size)

    def _prepare(self):
        if self.max_group_size < 1:
            raise ValueError("max_group_size must be at least 1.")

        graph = _as_weighted_digraph(self.graph)
        missing = [node for node in graph.nodes() if node not in self.quotas]
        if missing:
            sample = ", ".join(str(node) for node in missing[:5])
            raise ValueError(f"Missing quota for node(s): {sample}")

        quotas = {node: float(self.quotas[node]) for node in graph.nodes()}
        return graph, quotas


class BundleIndexCentrality(_QuotaBasedCentrality):
    def compute(self) -> dict:
        graph, quotas = self._prepare()
        scores = {}

        for node in graph.nodes():
            incoming = [
                (source, float(data.get("weight", 1.0)))
                for source, _, data in graph.in_edges(node, data=True)
                if source != node and float(data.get("weight", 1.0)) != 0
            ]
            max_size = min(self.max_group_size, len(incoming))
            count = 0

            for size in range(1, max_size + 1):
                for group in combinations(incoming, size):
                    if sum(weight for _, weight in group) >= quotas[node]:
                        count += 1

            scores[node] = float(count)

        self.scores = scores
        return self.scores


class PivotalIndexCentrality(_QuotaBasedCentrality):
    def compute(self) -> dict:
        graph, quotas = self._prepare()
        scores = {}

        for node in graph.nodes():
            incoming = [
                (source, float(data.get("weight", 1.0)))
                for source, _, data in graph.in_edges(node, data=True)
                if source != node and float(data.get("weight", 1.0)) != 0
            ]
            max_size = min(self.max_group_size, len(incoming))
            count = 0

            for size in range(1, max_size + 1):
                for group in combinations(incoming, size):
                    total_weight = sum(weight for _, weight in group)
                    if total_weight < quotas[node]:
                        continue

                    for _, weight in group:
                        if total_weight - weight < quotas[node]:
                            count += 1

            scores[node] = float(count)

        self.scores = scores
        return self.scores
