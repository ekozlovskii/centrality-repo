import heapq
from centrality.base import BaseCentrality


class DirectedWeightedBetweennessCentrality(BaseCentrality):

    def _distance(self, v, w):
        edge_data = self.graph.get_edge_data(v, w, default={})
        if self.graph.is_multigraph():
            weights = [data.get("weight", 1.0) for data in edge_data.values()]
            weight = max(weights) if weights else 1.0
        else:
            weight = edge_data.get("weight", 1.0)
        return 1.0 / weight if weight > 0 else float("inf")

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        nodes = list(self.graph.nodes())
        n = len(nodes)
        self.scores = {node: 0.0 for node in nodes}

        for source in nodes:
            stack = []
            predecessors = {node: [] for node in nodes}
            sigma = dict.fromkeys(nodes, 0.0)
            distance = dict.fromkeys(nodes, float("inf"))

            sigma[source] = 1.0
            distance[source] = 0.0
            heap = [(0.0, source)]

            while heap:
                dist_v, v = heapq.heappop(heap)
                if dist_v > distance[v]:
                    continue
                stack.append(v)

                for w in self.graph.successors(v):
                    vw_distance = dist_v + self._distance(v, w)
                    if vw_distance < distance[w]:
                        distance[w] = vw_distance
                        heapq.heappush(heap, (vw_distance, w))
                        sigma[w] = sigma[v]
                        predecessors[w] = [v]
                    elif abs(vw_distance - distance[w]) <= 1e-12:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            delta = dict.fromkeys(nodes, 0.0)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] != 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != source:
                    self.scores[w] += delta[w]

        norm = (n - 1) * (n - 2)
        if norm > 0:
            self.scores = {node: score / norm for node, score in self.scores.items()}

        return self.scores
