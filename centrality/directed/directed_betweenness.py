from collections import deque
from centrality.base import BaseCentrality


class DirectedBetweennessCentrality(BaseCentrality):

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
            distance = dict.fromkeys(nodes, -1)

            sigma[source] = 1.0
            distance[source] = 0
            queue = deque([source])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in self.graph.successors(v):
                    if distance[w] < 0:
                        queue.append(w)
                        distance[w] = distance[v] + 1
                    if distance[w] == distance[v] + 1:
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
