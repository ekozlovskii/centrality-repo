from centrality.base import BaseCentrality
import networkx as nx

class BetweennessCentrality(BaseCentrality):
    """
    Betweenness Centrality — как часто вершина лежит на кратчайшем
    пути между двумя другими вершинами.
    
    Формула: C(v) = sum( sigma(s,t|v) / sigma(s,t) ) для всех s != v != t
    где sigma(s,t) — количество кратчайших путей между s и t,
    sigma(s,t|v) — из них те что проходят через v.
    
    Идея: вершина центральна если она "мост" или "посредник" в сети.
    """

    def compute(self) -> dict:
        n = self.graph.number_of_nodes()
        self.scores = {node: 0.0 for node in self.graph.nodes()}

        for s in self.graph.nodes():
            # Поиск всех кратчайших путей от s
            paths = nx.single_source_shortest_path(self.graph, s)
            
            for t in self.graph.nodes():
                if s == t or t not in paths:
                    continue
                    
                path = paths[t]
                # Все промежуточные вершины на пути s -> t
                for v in path[1:-1]:
                    self.scores[v] += 1

        # Нормализация
        norm = (n - 1) * (n - 2)
        if norm > 0:
            self.scores = {k: v / norm for k, v in self.scores.items()}

        return self.scores