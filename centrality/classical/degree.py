from centrality.base import BaseCentrality
import networkx as nx

class DegreeCentrality(BaseCentrality):
    """
    Degree Centrality — нормализованное количество связей вершины.
    Формула: C(v) = deg(v) / (N - 1)
    где N — количество вершин в графе.
    """
    
    def compute(self) -> dict:
        n = self.graph.number_of_nodes()
        if n <= 1:
            return {node: 0.0 for node in self.graph.nodes()}
        
        self.scores = {
            node: self.graph.degree(node) / (n - 1)
            for node in self.graph.nodes()
        }
        return self.scores


class WeightedDegreeCentrality(BaseCentrality):
    """
    Взвешенная версия — учитывает веса рёбер (strength centrality).
    Формула: C(v) = sum(w_e) / (N - 1)
    """
    
    def compute(self) -> dict:
        n = self.graph.number_of_nodes()
        if n <= 1:
            return {node: 0.0 for node in self.graph.nodes()}
        
        self.scores = {}
        for node in self.graph.nodes():
            # Сумма весов всех рёбер вершины
            strength = sum(
                data.get('weight', 1.0) 
                for _, _, data in self.graph.edges(node, data=True)
            )
            self.scores[node] = strength / (n - 1)
        
        return self.scores