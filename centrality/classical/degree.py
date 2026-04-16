from centrality.base import BaseCentrality
import networkx as nx

class DegreeCentrality(BaseCentrality):

    
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