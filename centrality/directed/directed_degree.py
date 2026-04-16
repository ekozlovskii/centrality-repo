from centrality.base import BaseCentrality
import networkx as nx

class InDegreeCentrality(BaseCentrality):


    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")
        
        n = self.graph.number_of_nodes()
        self.scores = {
            node: self.graph.in_degree(node) / (n - 1)
            for node in self.graph.nodes()
        }
        return self.scores


class OutDegreeCentrality(BaseCentrality):


    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")
        
        n = self.graph.number_of_nodes()
        self.scores = {
            node: self.graph.out_degree(node) / (n - 1)
            for node in self.graph.nodes()
        }
        return self.scores