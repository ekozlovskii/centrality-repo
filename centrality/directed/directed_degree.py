from centrality.base import BaseCentrality
import networkx as nx

class InDegreeCentrality(BaseCentrality):
    """
    In-Degree Centrality — normalized number of incoming edges.
    
    Formula: C(v) = in_degree(v) / (N - 1)
    
    Idea: how many nodes point TO this node.
    High in-degree = popular, referenced, authoritative.
    Example: in citation networks — how many papers cite this paper.
    """

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
    """
    Out-Degree Centrality — normalized number of outgoing edges.
    
    Formula: C(v) = out_degree(v) / (N - 1)
    
    Idea: how many nodes this node points TO.
    High out-degree = active, influential, hub-like.
    Example: in social networks — how many people you follow.
    """

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")
        
        n = self.graph.number_of_nodes()
        self.scores = {
            node: self.graph.out_degree(node) / (n - 1)
            for node in self.graph.nodes()
        }
        return self.scores