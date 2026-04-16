from abc import ABC, abstractmethod
import networkx as nx
import numpy as np

class BaseCentrality(ABC):
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.scores = {}
    
    @abstractmethod
    def compute(self) -> dict:
        pass
    
    def normalize(self, scores: dict) -> dict:
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return {k: 0.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) 
                for k, v in scores.items()}
    
    def top_nodes(self, n: int = 5) -> list:
        if not self.scores:
            self.compute()
        return sorted(self.scores.items(), 
                      key=lambda x: -x[1])[:n]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(nodes={self.graph.number_of_nodes()})"