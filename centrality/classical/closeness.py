from centrality.base import BaseCentrality
import networkx as nx

class ClosenessCentrality(BaseCentrality):


    def compute(self) -> dict:
        self.scores = {}
        n = self.graph.number_of_nodes()

        for node in self.graph.nodes():
            # Считаем кратчайшие пути от node до всех остальных
            lengths = nx.single_source_shortest_path_length(self.graph, node)
            
            total = sum(lengths.values())
            reachable = len(lengths) - 1  # минус сама вершина

            if reachable == 0 or total == 0:
                self.scores[node] = 0.0
            else:
                # Нормализация на случай несвязного графа
                self.scores[node] = (reachable / (n - 1)) * (reachable / total)

        return self.scores