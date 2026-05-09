from centrality.base import BaseCentrality


def _edge_weight(data):
    return data.get("weight", 1.0)


class InDegreeCentrality(BaseCentrality):


    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")
        
        n = self.graph.number_of_nodes()
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

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
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

        self.scores = {
            node: self.graph.out_degree(node) / (n - 1)
            for node in self.graph.nodes()
        }
        return self.scores


class WeightedInDegreeCentrality(BaseCentrality):

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        n = self.graph.number_of_nodes()
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

        self.scores = {}
        for node in self.graph.nodes():
            if self.graph.is_multigraph():
                strength = sum(
                    _edge_weight(data)
                    for _, _, _, data in self.graph.in_edges(node, keys=True, data=True)
                )
            else:
                strength = sum(
                    _edge_weight(data)
                    for _, _, data in self.graph.in_edges(node, data=True)
                )
            self.scores[node] = strength / (n - 1)

        return self.scores


class WeightedOutDegreeCentrality(BaseCentrality):

    def compute(self) -> dict:
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed. Use nx.DiGraph().")

        n = self.graph.number_of_nodes()
        if n <= 1:
            self.scores = {node: 0.0 for node in self.graph.nodes()}
            return self.scores

        self.scores = {}
        for node in self.graph.nodes():
            if self.graph.is_multigraph():
                strength = sum(
                    _edge_weight(data)
                    for _, _, _, data in self.graph.out_edges(node, keys=True, data=True)
                )
            else:
                strength = sum(
                    _edge_weight(data)
                    for _, _, data in self.graph.out_edges(node, data=True)
                )
            self.scores[node] = strength / (n - 1)

        return self.scores
