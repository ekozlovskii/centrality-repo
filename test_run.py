import networkx as nx
from centrality.classical.degree import DegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality
from centrality.classical.eigenvector import EigenvectorCentrality
from utils.comparator import CentralityComparator

G = nx.karate_club_graph()

dc = DegreeCentrality(G)
cc = ClosenessCentrality(G)
bc = BetweennessCentrality(G)
ec = EigenvectorCentrality(G)

comp = CentralityComparator()
comp.add("Degree", dc)
comp.add("Closeness", cc)
comp.add("Betweenness", bc)
comp.add("Eigenvector", ec)

print("=== Top 10 Nodes ===")
comp.top_nodes(10)

comp.correlation_matrix()
comp.plot_comparison(10)