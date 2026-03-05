import networkx as nx
from centrality.classical.degree import DegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality
from centrality.classical.eigenvector import EigenvectorCentrality

G = nx.karate_club_graph()

dc = DegreeCentrality(G)
dc.compute()

cc = ClosenessCentrality(G)
cc.compute()

bc = BetweennessCentrality(G)
bc.compute()

ec = EigenvectorCentrality(G)
ec.compute()

print("=== Comparison of Four Centrality Indices ===")
print(f"{'Node':<8} {'Degree':<12} {'Closeness':<12} {'Betweenness':<14} {'Eigenvector':<12}")
print("-" * 58)

for node, score in dc.top_nodes(10):
    print(f"{node:<8} {score:<12.4f} {cc.scores[node]:<12.4f} {bc.scores[node]:<14.4f} {ec.scores[node]:<12.4f}")