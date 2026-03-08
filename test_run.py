import networkx as nx
import random
from centrality.classical.degree import DegreeCentrality, WeightedDegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality
from centrality.weighted.weighted_closeness import WeightedClosenessCentrality
from centrality.weighted.weighted_betweenness import WeightedBetweennessCentrality

G = nx.karate_club_graph()

random.seed(42)
for u, v in G.edges():
    G[u][v]['weight'] = random.uniform(0.1, 5.0)

# Классические
bc = BetweennessCentrality(G)
bc.compute()

# Взвешенные
wdc = WeightedDegreeCentrality(G)
wdc.compute()

wcc = WeightedClosenessCentrality(G)
wcc.compute()

wbc = WeightedBetweennessCentrality(G)
wbc.compute()

print("=== Weighted Indices Comparison ===")
print(f"{'Node':<8} {'W.Degree':<12} {'W.Closeness':<14} {'W.Betweenness':<14}")
print("-" * 48)

for node, score in wbc.top_nodes(10):
    print(f"{node:<8} {wdc.scores[node]:<12.4f} {wcc.scores[node]:<14.4f} {score:<14.4f}")