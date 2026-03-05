import networkx as nx
from centrality.classical.degree import DegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality

G = nx.karate_club_graph()

dc = DegreeCentrality(G)
dc.compute()

cc = ClosenessCentrality(G)
cc.compute()

bc = BetweennessCentrality(G)
bc.compute()

print("=== Сравнение трёх индексов ===")
print(f"{'Вершина':<10} {'Degree':<12} {'Closeness':<12} {'Betweenness':<12}")
print("-" * 46)

for node, score in dc.top_nodes(10):
    print(f"{node:<10} {score:<12.4f} {cc.scores[node]:<12.4f} {bc.scores[node]:<12.4f}")