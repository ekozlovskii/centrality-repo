import networkx as nx
from centrality.directed.directed_degree import InDegreeCentrality, OutDegreeCentrality
from centrality.directed.pagerank import PageRankCentrality

# Создаём направленный граф
DG = nx.scale_free_graph(50, seed=42)

idc = InDegreeCentrality(DG)
idc.compute()

odc = OutDegreeCentrality(DG)
odc.compute()

pr = PageRankCentrality(DG)
pr.compute()

print("=== Directed Indices Comparison ===")
print(f"{'Node':<8} {'In-Degree':<12} {'Out-Degree':<12} {'PageRank':<12}")
print("-" * 44)

for node, score in pr.top_nodes(10):
    print(f"{node:<8} {idc.scores[node]:<12.4f} {odc.scores[node]:<12.4f} {score:<12.4f}")