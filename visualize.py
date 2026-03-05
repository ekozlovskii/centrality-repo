import networkx as nx
import matplotlib.pyplot as plt
from centrality.classical.degree import DegreeCentrality

G = nx.karate_club_graph()

dc = DegreeCentrality(G)
dc.compute()

node_sizes = [dc.scores[node] * 3000 for node in G.nodes()]
node_colors = [dc.scores[node] for node in G.nodes()]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.YlOrRd,
    with_labels=True,
    font_size=8,
    edge_color='gray',
    alpha=0.9
)

plt.title("Degree Centrality — Karate Club Graph\nЧем больше и краснее вершина, тем она центральнее")

sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
sm.set_array(list(dc.scores.values()))
plt.colorbar(sm, ax=plt.gca(), label='Centrality Score')

plt.axis('off')
plt.tight_layout()
plt.savefig('degree_centrality.png', dpi=150)
plt.show()
print("График сохранён в degree_centrality.png")