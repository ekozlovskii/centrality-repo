# Repository of Models for Calculating Centrality Indices of Network Structures

This project is a Python library and a Streamlit application for calculating and comparing
centrality indices in network structures. It was developed as part of my bachelor's thesis in
Data Science and Business Analytics.

The main idea is to keep centrality measures as separate Python classes with a common interface.
The web application then uses these classes to calculate indices, compare results, and inspect
network data visually.

## What is implemented

### Classical indices
- Degree Centrality
- Closeness Centrality
- Betweenness Centrality
- Eigenvector Centrality

### Weighted indices
- Weighted Degree Centrality
- Weighted Closeness Centrality
- Weighted Betweenness Centrality
- Weighted Eigenvector Centrality

### Directed indices
- In-Degree Centrality
- Out-Degree Centrality
- Directed Closeness Centrality
- Directed Betweenness Centrality
- Directed Eigenvector Centrality
- PageRank

### Directed weighted indices
- Weighted In-Degree Centrality
- Weighted Out-Degree Centrality
- Directed Weighted Closeness Centrality
- Directed Weighted Betweenness Centrality
- Directed Weighted Eigenvector Centrality
- Weighted PageRank

### Quota-based indices
- Bundle Index
- Pivotal Index

These two indices use both the network structure and node attributes. In particular, each node
needs a quota value. The quota is a threshold of incoming influence for a node.

The Bundle Index counts groups of incoming neighbors whose total edge weight reaches the quota
of the target node. The Pivotal Index counts nodes that are decisive inside such groups: if a
pivotal node is removed from the group, the group no longer reaches the quota.

To keep the calculation feasible, the user sets `k`, the maximum size of a group of incoming
neighbors considered by the algorithm.

## Web application

The Streamlit app allows the user to:

- choose one of several built-in network datasets;
- upload an edge list from CSV;
- upload node attributes from CSV for quota-based indices;
- generate random graphs with parameters;
- select centrality indices by groups;
- compare centrality values in tables and distribution charts;
- visualize the graph with node size and color based on centrality;
- choose graph layouts such as Spring, Kamada-Kawai, Circular, Shell, and Spectral;
- inspect one node in detail;
- inspect nodes and edges in the Data Laboratory tab;
- compare two networks side by side;
- remove a node and see how centrality scores change;
- export calculated results as CSV.

For larger graphs the app includes a display mode with smaller nodes, more transparent edges,
optional labels, and physics stabilization. It does not solve every readability problem for dense
networks, but it makes graphs with 100-300 nodes easier to inspect.

## Data Laboratory

The Data Laboratory tab shows the data used by the current network.

The `Edges` table comes from the selected graph or uploaded edge CSV. It contains source nodes,
target nodes, and edge attributes such as weight.

The `Nodes` table comes from the nodes of the graph. If a node attributes CSV is uploaded, the
table also shows attributes such as quota, GDP, population, or any other columns provided by the
user. The selected quota column is used by Bundle Index and Pivotal Index.

## CSV formats

### Edge list

```text
source,target,weight
v1,v3,2
v1,v5,3
v3,v2,1
```

For a directed graph, select the directed graph option in the app when uploading the file.

### Node attributes

```text
node,quota,gdp,population
v1,1,100,20
v2,1,250,35
v3,1,180,28
```

The `node` column must match node identifiers in the graph. The `quota` column is required for
Bundle Index and Pivotal Index. Other columns are optional and can be used as additional node
attributes.

Example files for checking the quota-based indices are stored in `data/`:

- `quota_paper_edges.csv`
- `quota_paper_nodes_q1.csv`
- `quota_paper_nodes_q2.csv`
- `quota_demo_nodes.csv`

## Random graph models

The app currently supports four random graph models:

| Model | Main parameters | Notes |
|---|---|---|
| Erdős-Rényi | number of nodes, edge probability, seed | Basic random graph model |
| Barabási-Albert | number of nodes, edges per new node, seed | Preferential attachment and scale-free structure |
| Watts-Strogatz | number of nodes, nearest neighbors, rewiring probability, seed | Small-world network model |
| Scale-Free Directed | number of nodes, attachment probabilities, seed | Directed scale-free graph |

## Supported graph types

| Index group | Undirected | Weighted | Directed | Directed weighted | Requires node attributes |
|---|---:|---:|---:|---:|---:|
| Classical | yes | partly | no | no | no |
| Weighted | yes | yes | partly | partly | no |
| Directed | no | no | yes | partly | no |
| Directed weighted | no | no | yes | yes | no |
| Quota-based | yes | yes | yes | yes | yes |

For quota-based indices, undirected graphs are converted to reciprocal directed edges when needed.
If an edge has no weight, the default weight is 1.

## Installation

```bash
git clone https://github.com/ekozlovskii/centrality-repo.git
cd centrality-repo
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Library usage example

```python
import networkx as nx

from centrality.classical.degree import DegreeCentrality
from centrality.classical.betweenness import BetweennessCentrality
from utils.comparator import CentralityComparator

G = nx.karate_club_graph()

degree = DegreeCentrality(G)
degree.compute()

betweenness = BetweennessCentrality(G)
betweenness.compute()

comp = CentralityComparator()
comp.add("Degree", degree)
comp.add("Betweenness", betweenness)

print(comp.top_nodes(5))
print(comp.correlation_matrix())
```

## Quota-based usage example

```python
import networkx as nx

from centrality.quota_based import BundleIndexCentrality, PivotalIndexCentrality

G = nx.DiGraph()
G.add_weighted_edges_from([
    ("v1", "v3", 2),
    ("v1", "v5", 3),
    ("v3", "v2", 1),
    ("v4", "v3", 2),
    ("v5", "v2", 2),
])

quotas = {"v1": 1, "v2": 1, "v3": 1, "v4": 1, "v5": 1}

bundle = BundleIndexCentrality(G, quotas=quotas, max_group_size=2)
pivotal = PivotalIndexCentrality(G, quotas=quotas, max_group_size=2)

print(bundle.compute())
print(pivotal.compute())
```

## Project structure

```text
centrality_repo/
├── app.py
├── centrality/
│   ├── base.py
│   ├── classical/
│   ├── directed/
│   ├── quota_based/
│   └── weighted/
├── data/
├── tests/
├── utils/
└── requirements.txt
```

## Testing

```bash
python3 -m pytest
```

The test suite checks classical, weighted, directed, directed weighted, and quota-based
implementations. Some tests compare results with NetworkX, and the quota-based tests reproduce
examples from the referenced paper.

## Current limitations and next steps

- Very dense graphs can still be hard to read visually, even with large graph mode.
- Bundle Index and Pivotal Index are implemented for direct incoming influence.
- Node attributes are currently used mainly for quotas, but they can also support filtering and
  visualization later.

## Author

Evgeniy Kozlovskii  
Bachelor's Programme Data Science and Business Analytics  
HSE University, Faculty of Computer Science
