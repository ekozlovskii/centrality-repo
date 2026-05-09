# Repository of Models for Calculating Centrality Indices of Network Structures

This project is a Python library and a Streamlit application for calculating and comparing
centrality indices in network structures. It was developed as part of my bachelor's thesis in
Data Science and Business Analytics.

The main idea is simple: the centrality measures are implemented as separate classes with a
common interface, and the web application uses these classes to make the calculations easier to
explore visually.

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

## Web application

The Streamlit app allows the user to:

- choose one of several built-in network datasets;
- upload an edge list from CSV;
- generate random graphs with parameters;
- select centrality indices by groups;
- compare centrality values in tables and bar charts;
- visualize the graph with node size and color based on centrality;
- inspect one node in detail;
- compare two networks side by side;
- remove a node and see how centrality scores change;
- export calculated results as CSV.

For larger graphs the app includes a simple display mode with smaller nodes, more transparent
edges, optional labels, and automatic physics stabilization. This does not solve every readability
problem for dense networks, but it makes graphs with 100-200 nodes more usable than the default
visualization.

## Random graph models

The app currently supports four random graph models:

| Model | Main parameters | Notes |
|---|---|---|
| Erdős-Rényi | number of nodes, edge probability, seed | Basic random graph model |
| Barabási-Albert | number of nodes, edges per new node, seed | Preferential attachment / scale-free structure |
| Watts-Strogatz | number of nodes, nearest neighbors, rewiring probability, seed | Small-world network model |
| Scale-Free Directed | number of nodes, attachment probabilities, seed | Directed scale-free graph |

## Supported graph types

| Index group | Undirected | Weighted | Directed | Directed weighted |
|---|---:|---:|---:|---:|
| Classical | yes | partly | no | no |
| Weighted | yes | yes | partly | partly |
| Directed | no | no | yes | partly |
| Directed weighted | no | no | yes | yes |

Some weighted indices can technically be computed on directed graphs, but for interpretation it is
usually better to use the explicit directed weighted versions when the network has direction.

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

## Project structure

```text
centrality_repo/
├── app.py
├── centrality/
│   ├── base.py
│   ├── classical/
│   ├── weighted/
│   └── directed/
├── tests/
├── utils/
└── requirements.txt
```

## Testing

```bash
python3 -m pytest tests/ -v
```

At the moment the test suite checks the main classical, weighted, directed, and directed weighted
implementations on small graphs with known behavior or against NetworkX reference functions.

## Current limitations and next steps

- Multilayer centrality is planned but not implemented yet.
- Very dense graphs can still be hard to read visually, even with large graph mode.
- New centrality measures with node attributes or quotas are planned for the next stage of the thesis.

## Author

Evgeniy Kozlovskii  
Bachelor's Programme Data Science and Business Analytics  
HSE University, Faculty of Computer Science
