# Repository of Models for Calculating Centrality Indices of Network Structures

A Python library for computing classical and novel centrality indices
for weighted, directed, and multilayer networks, with an interactive web application.

## Description

This repository provides a unified, extensible toolkit for analyzing network structures
using centrality measures. The library is built around an abstract base class architecture,
ensuring a consistent interface across all implementations and making it straightforward
to add new indices.

## Implemented Indices

### Classical
- **Degree Centrality** — normalized number of direct connections
- **Closeness Centrality** — centrality based on average shortest path length
- **Betweenness Centrality** — centrality based on shortest path traversal
- **Eigenvector Centrality** — importance based on neighbor importance (power iteration)

### Weighted
- **Weighted Degree Centrality** — sum of edge weights (strength centrality)
- **Weighted Closeness Centrality** — closeness via Dijkstra with 1/weight distances
- **Weighted Betweenness Centrality** — betweenness on weighted shortest paths

### Directed
- **In-Degree Centrality** — normalized number of incoming edges
- **Out-Degree Centrality** — normalized number of outgoing edges
- **PageRank** — Google's random walk algorithm with damping factor 0.85

### In Development
- Novel indices accounting for node attributes and group interactions
- Multilayer network centrality

## Web Application

An interactive web application is available via Streamlit with the following features:

- 6 built-in network datasets (Karate Club, Les Misérables, Florentine Families, and more)
- CSV upload with automatic edge weight detection
- Interactive network graph (Pyvis) with node size/color reflecting centrality
- Node filter slider and node highlight search
- Interactive bar chart comparing multiple indices (Plotly)
- Pearson correlation heatmap
- Node Inspector with radar chart and percentile statistics
- Compare two networks side by side
- Node Impact Analysis — remove a node and see how scores change
- Export results as CSV

## Installation

```bash
git clone https://github.com/ekozlovskii/centrality-repo.git
cd centrality-repo
pip install -r requirements.txt
```

## Quick Start

```python
import networkx as nx
from centrality.classical.degree import DegreeCentrality
from centrality.classical.betweenness import BetweennessCentrality
from utils.comparator import CentralityComparator

G = nx.karate_club_graph()

comp = CentralityComparator()
comp.add("Degree", DegreeCentrality(G))
comp.add("Betweenness", BetweennessCentrality(G))

comp.top_nodes(5)
comp.correlation_matrix()
```

## Run Web Application

```bash
streamlit run app.py
```

## Project Structure

```
centrality_repo/
├── centrality/
│   ├── base.py              # Abstract base class
│   ├── classical/           # Degree, Closeness, Betweenness, Eigenvector
│   ├── weighted/            # Weighted Closeness, Weighted Betweenness
│   ├── directed/            # In-Degree, Out-Degree, PageRank
│   └── multilayer/          # Planned: multilayer indices
├── utils/
│   └── comparator.py        # Multi-index comparison and correlation
├── tests/
│   └── test_classical.py    # 16 automated pytest tests
├── app.py                   # Streamlit web application
└── requirements.txt
```

## Testing

```bash
python3 -m pytest tests/ -v
```

All 16 tests pass.

## Tech Stack

- **Python 3.9+**
- **NetworkX** — graph data structures
- **NumPy** — matrix operations
- **Streamlit** — web interface
- **Plotly** — interactive charts
- **Pyvis** — interactive network visualization
- **pytest** — automated testing

## Author

Evgeniy Kozlovskii — Data Science and Business Analytics, Bachelor's Thesis 2025–2026
HSE University, Faculty of Computer Science
Repository: [github.com/ekozlovskii/centrality-repo](https://github.com/ekozlovskii/centrality-repo)