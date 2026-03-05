# Repository of Models for Calculating Centrality Indices of Network Structures

A Python library for computing classical and novel centrality indices
for weighted, directed, and multilayer networks.

## Description

This repository provides a unified toolkit for analyzing network structures
using centrality measures. The library supports various network types and
is designed to be easily extensible with new mathematical models.

## Implemented Indices

### Classical
- **Degree Centrality** — centrality based on the number of direct connections
- **Closeness Centrality** — centrality based on average shortest path length
- **Betweenness Centrality** — centrality based on shortest path traversal

### In Development
- Weighted Degree Centrality
- Directed Centrality Indices
- Multilayer Network Centrality

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

G = nx.karate_club_graph()
dc = DegreeCentrality(G)
dc.compute()
print(dc.top_nodes(5))
```

## Project Structure
```
centrality/
├── base.py              # Abstract base class
├── classical/           # Classical centrality indices
├── weighted/            # Indices for weighted networks
├── directed/            # Indices for directed networks
└── multilayer/          # Indices for multilayer networks
```

## Tech Stack

- Python 3.9+
- NetworkX
- NumPy
- Matplotlib

## Author

Evgeniy Kozlovskii — Applied Data Analysis, Bachelor's Thesis 2025