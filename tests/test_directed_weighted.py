import pytest
import networkx as nx

from centrality.directed.directed_betweenness import DirectedBetweennessCentrality
from centrality.directed.directed_weighted_betweenness import DirectedWeightedBetweennessCentrality
from centrality.directed.directed_weighted_closeness import DirectedWeightedClosenessCentrality
from centrality.directed.directed_degree import (
    WeightedInDegreeCentrality,
    WeightedOutDegreeCentrality,
)
from centrality.directed.directed_eigenvector import (
    DirectedEigenvectorCentrality,
    DirectedWeightedEigenvectorCentrality,
)
from centrality.weighted.weighted_eigenvector import WeightedEigenvectorCentrality
from centrality.directed.weighted_pagerank import WeightedPageRankCentrality


def test_directed_betweenness_matches_networkx():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])

    scores = DirectedBetweennessCentrality(graph).compute()
    expected = nx.betweenness_centrality(graph, normalized=True)

    assert scores == pytest.approx(expected)


def test_weighted_directed_betweenness_matches_networkx_with_inverse_distance():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        (0, 1, 2.0),
        (1, 3, 2.0),
        (0, 2, 1.0),
        (2, 3, 1.0),
    ])

    graph_distance = graph.copy()
    for u, v, data in graph_distance.edges(data=True):
        data["distance"] = 1.0 / data["weight"]

    scores = DirectedWeightedBetweennessCentrality(graph).compute()
    expected = nx.betweenness_centrality(
        graph_distance,
        normalized=True,
        weight="distance",
    )

    assert scores == pytest.approx(expected)


def test_weighted_directed_closeness_matches_networkx_default_inward_mode():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        (0, 1, 2.0),
        (1, 2, 4.0),
        (0, 2, 1.0),
    ])

    graph_distance = graph.copy()
    for u, v, data in graph_distance.edges(data=True):
        data["distance"] = 1.0 / data["weight"]

    scores = DirectedWeightedClosenessCentrality(graph).compute()
    expected = nx.closeness_centrality(graph_distance, distance="distance")

    assert scores == pytest.approx(expected)


def test_weighted_in_and_out_degree_sum_edge_weights_by_direction():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        ("a", "c", 10.0),
        ("d", "c", 5.0),
        ("c", "b", 2.0),
        ("c", "e", 1.0),
    ])

    in_scores = WeightedInDegreeCentrality(graph).compute()
    out_scores = WeightedOutDegreeCentrality(graph).compute()

    assert in_scores["c"] == pytest.approx(15.0 / 4)
    assert out_scores["c"] == pytest.approx(3.0 / 4)


def test_directed_eigenvector_cycle_scores_are_equal():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    scores = DirectedEigenvectorCentrality(graph).compute()

    assert all(score == pytest.approx(next(iter(scores.values()))) for score in scores.values())


def test_weighted_eigenvector_star_center_has_highest_score():
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        (0, 1, 5.0),
        (0, 2, 2.0),
        (0, 3, 1.0),
    ])

    scores = WeightedEigenvectorCentrality(graph).compute()
    expected = nx.eigenvector_centrality(graph, weight="weight")

    assert scores == pytest.approx(expected, abs=1e-5)


def test_directed_weighted_eigenvector_prefers_node_with_strong_incoming_links():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        (0, 2, 5.0),
        (1, 2, 5.0),
        (2, 0, 1.0),
        (2, 1, 1.0),
    ])

    scores = DirectedWeightedEigenvectorCentrality(graph).compute()

    assert scores[2] == max(scores.values())


def test_weighted_pagerank_matches_networkx():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        (0, 1, 3.0),
        (0, 2, 1.0),
        (1, 2, 2.0),
        (2, 0, 1.0),
    ])

    scores = WeightedPageRankCentrality(graph).compute()

    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores[2] > scores[0] > scores[1]
