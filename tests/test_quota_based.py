import networkx as nx

from centrality.quota_based import BundleIndexCentrality, PivotalIndexCentrality


def paper_example_graph():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([
        ("v1", "v3", 2),
        ("v1", "v5", 3),
        ("v3", "v2", 1),
        ("v4", "v3", 2),
        ("v5", "v2", 2),
    ])
    graph.add_nodes_from(["v1", "v2", "v3", "v4", "v5"])
    return graph


def test_bundle_index_matches_paper_example_q1():
    graph = paper_example_graph()
    quotas = {node: 1 for node in graph.nodes()}

    scores = BundleIndexCentrality(graph, quotas=quotas, max_group_size=2).compute()

    assert scores == {"v1": 0.0, "v2": 3.0, "v3": 3.0, "v4": 0.0, "v5": 1.0}


def test_pivotal_index_matches_paper_example_q1():
    graph = paper_example_graph()
    quotas = {node: 1 for node in graph.nodes()}

    scores = PivotalIndexCentrality(graph, quotas=quotas, max_group_size=2).compute()

    assert scores == {"v1": 0.0, "v2": 2.0, "v3": 2.0, "v4": 0.0, "v5": 1.0}


def test_bundle_index_matches_paper_example_q2():
    graph = paper_example_graph()
    quotas = {node: 2 for node in graph.nodes()}

    scores = BundleIndexCentrality(graph, quotas=quotas, max_group_size=2).compute()

    assert scores == {"v1": 0.0, "v2": 2.0, "v3": 3.0, "v4": 0.0, "v5": 1.0}


def test_pivotal_index_matches_paper_example_q2():
    graph = paper_example_graph()
    quotas = {node: 2 for node in graph.nodes()}

    scores = PivotalIndexCentrality(graph, quotas=quotas, max_group_size=2).compute()

    assert scores == {"v1": 0.0, "v2": 2.0, "v3": 2.0, "v4": 0.0, "v5": 1.0}


def test_quota_indices_convert_undirected_unweighted_graph():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("c", "b")])
    quotas = {"a": 1, "b": 2, "c": 1}

    scores = BundleIndexCentrality(graph, quotas=quotas, max_group_size=2).compute()

    assert scores["b"] == 1.0
