# tests/test_classical.py
import pytest
import networkx as nx
from centrality.classical.degree import DegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality
from centrality.classical.eigenvector import EigenvectorCentrality


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def star_graph():
    """Star graph: node 0 is the center connected to all others."""
    return nx.star_graph(5)  # 6 nodes total

@pytest.fixture
def complete_graph():
    """Complete graph: every node connected to every other."""
    return nx.complete_graph(5)

@pytest.fixture
def path_graph():
    """Path graph: 0 - 1 - 2 - 3 - 4"""
    return nx.path_graph(5)

@pytest.fixture
def karate_graph():
    return nx.karate_club_graph()


# ── Degree Centrality ────────────────────────────────────────────────────────

class TestDegreeCentrality:

    def test_star_center_has_max_score(self, star_graph):
        """Center of star should have degree = 1.0"""
        dc = DegreeCentrality(star_graph)
        dc.compute()
        assert dc.scores[0] == pytest.approx(1.0)

    def test_star_leaves_have_min_score(self, star_graph):
        """Leaves of star should have degree = 1/5 = 0.2"""
        dc = DegreeCentrality(star_graph)
        dc.compute()
        for node in range(1, 6):
            assert dc.scores[node] == pytest.approx(0.2)

    def test_complete_graph_all_equal(self, complete_graph):
        """In complete graph all nodes should have equal centrality."""
        dc = DegreeCentrality(complete_graph)
        dc.compute()
        scores = list(dc.scores.values())
        assert all(s == pytest.approx(scores[0]) for s in scores)

    def test_scores_between_0_and_1(self, karate_graph):
        """All scores must be in [0, 1]."""
        dc = DegreeCentrality(karate_graph)
        dc.compute()
        for score in dc.scores.values():
            assert 0.0 <= score <= 1.0

    def test_top_nodes_returns_correct_count(self, karate_graph):
        dc = DegreeCentrality(karate_graph)
        dc.compute()
        assert len(dc.top_nodes(5)) == 5


# ── Closeness Centrality ─────────────────────────────────────────────────────

class TestClosenessCentrality:

    def test_center_of_star_has_highest_closeness(self, star_graph):
        """Center should have higher closeness than leaves."""
        cc = ClosenessCentrality(star_graph)
        cc.compute()
        center_score = cc.scores[0]
        for node in range(1, 6):
            assert center_score > cc.scores[node]

    def test_complete_graph_all_equal(self, complete_graph):
        """In complete graph all nodes are equally close."""
        cc = ClosenessCentrality(complete_graph)
        cc.compute()
        scores = list(cc.scores.values())
        assert all(s == pytest.approx(scores[0]) for s in scores)

    def test_path_endpoints_have_lowest_closeness(self, path_graph):
        """Endpoints of path (0 and 4) should have lower closeness than center (2)."""
        cc = ClosenessCentrality(path_graph)
        cc.compute()
        assert cc.scores[2] > cc.scores[0]
        assert cc.scores[2] > cc.scores[4]

    def test_scores_positive(self, karate_graph):
        cc = ClosenessCentrality(karate_graph)
        cc.compute()
        for score in cc.scores.values():
            assert score >= 0.0


# ── Betweenness Centrality ───────────────────────────────────────────────────

class TestBetweennessCentrality:

    def test_star_center_has_max_betweenness(self, star_graph):
        """All paths between leaves go through center."""
        bc = BetweennessCentrality(star_graph)
        bc.compute()
        center_score = bc.scores[0]
        for node in range(1, 6):
            assert center_score > bc.scores[node]

    def test_star_leaves_have_zero_betweenness(self, star_graph):
        """Leaves are never on shortest paths between other nodes."""
        bc = BetweennessCentrality(star_graph)
        bc.compute()
        for node in range(1, 6):
            assert bc.scores[node] == pytest.approx(0.0)

    def test_path_center_has_highest_betweenness(self, path_graph):
        """Center of path (node 2) has highest betweenness."""
        bc = BetweennessCentrality(path_graph)
        bc.compute()
        assert bc.scores[2] == max(bc.scores.values())

    def test_complete_graph_all_zero(self, complete_graph):
        """In complete graph no node is a bottleneck."""
        bc = BetweennessCentrality(complete_graph)
        bc.compute()
        for score in bc.scores.values():
            assert score == pytest.approx(0.0)


# ── Eigenvector Centrality ───────────────────────────────────────────────────

class TestEigenvectorCentrality:

    def test_star_center_has_max_eigenvector(self, star_graph):
        ec = EigenvectorCentrality(star_graph)
        ec.compute()
        assert ec.scores[0] == max(ec.scores.values())

    def test_complete_graph_all_equal(self, complete_graph):
        ec = EigenvectorCentrality(complete_graph)
        ec.compute()
        scores = list(ec.scores.values())
        assert all(s == pytest.approx(scores[0], abs=1e-4) for s in scores)

    def test_scores_positive(self, karate_graph):
        ec = EigenvectorCentrality(karate_graph)
        ec.compute()
        for score in ec.scores.values():
            assert score >= 0.0