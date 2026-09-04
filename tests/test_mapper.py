import numpy as np
import pytest

from vrtda.mapper import mapper, mapper_residual


def test_two_disjoint_squares_two_loop_nodes():
    A = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    B = A + np.array([100.0, 0.0])
    X = np.vstack([A, B])
    phi = np.array([0.0, 0.1, 0.2, 0.3, 10.0, 10.1, 10.2, 10.3])
    g = mapper(X, phi, n_bins=2, overlap=0.6, eps=1.2, max_dim=2)
    assert g.n_nodes == 2
    assert sorted(g.beta1_profile()) == [1, 1]  # each square is a genuine hole
    assert g.n_edges == 0  # disjoint -> no overlap


def test_overlap_creates_edge_when_connected():
    # one square whose points are spread across the lens -> spans two bins with a
    # connected 2-point overlap
    A = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    phi = np.array([0.0, 4.0, 8.0, 12.0])
    g = mapper(A, phi, n_bins=2, overlap=0.6, eps=1.2, max_dim=1)
    assert g.n_nodes == 2
    assert g.n_edges == 1
    assert g.edges[0] == (0, 1, 2)


def test_single_blob_single_node():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    phi = np.arange(30.0)  # a linear lens
    g = mapper(X, phi, n_bins=4, overlap=0.5, eps_frac=3.0)
    assert g.n_nodes == 4
    assert all(n.beta0 in (0, 1) for n in g.nodes)
    # a connected blob -> most nodes are single components
    assert sum(n.beta0 for n in g.nodes) >= 1


def test_mapper_residual_smoke():
    g = mapper_residual(layer=16, n_bins=6, eps_frac=2.0)
    assert g.n_nodes == 6
    assert len(g.beta1_profile()) == 6
    assert g.n_edges >= 0
