import numpy as np
import pytest

from vrtda.complexes import build_rips, make_torus_grid_complex
from vrtda.distances import pairwise_distances
from vrtda.persistence import persistent_homology
from vrtda.depth_persistence import (
    LayerResult,
    AttractorChain,
    _match_chains,
    betti_heatmap,
    depth_chains,
    depth_profile,
    stable_core,
    layer_barcodes,
)

A = frozenset({0, 1, 2})
A2 = frozenset({0, 1, 3})
B = frozenset({10, 11, 12})


def test_chain_consecutive_persistence():
    chains = _match_chains({0: [A], 1: [A], 2: [A]})
    assert len(chains) == 1
    c = chains[0]
    assert c.span == (0, 2)
    assert c.length == 3
    assert c.per_layer_tokens == {0: A, 1: A, 2: A}


def test_chain_two_separate_features():
    chains = _match_chains({0: [A], 1: [A], 2: [B]})
    spans = sorted(c.span for c in chains)
    assert spans == [(0, 1), (2, 2)]


def test_chain_gap_allowed_when_max_gap_big():
    # A reappears at layer 2 after a gap at layer 1 (B lives there)
    chains = _match_chains({0: [A], 1: [B], 2: [A2]}, min_overlap=0.3, max_gap=2)
    assert any(c.span == (0, 2) for c in chains)  # A linked across the gap


def test_chain_gap_blocked_when_max_gap_small():
    chains = _match_chains({0: [A], 1: [B], 2: [A2]}, min_overlap=0.3, max_gap=1)
    # gap of 2 > max_gap 1 -> A and A2 are separate
    assert all(c.length == 1 for c in chains)
    assert len(chains) == 3


def test_chain_no_match_below_threshold():
    # disjoint sets never match
    chains = _match_chains({0: [A], 1: [B]}, min_overlap=0.3)
    assert len(chains) == 2
    assert all(c.length == 1 for c in chains)


def test_stable_core_filters_short_chains():
    chains = [
        AttractorChain(per_layer_tokens={0: A, 1: A, 2: A}),
        AttractorChain(per_layer_tokens={1: B}),
    ]
    core = stable_core(chains, min_layer_fraction=0.5, total_layers=3)
    assert len(core) == 1
    assert core[0].span == (0, 2)


def test_betti_heatmap_torus_essential():
    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    total_essential = sum(1 for iv in bc.of_dim(1) if iv.is_essential)
    assert total_essential == 2
    lr = LayerResult(layer=0, complex=C, barcode=bc, nn=1.0, eps_max=float(C.values.max()),
                     labels=[str(i) for i in range(9)])
    H, fracs, layers = betti_heatmap({0: lr}, scale_fracs=[0.0, 10.0], dim=1, metric="essential")
    assert H.shape == (2, 1)
    assert layers == [0]
    assert H[1, 0] == pytest.approx(2.0)  # large scale -> both essential loops present
    assert H[0, 0] <= H[1, 0]


def test_betti_heatmap_betti_and_persistence_metrics():
    C = make_torus_grid_complex(2, (3, 3))
    bc = persistent_homology(C)
    lr = LayerResult(layer=0, complex=C, barcode=bc, nn=1.0, eps_max=float(C.values.max()),
                     labels=[str(i) for i in range(9)])
    Hb, _, _ = betti_heatmap({0: lr}, scale_fracs=[10.0], dim=1, metric="betti")
    Hp, _, _ = betti_heatmap({0: lr}, scale_fracs=[10.0], dim=1, metric="persistence")
    assert Hb[0, 0] == pytest.approx(2.0)
    assert Hp[0, 0] > 0.0


def test_depth_chains_smoke_on_real_data():
    lr = layer_barcodes(layers=[0, 64], eps_cap_frac=2.0, max_dim=2)
    assert set(lr) == {0, 64}
    chains = depth_chains(lr, min_overlap=0.2, max_gap=1)
    assert isinstance(chains, list)
    for c in chains:
        assert c.length >= 1


def _synthetic_layer_results():
    """Three 'layers' sharing one 4-cycle (tokens t0..t3) + isolated points.

    The square has edges of length 1 (present at cap 1.2) and absent diagonals
    (length sqrt(2) > 1.2), so it is a genuine persistent H_1 loop that is
    present identically in every layer."""
    base = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    extra = np.array([[10, 10], [11, 10], [10, 11], [11, 11]], float) + np.arange(4)[:, None] * 3
    X = np.vstack([base, extra])
    labels = [f"t{i}" for i in range(X.shape[0])]
    out = {}
    for L in [0, 1, 2]:
        D = pairwise_distances(X, "euclidean")
        C = build_rips(X, D, 1.2, max_dim=2)
        bc = persistent_homology(C)
        d = D.copy()
        np.fill_diagonal(d, np.inf)
        nn = float(d.min(axis=1).mean())
        out[L] = LayerResult(layer=L, complex=C, barcode=bc, nn=nn, eps_max=1.2, labels=labels)
    return out


def test_synthetic_loop_persists_across_layers():
    lr = _synthetic_layer_results()
    chains = depth_chains(lr, min_overlap=0.2, max_gap=1, top_k=10)
    assert any(c.length == 3 and c.tokens == {"t0", "t1", "t2", "t3"} for c in chains)


def test_synthetic_loop_absent_in_middle_layer_breaks_at_gap1():
    lr = _synthetic_layer_results()  # loop present at layers 0,1,2
    # spread the square's tokens far apart in layer 1 -> no edges, so no H_1 loop
    # is present there; with max_gap=1 the chain cannot bridge 0 -> 2
    base1 = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float)
    extra = np.array([[10, 10], [11, 10], [10, 11], [11, 11]], float) + np.arange(4)[:, None] * 3
    X1 = np.vstack([base1, extra])
    D = pairwise_distances(X1, "euclidean")
    C = build_rips(X1, D, 1.2, max_dim=2)
    bc = persistent_homology(C)
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    lr[1] = LayerResult(layer=1, complex=C, barcode=bc, nn=float(d.min(1).mean()),
                        eps_max=1.2, labels=[f"t{i}" for i in range(8)])
    chains = depth_chains(lr, min_overlap=0.2, max_gap=1, top_k=10)
    assert not any(c.span == (0, 2) and c.length >= 3 for c in chains)


def test_depth_profile_reports_activity():
    lr = _synthetic_layer_results()
    prof = depth_profile(lr, dim=1)
    assert set(prof) == {0, 1, 2}
    for L in prof:
        assert prof[L]["essential"] >= 1  # the square hole is essential
        assert prof[L]["total_persistence"] > 0.0
