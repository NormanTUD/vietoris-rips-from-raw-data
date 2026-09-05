"""Exact Betti-number validation for topologically known shapes.

The bug these tests guard: a Vietoris-Rips complex capped at too low a dimension
(`max_dim`) leaves every "(d+1)-shell" -- the boundary of a (d+1)-simplex -- as a
persistent d-cycle. Capping a 2-torus at triangles (max_dim=2) turned the single true
void into ~256 spurious ones (beta_2 = 257 instead of 1). The fix is to build the
complex up to dimension k+1 for a shape of intrinsic dimension k, which fills those
shells so that beta_0..beta_k are the *true* Betti numbers.

Rule enforced here:  for intrinsic dimension k, use  max_dim = k + 1,  and then
beta[0..k] must equal the known topology EXACTLY (no spurious excess in any H_d).

For k >= 3 the Rips complex hits a combinatorial wall (5-cliques explode), so those
shapes are validated with the EXACT cell complexes (make_torus_grid_complex /
make_sphere_complex), which are the ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from vrtda import pairwise_distances, build_rips, persistent_homology, betti_at
from vrtda import generators as G
from vrtda.complexes import make_torus_grid_complex, make_sphere_complex
from vrtda.beartype_guard import beartype_module


def _nn(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def _rip_betti(X: np.ndarray, frac: float = 1.6, max_dim: int = 3) -> list[int]:
    D = pairwise_distances(X)
    eps = frac * _nn(D)
    C = build_rips(X, D, eps, max_dim=max_dim)
    return persistent_homology(C).betti_at(eps)


def _pad(X: np.ndarray, ambient: int) -> np.ndarray:
    """Embed a point cloud in R^ambient by zero-padding (preserves all distances,
    so the Rips complex -- and hence the Betti numbers -- is unchanged)."""
    if X.shape[1] >= ambient:
        return X
    return np.hstack([X, np.zeros((X.shape[0], ambient - X.shape[1]))])


def _pad_to(b: list[int], n: int) -> list[int]:
    """First n Betti numbers, zero-padded to length n (ignores spurious H_{>=n})."""
    b = list(b)
    return (b + [0] * (n - len(b)))[:n]


def _circle(ambient: int) -> np.ndarray:
    return _pad(G.circle_grid(24), ambient)


def _torus2(ambient: int) -> np.ndarray:
    return _pad(G.product_torus_grid(2, 8), ambient)


# --------------------------------------------------------------------------- #
# Rips: 1-torus and 2-torus across ambient dimensions (1d / 2d / 3d / 10d / 50d)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ambient", [2, 3, 10, 50])
def test_circle_betti_across_ambient_dims(ambient: int) -> None:
    b = _rip_betti(_circle(ambient), frac=1.6, max_dim=2)
    assert _pad_to(b, 2) == [1, 1]


@pytest.mark.parametrize("ambient", [4, 10, 50])
def test_torus2_betti_across_ambient_dims(ambient: int) -> None:
    b = _rip_betti(_torus2(ambient), frac=1.6, max_dim=3)
    assert _pad_to(b, 3) == [1, 2, 1]


# --------------------------------------------------------------------------- #
# The regression itself: max_dim=k (old default) inflates the top Betti number,
# max_dim=k+1 (the fix) removes it.
# --------------------------------------------------------------------------- #
def test_torus2_old_default_maxdim2_has_spurious_h2() -> None:
    b = _rip_betti(G.product_torus_grid(2, 16), frac=1.6, max_dim=2)
    assert _pad_to(b, 3)[:2] == [1, 2]
    assert b[2] > 1, f"expected spurious excess H2 with max_dim=2, got beta={b}"


def test_torus2_fixed_maxdim3_has_no_spurious_h2() -> None:
    b = _rip_betti(G.product_torus_grid(2, 16), frac=1.6, max_dim=3)
    assert _pad_to(b, 3) == [1, 2, 1]


def test_rips_torus3_low_dims_exact() -> None:
    # Low dimensions are exact under Rips (max_dim=k=3); the top H3 needs the exact
    # complex below because 4-simplices are combinatorially infeasible at this scale.
    b = _rip_betti(G.product_torus_grid(3, 5), frac=1.6, max_dim=3)
    assert _pad_to(b, 3) == [1, 3, 3]


# --------------------------------------------------------------------------- #
# Exact cell complexes: ground-truth Betti numbers for higher intrinsic dimensions
# --------------------------------------------------------------------------- #
def test_exact_torus2() -> None:
    C = make_torus_grid_complex(2, (5, 5))
    assert betti_at(C, float(C.values.max())) == [1, 2, 1]


def test_exact_torus3() -> None:
    C = make_torus_grid_complex(3, (4, 4, 4))
    assert betti_at(C, float(C.values.max())) == [1, 3, 3, 1]


def test_exact_torus4() -> None:
    C = make_torus_grid_complex(4, (3, 3, 3, 3))
    assert betti_at(C, float(C.values.max())) == [1, 4, 6, 4, 1]


def test_exact_sphere2() -> None:
    C = make_sphere_complex(2)
    assert betti_at(C, float(C.values.max())) == [1, 0, 1]


def test_exact_sphere_euler_characteristic() -> None:
    # A triangulated S^2 has V - E + F == 2 for every subdivision level.
    for n_sub in (0, 1, 2):
        C = make_sphere_complex(n_sub)
        chi = C.count(0) - C.count(1) + C.count(2)
        assert chi == 2


def test_exact_torus3_essential_classes() -> None:
    C = make_torus_grid_complex(3, (3, 3, 3))
    bc = persistent_homology(C)
    ess = [len([iv for iv in bc.of_dim(d) if iv.is_essential]) for d in range(4)]
    assert ess == [1, 3, 3, 1]


beartype_module(__name__)
