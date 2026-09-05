"""Guard the 2-torus DISPLAY fix (and check it in higher dimensions).

The product 2-torus is a 2-torus embedded in R^4. A PCA-3D projection of that
4-fold-symmetric cloud collapses to a bare cylinder (the bug: "only an empty
cylinder, no outer bulge"). The fix shows the classic bagel (donut_grid, the same
(u,v) grid so it pairs 1:1 with the complex) while still computing topology in R^4.

These tests lock that in: the display must be a RING (radial thickness + a hole), the
topology must stay exact, and the check must also hold in higher ambient dimensions
(R^10, R^50) and for higher-k tori.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vrtda import pairwise_distances, build_rips, persistent_homology, betti_at
from vrtda import generators as G
from vrtda.complexes import make_torus_grid_complex
from vrtda.beartype_guard import beartype_module

ROOT = Path(__file__).resolve().parents[1]


def _load_tool() -> object:
    spec = importlib.util.spec_from_file_location("vrtda_interactive", ROOT / "tools" / "interactive.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vrtda_interactive"] = mod
    spec.loader.exec_module(mod)
    return mod


interactive = _load_tool()


def _args(shape: str, **kw: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        points=None, value_cols=None, index_cols=None, metric="euclidean",
        n=8, nper=16, k=2, frac=1.6, max_dim=2, title="",
    )
    base["shape"] = shape
    base.update(kw)
    return argparse.Namespace(**base)


def _radius(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p[:, :2], axis=1)


def is_bagel(p: np.ndarray) -> bool:
    """True if the 3D cloud is a torus ring (real radial thickness AND a hole),
    False for a degenerate cylinder (near-constant radius) or a solid disk."""
    rad = _radius(p)
    thickness = rad.max() / (rad.min() + 1e-9)
    hole = rad.min() > 0.4 * rad.max()
    return bool(thickness > 1.5 and hole)


def _rip_betti(X: np.ndarray, frac: float = 1.6, max_dim: int = 3) -> list[int]:
    D = pairwise_distances(X)
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    nn = float(d.min(axis=1).mean())
    C = build_rips(X, D, frac * nn, max_dim=max_dim)
    return persistent_homology(C).betti_at(frac * nn)


def _pad(X: np.ndarray, ambient: int) -> np.ndarray:
    if X.shape[1] >= ambient:
        return X
    return np.hstack([X, np.zeros((X.shape[0], ambient - X.shape[1]))])


# --------------------------------------------------------------------------- #
# The fix: product 2-torus renders as a bagel, not a PCA cylinder
# --------------------------------------------------------------------------- #
def test_product_torus2_displays_as_bagel() -> None:
    C, pts, proj, target, _ = interactive.build_source(_args("product", k=2, nper=16))
    assert pts.shape[1] == 3
    assert is_bagel(pts), (
        f"display degenerated (cylinder/disk): rad range {_radius(pts).min():.3f}..{_radius(pts).max():.3f}"
    )
    assert "bagel" in proj.lower() or "R4" in proj or "R\u2074" in proj


def test_product_torus2_topology_exact() -> None:
    C, *_ = interactive.build_source(_args("product", k=2, nper=16))
    assert betti_at(C, float(C.values.max()))[:3] == [1, 2, 1]


def test_default_torus_grid_displays_as_ring() -> None:
    _, pts, *_ = interactive.build_source(_args("torus-grid", n=10))
    assert pts.shape[1] == 3
    assert is_bagel(pts)


# --------------------------------------------------------------------------- #
# The predicate must actually discriminate (so a cylinder regression is caught)
# --------------------------------------------------------------------------- #
def test_is_bagel_rejects_cylinder_and_disk() -> None:
    assert is_bagel(G.donut_grid(12, 12))

    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    z = np.linspace(-0.4, 0.4, 8)
    AA, ZZ = np.meshgrid(a, z)
    cyl = np.column_stack([np.cos(AA).ravel(), np.sin(AA).ravel(), ZZ.ravel()])
    assert not is_bagel(cyl), "a unit cylinder must NOT be classified as a bagel"

    r = np.linspace(0.05, 1.0, 30)
    th = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    RR, TT = np.meshgrid(r, th)
    rr, tt = RR.ravel(), TT.ravel()
    disk = np.column_stack([np.cos(tt) * rr, np.sin(tt) * rr, np.zeros(rr.size)])
    assert not is_bagel(disk), "a solid disk (no hole) must NOT be classified as a bagel"


# --------------------------------------------------------------------------- #
# Higher ambient dimensions: topology of the 2-torus stays exact in R^4/R^10/R^50
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ambient", [4, 10, 50])
def test_torus2_topology_in_higher_ambient_dims(ambient: int) -> None:
    X = _pad(G.product_torus_grid(2, 8), ambient)
    assert _rip_betti(X, frac=1.6, max_dim=3)[:3] == [1, 2, 1]


# --------------------------------------------------------------------------- #
# Higher intrinsic dimension (k>=3): display stays 3D + low dims exact
# --------------------------------------------------------------------------- #
def test_product_torus3_displays_3d_and_low_dims_exact() -> None:
    C, pts, _proj, _target, report_dim = interactive.build_source(_args("product", k=3, nper=5))
    assert pts.shape[1] == 3
    assert report_dim <= 2  # capped for the fast beta_0..beta_2 visualizer
    assert betti_at(C, float(C.values.max()))[:3] == [1, 3, 3]


# --------------------------------------------------------------------------- #
# The epsilon slider must reach a fully-connected complex (the "only points /
# beta_0=24" bug: the old slider stopped at frac*nn, far short of connectivity)
# --------------------------------------------------------------------------- #
def test_slider_reaches_full_connectivity() -> None:
    C, *_ = interactive.build_source(_args("donut"))  # donut_grid(16,16), a coarse 2D cloud
    b = betti_at(C, float(C.values.max()))
    assert b[0] == 1, f"complex not fully connected at eps_max: beta_0={b[0]}"
    # and the slider bound is at least the connectivity threshold
    from vrtda import pairwise_distances
    X = G.donut_grid(16, 16)
    D = pairwise_distances(X)
    assert float(C.values.max()) >= interactive._connectivity_threshold(D)


def test_connectivity_threshold_matches_components() -> None:
    from collections import deque
    X = G.donut_grid(12, 16)
    D = pairwise_distances(X)
    T = interactive._connectivity_threshold(D)
    n = X.shape[0]

    def n_components(eps: float) -> int:
        adj = (D <= eps).astype(bool)
        seen = np.zeros(n, dtype=bool)
        comps = 0
        for s in range(n):
            if seen[s]:
                continue
            comps += 1
            q = deque([s]); seen[s] = True
            while q:
                u = q.popleft()
                for v in np.where(adj[u])[0]:
                    if not seen[v]:
                        seen[v] = True; q.append(int(v))
        return comps

    assert n_components(T * 1.0001) == 1
    assert n_components(T * 0.5) > 1


def test_higher_tori_exact_topology() -> None:
    assert betti_at(make_torus_grid_complex(3, (4, 4, 4)), float(make_torus_grid_complex(3, (4, 4, 4)).values.max())) == [1, 3, 3, 1]
    C4 = make_torus_grid_complex(4, (3, 3, 3, 3))
    assert betti_at(C4, float(C4.values.max())) == [1, 4, 6, 4, 1]


beartype_module(__name__)
