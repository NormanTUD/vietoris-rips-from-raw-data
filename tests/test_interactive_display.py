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

from vrtda import PointSet, pairwise_distances, build_rips, persistent_homology, betti_at
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
        eps_max=None, connect_margin=1.2,
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
# The donut: clean exact T^2 (default) vs honest full-range Rips (donut-rips)
# --------------------------------------------------------------------------- #
def test_donut_exact_clean_torus() -> None:
    # --shape donut is the exact T^2 cell complex (NOT Rips): a clean bagel that
    # reads exactly [1, 2, 1] at the top of the slider and builds instantly.
    C, pts, _proj, target, _rd = interactive.build_source(_args("donut", nper=16))
    assert pts.shape[1] == 3
    assert is_bagel(pts), "donut display must be a ring (inner + outer)"
    assert target == [1, 2, 1]
    assert betti_at(C, float(C.values.max()))[:3] == [1, 2, 1]


def test_donut_matches_torus_grid_topology() -> None:
    # donut and torus-grid share the exact T^2 engine -> identical topology at the top
    Cd, *_ = interactive.build_source(_args("donut", nper=8))
    Ct, *_ = interactive.build_source(_args("torus-grid", n=8))
    assert betti_at(Cd, float(Cd.values.max()))[:3] == [1, 2, 1]
    assert betti_at(Ct, float(Ct.values.max()))[:3] == [1, 2, 1]


def test_donut_rips_full_range_overfills() -> None:
    # REGRESSION GUARD for the dense-bagel Rips over-filling bug (see the IMPORTANT
    # block at the top of tools/interactive.py). --shape donut-rips is honest Rips over
    # the FULL range eps 0 -> max pairwise distance: the slider reaches Dmax and the
    # complex is fully connected there (beta_0 = 1), but the bagel OVER-FILLS so the
    # clean beta_1 = 2 is NOT read (beta_1 stays at the grid's loop count, beta_2
    # blows up). This documents that Rips genuinely cannot resolve a dense bagel to a
    # clean torus, so `--shape donut` (exact) remains the reliable path.
    C, *_ = interactive.build_source(_args("donut-rips", nper=8))
    X = G.donut_grid(8, 8)
    D = pairwise_distances(X)
    assert abs(float(C.values.max()) - float(D.max())) < 1e-9, "slider must reach max distance"
    b = betti_at(C, float(C.values.max()))
    assert b[0] == 1, f"not fully connected at max distance: beta_0={b[0]}"
    assert b[1] != 2, "over-filled Rips bagel should not read a clean beta_1 = 2"


# ---- dense-bagel over-filling: detection, auto-raise, feasibility cap, notes ---
def test_is_overfilling_helper() -> None:
    # A clean 2-manifold triangulation keeps ~1.5-2 triangles/vertex (NOT over-filling);
    # dense Rips keeps tens (over-filling). Zero points is degenerate (False).
    assert not interactive._is_overfilling(576, 1152)      # 2.0/vertex (exact torus scale)
    assert not interactive._is_overfilling(500, 1000)      # 2.0/vertex
    assert interactive._is_overfilling(1536, 33552)        # 21.8/vertex (the reported bug)
    assert interactive._is_overfilling(100, 600)           # 6.0/vertex
    assert not interactive._is_overfilling(0, 0)


def test_points_dense_bagel_capped_at_feasible(tmp_path: Path) -> None:
    # The "slider only goes to 0.188" fix, now spanning the FULL Rips range: a DENSE
    # NON-GRID bagel (so it is NOT reconstructed) loaded via --points must (a) be
    # detected as over-filling, and (b) have its slider capped below the infeasible
    # Dmax (which would be millions of simplices), at the largest feasible epsilon.
    X = G.donut(600, seed=0)
    assert interactive._detect_torus_grid(X) is None, "a random bagel is not a grid"
    csv = tmp_path / "bagel.csv"
    PointSet(X).to_csv(str(csv))
    D = pairwise_distances(X)
    Dmax = float(D.max())
    C, *_ = interactive.build_source(_args("donut", points=str(csv)))
    eps_max = float(C.values.max())
    n_faces = sum(1 for s in C.simplexes if len(s) == 3)
    assert eps_max < Dmax, "a dense bagel's Dmax is infeasible and must be capped"
    assert interactive._is_overfilling(X.shape[0], n_faces), \
        f"expected an over-filling dense bagel, got {n_faces}/{X.shape[0]} faces/vertex"


def test_max_feasible_eps_sparse_reaches_dmax() -> None:
    # The Dmax-as-maximum contract: a SPARSE cloud's max feasible epsilon is exactly
    # Dmax (the full Rips range is feasible); a DENSE bagel's is strictly below Dmax.
    sparse = G.circle_grid(24, radius=1.0)
    Ds = pairwise_distances(sparse)
    assert abs(interactive._max_feasible_eps(sparse, Ds, float(Ds.max()), budget=160_000)
               - float(Ds.max())) < 1e-9, "sparse cloud must reach Dmax in full"
    dense = G.donut_grid(20, 40)
    Dd = pairwise_distances(dense)
    assert interactive._max_feasible_eps(dense, Dd, float(Dd.max()), budget=160_000) < float(Dd.max())


def test_overfill_note_message() -> None:
    import io
    from rich.console import Console
    import _rich_ui
    buf = io.StringIO()
    _rich_ui.overfill_note(Console(file=buf), 1536, 33552)
    text = buf.getvalue()
    assert "over-filling" in text
    assert "--shape donut" in text


def test_make_torus_dense_donut_warns() -> None:
    import io
    import importlib.util
    from rich.console import Console
    spec = importlib.util.spec_from_file_location("vrtda_make_torus_x", ROOT / "tools" / "make_torus.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vrtda_make_torus_x"] = mod  # register so beartype_module(__name__) can find it
    spec.loader.exec_module(mod)
    args = argparse.Namespace(kind="donut", grid=True, nper=48, out="/tmp/bagel.csv")
    buf = io.StringIO()
    mod._warn_dense_donut(args, 24 * 48, Console(file=buf))     # dense -> warns
    assert "--shape donut" in buf.getvalue()
    buf2 = io.StringIO()
    mod._warn_dense_donut(args, 64, Console(file=buf2))         # sparse -> silent
    assert buf2.getvalue() == ""


# ---- torus-grid detection + exact T^2 reconstruction (the reliable --points fix) ---
def test_detect_torus_grid_clean() -> None:
    # A clean donut_grid is detected with the right (nu, nv, R, r).
    det = interactive._detect_torus_grid(G.donut_grid(24, 64))
    assert det is not None
    nu, nv, R, r = det
    assert (nu, nv) == (24, 64)
    assert abs(R - 1.0) < 1e-6 and abs(r - 0.35) < 1e-6


def test_even_grid_count() -> None:
    # A regular cyclic grid of cnt lines (each repeated) -> cnt; a random scatter -> None.
    grid = np.repeat(2 * np.pi * np.arange(24) / 24, 64)
    assert interactive._even_grid_count(grid) == 24
    rng = np.random.default_rng(0)
    assert interactive._even_grid_count(rng.uniform(0, 2 * np.pi, 100)) is None


def test_detect_torus_grid_rejects_non_grid() -> None:
    # A random bagel, a circle, and blobs are NOT a regular torus grid -> None.
    assert interactive._detect_torus_grid(G.donut(400, seed=0)) is None
    assert interactive._detect_torus_grid(G.circle_grid(48, radius=1.0)) is None
    assert interactive._detect_torus_grid(G.gmm(3, 90, 3)) is None


def test_points_clean_grid_reconstructs_torus(tmp_path: Path) -> None:
    # THE HEADLINE FIX: a clean donut_grid CSV loaded via --points is reconstructed as
    # the EXACT T^2 complex -> a clean torus (beta=[1,2,1]), not the over-filled Rips.
    X = G.donut_grid(24, 64)
    csv = tmp_path / "d.csv"
    PointSet(X).to_csv(str(csv))
    C, _pts, proj, target, _rd = interactive.build_source(_args("donut", points=str(csv)))
    assert target == [1, 2, 1]
    assert betti_at(C, float(C.values.max()))[:3] == [1, 2, 1]
    assert "exact" in proj


def test_points_reconstructed_torus_pairs_with_points(tmp_path: Path) -> None:
    # The reconstructed complex's vertices live on a genuine bagel surface (so the 3D
    # view draws edges between truly adjacent grid points = a complete torus) and there
    # are exactly 2 triangles per grid cell of the T^2 complex.
    X = G.donut_grid(24, 64)
    csv = tmp_path / "d.csv"
    PointSet(X).to_csv(str(csv))
    C, pts, _proj, _target, _rd = interactive.build_source(_args("donut", points=str(csv)))
    assert is_bagel(pts)
    n_faces = sum(1 for s in C.simplexes if len(s) == 3)
    assert n_faces == 2 * 24 * 64


def test_points_no_exact_torus_forces_rips(tmp_path: Path) -> None:
    # --no-exact-torus forces Rips even for a detected grid (over-fills, not a torus).
    X = G.donut_grid(24, 64)
    csv = tmp_path / "d.csv"
    PointSet(X).to_csv(str(csv))
    C, *_ = interactive.build_source(_args("donut", points=str(csv), no_exact_torus=True))
    n_faces = sum(1 for s in C.simplexes if len(s) == 3)
    assert interactive._is_overfilling(X.shape[0], n_faces)


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
