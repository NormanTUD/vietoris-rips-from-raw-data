# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18", "rich>=13"]
# ///
"""Build a self-contained interactive HTML/JS file for exploring TDA.

Two modes:

Filtration mode (default) -- drag the epsilon slider (or press Play) and watch
the Vietoris-Rips complex grow: points -> edges -> faces, while the Betti numbers
beta_0 / beta_1 / beta_2 (H0 components, H1 loops/holes, H2 voids) update live,
next to a live persistence diagram and the Betti function. Drag the 3D view to
rotate. Lower-dim clouds (2D/3D) are drawn directly; higher-dim clouds are shown
as a 3D PCA projection while the topology is still computed in the original D-dim
space (so the Betti numbers stay exact).

Layer/trajectory mode (--layers) -- treats the transformer layers as time steps.
The same 81 tokens are projected into ONE shared 3D frame (PCA of a few layers),
and you drag a time slider to watch the tokens MOVE across the surface as depth
increases, with their full trajectories drawn as trails (coloured by prompt),
plus a live convergence (spread-over-depth) curve.

Examples:
    # the clean, exact 2-torus (target beta = [1,2,1]), 3D donut you can rotate
    uv run tools/interactive.py --out interactive.html

    # a circle (beta_1 = 1) and a blob cloud
    uv run tools/interactive.py --shape circle --n 40 --out circle.html
    uv run tools/interactive.py --shape blobs --out blobs.html

    # your own data (any dimension; PCA-projected to 3D for display)
    uv run tools/interactive.py --points mydata.csv --value-cols x y z --out my.html
    uv run tools/interactive.py --points mydata.csv --metric cosine --max-dim 2 --out my.html

    # the 2-torus as a point cloud (product of two circles) via Rips
    uv run tools/interactive.py --shape product --k 2 --nper 10 --out product.html

    # watch the 81 tokens move across layers (time) - full depth or a subsample
    uv run tools/interactive.py --layers --out layers.html
    uv run tools/interactive.py --layers 0:64:8 --out layers.html
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
TOOLS = str(Path(__file__).resolve().parent)
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from rich.console import Console
from rich.progress import Progress

console = Console()

import _rich_ui
from vrtda import PointSet, pairwise_distances
from vrtda.complexes import FilteredComplex, build_rips, make_torus_grid_complex
from vrtda.errors import TooLargeError, VrtdaError
from vrtda.persistence import Barcode, Interval, persistent_homology
from vrtda import datasets, generators as G
from vrtda.beartype_guard import beartype_module


# =====================================================================
# IMPORTANT (DO NOT REMOVE) — the dense-bagel Rips over-filling bug.
# ---------------------------------------------------------------------
# ERROR (reproduced with `make_torus --kind donut --nper 64 --grid` -> a
#   1536-point bagel, then `--points that.csv`): the Betti numbers read
#   beta_1 = 25 (stuck at the grid's loop count) and beta_2 explodes
#   (~62,064 voids) instead of the true beta_1 = 2, beta_2 = 1. There is
#   NO epsilon at which a dense Rips bagel reads a clean beta_1 = 2: just
#   past connectivity the triangles already triangulate the holes away,
#   and the surface-completion epsilon is in the infeasible (>~300k
#   simplex) range, so the honest full-range Rips bagel never resolves.
# ROOT CAUSE: Vietoris-Rips on a dense 2-manifold keeps far more triangles
#   per vertex than a clean triangulation (~1.5), so its 2-skeleton fills
#   the 2-cycles (the torus void) and shreds beta_1 into many short loops.
#   This is a property of Rips on dense point clouds, NOT a code bug.
# MITIGATION (four layers, keep them in sync):
#   (a) `--points` AUTO-DETECTS a regular torus grid (evenly-spaced major x tube
#       angles, see _detect_torus_grid) and rebuilds the EXACT T^2 cell complex
#       (_exact_torus2) -> a clean torus, beta=[1,2,1], with a complete 3D view.
#       This is the reliable path for a clean `make_torus --grid` CSV. Use
#       `--no-exact-torus` to force the honest (over-filling) Rips instead.
#   (b) To SEE a clean torus without a CSV, use `--shape donut` -> the exact T^2
#       cell complex (fast, reads exactly [1, 2, 1]); see _exact_torus2 below.
#   (c) `--shape donut-rips` (honest full-range Rips) is capped at nper <= 8 so a
#       feasible epsilon exists.
#   (d) For a NON-grid dense cloud, `--points` runs Rips over the full pair-distance
#       range, auto-caps the slider at the largest feasible epsilon (_max_feasible_eps),
#       and prints _rich_ui.overfill_note() when it detects over-filling
#       (faces > _OVERFILL_FACE_RATIO * vertices), so you know the numbers are a
#       Rips artifact, not the true topology.
# Regression guard: tests/test_interactive_display.py::
#   test_donut_rips_full_range_overfills, test_detect_torus_grid_clean,
#   test_points_clean_grid_reconstructs_torus, test_points_no_exact_torus_forces_rips,
#   test_points_dense_bagel_capped_at_feasible.
# =====================================================================
# faces/vertex above which we flag a dense cloud as over-filling (a clean
# 2-manifold triangulation has ~1.5; over-filled Rips reaches tens).
_OVERFILL_FACE_RATIO: float = 5.0


def _is_overfilling(n_points: int, n_faces: int) -> bool:
    """SAFEGUARD (load time, testable): True when a Rips 2-skeleton keeps far more
    triangles per vertex than a clean triangulation (~1.5) -- the signature of a dense
    manifold point cloud over-filling its higher-dimensional voids. Kept as a pure
    function so it can be unit-tested without building a complex."""
    return n_points > 0 and n_faces > _OVERFILL_FACE_RATIO * n_points


def _nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def _connectivity_threshold(D: np.ndarray) -> float:
    """Smallest epsilon for which the distance graph (edges where D <= eps) is
    connected, i.e. the max edge weight of a minimum spanning tree. Below this the
    point cloud splits into several components; at/above it all points belong
    together (beta_0 = 1). Computed with a union-find over the sorted edges, so it
    stops as soon as the graph connects (fast, no homology needed)."""
    n = D.shape[0]
    if n <= 1:
        return 0.0
    i, j = np.triu_indices(n, 1)
    w = D[i, j]
    order = np.argsort(w, kind="stable")
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    comps = n
    thresh = 0.0
    for idx in order:
        a, b = int(i[idx]), int(j[idx])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            comps -= 1
            thresh = float(w[idx])
            if comps == 1:
                break
    return thresh


def _eps_max(D: np.ndarray, frac: float, connect_margin: float) -> float:
    """Slider upper bound: at least frac x mean-NN (the old default) AND at least
    connect_margin x the connectivity threshold, so the slider always reaches a
    fully-connected complex (all points in one component)."""
    return float(max(frac * _nn(D), connect_margin * _connectivity_threshold(D)))


def _build_rips_safe(X: np.ndarray, D: np.ndarray, eps_max: float, max_dim: int,
                     max_simplices: int = 2_000_000) -> FilteredComplex:
    """build_rips that, if the requested max_dim overflows max_simplices, retries with
    a lower max_dim down to 1 (edges only), so a larger slider range never crashes."""
    md = max_dim
    while True:
        try:
            return build_rips(X, D, eps_max, max_dim=md)
        except TooLargeError:
            if md <= 1:
                raise
            md -= 1


# Simplicity/render budget for the --points slider. A dense surface (e.g. a bagel)
# stops rendering at the connectivity epsilon if we cap there, leaving the view
# incomplete (missing the outer bulge). We therefore push the slider up to the largest
# epsilon whose Rips complex stays under these budgets (measured with max_dim=2, since
# faces drive the browser render). The DEFAULT budget keeps the view snappy; --eps-max
# lets the user opt into a larger (slower, more complete) complex, capped at the MAX
# budget. Both sit below the hard infeasibility wall (~300k simplices) where the
# pure-Python homology and the browser both choke.
_FEASIBLE_DEFAULT_BUDGET: int = 100_000
_FEASIBLE_MAX_BUDGET: int = 160_000


def _max_feasible_eps(X: np.ndarray, D: np.ndarray, eps_hi: float,
                      budget: int, lo: float = 0.0, iters: int = 6) -> float:
    """SAFEGUARD (feasibility): the largest eps in [lo, eps_hi] at which build_rips
    (max_dim=2) keeps <= budget simplices. The simplex count is monotone non-decreasing
    in eps, so a binary search finds it. Each probe is capped at `budget` simplices
    (max_simplices=budget) so an infeasible eps fails fast instead of building toward the
    2M hard cap. This lets the --points slider span the FULL max-pairwise-distance (Dmax)
    when feasible, and cap at the largest feasible epsilon otherwise (reported, no crash)."""
    lo = float(lo)
    hi = float(eps_hi)
    try:
        if build_rips(X, D, hi, max_dim=2, max_simplices=budget).n_simplices <= budget:
            return hi  # the full range up to Dmax is feasible
    except TooLargeError:
        pass
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        try:
            feasible = build_rips(X, D, mid, max_dim=2, max_simplices=budget).n_simplices <= budget
        except TooLargeError:
            feasible = False
        if feasible:
            lo = mid
        else:
            hi = mid
    return float(lo)


def _pca3(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _u, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def _to3d(X: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    """Return 3D display coordinates + a projection label."""
    if X.shape[1] == 2:
        return np.column_stack([X, np.zeros(X.shape[0])]), "identity (2D, z=0)"
    if X.shape[1] == 3:
        return X, "identity (3D)"
    return _pca3(X), f"PCA 3D projection (data is {X.shape[1]}D)"


def _torus_surface(nu: int, nv: int, R: float = 1.0, r: float = 0.35) -> np.ndarray:
    """3D donut-surface coordinates for torus-grid vertex index = i + j*nu."""
    idx = np.arange(nu * nv)
    i = idx % nu
    j = idx // nu
    u1 = 2.0 * np.pi * i / nu
    v1 = 2.0 * np.pi * j / nv
    x = (R + r * np.cos(v1)) * np.cos(u1)
    y = (R + r * np.cos(v1)) * np.sin(u1)
    z = r * np.sin(v1)
    return np.column_stack([x, y, z])


def _torus_betti(k: int) -> list[int]:
    """Betti numbers of the k-torus S^1 x ... x S^1: beta_d = C(k, d)."""
    from math import comb
    return [int(comb(k, d)) for d in range(k + 1)]


# ---- Betti vector -> homotopy-type recognition (the structure catalogue) ----
# Given the Betti numbers of a space, say what it is HOMOTOPY-EQUIVALENT to (not just
# "matches"). The catalogue is the classic "Betti fingerprints" reference set: spheres,
# tori, projective spaces, closed/open surfaces, graphs and common products -- the exact
# two-sheet list of the standard TDA cheat sheet. A match is a HYPOTHESIS about the
# homotopy type (it cannot be a ground-truth claim for arbitrary data), so the wording
# stays "homotopy equivalent to ..." and unknown signatures are reported honestly.

_FIXED_TOPO: tuple[tuple[tuple[int, ...], str], ...] = (
    ((1,),             "a contractible space (a point, a ball or a tree)"),
    ((1, 1),           "a circle S¹ (one independent loop / a single period)"),
    ((1, 0, 1),        "a 2-sphere S² (a closed surface wrapping one void)"),
    ((1, 0, 0, 1),     "a 3-sphere S³ (or a lens space / ℝP³)"),
    ((1, 1, 0),        "a cylinder or Möbius band (one 1-cycle + a boundary)"),
    ((1, 0, 0),        "a disk or plane (no cycles)"),
    ((1, 2, 1),        "a 2-torus T² = S¹×S¹ (two independent periods)"),
    ((1, 1, 1, 1),     "a product S¹ × S² (a loop wound around a sphere)"),
    ((1, 0, 2, 0, 1),  "a product S² × S² (two independent nested shells)"),
    ((1, 2, 2, 1),     "a Heisenberg nilmanifold (β equals T³, but is NOT a torus!)"),
    ((1, 0, 1, 0, 1),  "complex projective space ℂP²"),
)


def _strip_trailing_zeros(betti: list[int]) -> list[int]:
    """Drop trailing zeros. Used ONLY where the extra zero-dimensions would be noise
    (transition detection); for structural matching the full-length vector is kept,
    because (1,2) a graph and (1,2,0) an open surface are different signatures."""
    v = [int(b) for b in betti]
    while v and v[-1] == 0:
        v.pop()
    return v or [0]


def _pad(v: list[int], L: int) -> list[int]:
    """Pad (never truncate) a signature to length L with zeros."""
    vv = [int(b) for b in v]
    return vv + [0] * (L - len(vv))


def _sig_matches(sig: list[int], v: list[int]) -> bool:
    """Signature equality, ignoring differences made only of trailing zero columns."""
    return _pad(sig, max(len(sig), len(v))) == _pad(v, max(len(sig), len(v)))


def _topology_name(betti: list[int]) -> str:
    """Best homotopy-type description for a Betti vector. Returns a hypothesis phrased as
    "homotopy equivalent to ..." (never a ground-truth claim). Tries exact catalogue
    signatures first, then parametric families (spheres/tori/surfaces/projective spaces),
    and finally reports the vector as unrecognized if nothing plausible fits. The FULL
    vector is matched (length = reported maxdim + 1): (1,2) and (1,2,0) are different."""
    from math import comb
    v = [int(b) for b in betti]
    for sig, name in _FIXED_TOPO:
        if list(sig) == v:
            return f"homotopy equivalent to {name}"
    if len(v) == 1:
        return ("a single isolated point (contractible)"
                if v[0] == 1
                else f"{v[0]} disjoint points / clusters (a wedge of {v[0]} copies of S⁰)")
    if len(v) >= 2 and v[0] > 1 and all(b == 0 for b in v[1:]):
        return f"{v[0]} disjoint points / clusters (a wedge of {v[0]} copies of S⁰)"
    if len(v) == 2 and v[0] == 1 and v[1] >= 1:
        return f"a connected graph with {v[1]} independent cycle(s) (cycle rank {v[1]})"
    if len(v) == 2 and v[0] == 1 and v[1] == 0:
        return "a connected tree-like region (no cycles yet)"
    if len(v) == 3 and v[0] == 1 and v[2] == 0 and v[1] >= 1:
        return f"an open surface / disk with {v[1]} hole(s)"
    if len(v) == 3 and v[0] == 1 and v[2] == 1 and v[1] >= 2 and v[1] % 2 == 0:
        return f"a closed orientable surface of genus {v[1] // 2}"
    if len(v) >= 3 and v[0] == 1 and all(b == 0 for b in v[1:]):
        return "a contractible space (a point, a ball or a tree)"
    if len(v) >= 2 and v[0] == 1:
        nz = [i for i, b in enumerate(v) if b]
        if nz and nz[-1] >= 1:
            n = nz[-1]
            p: list[int] = [comb(n, k) for k in range(n + 1)]
            if sum(p) > 1 and _pad(p, len(v)) == v:
                return f"the {n}-torus Tⁿ = (S¹)ⁿ  (βₖ = C({n}, k), {2 ** n} classes total)"
    if len(v) >= 3 and v[0] == 1 and v.count(1) == 2 and all(0 <= b <= 1 for b in v):
        nz = [i for i, b in enumerate(v) if b == 1]
        if nz == [0, len(v) - 1]:
            return f"the {len(v) - 1}-sphere S^{len(v) - 1}"
    if len(v) >= 5 and len(v) % 2 == 1 and all(v[k] == 1 for k in range(0, len(v), 2)) \
            and all(v[k] == 0 for k in range(1, len(v), 2)):
        return f"complex projective space ℂP^{(len(v) - 1) // 2}"
    if len(v) >= 5 and len(v) % 4 == 1 and all(v[k] == 1 for k in range(0, len(v), 4)) \
            and all(v[k] == 0 for k in range(len(v)) if k % 4 != 0):
        return f"quaternion projective space ℍP^{(len(v) - 1) // 4}"
    if len(v) >= 2 and v[0] == 1:
        nz = [b for b in v[1:] if b != 0]
        dims = [d for d, b in enumerate(v[1:], 1) if b != 0]
        if len(dims) == 1 and all(b == nz[0] for b in nz):
            return f"a wedge of {nz[0]} copies of S^{dims[0]}"
    nz_pos = [(d, b) for d, b in enumerate(v) if b]
    if len(nz_pos) == 2 and nz_pos[0][0] == 0 and len(v) >= 2 \
            and all(b == nz_pos[0][1] for _, b in nz_pos) \
            and nz_pos[0][1] >= 2:
        c = nz_pos[0][1]
        d = nz_pos[1][0]
        return (f"the disjoint union of {c} copies of S^{d} "
                f"({c} separate components, each with its own H_{d})")
    return (f"an unrecognized Betti signature ({v}) — "
            "report the vector and its construction")


def _reference_signatures() -> list[tuple[str, list[int]]]:
    """The nearest-neighbour retina: representative Betti fingerprints of the standard
    catalogue (spheres/tori/surfaces/projective spaces/graphs/products), used to find
    topological STRUCTURE SIMILARITY when the exact signature is unknown."""
    from math import comb
    refs: list[tuple[str, list[int]]] = []
    for m in range(1, 9):
        sv = [0] * (m + 1)
        sv[0] = sv[m] = 1
        refs.append((f"m-sphere S^{m}", sv))
    for n in range(1, 9):
        refs.append((f"n-torus T^{n}", [comb(n, k) for k in range(n + 1)]))
    for g in range(1, 6):
        refs.append((f"closed surface of genus {g}", [1, 2 * g, 1]))
    for n in range(1, 5):
        refs.append((f"ℂP^{n}", [1 if k % 2 == 0 else 0 for k in range(2 * n + 1)]))
    for n in range(1, 4):
        refs.append((f"ℍP^{n}", [1 if k % 4 == 0 else 0 for k in range(4 * n + 1)]))
    for c in range(2, 9):
        refs.append((f"connected graph with {c} cycles", [1, c]))
        refs.append((f"open surface with {c} holes", [1, c, 0]))
    for k in range(2, 6):
        refs.append((f"{k} disjoint points", [k]))
    refs.append(("product S¹ × S²", [1, 1, 1, 1]))
    refs.append(("product S² × S²", [1, 0, 2, 0, 1]))
    refs.append(("Heisenberg nilmanifold", [1, 2, 2, 1]))
    return refs


def _closest_topologies(betti: list[int], top: int = 3) -> list[dict[str, object]]:
    """Nearest known topologies to a Betti vector (STRUCTURE SIMILARITY). Distance is the
    L1 difference of the Betti vectors (padded to a common length) plus a penalty for a
    different top non-zero dimension, so e.g. (1,3,2,1) is close to a 3-torus T³=(1,3,3,1)
    but (1,2) a graph stays far from (1,2,0) an open surface. Returns the `top` closest
    signatures that are not the vector itself, each as {"name", "dist"}."""
    v = [int(b) for b in betti]
    scored: list[tuple[int, str]] = []
    for name, sig in _reference_signatures():
        if _sig_matches(v, sig):
            continue
        L = max(len(v), len(sig))
        dist = sum(abs(a - b) for a, b in zip(_pad(v, L), _pad(sig, L)))
        dim_v = len(v) - 1 - ([0] + list(reversed(v))).index(0) if v else 0
        dim_s = len(sig) - 1 - ([0] + list(reversed(sig))).index(0) if sig else 0
        dist += abs(dim_v - dim_s)
        scored.append((dist, name))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [{"name": name, "dist": int(dist)} for dist, name in scored[:top]]


# ---- Betti vector -> ATTRACTOR / REPELLOR / BIFURCATION recognition ---------
# The dynamics cheat sheet: what will-be/should-be-attractor structures the current
# (or evolving) Betti signature is compatible with. Attractors and REPELLORS are
# topologically identical (reversing time turns one into the other), so these are
# HYPOTHESES: they say "under an attractor interpretation this is the dynamics you'd
# expect", never a proof. Bifurcation labels come from comparing consecutive slider
# steps ((1,0) -> (1,1) = Hopf, (1,1) -> (1,2,1) = Neimark–Sacker, ...).

_DYNAMICS_FIXED: tuple[tuple[tuple[int, ...], str], ...] = (
    ((1,), "a single stable fixed point (everything converges to one state)"),
    ((1, 1), "a limit-cycle / periodic attractor (homotopy type of S¹ — one period)"),
    ((1, 0, 1), "a closed energy-shell state space S² (a stable core it circles)"),
    ((1, 0, 0, 1), "a 3D state space closed on itself (S³ or lens-type)"),
    ((1, 0, 0, 0, 1), "a 4D state space closed on itself (S⁴ or lens-type)"),
    ((1, 2, 1), "a quasi-periodic 2-torus attractor (two incommensurable frequencies)"),
    ((1, 3, 3, 1), "a quasi-periodic 3-torus (rare — usually on the route to chaos, Ruelle–Takens)"),
    ((1, 4, 6, 4, 1), "a quasi-periodic 4-torus attractor (four incommensurable frequencies)"),
    ((1, 2, 0), "a Lorenz-/Chua-type chaotic attractor (two 'wings', no bulk) or two merged limit cycles"),
    ((1, 1, 0), "a Rössler-/Duffing-type folded attractor (one dominant loop) or a limit cycle on a surface"),
    ((1, 0, 1, 0, 1), "a complex-projective-type state space ℂP²"),
)


def _dynamics_name(betti: list[int]) -> str:
    """Best ATTRACTOR-style dynamics hypothesis compatible with a Betti vector. Always a
    hypothesis (attractor/repellor are indistinguishable topologically), phrased as such.
    The FULL vector is matched (length = reported maxdim + 1): (1,1) a limit cycle and
    (1,1,0) a Rössler-folded surface are DIFFERENT signatures."""
    from math import comb
    v = [int(b) for b in betti]
    for sig, name in _DYNAMICS_FIXED:
        if list(sig) == v:
            return (f"dynamics: {name} "
                    "(repellors look identical — reverse time t→−t to expose them)")
    if len(v) == 1:
        if v[0] == 1:
            return "dynamics: a single stable fixed point (repellors look identical — reverse time)"
        return (f"dynamics: {v[0]} coexisting stable fixed points "
                f"(multistability — {v[0]} basins)")
    if len(v) == 2 and v[0] == v[1] and v[0] >= 1:
        return (f"dynamics: {v[0]} disjoint limit cycles "
                f"({v[0]} independent periods / basins)")
    if len(v) == 2 and v[0] == 1 and v[1] == 0:
        return "dynamics: one connected region with no 1-cycles (a growing basin)"
    if len(v) == 2 and v[0] >= 1 and v[1] == 0:
        return f"dynamics: {v[0]} disjoint growing basins (merging as ε grows)"
    if len(v) >= 2 and v[0] == 1 and v[-1] == 1:
        n = len(v) - 1
        if all(b == comb(n, k) for k, b in enumerate(v)):
            return (f"dynamics: a quasi-periodic {n}-torus attractor "
                    f"(KAM / integrable: {n} incommensurable frequencies)")
    if len(v) == 3 and v[0] == 1 and v[2] == 0 and v[1] >= 1:
        if v[1] >= 8:
            return (f"dynamics: a richly structured basin network ({v[1]} holes — "
                    "check for Wada / riddled basins)")
        return f"dynamics: {v[1]} basin region(s) / islands (a Swiss-cheese state space — repellors sit in the holes)"
    if len(v) == 3 and v[0] == 1 and v[2] == 1 and v[1] >= 2 and v[1] % 2 == 0:
        return (f"dynamics: a genus-{v[1] // 2} handle-body state space "
                f"(g = {v[1] // 2} topological holes — e.g. synchronized dynamics on a 2D torus-like shell)")
    if len(v) == 2 and v[0] == 1 and v[1] >= 1:
        if v[1] == 2:
            return "dynamics: two linked cycles (figure-eight — homoclinic/heteroclinic candidates)"
        return f"dynamics: a limit-cycle network with {v[1]} cycles"
    if len(v) >= 2 and v[0] == 1 and all(b == 0 for b in v[1:]):
        return "dynamics: one connected region free of cycles/holes — a single attracting point or basin (a sink)"
    return f"dynamics: unrecognized ({v}) — typically multistability or a high-dim manifold"


_DYNAMICS_TRANSITIONS: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...] = (
    ((1,), (1, 1), "Hopf bifurcation: a fixed point spawns a limit cycle (β₁: 0 → 1)"),
    ((1,), (2,), "pitchfork / symmetry-breaking: one fixed point splits in two (β₀: 1 → 2)"),
    ((2,), (1,), "saddle–node / crisis: two fixed points annihilate into one (β₀: 2 → 1)"),
    ((1, 1), (1, 2, 1), "Neimark–Sacker: a limit cycle grows a second frequency → 2-torus"),
    ((1, 2, 1), (1,), "torus collapse: quasi-periodicity dies back to a point (β₁: 2 → 0, β₂: 1 → 0)"),
    ((1, 3, 3, 1), (1,), "Ruelle–Takens–Newhouse: the 3-torus breaks down, chaos candidate"),
    ((2, 2), (1, 1), "two limit cycles merge into one (attractor collision)"),
    ((1, 1), (1,), "blue-sky catastrophe / saddle-node on a cycle: the loop shrinks to a point (β₁ → 0)"),
)


def _dynamics_transition(prev: list[int], cur: list[int]) -> str:
    """If consecutive slider states (prev, cur) cross a known BIFURCATION, name it. Empty
    string when nothing classic matches."""
    a = _strip_trailing_zeros(prev)
    b = _strip_trailing_zeros(cur)
    for pa, pb, name in _DYNAMICS_TRANSITIONS:
        if list(pa) == a and list(pb) == b:
            return name
    return ""


_DYNAMICS_REFS: tuple[tuple[str, list[int]], ...] = (
    ("fixed point", [1]), ("2 fixed points", [2]), ("3 fixed points", [3]),
    ("limit cycle", [1, 1]), ("2 disjoint cycles", [2, 2]),
    ("2-torus T² attractor", [1, 2, 1]), ("3-torus T³ attractor", [1, 3, 3, 1]),
    ("4-torus T⁴ attractor", [1, 4, 6, 4, 1]),
    ("Lorenz/Chua attractor", [1, 2, 0]), ("Rössler-type attractor", [1, 1, 0]),
    ("sphere S²", [1, 0, 1]), ("sphere S³", [1, 0, 0, 1]), ("sphere S⁴", [1, 0, 0, 0, 1]), ("ℂP² state space", [1, 0, 1, 0, 1]),
    ("basin network", [1, 4, 0]), ("basin islands", [1, 2, 0]),
)


def _closest_dynamics(betti: list[int], top: int = 3) -> list[dict[str, object]]:
    """Nearest attractor/repellor families to a Betti vector (dynamics similarity)."""
    v = [int(b) for b in betti]
    scored: list[tuple[int, str]] = []
    for name, sig in _DYNAMICS_REFS:
        if _sig_matches(v, sig):
            continue
        L = max(len(v), len(sig))
        dist = sum(abs(a - b) for a, b in zip(_pad(v, L), _pad(sig, L)))
        for i in range(L):
            pass  # the padded L1 above already covers the dim-info columns
        if L <= 4:
            dist += abs(sum(v[i] for i in range(2, len(v)) if i % 2 == 0 and i >= 2)
                        - sum(sig[i] for i in range(2, len(sig)) if i % 2 == 0 and i >= 2))
        scored.append((dist, name))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [{"name": name, "dist": int(dist)} for dist, name in scored[:top]]


def _exact_torus2(nu: int, nv: int, kind: str, label: str,
                  R: float = 1.0, r: float = 0.35) -> tuple[FilteredComplex, np.ndarray, str, list[int], int]:
    """The exact T^2 cell complex with a positional filtration (vertices, then edges,
    then faces), displayed as a bagel surface. Clean beta = [1, 2, 1] and instant --
    unlike a Rips bagel, which over-fills and never reads a clean beta_1 = 2.

    Vertex index = i + j*nu (i in [0,nu) major, j in [0,nv) tube), which matches
    generators.donut_grid(nu, nv) and _torus_surface(nu, nv, R, r) exactly, so a CSV of
    a donut_grid pairs 1:1 with this complex's vertices."""
    C0 = make_torus_grid_complex(2, (nu, nv))
    simps = list(C0.simplexes)

    def i_of(j: int) -> int:
        return j % nu  # major index (drives the positional filtration phase)

    vals: list[float] = []
    for s in simps:
        d = len(s) - 1
        if d == 0:
            vals.append(0.0)
        elif d == 1:
            ph = (i_of(s[0]) + i_of(s[1])) / 2.0 / nu
            vals.append(1.0 + 0.8 * ph)
        else:
            ph = (i_of(s[0]) + i_of(s[1]) + i_of(s[2])) / 3.0 / nu
            vals.append(2.0 + 0.8 * ph)
    C = FilteredComplex(
        simps, np.array(vals, dtype=np.float64),
        np.array([len(s) - 1 for s in simps], dtype=np.int64), kind,
        {"nu": nu, "nv": nv},
    )
    pts = _torus_surface(nu, nv, R, r)
    C.params["raw"] = pts
    # SAFEGUARD (runtime, reliability): the exact T^2 complex MUST read [1, 2, 1] at the
    # top of the slider. If a refactor ever changes this, fail loudly instead of
    # silently showing a wrong torus. The exact complex is small, so this check is cheap.
    got = persistent_homology(C).betti_at(float(C.values.max()))[:3]
    if got != [1, 2, 1]:
        raise VrtdaError(
            f"internal error: exact T^2 complex (nu={nu}, nv={nv}, kind={kind!r}) read beta={got}, "
            f"expected [1, 2, 1] -- the reliable clean-torus path has regressed."
        )
    return C, pts, label, [1, 2, 1], int(C.max_dim())


def _even_grid_count(angles: np.ndarray, max_count: int = 512) -> int | None:
    """If `angles` sit on a regular cyclic grid of `cnt` evenly-spaced lines (each line
    possibly holding several points), return the FINEST such cnt (searched high->low);
    else None. This is how a torus-grid's major/tube index count is recovered."""
    n = len(angles)
    for cnt in range(min(max_count, n), 1, -1):
        if n % cnt:
            continue
        width = 2.0 * np.pi / cnt
        line = np.round((angles % (2.0 * np.pi)) / width).astype(np.int64) % cnt
        per_bin = np.bincount(line, minlength=cnt)
        if per_bin.min() != n // cnt:      # uneven occupancy -> not a regular grid
            continue
        res = np.abs(angles - line * width)
        res = np.minimum(res, 2.0 * np.pi - res)
        if res.max() < 0.45 * width:       # every point close to a grid line
            return cnt
    return None


def _detect_torus_grid(X: np.ndarray, max_count: int = 512) -> tuple[int, int, float, float] | None:
    """Detect whether X is a regular grid on a torus (bagel) surface. Returns
    (nu, nv, R, r) or None. A donut_grid(nu, nv, R, r) has nu evenly-spaced major angles
    (u=atan2(y,x)) and nv evenly-spaced tube angles (v=atan2(z, rho-R)), with
    rho=sqrt(x^2+y^2) in [R-r, R+r] and n = nu*nv. Used to reconstruct the EXACT T^2
    complex from such a CSV (a clean torus) instead of Rips (which over-fills)."""
    n = X.shape[0]
    if n < 16 or X.shape[1] < 3 or not np.isfinite(X).all():
        return None
    u = np.arctan2(X[:, 1], X[:, 0])
    rho = np.hypot(X[:, 0], X[:, 1])
    R = 0.5 * (float(rho.max()) + float(rho.min()))
    r = 0.5 * (float(rho.max()) - float(rho.min()))
    if r <= 1e-6 or R <= 1e-6 or r >= R:      # r>=R is not a bagel (no hole)
        return None
    v = np.arctan2(X[:, 2], rho - R)
    nu = _even_grid_count(u, max_count)
    nv = _even_grid_count(v, max_count)
    if nu is None or nv is None or nu * nv != n:
        return None
    return int(nu), int(nv), float(R), float(r)


# ---- robust torus detection + fitted T^2 reconstruction -------------------
# A torus point cloud does NOT have to be a perfectly regular grid to be shown
# as a clean (closed) torus: we detect the torus SHAPE (ring R, tube r) and the
# grid size (nu, nv) from the point count, then rebuild the exact T^2 cell
# complex ON TOP OF THE ACTUAL POINTS (arranged by their major/tube angles).
# At the top of the slider the complex is a closed T^2 -> beta = [1, 2, 1],
# i.e. everything is connected and no holes remain -- the maximum the slider can
# reach. Vietoris-Rips on a dense bagel can NEVER do this (it over-fills; see the
# IMPORTANT block at the top of this file); the fitted T^2 can, and it uses the
# user's real points. EVERY step fails loudly (a VrtdaError carrying the full
# state and the reason) instead of silently showing a wrong torus.


def _torus_error(why: str, **state: object) -> VrtdaError:
    """Build a VrtdaError that says WHY a torus-reconstruction step failed and dumps the
    full relevant state, so a misstep at ANY calculation is loud, not silent."""
    lines = [f"[torus reconstruction] {why}"]
    for k in sorted(state):
        lines.append(f"    {k}: {state[k]}")
    return VrtdaError("\n".join(lines))


def _covers_circle(angles: np.ndarray, max_gap_frac: float = 0.5) -> bool:
    """True iff `angles` wrap around nearly the whole circle (the largest gap -- including
    the wrap-around gap -- is < max_gap_frac * 2*pi). A full torus ring/tube must."""
    s = np.sort(angles % (2.0 * np.pi))
    if s.size < 2:
        return False
    gaps = np.diff(s)
    wrap_gap = float(s[0] + 2.0 * np.pi - s[-1])
    largest = float(max(float(gaps.max()), wrap_gap))
    return largest < max_gap_frac * (2.0 * np.pi)


def _all_factors(n: int) -> list[tuple[int, int]]:
    """All (a, b) with a*b = n and 2 <= a <= b. Empty if n is prime or < 4."""
    from math import isqrt
    return [(a, n // a) for a in range(2, isqrt(n) + 1) if n % a == 0]


def _rank_assign(u: np.ndarray, v: np.ndarray, nu: int, nv: int) -> np.ndarray:
    """Assign the n = nu*nv points to the T^2 grid's vertices by (u, v) rank: the i-th
    major-angle group (u-sorted, nu consecutive groups of nv) becomes ring i, and within a
    ring the v-sort gives the tube position j. Vertex k = i + j*nu gets the assigned point.
    Requires n % nu == 0 (true for an exact factorization)."""
    n = len(u)
    per = n // nu
    order_u = np.argsort(u, kind="stable")
    assign = np.empty(n, dtype=np.int64)
    for i in range(nu):
        ring = order_u[i * per:(i + 1) * per]
        for j, pidx in enumerate(ring[np.argsort(v[ring], kind="stable")]):
            assign[i + j * nu] = pidx
    return assign


def _grid_edge_median(X: np.ndarray, assign: np.ndarray, nu: int, nv: int) -> float:
    """Median Euclidean length of the 2 forward grid-neighbour edges per vertex (the full
    grid has 2n such edges). A factorization that matches the cloud's real ring/tube
    structure keeps these short; a wrong one strings far-apart points together."""
    pts = X[assign]
    n = pts.shape[0]
    lens = np.empty(2 * n, dtype=np.float64)
    for k in range(n):
        i = k % nu
        j = k // nu
        n_major = ((i + 1) % nu) + j * nu
        n_tube = i + ((j + 1) % nv) * nu
        lens[2 * k] = np.linalg.norm(pts[k] - pts[n_major])
        lens[2 * k + 1] = np.linalg.norm(pts[k] - pts[n_tube])
    return float(np.median(lens))


def _torus_fit(X: np.ndarray, max_count: int = 512) -> tuple[int, int, float, float] | None:
    """Robustly decide whether X is a (possibly IRREGULAR) sampling of a torus surface and
    return the fitted grid + radii (nu, nv, R, r), or None. A cloud is accepted as a torus
    iff it
      (a) is a bagel: rho = sqrt(x^2+y^2) spans [R-r, R+r] with a real hole, 0 < r < R;
      (b) is NOT a sphere/blob: a torus surface keeps r/R < 0.70 (measured 0.35-0.46 even
          for fat/heritage/noisy tori; spheres/blobs measure 0.93-0.99);
      (c) wraps around BOTH the ring and the tube (full 2pi circles, no more than a
          half-circle gap);
      (d) is a thin 2D surface: median tube-distance residual < 0.35 (true tori measure
          ~0, noisy ones ~0.24; filled volumes / blobs fail this AND (b)).
    For a clean grid the exact (nu, nv) is recovered; otherwise the 2-factor (nu, nv) of n
    whose grid-neighbour edges are shortest (the fit matching the real ring structure) is
    chosen. Every check is a hard rejection, so a non-torus cloud returns None."""
    n = X.shape[0]
    if n < 16 or X.shape[1] < 3 or not np.isfinite(X).all():
        return None
    rho = np.hypot(X[:, 0], X[:, 1])
    R = float(0.5 * (float(rho.max()) + float(rho.min())))
    r = float(0.5 * (float(rho.max()) - float(rho.min())))
    if not (1e-9 < r and r < 0.70 * R):   # no real hole, or sphere/blob-like fatness
        return None
    u = np.arctan2(X[:, 1], X[:, 0])
    v = np.arctan2(X[:, 2], rho - R)
    if not (_covers_circle(u) and _covers_circle(v)):
        return None                       # must wrap around the ring AND the tube
    tube_dist = np.hypot(X[:, 2], rho - R)
    resid = np.abs(tube_dist - r) / max(r, 1e-12)
    if float(np.median(resid)) > 0.35 or float(np.percentile(resid, 90)) > 0.60:
        return None                       # not a thin torus surface (blob/filled volume)
    nu = _even_grid_count(u, max_count)
    nv = _even_grid_count(v, max_count)
    if nu is not None and nv is not None and nu * nv == n:
        return int(nu), int(nv), R, r     # clean regular grid -> exact counts
    facs = _all_factors(n)
    if not facs:
        return None                       # no 2-factor grid -> cannot rebuild a T^2
    best: tuple[int, int] | None = None
    best_med = float("inf")
    for fa, fb in facs:
        med = _grid_edge_median(X, _rank_assign(u, v, fa, fb), fa, fb)
        if med < best_med:
            best_med = med
            best = (fa, fb)
    if best is None:
        return None
    return int(best[0]), int(best[1]), R, r   # irregular sampling -> best-fit (nu, nv)


def _assign_torus_grid(X: np.ndarray, nu: int, nv: int, R: float, r: float) -> np.ndarray:
    """Assign the n = nu*nv points to the T^2 grid's vertices so that complex vertex
    k = i + j*nu (i = major ring, j = tube position) sits at the point with the i-th
    major-angle rank and, within its ring, the j-th tube-angle rank (the EXACT ranking
    that _torus_fit scored). Returns `assign` where assign[k] is the input row shown at
    complex vertex k. Verifies (a) the grid fits the point count and (b) the assignment
    is a bijection over all n points -- otherwise the T^2 would silently show a wrong
    torus, so this fails loudly."""
    n = X.shape[0]
    if nu * nv != n:
        raise _torus_error("grid does not fit the point count",
                           nu=nu, nv=nv, n=n, nu_nv=nu * nv, r=r, R=R)
    rho = np.hypot(X[:, 0], X[:, 1])
    u = np.arctan2(X[:, 1], X[:, 0])
    v = np.arctan2(X[:, 2], rho - R)
    if n % nu != 0 or n // nu != nv:
        raise _torus_error("point count not divisible by the ring count",
                           n=n, nu=nu, nv=nv, per=n // nu)
    assign = _rank_assign(u, v, nu, nv)
    used = set(int(x) for x in assign)
    if len(used) != n or min(used) != 0 or max(used) != n - 1:
        raise _torus_error("grid assignment is not a bijection over the points",
                           n=n, nu=nu, nv=nv, n_used=len(used),
                           min_used=(min(used) if used else None),
                           max_used=(max(used) if used else None))
    return assign


def _fit_torus2(X: np.ndarray, nu: int, nv: int, R: float, r: float,
                kind: str, label: str) -> tuple[FilteredComplex, np.ndarray, str, list[int], int]:
    """Rebuild the EXACT T^2 cell complex fitted onto the ACTUAL point cloud X (which
    need not be a regular grid), with a GEOMETRIC filtration:
        vertex value 0; edge value = Euclidean distance between its two real points;
        face value = max of its three edge values.
    At the top of the slider every edge and face is present, so the complex is a closed
    T^2 -- beta exactly [1, 2, 1], "everything connected, no holes" -- the maximum the
    slider can reach. The points shown are the user's REAL points (in grid order), so the
    3D view is their own bagel, fully filled. Guardrails below fail loudly on any
    misstep (non-finite/zero edge, broken filtration, beta != [1, 2, 1] at max)."""
    assign = _assign_torus_grid(X, nu, nv, R, r)
    pts = X[assign]
    C0 = make_torus_grid_complex(2, (nu, nv))
    simps = list(C0.simplexes)

    _edge_val: dict[tuple[int, int], float] = {}

    def _edge(a: int, b: int) -> float:
        key = (a, b) if a < b else (b, a)
        if key in _edge_val:
            return _edge_val[key]
        d = float(np.linalg.norm(pts[a] - pts[b]))
        if not np.isfinite(d):
            raise _torus_error("non-finite edge distance",
                               a=a, b=b, d=d, nu=nu, nv=nv)
        if d <= 0.0:
            raise _torus_error("zero edge distance (coincident grid-neighbours)",
                               a=a, b=b, nu=nu, nv=nv, pt_a=pts[a].tolist(), pt_b=pts[b].tolist())
        _edge_val[key] = d
        return d

    vals: list[float] = []
    for s in simps:
        d = len(s) - 1
        if d == 0:
            vals.append(0.0)
        elif d == 1:
            vals.append(_edge(s[0], s[1]))
        else:
            vals.append(max(_edge(s[0], s[1]), _edge(s[0], s[2]), _edge(s[1], s[2])))
    # Monotonicity guard: every face must be born at or after its edges (face value is
    # the max of its edge values by construction; verify that the model agrees).
    for s, v in zip(simps, vals):
        if len(s) == 3:
            a, b, c = s
            mx = max(_edge(a, b), _edge(a, c), _edge(b, c))
            if float(v) + 1e-12 < mx:
                raise _torus_error("face born before an edge (broken filtration)",
                                   nu=nu, nv=nv, face=s,
                                   face_value=float(v), max_edge_value=float(mx))
    C = FilteredComplex(
        simps, np.array(vals, dtype=np.float64),
        np.array([len(s) - 1 for s in simps], dtype=np.int64),
        kind, {"nu": nu, "nv": nv, "R": R, "r": r, "fitted": True, "raw": pts},
    )
    # SAFEGUARD (runtime, reliability): at the top of the slider the fitted T^2 MUST
    # read exactly [1, 2, 1] -- a closed torus with no holes. If it ever does not, fail
    # loudly with the full state instead of silently showing a wrong topology.
    got = persistent_homology(C).betti_at(float(C.values.max()))[:3]
    if got != [1, 2, 1]:
        raise _torus_error("fitted T^2 did not close up (beta != [1, 2, 1] at max eps)",
                           nu=nu, nv=nv, R=R, r=r, got=got,
                           n_points=X.shape[0], n_simplices=C.n_simplices,
                           value_max=float(C.values.max()))
    return C, pts, label, [1, 2, 1], int(C.max_dim())


def build_source(args: argparse.Namespace) -> tuple[FilteredComplex, np.ndarray, str, list[int] | None, int]:
    """Return (complex, display_points_3d, projection_label, target_betti_or_None, report_dim).

    report_dim is the highest Betti number to display. For a shape of intrinsic
    dimension k the Rips complex is built with max_dim = k + 1, so simplices up to
    dimension k + 1 fill the "(d+1)-shells" that would otherwise persist as spurious
    H_d (this is what produced beta_2 = 257 on a 2-torus when capped at triangles).
    We then report only beta_0..beta_k, which is the true topology. Validation suite:
    tests/test_betti_shapes.py.
    """
    if args.points is not None:
        # The point-cloud path. Two sub-cases:
        #  (1) The cloud is a torus SURFACE (a regular grid OR an irregular sampling, e.g.
        #      `make_torus --kind donut --grid`). We REBUILD THE EXACT T^2 CELL COMPLEX on
        #      the actual points (_fit_torus2) -> a closed torus, beta=[1,2,1], no holes at
        #      the max epsilon -- the reliable answer to "load my bagel, show me a torus".
        #      Vietoris-Rips on a dense bagel over-fills and never closes up (see the
        #      IMPORTANT block above), so this fitted complex is what actually makes it a
        #      torus. --no-exact-torus forces the honest Rips instead.
        #  (2) Any other cloud: plain Vietoris-Rips over the full Dmax range.
        X = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols).data
        if not getattr(args, "no_exact_torus", False):
            fit = _torus_fit(X)
            if fit is not None:
                nu, nv, R, r = fit
                console.print(f"[cyan]NOTE: detected a torus surface in {args.points} "
                              f"({nu}x{nv} grid, R={R:.3f}, r={r:.3f}); rebuilding the exact "
                              f"T^2 cell complex ON your points (a closed torus, beta=[1,2,1], "
                              f"no holes at the max epsilon) instead of the Vietoris-Rips "
                              f"complex that over-fills a dense bagel. Use --no-exact-torus "
                              f"to force Rips.[/cyan]")
                C, pts, _label, target, rd = _fit_torus2(X, nu, nv, R, r, "donut",
                                                         "torus (exact T^2 fitted to your points)")
                return C, pts, "exact T^2 (fitted to points)", target, rd

        # (2) Arbitrary point cloud via Rips.
        D = pairwise_distances(X, args.metric)
        max_dim = max(3, args.max_dim)
        # The slider spans the FULL Vietoris-Rips range: eps 0 -> max pairwise distance
        # (Dmax), or 0 -> --eps-max if the user gives an explicit (smaller) ceiling.
        Dmax = float(D.max())
        eps_hi = Dmax if args.eps_max is None else min(float(args.eps_max), Dmax)
        # A DENSE cloud's Dmax is infeasible (the Rips complex would have millions of
        # simplices -> the browser and the pure-Python homology both choke). We cap at
        # the largest FEASIBLE epsilon and say so. This is not ignoring --eps-max: it is
        # the largest value that can actually be built. Sparse clouds reach Dmax in full.
        eps_max = _max_feasible_eps(X, D, eps_hi, budget=_FEASIBLE_MAX_BUDGET)
        if eps_max < eps_hi - 1e-9:
            console.print(f"[yellow]NOTE: max distance {eps_hi:.3f} is infeasible for this "
                          f"{X.shape[0]}-pt cloud (the Rips complex would have millions of "
                          f"simplices); capping the slider at the largest feasible "
                          f"{eps_max:.3f}. For a clean torus use `--shape donut` (or drop "
                          f"--no-exact-torus if this is a torus grid).[/yellow]")
        C = _build_rips_safe(X, D, eps_max, max_dim)
        n_faces = sum(1 for s in C.simplexes if len(s) == 3)
        if _is_overfilling(X.shape[0], n_faces):
            _rich_ui.overfill_note(console, X.shape[0], n_faces)
        pts, proj = _to3d(X, "rips")
        _attach_raw(C, X)
        return C, pts, proj, None, min(max_dim - 1, 3)

    if args.shape == "torus-grid":
        return _exact_torus2(args.n, args.n, "torus_grad", "torus-grid surface (exact T^2 cell complex)")

    if args.shape == "donut":
        # MITIGATION (a): the clean bagel = exact T^2 cell complex (NOT Rips). Fast and
        # reads exactly [1, 2, 1] at the top, both rings visible. THIS is the shape to
        # use to SEE a torus reliably. (Rips on a dense bagel over-fills -- see the
        # IMPORTANT note block at the top of this file.)
        m = max(args.nper, 8)
        return _exact_torus2(m, m, "donut", "donut / bagel (exact T^2 cell complex)")

    if args.shape == "donut-rips":
        # MITIGATION (b): Real Vietoris-Rips on a bagel, slider over the FULL range
        # eps 0 -> max pairwise distance. A genuine Rips demonstration, but the bagel
        # over-fills: beta_1 never cleanly reads 2 (it spikes then collapses) and
        # beta_2 blows up as the holes get triangulated away (see the IMPORTANT note
        # block at the top). Homology is slow, so cap the grid at nper <= 8 so a
        # feasible epsilon exists.
        nper = min(args.nper, 8)
        if nper != args.nper:
            print(f"[donut-rips] full-range Rips homology is slow; capping grid "
                  f"{args.nper}x{args.nper} -> {nper}x{nper}", file=sys.stderr)
        X = G.donut_grid(nper, nper)
        D = pairwise_distances(X, args.metric)
        eps_max = float(D.max())  # max pairwise distance -> slider 0 .. Dmax
        max_dim = 2  # edges + triangles (the bagel surface); skip slow tetrahedra
        with _rich_ui.timed(Console(), "Building Rips bagel (full range 0 -> max distance)"):
            C = _build_rips_safe(X, D, eps_max, max_dim)
        pts, proj = _to3d(X, "rips")
        _attach_raw(C, X)
        return C, pts, proj, [1, 2, 1], 2

    # Rips-based synthetic sources: each has a known intrinsic dimension k, and (for the
    # clean samplings) a known target Betti vector.
    if args.shape == "circle":
        X = G.circle_grid(args.n, radius=1.0); k = 1; target = [1, 1]
    elif args.shape == "product":
        X = G.product_torus_grid(args.k, args.nper); k = int(args.k)
        target = _torus_betti(k) if k <= 2 else None
    elif args.shape == "sphere":
        # --k is the sphere dimension: --k 2 -> S^2 (a ball in R^3). Use enough points
        # to read the surface (honours --n and --nper, whichever is larger).
        nsp = max(args.n, args.nper)
        X = G.sphere(nsp, dim=max(1, args.k), radius=1.0); k = max(1, args.k); target = None
    elif args.shape == "blobs":
        X = G.gmm(3, args.n, 3); k = 0; target = None
    else:
        raise SystemExit(f"unknown shape {args.shape!r}")

    D = pairwise_distances(X, args.metric)
    # Slider must span the FULL feasible Vietoris-Rips range (0 .. max pairwise
    # distance), exactly as the --points path does -- a connectivity-scaled ceiling is
    # too small to ever close a space. Capped at the largest FEASIBLE epsilon so a
    # dense cloud can't blow up the pure-Python homology or the browser. Exception:
    # the sparse product T^2 keeps its connectivity ceiling -- its clean [1,2,1]
    # plateau is exactly why the (already exact-torus-path) demo exists.
    if args.shape == "product" and k == 2:
        eps_max = _eps_max(D, args.frac, getattr(args, "connect_margin", 1.2))
    else:
        eps_hi = float(D.max())
        eps_max = _max_feasible_eps(X, D, eps_hi, budget=_FEASIBLE_MAX_BUDGET)
        if eps_max < eps_hi - 1e-9:
            console.print(f"[yellow]NOTE: max distance {eps_hi:.3f} is infeasible for this "
                          f"{X.shape[0]}-pt Rips complex; capping the slider at the largest "
                          f"feasible {eps_max:.3f}.[/yellow]")
    # Cap at 3-simplices: this is a fast beta_0..beta_2 visualizer, and (k+1)-simplices
    # are combinatorially infeasible for k>=3 (5-cliques explode). For k>=3 the low
    # dimensions beta_0..beta_2 are still exact under Rips; the top class needs the
    # exact cell complex (see tests/test_betti_shapes.py).
    max_dim = min(max(k + 1, 2, args.max_dim), 3)
    C = _build_rips_safe(X, D, eps_max, max_dim)
    _attach_raw(C, X)
    if args.shape == "product" and k == 2:
        # The R^4 product 2-torus is 4-fold symmetric, so a PCA-3D view collapses to a
        # bare cylinder. Show the classic bagel instead: donut_grid(n,n) is the exact
        # same (u_i, v_j) grid (index = i*n + j), so its points pair 1:1 with the
        # complex vertices while the topology is still computed in the clean R^4 cloud.
        pts = G.donut_grid(args.nper, args.nper)
        proj = "2-torus bagel (R³ view) · topology computed in R⁴"
    else:
        pts, proj = _to3d(X, "rips")
    return C, pts, proj, target, min(k, 2)


def _round_list(x: list[float], nd: int = 5) -> list[float]:
    return [round(float(v), nd) for v in x]


def _attach_raw(C: FilteredComplex, X: np.ndarray) -> None:
    """Attach the original (possibly multi-dimensional) coordinates of the complex's
    vertices to C.params, so the browser can re-project the view (PCA / raw dims /
    random) without changing the (independent) topology. Vertex i <-> cloud row i."""
    C.params["raw"] = X


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    C, pts, proj, target, report_dim = build_source(args)
    # Persistent homology is pure-Python and slow on large complexes (e.g. the
    # full-range Rips bagel). Show a transient progress bar when it will take a while.
    n_simp = C.n_simplices
    if n_simp > 30000:
        with Progress(transient=True) as progress:
            task = progress.add_task("persistent homology", total=n_simp)

            def _cb(j: int, n: int) -> None:
                progress.update(task, completed=j)

            bc = persistent_homology(C, progress_cb=_cb)
    else:
        bc = persistent_homology(C)
    eps_max = float(C.values.max())
    n_grid = args.n_grid
    grid = np.linspace(0.0, eps_max, n_grid)
    table = bc.betti_function(grid)
    maxdim = min(report_dim, int(table.shape[1] - 1))
    table = table[:, :maxdim + 1]

    # Per-slider-position homotopy-type recognition. The BETA VECTOR read straight off a
    # Rips complex at a single epsilon over-fills (dense clouds never read true) and its
    # TOP dimension is unpaired garbage -- so the recognition layer runs on the PROMINENT
    # (persistent) classes that are alive at each grid point. For exact cell complexes
    # the top dimension is claimed too (its classes are genuine). The raw table is still
    # what the cards show; only the hypothesis text uses the persistent structure.
    topo_rows: list[list[int]] = table.astype(int).tolist()
    top_claimable = C.kind != "rips"
    prom = _prominent_intervals(bc, eps_max, maxdim, top_claimable=top_claimable)
    struct_rows = _struct_at(bc, grid, maxdim, top_claimable=top_claimable, prom=prom)
    topo_messages = [_topology_name(row) for row in struct_rows]
    topo_closest = [_closest_topologies(row) for row in struct_rows]
    dyn_messages = [_dynamics_name(row) for row in struct_rows]
    dyn_closest = [_closest_dynamics(row) for row in struct_rows]
    dyn_transitions: list[str] = [""] * len(topo_rows)
    for i in range(1, len(topo_rows)):
        dyn_transitions[i] = _dynamics_transition(topo_rows[i - 1], topo_rows[i])
    # systematic per-dimension readings for the H₀/H₁/H₂ toggles: (a) one reading line
    # per dim per grid row, (b) the recognition for EVERY active-dim mask (the 'only in
    # single dims' view), (c) the prominent interval cards per dim.
    dim_summaries = [_dimension_summaries(r, top_claimable) for r in struct_rows]
    masked_messages = _masked_recognitions(struct_rows, maxdim)
    dim_intervals = _dim_interval_table(prom, eps_max)
    # Original (multi-dimensional) coordinates of the displayed points, so the browser can
    # re-project interactively (PCA / raw dims / random) without touching the topology.
    raw_cloud = np.asarray(C.params.get("raw", pts), dtype=np.float64)
    if raw_cloud.shape[0] != len(pts):
        raise VrtdaError(
            f"internal error: raw cloud size {raw_cloud.shape[0]} != display points "
            f"{len(pts)} (projection panel would silently misalign); fix build_source.")
    if np.ndim(raw_cloud) != 2 or raw_cloud.shape[1] < 1:
        raise VrtdaError(f"internal error: unplottable raw cloud shape={raw_cloud.shape}")
    shell_facts = _shell_facts(raw_cloud)

# FEATURE: the persistent STRUCTURE -- the honest one-line answer. Priority:
#   (1) EXACT cell complex      -> the closed complex at the top IS the answer.
#   (2) GEOMETRIC ROUND SHELL   -> read the sphere/torus purely from geometry; only the
#                                  number of STRONG 1-cycles (length >= 0.30 x full range)
#                                  distinguishes a k-sphere (0) from a torus-hypersurface.
#   (3) Rips cloud (no shell)   -> longest STABLE PLATEAU of the Betti signature: the ε
#                                  window where nothing changes (a torus sits at [1,2,0]
#                                  for a long stretch while the over-filling tail and the
#                                  ε=0 "all-separate" start are both length-1 noise).
# Priority (3) result is a hypothesis only -- Rips dense clouds never guarantee a read.
    top_claimable = C.kind != "rips"
    feat_index = int(len(table) - 1)
    feat_route = "exact cell complex (closed complex at max ε)"
    if C.kind != "rips":
        feat_row = [int(b) for b in table[-1]]
    else:
        feat_row = [0] * (maxdim + 1)
        feat_index = 0
        if shell_facts and int(shell_facts["tangent"]) >= 1:
            h1 = sum(1 for iv in bc.intervals
                     if int(iv.dim) == 1 and float(iv.length) >= 0.30 * eps_max)
            k = int(shell_facts["tangent"])
            feat_route = "geometric shell read (all points ≈R from centroid, SVD tangent dim)"
            if k == 1:
                feat_row = [1, 1] + [0] * (maxdim - 1)
                feat_index = len(table) - 1
            else:
                f1 = min(h1, k)
                feat_row = [1] + [f1] + [0] * (maxdim - 1)
                feat_index = len(table) - 1

        if feat_index == 0:
            # (3) plateau read
            feat_route = "persistent-plateau hypothesis (Rips, longest stable run of the Betti signature)"
            capped = [[(int(v) if (i < maxdim or maxdim == 0) else 0)
                       for i, v in enumerate(row)] for row in table]

            def _trivial(row: list[int]) -> bool:
                return row[0] == 1 and all(v == 0 for v in row[1:])

            runs: list[tuple[int, int, list[int]]] = []
            cur: list[int] | None = None
            cur_start = 0
            for i in range(len(capped) + 1):
                row = capped[i] if i < len(capped) else None
                if row != cur:
                    if cur is not None:
                        runs.append((cur_start, i - cur_start, cur))
                    cur = row
                    cur_start = i
            saw_structure = any(s[0] > 1 or any(v != 0 for v in s[1:])
                                for _, _, s in runs)
            cands = [(st, L, s) for (st, L, s) in runs
                     if not (saw_structure and (_trivial(s) or st == 0))]
            if not cands:
                cands = runs
            best_score = -1.0
            best_mid = 0
            for st, L, s in cands:
                mid = st + (L - 1) // 2
                score = float(L) + 0.5 * (1.0 if s[0] > 1 else 0.0) \
                    + 0.3 * (1.0 if any(v != 0 for v in s[1:]) else 0.0) \
                    - 0.3 * (mid / max(1, len(capped)))
                if score > best_score:
                    best_score = score
                    best_mid = mid
            feat_index = best_mid
            feat_row = [(int(v) if (i < maxdim or maxdim == 0) else 0)
                        for i, v in enumerate(table[feat_index])]
    topo_feature = _topology_name(feat_row)
    dyn_feature = _dynamics_name(feat_row)
    f1 = feat_row[1] if maxdim >= 1 else 0
    if shell_facts:
        k = int(shell_facts["tangent"])
        R = float(shell_facts["R"])
        rms = float(shell_facts["rms"])
        geo = f"every point sits ≈R={R:.3f} from the centroid (rms {rms:.1%}) — a {k}D tangent patch"
        if k == 1:
            topo_feature = (f"a 1-sphere / circle S¹ (radius R≈{R:.2f}): {geo}; one "
                            f"independent loop, no enclosed volume.")
            dyn_feature = _dynamics_name([1, 1])
        elif f1 == 0:
            topo_feature = (f"a {k}-sphere S^{k} shell (radius R≈{R:.2f}): {geo}; the enclosed "
                            f"volume is the TOP class — certify it with an explicit cell "
                            f"complex, a Rips cloud cannot (documented over-fill).")
            dyn_feature = _dynamics_name([int(feat_row[0]), 0, 0])
        else:
            topo_feature = (f"a round torus-hypersurface T^{f1} (radius R≈{R:.2f}): "
                            f"{geo}; the {f1} independent H_1 cycles are the frequencies.")

    n = len(pts)
    edges = [[int(a), int(b), round(float(C.values[C.index_of((int(a), int(b)))]), 5)]
             for (a, b) in (s for s in C.simplexes if len(s) == 2)]
    faces = [[int(a), int(b), int(c), round(float(C.values[C.index_of((int(a), int(b), int(c)))]), 5)]
             for (a, b, c) in (s for s in C.simplexes if len(s) == 3)]
    intervals = []
    for iv in bc.intervals:
        if iv.dim > maxdim:
            continue
        death = eps_max if not np.isfinite(iv.death) else float(iv.death)
        intervals.append([int(iv.dim), round(float(iv.birth), 5), round(death, 5)])

    title = args.title or C.kind
    extra = ""
    if "PCA" in proj:
        extra = f" · topology computed in the original high-dim space"
    sub = (f"{proj}{extra}  ·  {n} points  ·  {len(edges)} edges / {len(faces)} faces  ·  "
           f"ε_max = {eps_max:.3f}" + (f"  ·  target β = {target}" if target else ""))

    return {
        "mode": "filtration",
        "title": title,
        "sub": sub,
        "metric": args.metric,
        "eps_max": round(eps_max, 5),
        "projection": proj,
        "target": target,
        "maxdim": maxdim,
        "points": [[round(float(v), 5) for v in row] for row in pts],
        "raw": [[round(float(v), 5) for v in row] for row in raw_cloud],
        "edges": edges,
        "faces": faces,
        "betti": {"grid": _round_list(list(grid)), "table": table.tolist(), "maxdim": maxdim},
        "topo_messages": topo_messages,
        "topo_closest": topo_closest,
        "dyn_messages": dyn_messages,
        "dyn_closest": dyn_closest,
        "dyn_transitions": dyn_transitions,
        "struct_rows": struct_rows,
        "dim_summaries": dim_summaries,
        "masked_messages": masked_messages,
        "dim_intervals": dim_intervals,
        "topo_feature": topo_feature,
        "dyn_feature": dyn_feature,
        "feat_row": feat_row,
        "feat_route": feat_route,
        "shell_facts": shell_facts,
        "intervals": intervals,
    }


def parse_layers(spec: str | None, data_dir: str | Path | None = None) -> list[int]:
    """Parse a --layers spec: 'start:stop[:step]' (Python range, inclusive stop),
    a comma list ('0,16,32,64'), or an explicit space/comma list of ints. None -> all."""
    all_l = datasets.list_layers(data_dir=data_dir)
    if not spec or spec == "all":
        return all_l
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        a = int(parts[0]); b = int(parts[1])
        st = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        st = max(1, st)
        out = list(range(a, b + 1, st)) if a <= b else list(range(a, b - 1, -st))
        return [L for L in out if L in all_l]
    toks = [t for t in spec.replace(",", " ").split() if t]
    if not toks:
        return all_l
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        raise SystemExit(f"cannot parse --layers {spec!r}")
    return [L for L in vals if L in all_l]


def _cloud_dynamics_fingerprint(
        X3: np.ndarray, sub: int = 60, max_simplices: int = 40000) -> list[int]:
    """Cheap per-layer dynamics fingerprint for ONE time step's point cloud (already
    PCA-3D). Builds a Vietoris-Rips complex (max_dim=2, hard simplex budget) and reads
    THREE scale-free, persistence-based numbers instead of a raw single-scale Betti:
      beta_0' = components still separate past ~1.2x median-NN  (multistability, k basins)
      beta_1' = 1-cycles living >= 10% of the full distance range (loops / limit cycles)
      beta_2' = essential 2-voids (never filled: torus / sphere shells)
    These map straight onto the dynamics catalogue (fixed point / limit cycle / torus /
    Lorenz cycles / basins) and are robust because they ignore Rips "over-fill" garbage.
    Returns a hypothesis vector of length 3, capped so absurd values can't leak out."""
    from vrtda.complexes import build_rips
    n = len(X3)
    if n < 4:
        return [max(1, n), 0, 0]
    if n > sub:
        idx = np.linspace(0, n - 1, sub).astype(int)
        X3 = X3[idx]
        n = sub
    D = pairwise_distances(X3, "euclidean")
    dmax = float(np.max(np.triu(D, 1)))
    m = D + np.eye(n) * 1e15
    submin = float(np.median(m.min(1)))
    C = None
    for md in (2, 1):
        try:
            C = build_rips(X3, D, dmax, max_dim=md, max_simplices=max_simplices)
            break
        except TooLargeError:
            C = None
    if C is None:
        return [1, 0, 0]
    bc = persistent_homology(C)
    comps = 0
    h1 = 0
    h2 = 0
    for it in bc.intervals:
        d = int(it.dim)
        birth = float(it.birth)
        death = float(it.death)
        if d == 0:
            if death >= 1.2 * submin:
                comps += 1
        elif d == 1:
            if np.isinf(death):
                h1 += 1
            elif death - birth >= 0.10 * dmax:
                h1 += 1
        elif d == 2 and np.isinf(death):
            h2 += 1
    return [min(comps, 8), min(h1, 8), min(h2, 4)]


def _prominent_intervals(bc: Barcode, eps_max: float, maxdim: int,
                         thr: float = 0.15,
                         top_claimable: bool = False) -> dict[int, list[Interval]]:
    """PROMINENT (persistent) intervals per dimension: the classes the recognition layer
    trusts. Robust de-noising, scale-aware:

      * floor: a class must live >= thr x the FULL filtration range;
      * scale-gate (dim >= 1): if even the LONGEST survivor sits below half the filtration
        range, the deck is sub-structural noise -> claim nothing (0 classes);
      * gap-select: among the survivors, keep the prefix above the largest length gap;
      * top dimension: claimed ONLY for EXACT complexes (top_claimable=True) where the top
        classes are genuine. For Rips the top dimension is NEVER claimed: every class is
        essential there (nothing cancels it above the cap) and Rips over-fills the real
        topology (see the IMPORTANT note at the top) -- that is exactly the "the top class
        needs the exact cell complex" case. Silent "no claim" is a safe failure; a wrong
        number is not. Dimension 0 is always claimed: components kill cleanly everywhere.
    """
    out: dict[int, list[Interval]] = {d: [] for d in range(maxdim + 1)}
    floor = thr * float(eps_max)
    for d in range(maxdim + 1):
        ivs = [iv for iv in bc.intervals if int(iv.dim) == d]
        if d == maxdim and maxdim > 0 and not top_claimable:
            out[d] = []
            continue
        fin = sorted((iv for iv in ivs if np.isfinite(iv.length) and iv.length >= floor),
                     key=lambda iv: float(iv.length), reverse=True)
        ess = [iv for iv in ivs if iv.is_essential]
        keep: list[Interval] = []
        if d == 0:
            if fin:
                lens = [float(iv.length) for iv in fin]
                if len(lens) > 1:
                    gaps = [lens[i] - lens[i + 1] for i in range(len(lens) - 1)]
                    gi = int(max(range(len(gaps)), key=lambda i: gaps[i]))
                    if gaps[gi] >= 0.35 * lens[0]:
                        fin = fin[:max(gi + 1, 1)]
                keep = fin
                if not fin:
                    keep = fin
            keep = keep + ess
        else:
            if fin and float(fin[0].length) >= 0.5 * float(eps_max):
                lens = [float(iv.length) for iv in fin]
                if len(lens) > 1:
                    gaps = [lens[i] - lens[i + 1] for i in range(len(lens) - 1)]
                    gi = int(max(range(len(gaps)), key=lambda i: gaps[i]))
                    if gaps[gi] >= 0.35 * lens[0]:
                        fin = fin[:max(gi + 1, 1)]
                keep = fin + ess
        out[d] = keep
    return out


def _shell_facts(X: np.ndarray, rms_tol: float = 0.20,
                 k_nn: int = 4) -> dict[str, object] | None:
    """GEOMETRIC recognition of a round shell: are all points (almost) equidistant from
    the centroid? If yes, estimate the tangent dimension k by counting the substantial
    singular values of local k-NN patches (k_nn=4, a tight patch: enough to see the
    tangent plane, small enough not to be swamped by the curvature of a sphere sample).
    Returns raw facts only -- whether the shell is a sphere or a torus-like
    hypersurface is decided with the prominent Betti (a sphere has no H_1, a T^k has k)."""
    n, d = X.shape
    if n < k_nn + 2 or d < 2:
        return None
    c = X - X.mean(0)
    r = np.linalg.norm(c, axis=1)
    R = float(r.mean())
    if R < 1e-9:
        return None
    rms = float(np.sqrt(((r - R) ** 2).mean()) / R)
    if rms > rms_tol:
        return None
    D = pairwise_distances(X, "euclidean")
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k_nn]
    dims: list[int] = []
    for i in range(n):
        V = c[idx[i]] - c[i]
        s = np.linalg.svd(V, compute_uv=False)
        dims.append(int((s > 0.15 * s[0]).sum()))
    return {"round": True, "R": R, "rms": rms, "tangent": int(np.median(dims))}


def _dimension_summaries(row: list[int], top_claimable: bool) -> list[str]:
    """One systematic reading PER homology dimension of a structure row: what the k-th
    Betti number alone claims (components / loops / voids / top class). This is the
    per-dimension 'show me structure' view -- each line is what THAT dim contributes,
    decoupled from the others. The top-dim caveat (Rips over-fill) is attached here."""
    v = [int(x) for x in row]
    out: list[str] = []
    for d, b in enumerate(v):
        if d == 0:
            if b <= 0:
                out.append("no component at prominence scale")
            elif b == 1:
                out.append("1 connected component — the cloud is a single piece")
            else:
                out.append(f"{b} components / clusters still separate (as dynamics "
                           f"{b} basins) — merge as ε grows")
        elif d == 1:
            if b <= 0:
                out.append("no persistent 1-cycle — nothing winds around (no period, no handle)")
            elif b == 1:
                out.append("1 independent loop — one frequency / period (a limit-cycle or "
                           "S¹ factor; a Tⁿ skeleton needs n loops)")
            else:
                out.append(f"{b} independent loops — {b} frequencies (a T^{b}-type "
                           f"skeleton or a figure-8)")
        elif b <= 0:
            out.append(f"no persistent {d}-cycle — no enclosed {d}-void")
        elif b == 1:
            caveat = ("" if top_claimable else
                      " — Rips over-fill is documented, treat as hypothesis until an exact "
                      "cell complex certifies it")
            out.append(f"1 independent {d}-cycle — one enclosed {d}-void (a closed "
                       f"S^{d}/T^{d} top class{caveat})")
        else:
            out.append(f"{b} independent {d}-cycles — e.g. a genus-{b // 2} closed "
                       f"surface or a T^{d}-type void lattice")
    return out


def _masked_recognitions(rows: list[list[int]],
                         maxdim: int) -> list[dict[int, str]]:
    """Structure hypothesis for EVERY active-dimension subset (mask bit k = H_k kept):
    the systematic 'what if only these dims matter' view behind the H₀/H₁/H₂ toggles.
    A single-dim mask reads as the PURE that-dim structure; multi-dim masks rerun the
    full catalogue on the masked row. Bounded to 5 dims (32 masks) -- above that only
    the full mask is shipped (the toggles still hide bars/cards, the badge just says
    'masked recognition unavailable')."""
    D = maxdim + 1
    full = (1 << D) - 1
    if D <= 5:
        out: list[dict[int, str]] = []
        for r in rows:
            msgs: dict[int, str] = {}
            for m in range(1, full + 1):
                masked = [int(r[i]) if (m >> i) & 1 else 0 for i in range(D)]
                if m & (m - 1) == 0:  # single dimension kept -> the PURE that-dim structure
                    d = m.bit_length() - 1
                    b = masked[d]
                    if d == 0:
                        if b == 1:
                            msgs[m] = ("only H₀: everything is one component — the whole "
                                       "cloud collapses to a single piece")
                        else:
                            msgs[m] = (f"only H₀: {b} separate component(s) — the "
                                       f"cluster / basin count")
                    elif b == 1:
                        msgs[m] = (f"only H_{d}: 1 independent {d}-cycle — a single "
                                   f"{d}-dimensional feature (enclosed {d}-void / frequency)")
                    else:
                        msgs[m] = (f"only H_{d}: {b} independent {d}-cycle(s) — the pure "
                                   f"{d}-dimensional structure")
                else:
                    msgs[m] = _topology_name(masked)
            out.append(msgs)
        return out
    return [{full: _topology_name(r)} for r in rows]


def _dim_interval_table(prom: dict[int, list[Interval]],
                        eps_max: float) -> dict[int, list[list[float | None]]]:
    """Serializable interval cards per dimension (for the systematic panel): birth,
    death (None when essential), length, share of the full range in %, essential flag.
    'Alive at the current ε' is derived client-side from birth/death/essential."""
    tbl: dict[int, list[list[float | None]]] = {}
    for d, ivs in prom.items():
        rows: list[list[float | None]] = []
        for iv in ivs:
            death = None if not np.isfinite(float(iv.death)) else round(float(iv.death), 5)
            rows.append([round(float(iv.birth), 5), death,
                         round(float(iv.length), 5),
                         round(float(iv.birth) / eps_max * 100.0, 1) if eps_max > 0 else 0.0,
                         1.0 if bool(iv.is_essential) else 0.0])
        if rows:
            tbl[int(d)] = rows
    return tbl


def _struct_at(bc: Barcode, eps_grid: np.ndarray, maxdim: int,
               top_claimable: bool,
               prom: dict[int, list[Interval]] | None = None) -> list[list[int]]:
    """Prominent-Betti 'structure rows' at every slider grid point: which prominent
    classes are ALIVE there. The top dim is claimed only when the complex is exact.
    `prom` can be passed in when the caller computed it anyway (avoids re-derivation)."""
    if prom is None:
        prom = _prominent_intervals(bc, float(eps_grid[-1]), maxdim,
                                    top_claimable=top_claimable)
    rows: list[list[int]] = []
    for e in eps_grid:
        rows.append([sum(1 for iv in prom.get(d, []) if iv.alive_at(float(e)))
                     for d in range(maxdim + 1)])
    return rows


def build_layer_trajectory(args: argparse.Namespace) -> dict[str, object]:
    """Layer/trajectory mode: project the token clouds of several layers into one
    shared 3D frame (PCA of a few reference layers) so the tokens can be seen
    MOVING across the surface as the layer index (time) advances."""
    layers = parse_layers(args.layers, data_dir=args.data_dir)
    if not layers:
        raise SystemExit(f"no layers matched {args.layers!r}; available: {datasets.list_layers(data_dir=args.data_dir)}")
    ref_idx = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    ref_layers = [layers[i] for i in ref_idx]
    ref = [datasets.load_token_cloud(data_dir=args.data_dir, layer=L).data for L in ref_layers]
    stack = np.vstack(ref)
    center = stack.mean(0)
    _u, _s, Vt = np.linalg.svd(stack - center, full_matrices=False)
    frame = Vt[:3]

    traj = np.zeros((len(layers), ref[0].shape[0], 3), dtype=np.float64)
    spread = np.zeros(len(layers), dtype=np.float64)
    for i, L in enumerate(layers):
        X = datasets.load_token_cloud(data_dir=args.data_dir, layer=L).data
        P3 = (X - center) @ frame.T
        traj[i] = P3
        spread[i] = float(np.linalg.norm(P3 - P3.mean(0), axis=1).mean())

    ref_ps = datasets.load_token_cloud(data_dir=args.data_dir, layer=layers[0])
    labels = list(ref_ps.labels)
    ntok = len(labels)
    prompts = [int(lbl.split("_")[0]) for lbl in labels]
    order = sorted(set(prompts))
    group_of = [order.index(q) for q in prompts]
    try:
        texts = datasets.token_texts(data_dir=args.data_dir, layer=layers[0])
        ptexts = [next((texts[t] for t in range(ntok) if group_of[t] == g), str(order[g])) for g in range(len(order))]
    except Exception:
        ptexts = [str(order[g]) for g in range(len(order))]

    # STRUCTURE + ATTRACTOR DETECTION (per sampled layer): each layer is one time step of
    # the system, so a Betti fingerprint per layer is read against BOTH the topology
    # catalogue (what homotopy type the token cloud looks like) and the dynamics catalogue
    # (fixed points / limit cycles / tori / Lorenz-type / basin networks / bifurcation
    # signatures). Hypotheses only -- see the W-caveats below.
    try:
        strided = layers[::max(1, len(layers) // 12)]
        dyn_rows: dict[int, list[int]] = {}
        for L in strided:
            i = layers.index(L)
            dyn_rows[int(L)] = _cloud_dynamics_fingerprint(traj[i])
        all_L: list[int] = list(dyn_rows.keys())
        topo_rows: dict[int, list[int]] = dict(dyn_rows)
        topo_messages: dict[int, str] = {
            int(L): _topology_name(r) for L, r in dyn_rows.items()}
        topo_closest: dict[int, list[dict[str, object]]] = {
            int(L): _closest_topologies(r) for L, r in dyn_rows.items()}
        dyn_messages: dict[int, str] = {
            int(L): _dynamics_name(r) for L, r in dyn_rows.items()}
        dyn_closest: dict[int, list[dict[str, object]]] = {
            int(L): _closest_dynamics(r) for L, r in dyn_rows.items()}
        dyn_transitions: dict[int, str] = {}
        for a, b in zip(all_L[:-1], all_L[1:]):
            dyn_transitions[int(b)] = _dynamics_transition(
                dyn_rows[a], dyn_rows[b])
        # a whole-trajectory "attractor-set" probe on a downsampled union of the shared frame
        union = np.vstack([traj[layers.index(L)][:: max(1, ntok // 30)] for L in strided])
        global_betti = _cloud_dynamics_fingerprint(union, sub=120)
        dyn_global = _dynamics_name(global_betti)
        topo_global = _topology_name(global_betti)
        # per-layer systematic views (same masked-panel mechanism as filtration, D=3)
        row_list: list[list[int]] = [r for _, r in dyn_rows.items()]
        key_list: list[int] = [int(L) for L in dyn_rows.keys()]
        masked_flat = _masked_recognitions(row_list, 2)
        topo_masked: dict[int, dict[int, str]] = {
            k: m for k, m in zip(key_list, masked_flat)}
        topo_dims: dict[int, list[str]] = {
            int(L): _dimension_summaries(r, False) for L, r in dyn_rows.items()}
    except Exception:
        dyn_messages, dyn_closest, dyn_transitions, dyn_global = {}, {}, {}, ""
        topo_messages, topo_closest, topo_rows, topo_global = {}, {}, {}, ""
        topo_masked, topo_dims = {}, {}

    title = args.title or "Token trajectories across layers"
    sub = (f"layers {layers[0]}…{layers[-1]} ({len(layers)} steps)  ·  {ntok} tokens in a shared "
           f"3D PCA frame  ·  each layer = one time step  ·  drag to rotate")
    return {
        "mode": "trajectory",
        "title": title,
        "sub": sub,
        "n_layers": len(layers),
        "layers": [int(L) for L in layers],
        "n_tokens": ntok,
        "token_labels": labels,
        "traj": [[[round(float(v), 1) for v in row] for row in traj.transpose(1, 0, 2)[tok]] for tok in range(ntok)],
        "spread": [round(float(v), 3) for v in spread],
        "group_of": group_of,
        "prompt_labels": ptexts,
        "dyn_messages": dyn_messages,
        "dyn_rows": {int(L): [int(v) for v in r] for L, r in dyn_rows.items()},
        "dyn_closest": dyn_closest,
        "dyn_transitions": dyn_transitions,
        "dyn_global": dyn_global,
        "topo_rows": {int(L): [int(v) for v in r] for L, r in topo_rows.items()},
        "topo_messages": topo_messages,
        "topo_masked": topo_masked,
        "topo_dims": topo_dims,
        "topo_closest": topo_closest,
        "topo_global": topo_global,
    }


def render_html(data: dict[str, object]) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data))


# The HTML/JS is a plain (non-f) template; only the __DATA__ token is substituted.
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#2b3340; --txt:#e6edf3; --mut:#8b98a9; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: var(--bg); color: var(--txt); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         display: flex; flex-direction: column; overflow: hidden; }
  #top { padding: 10px 16px 8px; border-bottom: 1px solid var(--line); }
  #top h1 { font-size: 17px; margin: 0 0 3px; font-weight: 650; }
  #top #sub { font-size: 12px; color: var(--mut); }
  #main { flex: 1; display: flex; min-height: 0; }
  #left { flex: 1.25; position: relative; min-width: 0; }
  #scene { width: 100%; height: 100%; display: block; cursor: grab; }
  #scene:active { cursor: grabbing; }
  #hint { position: absolute; left: 12px; bottom: 10px; font-size: 11px; color: var(--mut); pointer-events: none; }
  #right { flex: 1; min-width: 340px; max-width: 460px; border-left: 1px solid var(--line);
           display: flex; flex-direction: column; overflow-y: auto; }
  #cards { display: flex; gap: 8px; padding: 12px; }
  .card { flex: 1; background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 10px 8px; text-align: center; transition: border-color .2s, box-shadow .2s; }
  .card.match { border-color: #2ea043; box-shadow: 0 0 0 1px #2ea043 inset; }
  .card .lbl { font-size: 11.5px; color: var(--mut); line-height: 1.25; }
  .card .num { font-size: 34px; font-weight: 750; line-height: 1.1; margin-top: 4px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
  .panel { padding: 8px 14px 12px; border-bottom: 1px solid var(--line); }
  .panel h3 { margin: 6px 0 4px; font-size: 12.5px; font-weight: 650; color: #cdd6e0; letter-spacing:.2px; }
  .panel canvas { width: 100%; display: block; }
  #controls { border-top: 1px solid var(--line); padding: 10px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  button { background:#1f2937; color: var(--txt); border: 1px solid #3a4553; border-radius: 8px;
           padding: 7px 14px; cursor: pointer; font-size: 13px; }
  button:hover { background:#283241; }
  input[type=range] { flex: 1; min-width: 160px; accent-color: #4ea1ff; }
  #eps-readout { font-variant-numeric: tabular-nums; font-size: 13px; color: var(--mut); min-width: 130px; }
  .toggle { display:flex; align-items:center; gap:5px; font-size: 12.5px; color: var(--mut); cursor:pointer; user-select:none; }
  #badge { font-size: 11.5px; color:#2ea043; font-weight:650; }
  #dynbadge { font-size: 11.5px; color:#d39b52; font-weight:600; margin-left: 10px; }
  #dimsrow { display:inline-flex; flex-wrap:wrap; align-items:center; gap:2px 12px;
             margin-left:8px; vertical-align: middle; }
  #dimsrow label { display:inline-flex; align-items:center; gap:4px; cursor:pointer;
                   user-select:none; color:var(--mut); font-size:12.5px; }
  #syst { font-size:12px; line-height:1.65; color:var(--txt); }
  #syst .sl { display:flex; gap:8px; align-items:baseline; }
  #syst .sl .htag { min-width:28px; font-weight:700; }
  .card { cursor: help; }
  .card.match { background:#10251a; }
  #legend { display:flex; flex-wrap:wrap; gap:6px 12px; font-size:11.5px; color:var(--mut); }
  #legend .dot { width:10px; height:10px; }
  #errbox { display:none; position:fixed; left:16px; right:16px; bottom:14px; z-index:1000;
            background:#3d1115; border:1px solid #f85149; border-left:6px solid #f85149;
            border-radius:10px; padding:12px 16px; box-shadow:0 8px 28px rgba(0,0,0,.55); }
  #errbox h4 { margin:0 0 6px; font-size:14px; color:#ffa28b; }
  #errbox pre { margin:0; font-size:11.5px; line-height:1.45; color:#ffd7d0; max-height:38vh;
                overflow:auto; white-space:pre-wrap; word-break:break-word; }
</style>
</head>
<body>
  <div id="top">
    <h1 id="title"></h1>
    <div id="sub"></div>
  </div>
  <div id="main">
    <div id="left">
      <canvas id="scene"></canvas>
      <div id="hint"></div>
    </div>
    <div id="right">
      <div id="cards"></div>
      <div class="panel" id="p-bfun">
        <h3>Betti numbers over ε <span id="dimsrow"></span><span id="badge"></span><span id="dynbadge"></span></h3>
        <canvas id="bfun" height="150"></canvas>
        <div id="syst" style="margin-top:6px;"></div>
      </div>
      <div class="panel" id="p-diag">
        <h3>Persistence diagram (birth → death)</h3>
        <canvas id="diag" height="230"></canvas>
      </div>
      <div class="panel" id="p-conv" style="display:none">
        <h3>Convergence — spread over depth</h3>
        <canvas id="conv" height="150"></canvas>
      </div>
      <div class="panel" id="p-toggles">
        <div class="toggle"><input type="checkbox" id="t-points" checked><label for="t-points">Points</label></div>
        <div class="toggle"><input type="checkbox" id="t-edges" checked><label for="t-edges">Edges (H¹)</label></div>
        <div class="toggle"><input type="checkbox" id="t-faces" checked><label for="t-faces">Faces (fill, H²)</label></div>
        <div class="toggle"><input type="checkbox" id="t-shade" checked><label for="t-shade">Shade (3D depth)</label></div>
      </div>
      <div class="panel" id="p-legend" style="display:none">
        <h3>Prompts</h3>
        <div id="legend"></div>
      </div>
    </div>
  </div>
  <div id="controls">
    <button id="play">▶ Play</button>
    <button id="reset">Reset</button>
    <button id="resetview">Reset view</button>
    <input type="range" id="slider" min="0" max="1000" value="0">
    <div id="eps-readout"></div>
  </div>
  <div id="errbox">
    <h4 id="err-title"></h4>
    <pre id="err-body"></pre>
  </div>

<script>
const DATA = __DATA__;
const MODE = DATA.mode || "filtration";
const DIM_NAME = {0:"H₀ components", 1:"H₁ loops / holes", 2:"H₂ voids", 3:"H₃", 4:"H₄"};
const DIM_COLOR = {0:"#4ea1ff", 1:"#3fd07a", 2:"#ff9f45", 3:"#e060c0", 4:"#c9d16a"};
const EMAX = (DATA.eps_max || 1);
const MD = DATA.maxdim || 0;
const P = DATA.points || [], E = DATA.edges || [], F = DATA.faces || [], IV = DATA.intervals || [];
const GRID = (DATA.betti && DATA.betti.grid) || [0,1];
const TABLE = (DATA.betti && DATA.betti.table) || [[0]];

// Face birth values pre-sorted so "how many faces are active at eps" is O(log n).
const F_EPS = F.map(f => f[3]).sort((a,b) => a-b);
function countActiveFaces(e){
  let lo = 0, hi = F_EPS.length;
  while (lo < hi){ const m = (lo+hi)>>1; if (F_EPS[m] <= e) lo = m+1; else hi = m; }
  return lo;
}

const N_L = DATA.n_layers || 0;
const TRAJ = DATA.traj || [];
const SPREAD = DATA.spread || [];
const N_TOK = DATA.n_tokens || 0;
const GROUP = DATA.group_of || [];

let rx = -0.45, ry = 0.7, eps = 0, t = 0, playing = false, raf = null, lastT = 0;
let showPoints = true, showEdges = true, showFaces = true, shade = true;
let fitR = 1.0;
// Cached depth (painter's) order of the faces. It depends only on the view
// (rx, ry), never on eps, so we recompute it only when the view rotates.
let faceOrder = null, faceOrderKey = "";

const scene = document.getElementById("scene"), sctx = scene.getContext("2d");
const bfun  = document.getElementById("bfun"),  bctx = bfun.getContext("2d");
const diag  = document.getElementById("diag"),  dctx = diag.getContext("2d");
const conv  = document.getElementById("conv"),  cctx = conv.getContext("2d");

document.getElementById("title").textContent = DATA.title;
document.getElementById("sub").textContent = DATA.sub;
document.getElementById("hint").textContent = MODE === "trajectory"
  ? "drag to rotate · slider / ▶ Play to step through layers (time)"
  : "drag to rotate · slider / ▶ Play to step the filtration ε";

// ---- side cards ----------------------------------------------------------
const cardsEl = document.getElementById("cards");
const cardNums = [];
if (MODE === "filtration"){
  for (let d = 0; d <= MD; d++){
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<div class="lbl"><span class="dot" style="background:${DIM_COLOR[d]}"></span>${DIM_NAME[d] || ("H"+d)}<br>β<sub>${d}</sub></div><div class="num" id="b${d}">0</div>`;
    cardsEl.appendChild(el);
    cardNums.push(document.getElementById("b" + d));
  }
} else {
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="lbl">layer (time)</div><div class="num" id="layer-num" style="font-size:26px">0</div>`;
  cardsEl.appendChild(el);
  const el2 = document.createElement("div");
  el2.className = "card";
  el2.innerHTML = `<div class="lbl">tokens moving</div><div class="num" id="tok-num" style="font-size:26px">0</div>`;
  cardsEl.appendChild(el2);
}

// ---- per-dimension toggles: see the structure in single dimensions ---------
// Each dimension gets a checkbox. Turning a dimension off hides its Betti curve,
// its side card, and reprojects the structure recognition onto the ACTIVE dims
// (server-precomputed masked_messages): "nur in einzelnen Dims diese Strukturen".
const dimsRow = document.getElementById("dimsrow");
const dimsOn = [];
const N_D = MODE === "filtration" ? MD + 1
          : Math.max(1, ...Object.keys(DATA.dyn_rows || {}).map(L => (DATA.dyn_rows[L] || []).length));
const fullMask = (1 << N_D) - 1;
const SUBD = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089";
function hsub(d){ return "H" + (SUBD[d] || ("_{" + d + "}")); }
function activeMask(){
  let m = 0;
  for (let d = 0; d < N_D; d++) if (dimsOn[d]) m |= (1 << d);
  return m;
}
function maskLabel(mask){
  if (!mask) return "";
  const s = [];
  for (let d = 0; d < N_D; d++) if (mask & (1 << d)) s.push(hsub(d));
  return "[" + s.join("\u00b7") + "]";
}
function maskedMsg(col, mask){
  if (!col) return "";
  if (col[mask] != null) return col[mask];
  if (mask === fullMask) return "";
  if (Object.keys(col).length === 1){
    return "nur volle Struktur analysierbar (zu viele topologische Dimensionen f\u00fcr die Maskierung)";
  }
  return "";
}
for (let d = 0; d < N_D; d++){
  dimsOn.push(true);
  const lab = document.createElement("label");
  lab.title = "Struktur nur in dieser Dimension anzeigen" + (d === 0 ? " (aus \u2014 die anderen Dimensionen werden maskiert)" : "");
  lab.innerHTML = `<input type="checkbox" checked><span style="color:${DIM_COLOR[d] || "#ccc"}">${hsub(d)}</span>`;
  lab.querySelector("input").addEventListener("change", ev => { dimsOn[d] = ev.target.checked; render(); });
  dimsRow.appendChild(lab);
}

// ---- canvas sizing -------------------------------------------------------
function fit(c){
  const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  c.width = Math.max(2, Math.round(r.width * dpr));
  c.height = Math.max(2, Math.round(r.height * dpr));
  return dpr;
}
function fitAll(){
  fit(scene); fit(bfun); fit(diag); fit(conv);
  if (MODE !== "filtration"){ document.getElementById("p-diag").style.display="none"; }
  if (MODE !== "trajectory"){ document.getElementById("p-conv").style.display="none"; document.getElementById("p-legend").style.display="none"; }
}
window.addEventListener("resize", () => { fitAll(); render(); });

// ---- 3D -> 2D projection -------------------------------------------------
function project(p){
  let x=p[0], y=p[1], z=p[2];
  let cy=Math.cos(ry), sy=Math.sin(ry);
  let x1=x*cy+z*sy, z1=-x*sy+z*cy;
  let cx=Math.cos(rx), sx=Math.sin(rx);
  let y1=y*cx-z1*sx, z2=y*sx+z1*cx;
  return [x1, y1, z2];
}
function computeFit(){
  let mx=0;
  if (MODE === "trajectory"){
    for (const tok of TRAJ) for (const p of tok){ const q=project(p); const r=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]; if(r>mx) mx=r; }
  } else {
    for (const p of P){ const q=project(p); const r=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]; if(r>mx) mx=r; }
  }
  fitR = Math.sqrt(mx) || 1;
}
function toScreen(q, w, h){
  const s = Math.min(w, h) * 0.42 / fitR;
  return [ w/2 + q[0]*s, h/2 - q[1]*s, q[2] ];
}

// ---- color ----------------------------------------------------------------
const STOPS = [[0,[30,80,220]],[0.25,[20,180,220]],[0.5,[45,200,95]],[0.75,[240,210,45]],[1,[232,64,64]]];
function colormap(t){
  t = Math.max(0, Math.min(1, t));
  for (let i=0;i<STOPS.length-1;i++){
    if (t <= STOPS[i+1][0]){
      const f=(t-STOPS[i][0])/(STOPS[i+1][0]-STOPS[i][0]);
      const a=STOPS[i][1], b=STOPS[i+1][1];
      return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
    }
  }
  return STOPS[STOPS.length-1][1];
}
const rgba=(c,a)=>`rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a==null?1:a})`;
const _ll=Math.hypot(-0.45,0.5,0.74);
const LIGHT=[-0.45/_ll,0.5/_ll,0.74/_ll];
function shadeColor(a,b,c,base,fitR){
  // two-sided Lambert lighting + a front-to-back depth cue, so the 3D shape reads
  // without having to rotate: near faces are brighter, the light gives each patch a
  // normal-based tone, and far faces recede.
  let nx=(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]);
  let ny=(b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]);
  let nz=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const nl=Math.hypot(nx,ny,nz)||1;
  const lambert=0.55+0.45*Math.abs((nx*LIGHT[0]+ny*LIGHT[1]+nz*LIGHT[2])/nl);
  const z=(a[2]+b[2]+c[2])/3;
  const depth=0.5+0.5*((z/fitR+1)/2);
  const br=depth*lambert;
  return [base[0]*br, base[1]*br, base[2]*br];
}
function depthT(z,fitR){ return 0.5+0.5*((z/fitR+1)/2); }
const PROMPT_COLORS = ["#4ea1ff","#3fd07a","#ff9f45","#e060c0","#f5d442","#5ad1e6","#ff6b81","#9d7bff","#6bde7f","#e8873a","#7aa2ff","#c9d16a"];
function groupColor(g){ return PROMPT_COLORS[((g % PROMPT_COLORS.length) + PROMPT_COLORS.length) % PROMPT_COLORS.length]; }

// ---- filtration scene ----------------------------------------------------
function renderScene(){
  const dpr = window.devicePixelRatio || 1;
  const w = scene.width/dpr, h = scene.height/dpr;
  sctx.save(); sctx.scale(dpr, dpr);
  sctx.clearRect(0,0,w,h);
  const proj = P.map(project);
  const scr = proj.map(q => toScreen(q, w, h));

  if (showFaces && F.length){
    // Painter's (back-to-front) order depends only on the view, not eps. Recompute
    // it only when the view rotates; the rounded key throttles re-sorts during a
    // drag. (Re-sorting every face on every frame is what made this ultra-slow.)
    const vkey = (Math.round(rx*50)) + "," + (Math.round(ry*50));
    if (faceOrderKey !== vkey){
      faceOrder = F.map((_, i) => i);
      faceOrder.sort((a,b)=>{
        const fa=F[a], fb=F[b];
        return (proj[fa[0]][2]+proj[fa[1]][2]+proj[fa[2]][2])
             - (proj[fb[0]][2]+proj[fb[1]][2]+proj[fb[2]][2]);
      });
      faceOrderKey = vkey;
    }
    const alpha = shade ? 0.6 : 0.16;
    if (F.length <= 2000){
      // small mesh: one path per face (keeps the crisp per-face wireframe)
      sctx.lineWidth = 0.6;
      for (const i of faceOrder){
        const f = F[i]; if (f[3] > eps) continue;
        const base = colormap(f[3]/EMAX);
        const col = shade ? shadeColor(proj[f[0]],proj[f[1]],proj[f[2]],base,fitR) : base;
        sctx.beginPath();
        sctx.moveTo(scr[f[0]][0], scr[f[0]][1]);
        sctx.lineTo(scr[f[1]][0], scr[f[1]][1]);
        sctx.lineTo(scr[f[2]][0], scr[f[2]][1]);
        sctx.closePath();
        sctx.fillStyle = rgba(col, alpha);
        sctx.fill();
        if (shade){ sctx.strokeStyle = rgba(col, 0.85); sctx.stroke(); }
      }
    } else {
      // dense mesh: bucket faces by quantised colour and issue ONE fill per bucket
      // (41k individual fills -> a few hundred), which keeps shading but is fast.
      const buckets = new Map();
      for (const i of faceOrder){
        const f = F[i]; if (f[3] > eps) continue;
        const base = colormap(f[3]/EMAX);
        const col = shade ? shadeColor(proj[f[0]],proj[f[1]],proj[f[2]],base,fitR) : base;
        const key = (((col[0]|0)>>5)&7)*64 + (((col[1]|0)>>5)&7)*8 + (((col[2]|0)>>5)&7);
        let g = buckets.get(key);
        if (!g){ g = {col: col, faces: []}; buckets.set(key, g); }
        g.faces.push(f);
      }
      for (const g of buckets.values()){
        sctx.fillStyle = rgba(g.col, alpha);
        sctx.beginPath();
        for (const f of g.faces){
          sctx.moveTo(scr[f[0]][0], scr[f[0]][1]);
          sctx.lineTo(scr[f[1]][0], scr[f[1]][1]);
          sctx.lineTo(scr[f[2]][0], scr[f[2]][1]);
          sctx.closePath();
        }
        sctx.fill();
      }
    }
  }
  if (showEdges){
    sctx.lineWidth = 1.1;
    for (const e of E){
      if (e[2] > eps) continue;
      const z = (proj[e[0]][2]+proj[e[1]][2])/2;
      const a = shade ? 0.3+0.6*depthT(z,fitR) : 0.9;
      sctx.strokeStyle = rgba(colormap(e[2]/EMAX), a);
      sctx.beginPath();
      sctx.moveTo(scr[e[0]][0], scr[e[0]][1]);
      sctx.lineTo(scr[e[1]][0], scr[e[1]][1]);
      sctx.stroke();
    }
  }
  if (showPoints){
    for (let i=0;i<P.length;i++){
      const dpt = depthT(proj[i][2], fitR);
      const r = shade ? (1.3+1.4*dpt) : 2.0;
      sctx.fillStyle = rgba([235,241,247], shade ? 0.4+0.6*dpt : 1);
      sctx.beginPath();
      sctx.arc(scr[i][0], scr[i][1], r, 0, Math.PI*2);
      sctx.fill();
    }
  }
  sctx.restore();
}

// ---- trajectory scene (points moving across layers) ----------------------
function lerpPos(tok, f){
  const i0 = Math.max(0, Math.min(N_L-1, Math.floor(f)));
  const i1 = Math.max(0, Math.min(N_L-1, i0+1));
  const fr = f - i0;
  const a = TRAJ[tok][i0], b = TRAJ[tok][i1];
  return [ a[0]+(b[0]-a[0])*fr, a[1]+(b[1]-a[1])*fr, a[2]+(b[2]-a[2])*fr ];
}
function renderTrajScene(){
  const dpr = window.devicePixelRatio || 1;
  const w = scene.width/dpr, h = scene.height/dpr;
  sctx.save(); sctx.scale(dpr, dpr);
  sctx.clearRect(0,0,w,h);
  // trails: each token's path from layer 0 up to the current time
  sctx.lineWidth = 1.1;
  for (let tok=0; tok<N_TOK; tok++){
    const col = groupColor(GROUP[tok]);
    sctx.strokeStyle = col; sctx.globalAlpha = 0.30;
    sctx.beginPath();
    let started = false;
    for (let L=0; L<=t; L++){
      const s = toScreen(project(TRAJ[tok][L]), w, h);
      if (!started){ sctx.moveTo(s[0], s[1]); started = true; } else sctx.lineTo(s[0], s[1]);
    }
    sctx.stroke();
  }
  sctx.globalAlpha = 1;
  // current positions (bright, coloured by prompt)
  for (let tok=0; tok<N_TOK; tok++){
    const s = toScreen(project(lerpPos(tok, t)), w, h);
    sctx.fillStyle = groupColor(GROUP[tok]);
    sctx.beginPath(); sctx.arc(s[0], s[1], 3.0, 0, Math.PI*2); sctx.fill();
  }
  sctx.restore();
}

// ---- persistence diagram -------------------------------------------------
function renderDiagram(){
  const dpr = window.devicePixelRatio || 1;
  const w = diag.width/dpr, h = diag.height/dpr;
  dctx.save(); dctx.scale(dpr, dpr);
  dctx.clearRect(0,0,w,h);
  const m = 26, pw = w-m-8, ph = h-m-8;
  const X = v => m + (v/EMAX)*pw, Y = v => h-m - (v/EMAX)*ph;
  dctx.strokeStyle = "#3a4553"; dctx.lineWidth = 1;
  dctx.strokeRect(m, 8, pw, ph);
  dctx.strokeStyle = "#5a6675"; dctx.setLineDash([4,4]);
  dctx.beginPath(); dctx.moveTo(X(0),Y(0)); dctx.lineTo(X(EMAX),Y(EMAX)); dctx.stroke(); dctx.setLineDash([]);
  dctx.strokeStyle = "#e6edf3"; dctx.globalAlpha=0.5; dctx.lineWidth=1;
  dctx.beginPath(); dctx.moveTo(X(eps),8); dctx.lineTo(X(eps),h-m); dctx.stroke();
  dctx.beginPath(); dctx.moveTo(m,Y(eps)); dctx.lineTo(w-8,Y(eps)); dctx.stroke(); dctx.globalAlpha=1;
  for (const iv of IV){
    const alive = iv[1] <= eps && eps < iv[2];
    const dead = eps >= iv[2];
    dctx.globalAlpha = alive ? 1 : (dead ? 0.18 : 0.4);
    dctx.fillStyle = DIM_COLOR[iv[0]];
    dctx.beginPath(); dctx.arc(X(iv[1]), Y(iv[2]), alive?3.4:2.4, 0, Math.PI*2); dctx.fill();
  }
  dctx.globalAlpha=1;
  dctx.fillStyle = "#8b98a9"; dctx.font = "10px system-ui";
  dctx.fillText("birth →", m, h-4);
  dctx.save(); dctx.translate(9, m+ph/2); dctx.rotate(-Math.PI/2); dctx.fillText("death ↑", 0, 0); dctx.restore();
  dctx.restore();
}

// ---- betti function ------------------------------------------------------
function renderBfun(){
  const dpr = window.devicePixelRatio || 1;
  const w = bfun.width/dpr, h = bfun.height/dpr;
  bctx.save(); bctx.scale(dpr, dpr);
  bctx.clearRect(0,0,w,h);
  const m = 22, pw = w-m-8, ph = h-14-8;
  let maxB = 1; for (const row of TABLE) for (const v of row) if (v>maxB) maxB=v;
  const X = i => m + (i/(GRID.length-1))*pw, Y = v => 8 + ph - (v/maxB)*ph;
  bctx.strokeStyle="#3a4553"; bctx.strokeRect(m,8,pw,ph);
  for (let d=0; d<=MD; d++){
    if (!dimsOn[d]) continue;
    bctx.strokeStyle = DIM_COLOR[d]; bctx.lineWidth=1.6; bctx.beginPath();
    for (let i=0;i<GRID.length;i++){
      const x=X(i), y=Y(TABLE[i][d]);
      if (i===0) bctx.moveTo(x,y); else bctx.lineTo(x,y);
    }
    bctx.stroke();
  }
  const cx = m + (eps/EMAX)*pw;
  bctx.strokeStyle="#e6edf3"; bctx.globalAlpha=0.6; bctx.beginPath();
  bctx.moveTo(cx,8); bctx.lineTo(cx,8+ph); bctx.stroke(); bctx.globalAlpha=1;
  let lx = m+6;
  for (let d=0; d<=MD; d++){
    if (!dimsOn[d]) continue;
    bctx.fillStyle = DIM_COLOR[d]; bctx.fillRect(lx, h-6, 9, 3);
    bctx.fillStyle="#8b98a9"; bctx.font="10px system-ui";
    bctx.fillText("β"+d, lx+11, h-1); lx += 34;
  }
  bctx.restore();
}

// Betti-rows of each trajectory layer we checked, drawn as colored bars (dynamics
// fingerprint per depth). The cursor shows the current layer.
function renderBfunTraj(){
  const dpr = window.devicePixelRatio || 1, w = bfun.width/dpr, h = bfun.height/dpr;
  bctx.save(); bctx.scale(dpr, dpr);
  bctx.clearRect(0,0,w,h);
  const rows = DATA.dyn_rows || {};
  const Ls = Object.keys(rows).map(Number).sort((a,b)=>a-b);
  const D = Math.max(1, ...Ls.map(L=>rows[L].length));
  const m = 22, pw = w-m-8, ph = h-16-8;
  bctx.strokeStyle="#3a4553"; bctx.strokeRect(m,8,pw,ph);
  const cw = pw/Math.max(1,Ls.length);
  let maxB = 1; for (const L of Ls) for (const v of rows[L]) if (v>maxB) maxB=v;
  for (const L of Ls){
    const x = m + Ls.indexOf(L)*cw;
    for (let d=0; d<rows[L].length; d++){
      if (!dimsOn[d]) continue;
      bctx.fillStyle = DIM_COLOR[d];
      const bh = (rows[L][d]/maxB)*ph;
      bctx.fillRect(x+2+d*cw*0.28, 8+ph-bh, cw*0.24, bh);
    }
  }
  const li = Math.max(0, Math.min(N_L-1, Math.round(t)));
  const cx = m + (li/Math.max(1,N_L-1))*pw;
  bctx.strokeStyle="#e6edf3"; bctx.globalAlpha=0.6; bctx.beginPath();
  bctx.moveTo(cx,8); bctx.lineTo(cx,8+ph); bctx.stroke(); bctx.globalAlpha=1;
  let lx = m+6;
  for (let d=0; d<Math.max(1,...Ls.map(L=>rows[L].length)); d++){
    if (!dimsOn[d]) continue;
    bctx.fillStyle = DIM_COLOR[d]; bctx.fillRect(lx, h-6, 9, 3);
    bctx.fillStyle="#8b98a9"; bctx.font="10px system-ui";
    bctx.fillText("β"+d, lx+11, h-1); lx += 34;
  }
  bctx.restore();
}
function renderConv(){
  const dpr = window.devicePixelRatio || 1;
  const w = conv.width/dpr, h = conv.height/dpr;
  cctx.save(); cctx.scale(dpr, dpr);
  cctx.clearRect(0,0,w,h);
  const m = 22, pw = w-m-8, ph = h-16-8;
  const n = SPREAD.length;
  let mn=Infinity, mx=-Infinity; for (const v of SPREAD){ if(v<mn)mn=v; if(v>mx)mx=v; }
  if (!isFinite(mn)){ mn=0; mx=1; }
  if (mx-mn < 1e-9) mx = mn+1;
  const X = i => m + (i/Math.max(1,n-1))*pw, Y = v => 8 + ph - ((v-mn)/(mx-mn))*ph;
  cctx.strokeStyle="#3a4553"; cctx.strokeRect(m,8,pw,ph);
  cctx.strokeStyle="#4ea1ff"; cctx.lineWidth=1.8; cctx.beginPath();
  for (let i=0;i<n;i++){ const x=X(i), y=Y(SPREAD[i]); if(i===0) cctx.moveTo(x,y); else cctx.lineTo(x,y); }
  cctx.stroke();
  const cx = X(t);
  cctx.strokeStyle="#e6edf3"; cctx.globalAlpha=0.6; cctx.beginPath();
  cctx.moveTo(cx,8); cctx.lineTo(cx,8+ph); cctx.stroke(); cctx.globalAlpha=1;
  cctx.fillStyle="#8b98a9"; cctx.font="10px system-ui";
  cctx.fillText("depth (layer) →", m, h-3);
  cctx.restore();
}

// ---- prompt legend -------------------------------------------------------
function buildLegend(){
  const gmax = (GROUP.length ? Math.max.apply(null, GROUP) : 0) + 1;
  let html = "";
  for (let g=0; g<gmax; g++){
    const cnt = GROUP.reduce((a,b)=>a+(b===g?1:0), 0);
    const lbl = DATA.prompt_labels && DATA.prompt_labels[g] ? DATA.prompt_labels[g] : ("prompt "+g);
    html += `<span><span class="dot" style="background:${groupColor(g)}"></span>${lbl} <span style="opacity:.6">(${cnt})</span></span>`;
  }
  document.getElementById("legend").innerHTML = html;
}

// ---- cards update --------------------------------------------------------
function bettiAt(e){
  let idx = Math.round(e/EMAX*(GRID.length-1));
  idx = Math.max(0, Math.min(GRID.length-1, idx));
  return TABLE[idx];
}

// ---- guardrails ----------------------------------------------------------
// Any calculation misstep must surface LOUDLY in the browser: a sticky red error box
// with the failing state (ε, target, …), plus full detail on the JS console.
let errSource = null;
function showError(title, body, source){
  errSource = source || "misc";
  const box = document.getElementById("errbox");
  const t = document.getElementById("err-title");
  const b = document.getElementById("err-body");
  t.textContent = title;
  b.textContent = body + "\n\nstate: eps=" + eps + "/" + EMAX +
    " · sliderMax=" + (DATA.eps_max ?? "?") +
    (DATA.target ? " · target=[" + DATA.target.join(",") + "]" + " · got=[" + bettiAt(eps).slice(0, DATA.target.length).join(",") + "]" : "");
  box.style.display = "block";
  console.error(title + "\n" + body);
}
function hideError(){
  if (errSource === "runtime") return;   // runtime errors stay visible until reload
  document.getElementById("errbox").style.display = "none";
}
window.addEventListener("error", ev => {
  showError("JS runtime error",
    (ev && ev.message || "unknown error") +
    "\n\nstack:\n" + (ev && ev.error && ev.error.stack || "(none)"), "runtime");
});
window.addEventListener("unhandledrejection", ev => {
  showError("Unhandled promise rejection",
    (ev && ev.reason && (ev.reason.stack || ev.reason.message || String(ev.reason)) || "unknown"),
    "runtime");
});
let mismatchShown = false;

function updateCards(){
  if (MODE === "filtration"){
    const b = bettiAt(eps);
    for (let d=0; d<=MD; d++){
      cardNums[d].textContent = b[d];
      cardNums[d].style.color = DIM_COLOR[d];
    }
    const badge = document.getElementById("badge");
    const dynBadge = document.getElementById("dynbadge");
    const ki = Math.max(0, Math.min((DATA.struct_rows || []).length - 1,
                                    Math.round(eps / EMAX * ((DATA.struct_rows || []).length - 1))));
    const firstClause = (s) => {
      if (!s) return "";
      const cut = Math.min(s.indexOf(":"), s.indexOf(";"), s.indexOf("("), s.length);
      const seg = cut > 0 ? s.slice(0, cut) : s;
      return seg.slice(0, 140);
    };
    const featOK = DATA.target ? DATA.feat_row.slice(0, DATA.target.length)
          .every((v, i) => v === DATA.target[i]) : false;
    const atEps = (DATA.topo_messages && DATA.topo_messages[ki]) || "";
    badge.textContent = (featOK ? "✓ " : "") + firstClause(atEps);
    badge.title = "Topology (persistent structure): " + DATA.topo_feature +
                  "\nFeature β = [" + DATA.feat_row.join(", ") + "]";
    dynBadge.textContent = firstClause(DATA.dyn_feature);
    dynBadge.title = "Dynamics hypothesis: " + DATA.dyn_feature;
    if (DATA.target){
      const m = b.slice(0, DATA.target.length).every((v,i)=>v===DATA.target[i]);
      for (let d=0; d<=MD; d++) document.querySelectorAll(".card")[d].classList.toggle("match", m);
      // Guardrail: at the top of the slider the computed Betti vector MUST equal the
      // declared target (e.g. a rebuilt exact T^2 must read [1,2,1]). A mismatch is a
      // loud error, never a silent wrong answer.
      if (eps === EMAX){
        if (!m && !mismatchShown){
          mismatchShown = true;
          showError("Topology mismatch at max ε",
            "Betti(" + EMAX + ") = [" + b.slice(0, DATA.target.length).join(", ") +
            "] but DATA.target = [" + DATA.target.join(", ") +
            "].\n\nThe slider is at its maximum, so every simplex in the dataset must be " +
            "present — this build should yield exactly the declared target topology. The " +
            "data or the build path is inconsistent (look at the Python-side traceback).",
            "target");
        }
      } else {
        mismatchShown = false;
        hideError();
      }
    }
  } else {
    const li = Math.max(0, Math.min(N_L-1, Math.round(t)));
    const ln = document.getElementById("layer-num"); if (ln) ln.textContent = DATA.layers[li] + "  /  " + DATA.layers[N_L-1];
    const tn = document.getElementById("tok-num"); if (tn) tn.textContent = N_TOK;
    const badge = document.getElementById("badge");
    const dynBadge = document.getElementById("dynbadge");
    const L = DATA.layers[li];
    const tm = (DATA.topo_messages && DATA.topo_messages[L]) || "";
    const msg = (DATA.dyn_messages && DATA.dyn_messages[L]) || "";
    const tr = (DATA.dyn_transitions && DATA.dyn_transitions[L]) || "";
    if (badge) badge.textContent = tm ? tm.split("(")[0].trim().slice(0, 130) : "";
    if (badge) badge.title = "Topology at layer " + L + ":\n" + tm +
      (DATA.topo_global ? "\n\nWhole-trajectory union guess:\n" + DATA.topo_global : "");
    if (dynBadge){
      dynBadge.textContent = (msg ? msg.split("(")[0].trim().slice(0, 70) : "") +
                             (tr ? " — ⇥ " + tr.slice(0, 60) : "");
      dynBadge.title = "Dynamics at layer " + L + ":\n" + msg +
        (tr ? "\nBifurcation into this layer:\n" + tr : "") +
        (DATA.dyn_global ? "\n\nWhole-trajectory attractor-set guess:\n" + DATA.dyn_global : "");
    }
  }
}

function render(){
  computeFit();
  if (MODE === "trajectory"){ renderTrajScene(); renderConv(); renderBfunTraj(); }
  else { renderScene(); renderDiagram(); renderBfun(); }
  updateCards();
  const ro = document.getElementById("eps-readout");
  if (MODE === "trajectory"){
    const li = Math.max(0, Math.min(N_L-1, Math.round(t)));
    ro.textContent = `layer ${DATA.layers[li]}  ·  t = ${t.toFixed(1)}/${N_L-1}`;
  } else {
    ro.textContent = `ε = ${eps.toFixed(3)}  /  ${EMAX.toFixed(3)}`;
  }
}

// ---- controls ------------------------------------------------------------
const slider = document.getElementById("slider");
if (MODE === "trajectory"){ slider.max = Math.max(1, N_L-1); }
function setSlider(){ slider.value = Math.round(MODE === "trajectory" ? t : eps/EMAX*1000); }
slider.addEventListener("input", () => {
  stopPlay();
  if (MODE === "trajectory") t = +slider.value; else eps = +slider.value/1000*EMAX;
  render();
});
const playBtn = document.getElementById("play");
function stopPlay(){ playing=false; if(raf) cancelAnimationFrame(raf); playBtn.textContent="▶ Play"; }
function tick(now){
  if(!playing) return;
  if(!lastT) lastT=now;
  const dt=(now-lastT)/1000; lastT=now;
  if (MODE === "trajectory"){
    t += dt * (Math.max(1, N_L-1)/7);
    if (t >= N_L-1){ t = N_L-1; stopPlay(); }
  } else {
    eps += dt * (EMAX/8);
    if (eps >= EMAX){ eps = EMAX; stopPlay(); }
  }
  setSlider(); render();
  if (playing) raf = requestAnimationFrame(tick);
}
playBtn.addEventListener("click", () => {
  if (playing){ stopPlay(); return; }
  if (MODE === "trajectory"){ if (t >= N_L-1) t = 0; }
  else { if (eps >= EMAX) eps = 0; }
  playing=true; lastT=0; playBtn.textContent="⏸ Pause";
  raf = requestAnimationFrame(tick);
});
document.getElementById("reset").addEventListener("click", () => { stopPlay(); if (MODE==="trajectory") t=0; else eps=0; setSlider(); render(); });
document.getElementById("resetview").addEventListener("click", () => { rx=-0.45; ry=0.7; render(); });

document.getElementById("t-points").addEventListener("change", e=>{showPoints=e.target.checked; render();});
document.getElementById("t-edges").addEventListener("change", e=>{showEdges=e.target.checked; render();});
document.getElementById("t-faces").addEventListener("change", e=>{showFaces=e.target.checked; render();});
document.getElementById("t-shade").addEventListener("change", e=>{shade=e.target.checked; render();});

// drag to rotate
let dragging=false, lx=0, ly=0;
scene.addEventListener("mousedown", e=>{dragging=true; lx=e.clientX; ly=e.clientY;});
window.addEventListener("mouseup", ()=>dragging=false);
window.addEventListener("mousemove", e=>{
  if(!dragging) return;
  ry += (e.clientX-lx)*0.01;
  rx += (e.clientY-ly)*0.01;
  rx = Math.max(-1.5, Math.min(1.5, rx));
  lx=e.clientX; ly=e.clientY;
  render();
});
scene.addEventListener("touchstart", e=>{const p=e.touches[0];dragging=true;lx=p.clientX;ly=p.clientY;},{passive:true});
scene.addEventListener("touchend", ()=>dragging=false);
scene.addEventListener("touchmove", e=>{const p=e.touches[0];ry+=(p.clientX-lx)*0.01;rx+=(p.clientY-ly)*0.01;
  rx=Math.max(-1.5,Math.min(1.5,rx)); lx=p.clientX; ly=p.clientY; render();},{passive:true});

// ---- go ------------------------------------------------------------------
fitAll();
if (MODE === "trajectory"){ buildLegend(); t = N_L-1; }
setSlider();
render();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--layers", nargs="?", const="all", default=None,
                   help="layer/trajectory mode: treat transformer layers as time steps. "
                        "Pass 'all' (or nothing) for all layers, a range 'start:stop[:step]', "
                        "or a list '0,16,32,64'. Overrides --shape/--points.")
    p.add_argument("--data-dir", default=None, help="transformer data dir (default: bundled capital_berlin_multilingual)")
    p.add_argument("--shape", default="torus-grid",
                   choices=["torus-grid", "circle", "donut", "donut-rips", "product", "sphere", "blobs"],
                   help="synthetic source (default: torus-grid = exact T^2; donut = clean bagel, "
                        "donut-rips = honest Rips bagel that over-fills)")
    p.add_argument("--points", default=None, help="point-cloud CSV (overrides --shape)")
    p.add_argument("--no-exact-torus", action="store_true",
                   help="if the CSV is a regular torus grid, force Vietoris-Rips instead of "
                        "reconstructing the exact T^2 cell complex (a clean torus, beta=[1,2,1])")
    p.add_argument("--value-cols", nargs="*", default=None)
    p.add_argument("--index-cols", nargs="*", default=None)
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "squared", "manhattan", "cosine", "normalized_euclidean"])
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--n", type=int, default=8, help="torus-grid per-axis cells / circle points / sphere points")
    p.add_argument("--nper", type=int, default=10, help="points per circle (donut/product grids)")
    p.add_argument("--k", type=int, default=2, help="ambient dim for product/sphere")
    p.add_argument("--frac", type=float, default=1.6, help="Rips: eps_max as a fraction of mean nearest-neighbour distance")
    p.add_argument("--eps-max", type=float, default=None,
                   help="force the slider's max epsilon. Default: the full Vietoris-Rips range "
                        "(0 -> max pairwise distance), auto-capped at the largest feasible "
                        "epsilon for dense clouds (so it never crashes on millions of simplices). "
                        "Lower it to render a sparser surface faster; it is capped at feasibility.")
    p.add_argument("--connect-margin", type=float, default=1.2,
                   help="slider reaches at least this many x the connectivity threshold, so at max "
                        "epsilon all points form one connected complex (default 1.2). Use 0 to disable.")
    p.add_argument("--n-grid", type=int, default=140, help="epsilon grid resolution for Betti curve / slider")
    p.add_argument("--title", default="")
    p.add_argument("--out", default="interactive.html", help="output HTML file")
    args = p.parse_args()

    console = Console()
    _rich_ui.params_table(p, args, console)

    t0 = time.time()
    try:
        if args.layers is not None:
            data = build_layer_trajectory(args)
        else:
            data = build_payload(args)
    except VrtdaError as e:
        # Loud, state-bearing failure: no silent wrong output. Dump the reason + state
        # (a VrtdaError from the torus path already carries why/which step), then the
        # full traceback so the whole calculation is reconstructable.
        console.print("\n[bold red]calculation failed (VrtdaError)[/bold red]")
        console.print(str(e))
        console.print("[dim]full traceback:[/dim]")
        traceback.print_exc()
        return 2
    except TooLargeError as e:
        console.print("\n[bold red]too large to compute[/bold red]")
        console.print(f"[red]{e}[/red]")
        traceback.print_exc()
        return 2
    except Exception as e:  # any misstep must fail loudly, never silently
        console.print(f"\n[bold red]unexpected error: {type(e).__name__}[/bold red]")
        console.print(f"[red]{e}[/red]")
        traceback.print_exc()
        return 3
    html = render_html(data).replace("__TITLE__", data["title"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if data["mode"] == "trajectory":
        console.print(f"[bold green]wrote[/bold green] {out}  "
                      f"(trajectory: {data['n_tokens']} tokens x {data['n_layers']} layers)")
    else:
        console.print(f"[bold green]wrote[/bold green] {out}  "
                      f"({len(data['points'])} points, {len(data['edges'])} edges, {len(data['faces'])} faces)")
    console.print(f"[dim]open it in a browser:  file://{out.resolve()}   [{time.time()-t0:.1f}s][/dim]")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
