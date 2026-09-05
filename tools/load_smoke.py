# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18", "rich>=13"]
# ///
"""Phase-2 smoke: load the real transformer data and run a small Rips + betti sweep."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
TOOLS = str(Path(__file__).resolve().parent)
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from rich.console import Console

import _rich_ui
import vrtda as V
from vrtda.beartype_guard import beartype_module

_console = Console()


def nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def betti_line(ps: "V.PointSet", metric: str, max_dim: int, lo: float, hi: float, n: int = 12) -> tuple[float, int, list[tuple[int, ...]]]:
    D = V.pairwise_distances(ps.data, metric)
    nnn = nn(D)
    eps_max = hi * nnn
    C = V.build_rips(ps.data, D, eps_max, max_dim=max_dim)
    bc = _rich_ui.homology_progress(C, _console)
    eps = np.linspace(lo * nnn, hi * nnn, n)
    md = max(max_dim, bc.max_dim())
    line: list[tuple[int, ...]] = []
    for e in eps:
        b = bc.betti_at(float(e))
        b = list(b) + [0] * (md + 1 - len(b))
        line.append(tuple(b[: max_dim + 1]))
    return nnn, C.n_simplices, line


def show(title: str, ps: "V.PointSet", metric: str, max_dim: int, lo: float = 0.9, hi: float = 2.0) -> None:
    with _rich_ui.timed(_console, f"Rips + homology: {title}"):
        nnn, nsimp, line = betti_line(ps, metric, max_dim, lo, hi)
    print(f"\n== {title} == n={ps.n} dim={ps.dim} nn={nnn:.4g} nsimp={nsimp}")
    print("   eps-frac : " + "  ".join(f"{v:>6}" for v in np.linspace(lo, hi, 12).round(2)))
    for i, b in enumerate(line):
        print(f"   b({lo + (hi-lo)*i/11:.2f}) : " + "  ".join(f"{v:>6}" for v in b))


def main() -> int:
    dd = V.datasets._data_root()
    layers = V.datasets.list_layers()
    print(f"layers: {layers[0]}..{layers[-1]} ({len(layers)} files)")

    # 1) token cloud, full 5120 dims, layer 0
    tc = V.datasets.load_token_cloud(dd, layer=0)
    show("token cloud layer_000 FULL 5120d", tc, "euclidean", 2)

    # 2) token cloud, feature selection (first 16 dims), cosine metric
    tc16 = tc.select_dims(list(range(16)), name="layer_000_16d")
    show("token cloud layer_000 first-16-dim (cosine)", tc16, "cosine", 2)

    # 3) residual norms: 81 tokens x 65 layers (each token's norm trajectory)
    mat, labels = V.datasets.load_residual_matrix(dd, kind="norms")
    norms_ps = V.PointSet(mat, labels=labels, name="residual_norms_traj")
    show("residual-norm trajectories (81 x 65)", norms_ps, "euclidean", 2)

    # 4) layer_points shape check (few layers x few dims to stay small)
    lp = V.datasets.load_layer_points(dd, layers=[0, 16, 32, 48, 64], value_cols=None)
    lp8 = lp.select_dims(list(range(8)), name="layer_points_8d")
    print(f"\n== layer_points (5 layers) == n={lp.n} full_dim={lp.dim} -> 8d: n={lp8.n} dim={lp8.dim}")
    print("   sample labels:", lp.labels[:2], "...", lp.labels[-1])
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
