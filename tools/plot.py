# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "matplotlib>=3.7", "beartype>=0.18", "rich>=13"]
# ///
"""Plot Betti function, persistence barcode, and 2D point cloud of a point-cloud CSV.

Produces <out-dir>/betti.png, <out-dir>/barcode.png, <out-dir>/cloud.png.
Requires matplotlib (already declared in the PEP-723 header above).

Run:
    uv run tools/plot.py --points mydata.csv --out-dir plots/
    uv run tools/plot.py --points mydata.csv --value-cols x y --metric cosine
"""
import argparse
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
from vrtda import (
    PointSet,
    pairwise_distances,
    build_rips,
    plotting,
)
from vrtda.beartype_guard import beartype_module


def _nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", required=True, help="point-cloud CSV (one point per row)")
    p.add_argument("--value-cols", nargs="*", default=None)
    p.add_argument("--index-cols", nargs="*", default=None)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac", type=float, default=1.6, help="eps_max as fraction of mean-nn")
    p.add_argument("--min-dim", type=int, default=1, help="only show homology dims >= this (1 = drop dim-0 noise)")
    p.add_argument("--min-persistence", type=float, default=0.05, help="drop barcode bars shorter than this fraction of eps_max")
    p.add_argument("--max-bars", type=int, default=12, help="max bars per dimension in the barcode panel")
    p.add_argument("--title", default="")
    p.add_argument("--out-dir", required=True, help="directory for the PNGs")
    args = p.parse_args()

    console = Console()
    _rich_ui.params_table(p, args, console)

    ps = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols)
    D = pairwise_distances(ps.data, args.metric)
    nn = _nn(D)
    eps_max = args.frac * nn
    with _rich_ui.timed(console, f"Building Rips complex (max_dim={args.max_dim})"):
        C = build_rips(ps.data, D, eps_max, max_dim=args.max_dim)
    bc = _rich_ui.homology_progress(C, console)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = args.title or ps.name
    eps = np.linspace(0.9 * nn, eps_max, 16)
    arr = bc.betti_function(list(eps))
    plotting.plot_betti_function(eps, arr, out / "betti.png", title=f"Betti function — {t}")
    plotting.plot_persistence_summary(
        bc.intervals, out / "barcode.png", title=f"Persistence summary — {t}",
        min_dim=args.min_dim, min_persistence_frac=args.min_persistence,
        max_bars=args.max_bars, eps_max=eps_max,
    )
    plotting.plot_point_cloud_2d(ps.data, out / "cloud.png", labels=ps.labels, title=f"Point cloud — {t}")
    print(f"wrote {out}/betti.png, barcode.png, cloud.png (n={ps.n}, dim={ps.dim})")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
