# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18", "rich>=13"]
# ///
"""Sweep epsilon and print the Betti function beta_k(eps) for a point cloud."""
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
from vrtda import PointSet, pairwise_distances, build_rips
from vrtda.beartype_guard import beartype_module


def _nn_mean(D: np.ndarray) -> float:
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", required=True, help="point cloud CSV (one point per row)")
    p.add_argument("--value-cols", nargs="*", default=None, help="numeric columns (default: all non-index)")
    p.add_argument("--index-cols", nargs="*", default=None, help="label columns")
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac-lo", type=float, default=0.9, help="lower eps as fraction of mean-nn")
    p.add_argument("--frac-hi", type=float, default=2.0, help="upper eps as fraction of mean-nn")
    p.add_argument("--eps-lo", type=float, default=None, help="explicit lower eps (overrides frac-lo)")
    p.add_argument("--eps-hi", type=float, default=None, help="explicit upper eps (overrides frac-hi)")
    p.add_argument("--n", type=int, default=30, help="number of epsilon samples")
    p.add_argument("--out", default=None, help="optional output CSV of the sweep")
    args = p.parse_args()

    console = Console()
    _rich_ui.params_table(p, args, console)

    ps = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols)
    D = pairwise_distances(ps.data, args.metric)
    nn = _nn_mean(D)
    lo = args.eps_lo if args.eps_lo is not None else args.frac_lo * nn
    hi = args.eps_hi if args.eps_hi is not None else args.frac_hi * nn

    with _rich_ui.timed(console, f"Building Rips complex (max_dim={args.max_dim})"):
        C = build_rips(ps.data, D, hi, max_dim=args.max_dim)
    bc = _rich_ui.homology_progress(C, console)
    md = max(args.max_dim, bc.max_dim())

    epsilons = np.linspace(lo, hi, args.n)
    rows = []
    header = ["eps"] + [f"b{k}" for k in range(md + 1)]
    print("  " + "  ".join(f"{h:>8}" for h in header))
    for e in epsilons:
        b = bc.betti_at(float(e))
        b = list(b) + [0] * (md + 1 - len(b))
        rows.append([float(e)] + b)
        print("  " + "  ".join(f"{v:8.4g}" for v in rows[-1]))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(",".join(header) + "\n")
            for r in rows:
                fh.write(",".join(f"{v:.6g}" for v in r) + "\n")
        print(f"wrote {args.out}")
    print(f"n={ps.n} dim={ps.dim} metric={args.metric} nn={nn:.4g} eps=[{lo:.4g},{hi:.4g}] nsimp={C.n_simplices}")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
