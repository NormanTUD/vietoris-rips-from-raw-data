# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""End-to-end attractor analysis on the capital_berlin_multilingual data.

Pipeline: load token clouds (per layer) -> Rips -> persistent homology -> attractor
summary, plus a reduced 2D view for a couple of layers. Writes a markdown report and
(optionally, if matplotlib is installed) a couple of plots.

Run:
    uv run examples/attractor_analysis.py --layers 0 16 32 48 64 --out examples/report.md
"""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json

import numpy as np

from vrtda import (
    PointSet,
    pairwise_distances,
    build_rips,
    persistent_homology,
    datasets,
    attractors,
    reports,
    reduction,
)
from vrtda.beartype_guard import beartype_module
from vrtda.complexes import FilteredComplex
from vrtda.persistence import Barcode


def _nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def layer_analysis(layer: int, metric: str, max_dim: int, frac: float) -> tuple[PointSet, Barcode, float, FilteredComplex]:
    ps = datasets.load_token_cloud(layer=layer)
    D = pairwise_distances(ps.data, metric)
    nn = _nn(D)
    eps_max = frac * nn
    C = build_rips(ps.data, D, eps_max, max_dim=max_dim)
    bc = persistent_homology(C)
    return ps, bc, eps_max, C


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="*", default=[0, 16, 32, 48, 64])
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--frac", type=float, default=1.5)
    p.add_argument("--min-fraction", type=float, default=0.1)
    p.add_argument("--out", default="examples/report.md")
    p.add_argument("--plot-dir", default=None, help="if set, save plots there (needs matplotlib)")
    args = p.parse_args()

    dd = datasets._data_root()
    model_info = json.loads((dd / "model_info.json").read_text())
    group_info = json.loads((dd / "group_info.json").read_text())

    rep = reports.Report("Attractor Analysis — capital_berlin_multilingual")
    rep.section("Data", [
        f"model: {model_info.get('model_name')}",
        f"d_model={model_info.get('d_model')}, n_layers={model_info.get('n_layers')}, "
        f"n_heads={model_info.get('n_heads')}",
        f"prompts: {group_info.get('n_prompts')} (multilingual), expected answer: "
        f"{group_info.get('expected_answer')}",
        f"point cloud per layer: 81 token hidden states x {model_info.get('d_model')} dims",
        f"metric={args.metric}, max_dim={args.max_dim}, eps_max={args.frac} x mean-nn",
    ])

    rows = []
    barcodes = {}
    for L in args.layers:
        ps, bc, eps_max, C = layer_analysis(L, args.metric, args.max_dim, args.frac)
        barcodes[f"layer_{L:03d}"] = (bc, eps_max)
        s = attractors.per_dim_summary(bc, eps_max, args.min_fraction)
        rows.append([
            f"layer_{L:03d}", ps.n, C.n_simplices, round(eps_max, 3),
        ] + [s[d][m] for d in range(args.max_dim + 1) for m in
             ("essential", "long_lived", "total_persistence")])

    hdr = (["layer", "n", "nsimp", "eps_max"] +
           [f"b{d}_{m}" for d in range(args.max_dim + 1)
            for m in ("essential", "long_lived", "persist")])
    rep.table("Persistent features across depth (attractors)", hdr, rows)

    # auto findings
    findings = []
    for d in range(1, args.max_dim + 1):
        key = f"b{d}_essential"
        best = max(rows, key=lambda r: r[hdr.index(key)])
        findings.append(
            f"b{d}_essential peaks at {best[0]} with {best[hdr.index(key)]} persistent "
            f"feature(s)."
        )
    rep.section("Findings", findings)

    # betti function for the first layer (table)
    bc0, eps0 = barcodes[f"layer_{args.layers[0]:03d}"]
    epsilons = np.linspace(0.9 * eps0 / args.frac, eps0, 12)
    arr = bc0.betti_function(list(epsilons))
    bhdr, brows = reports.betti_table(epsilons, arr)
    rep.table(f"Betti function, layer_{args.layers[0]:03d}", bhdr, brows)

    out = Path(args.out)
    rep.write(out, fmt="md")
    print(f"wrote {out}")

    if args.plot_dir:
        from vrtda import plotting
        pdir = Path(args.plot_dir); pdir.mkdir(parents=True, exist_ok=True)
        plotting.plot_betti_function(epsilons, arr, pdir / f"betti_layer_{args.layers[0]:03d}.png")
        plotting.plot_barcode(bc0.intervals, pdir / f"barcode_layer_{args.layers[0]:03d}.png")
        print(f"wrote plots to {pdir}")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
