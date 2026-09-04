# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""Unified, selectable CLI for the attractor-analysis methods.

Each method is a subcommand; run `uv run tools/methods.py --list` to see them.
Plots require matplotlib: `uv run --with matplotlib tools/methods.py ...`
Wasserstein on large diagrams requires scipy: `uv run --with scipy tools/methods.py ...`

    metrics    persistence entropy / landscape / image for a barcode
    distance   bottleneck / p-Wasserstein between two barcodes
    depth      cross-LAYER attractors (heatmap, chains, stable core, profile)
    mapper     1D Mapper graph (lens -> Rips per bin -> betti)
    dynamics   dynamical attractors (convergence, per-language, flow-SVD, attention)
"""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import (
    PointSet,
    pairwise_distances,
    build_rips,
    persistent_homology,
    datasets,
    attractors,
    persistence_metrics as pm,
    distance as dist,
    depth_persistence as dp,
    mapper as mapper_mod,
    dynamics as dyn,
)
from vrtda.beartype_guard import beartype_function, beartype_module
from vrtda.persistence import Barcode

METHODS = {
    "metrics": "persistence entropy / landscape / image for a barcode",
    "distance": "bottleneck / p-Wasserstein between two barcodes",
    "depth": "cross-LAYER attractors: heatmap, chains, stable core, profile",
    "mapper": "1D Mapper graph (lens -> Rips per bin -> betti)",
    "dynamics": "dynamical attractors: convergence, per-language, flow-SVD, attention",
}
METHODS_DYNAMIC = ["convergence", "per_language", "flow", "attention"]


def _try_plot(fn: str, *a: object, **k: object) -> object:
    try:
        from vrtda import plotting

        return getattr(plotting, fn)(*a, **k)
    except Exception as e:  # matplotlib missing / other
        print(f"[plot skipped] {e}")


def _cloud(args: argparse.Namespace, which: str) -> tuple[str, PointSet]:
    """Return (name, PointSet) from --{which}-layer or --{which}-csv."""
    layer = getattr(args, f"{which}_layer", None)
    csv = getattr(args, f"{which}_csv", None)
    if csv:
        ps = PointSet.from_csv(csv, value_cols=args.value_cols, index_cols=args.index_cols)
        return Path(csv).stem, ps
    L = layer if layer is not None else getattr(args, "layer", 0)
    return f"layer_{int(L):03d}", datasets.load_token_cloud(args.data_dir, int(L))


def _barcode(pts: np.ndarray, metric: str, max_dim: int, frac: float) -> tuple[Barcode, float]:
    D = pairwise_distances(pts, metric)
    d = D.copy(); np.fill_diagonal(d, np.inf)
    nn = float(d.min(1).mean())
    C = build_rips(pts, D, frac * nn, max_dim=max_dim)
    return persistent_homology(C), frac * nn


def _default_layers(args: argparse.Namespace) -> list[int]:
    if args.layers is not None:
        return args.layers
    all_l = datasets.list_layers(args.data_dir)
    return all_l[:: max(1, len(all_l) // 8)]


def cmd_metrics(args: argparse.Namespace) -> int:
    name, ps = _cloud(args, "a")
    bc, eps_max = _barcode(ps.data, args.metric, args.max_dim, args.frac)
    dim = args.dim
    print(f"[metrics] {name}  (n={ps.n}, dim={ps.dim}, eps_max={eps_max:.4g})")
    print(f"  entropy(d={dim})        = {pm.persistence_entropy(bc, dim):.4f}")
    print(f"  entropy_all(finite)     = {pm.persistence_entropy(bc):.4f}")
    xgrid, F = pm.persistence_landscape(bc, dim)
    print(f"  landscape(d={dim})       = {F.shape[0]} levels, top peak {F[0].max():.4f}")
    img = pm.persistence_image(bc, dim, eps_max=eps_max)
    print(f"  image(d={dim})          = {img.shape}, max {img.max():.4f}")
    if args.plot:
        _try_plot("plot_persistence_diagram", pm.persistence_diagram(bc, dim), args.plot + ".diag.png")
        _try_plot("plot_persistence_landscape", xgrid, F, args.plot + ".landscape.png")
        _try_plot("plot_persistence_image", img, args.plot + ".image.png")
    return 0


def cmd_distance(args: argparse.Namespace) -> int:
    na, pa = _cloud(args, "a")
    nb, pb = _cloud(args, "b")
    bca, _ = _barcode(pa.data, args.metric, args.max_dim, args.frac)
    bcb, _ = _barcode(pb.data, args.metric, args.max_dim, args.frac)
    print(f"[distance] {na} vs {nb}  (dim={args.dim}, top_k={args.top_k})")
    for which in args.which:
        if which == "bottleneck":
            print(f"  bottleneck        = {dist.bottleneck(bca, bcb, args.dim, args.top_k):.6g}")
        elif which == "wasserstein":
            try:
                w = dist.p_wasserstein(bca, bcb, args.dim, args.p, args.top_k)
                print(f"  wasserstein(p={args.p}) = {w:.6g}")
            except ImportError:
                print(f"  wasserstein(p={args.p}) = <needs scipy: uv run --with scipy tools/methods.py ...>")
    return 0


def cmd_depth(args: argparse.Namespace) -> int:
    layers = _default_layers(args)
    print(f"[depth] layers={layers}  eps_cap_frac={args.eps_cap_frac}  max_dim={args.max_dim}")
    lr = dp.layer_barcodes(args.data_dir, layers=layers, metric=args.metric,
                           eps_cap_frac=args.eps_cap_frac, max_dim=args.max_dim, texts=True)
    # profile over depth
    prof = dp.depth_profile(lr, dim=args.dim)
    print("\nlayer | nn | #intervals | total_persistence | beta_peak@frac")
    for L in layers:
        p = prof[L]
        print(f"  {L:3d} | {p['nn']:7.1f} | {p['n_intervals']:6d} | {p['total_persistence']:8.1f} | {p['beta_peak']:2d}@{p['beta_peak_frac']:.2f}")

    # heatmap
    H, fracs, Ls = dp.betti_heatmap(lr, scale_fracs=np.linspace(0.5, 2.5, 9), dim=args.dim, metric=args.heat_metric)
    print(f"\n{args.heat_metric} heatmap (rows=frac x nn, cols=layer):")
    print("        " + "".join(f"{L:>5d}" for L in Ls))
    for s, f in enumerate(fracs):
        print(f"{f:5.2f}   " + "".join(f"{H[s, t]:5.0f}" for t in range(len(Ls))))

    # chains
    chains = dp.depth_chains(lr, min_overlap=args.min_overlap, max_gap=args.max_gap,
                             essential_only=args.essential, top_k=args.top_k, dim=args.dim)
    long = [c for c in chains if c.length >= 2]
    print(f"\n# chains: {len(chains)}  spanning >=2 layers: {len(long)}")
    for c in sorted(long, key=lambda c: -c.length)[: args.show]:
        toks = set()
        for L in c.layers():
            for t in c.per_layer_tokens[L]:
                i = lr[L].labels.index(t)
                if lr[L].texts:
                    toks.add(lr[L].texts[i])
        print(f"  {c.span} len={c.length} ntok={len(c.tokens)} {sorted(toks)[:5]}")
    core = dp.stable_core(chains, min_layer_fraction=args.core_fraction, total_layers=len(layers))
    print(f"# stable core (>= {args.core_fraction:.0%} of depth): {len(core)}")

    if args.plot:
        _try_plot("plot_depth_heatmap", H, fracs, Ls, args.plot + ".heatmap.png")
        _try_plot("plot_betti_over_depth", prof, args.plot + ".betti_depth.png")
        if long:
            from vrtda.reduction import pca

            c = sorted(long, key=lambda c: -c.length)[0]
            L = c.layers()[0]
            ps = datasets.load_token_cloud(args.data_dir, L)
            xy = pca(ps.data, 2)[0]
            idx = [ps.labels.index(t) for t in c.per_layer_tokens[L] if t in ps.labels]
            _try_plot("plot_attractor_overlay", xy, [idx] if len(idx) >= 2 else [],
                      args.plot + ".overlay.png", title=f"attractor layers {c.span} @ L{L}")
    return 0


def cmd_mapper(args: argparse.Namespace) -> int:
    if args.csv:
        ps = PointSet.from_csv(args.csv, value_cols=args.value_cols, index_cols=args.index_cols)
        name = Path(args.csv).stem
        phi = np.linalg.norm(ps.data, axis=1)  # default lens: distance from origin
    else:
        L = int(args.layer)
        ps = datasets.load_token_cloud(args.data_dir, L)
        norms, _labels = datasets.load_residual_matrix(args.data_dir, "norms")
        phi = norms[:, L]  # residual-norm lens
        name = f"layer_{L:03d}"
    g = mapper_mod.mapper(ps.data, phi, n_bins=args.n_bins, overlap=args.overlap,
                          eps=args.eps, eps_frac=args.eps_frac, max_dim=args.max_dim)
    print(f"[mapper] {name}  lens range=[{phi.min():.3g},{phi.max():.3g}]  nodes={g.n_nodes} edges={g.n_edges}")
    print("  bin | interval | n | b0 | b1")
    for i, n in enumerate(g.nodes):
        print(f"  {i:2d}  | [{n.interval[0]:7.3g},{n.interval[1]:7.3g}] | {n.n_points:3d} | {n.beta0} | {n.beta1}")
    print(f"  beta1 profile: {g.beta1_profile()}  total loops = {sum(g.beta1_profile())}")
    if args.plot:
        _try_plot("plot_mapper", g, args.plot + ".mapper.png")
    return 0


def cmd_dynamics(args: argparse.Namespace) -> int:
    which = args.which
    if "convergence" in which:
        conv = dyn.convergence(args.data_dir)
        s = dyn.convergence_summary(conv)
        print("[dynamics] convergence summary:")
        for k, v in s.items():
            print(f"  {k} = {v}")
        if args.plot:
            _try_plot("plot_convergence", conv, args.plot + ".convergence.png")
    if "per_language" in which:
        layers, mat, prompts = dyn.per_language_final_token_distance(args.data_dir, args.layers)
        print(f"\n[dynamics] per-language answer-token distance to centroid (layers={layers}):")
        print("        " + "".join(f"{L:>8d}" for L in layers))
        for i in range(mat.shape[0]):
            print(f"  {prompts[i][:20]:20s} " + "".join(f"{mat[i, j]:8.1f}" for j in range(mat.shape[1])))
    if "flow" in which:
        comp, var, ls = dyn.flow_svd(args.data_dir, args.layers)
        print(f"\n[dynamics] flow SVD (layers={ls}) explained variance: {np.round(var, 4)}")
    if "attention" in which:
        lyr, curve, peak = dyn.attention_over_depth(args.data_dir, metric=args.attention)
        print(f"\n[dynamics] attention[{args.attention}] over depth: peak layer {peak}")
        print("  curve@0,8,16,24,32,40,48,56,63:", np.round(curve[[0, 8, 16, 24, 32, 40, 48, 56, 63]], 3))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list available methods")
    p.add_argument("--data-dir", default=None, help="override the dataset root")
    sub = p.add_subparsers(dest="cmd")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--plot", default=None, help="prefix for PNG output (needs --with matplotlib)")
        sp.add_argument("--metric", default="euclidean")
        sp.add_argument("--max-dim", type=int, default=2)
        sp.add_argument("--frac", type=float, default=3.0)
        sp.add_argument("--dim", type=int, default=1)
        sp.add_argument("--value-cols", nargs="*", default=None)
        sp.add_argument("--index-cols", nargs="*", default=None)

    common = beartype_function(common)

    sp = sub.add_parser("metrics", help=METHODS["metrics"])
    common(sp)
    sp.add_argument("--a-layer", type=int, default=None)
    sp.add_argument("--a-csv", default=None)
    sp.set_defaults(fn=cmd_metrics)

    sp = sub.add_parser("distance", help=METHODS["distance"])
    common(sp)
    sp.add_argument("--a-layer", type=int, default=None)
    sp.add_argument("--a-csv", default=None)
    sp.add_argument("--b-layer", type=int, default=None)
    sp.add_argument("--b-csv", default=None)
    sp.add_argument("--which", nargs="*", default=["bottleneck", "wasserstein"])
    sp.add_argument("--p", type=float, default=2.0)
    sp.add_argument("--top-k", type=int, default=20, help="restrict each diagram to its k most persistent points")
    sp.set_defaults(fn=cmd_distance)

    sp = sub.add_parser("depth", help=METHODS["depth"])
    common(sp)
    sp.add_argument("--layers", type=int, nargs="*", default=None, help="default: every ~8th layer")
    sp.add_argument("--eps-cap-frac", type=float, default=2.5)
    sp.add_argument("--heat-metric", default="betti", choices=["betti", "essential", "persistence"])
    sp.add_argument("--top-k", type=int, default=10)
    sp.add_argument("--min-overlap", type=float, default=0.2)
    sp.add_argument("--max-gap", type=int, default=1)
    sp.add_argument("--essential", action="store_true", help="track only essential (never-dying) loops")
    sp.add_argument("--core-fraction", type=float, default=0.5)
    sp.add_argument("--show", type=int, default=15)
    sp.set_defaults(fn=cmd_depth)

    sp = sub.add_parser("mapper", help=METHODS["mapper"])
    common(sp)
    sp.add_argument("--layer", type=int, default=16)
    sp.add_argument("--csv", default=None)
    sp.add_argument("--lens", default=None, help="numeric column for --csv (default: norm)")
    sp.add_argument("--n-bins", type=int, default=8)
    sp.add_argument("--overlap", type=float, default=0.5)
    sp.add_argument("--eps", type=float, default=None)
    sp.add_argument("--eps-frac", type=float, default=2.0)
    sp.set_defaults(fn=cmd_mapper)

    sp = sub.add_parser("dynamics", help=METHODS["dynamics"])
    common(sp)
    sp.add_argument("--which", nargs="*", default=list(METHODS_DYNAMIC))
    sp.add_argument("--layers", type=int, nargs="*", default=None)
    sp.add_argument("--attention", default="to_self", choices=["to_self", "from_self_entropy", "max_source"])
    sp.set_defaults(fn=cmd_dynamics)

    args = p.parse_args()
    if args.list or args.cmd is None:
        print("available methods (uv run tools/methods.py <method> [...]):")
        for name, desc in METHODS.items():
            print(f"  {name:10s} {desc}")
        return 0
    return args.fn(args)


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
