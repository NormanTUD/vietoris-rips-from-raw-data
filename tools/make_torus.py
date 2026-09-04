# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Generate a ground-truth point cloud CSV for TDA validation."""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import PointSet, generators as G


def build(args) -> np.ndarray:
    if args.kind == "circle":
        if args.grid:
            return G.circle_grid(args.n, radius=args.radius)
        return G.circle(args.n, radius=args.radius, seed=args.seed, noise=args.noise)
    if args.kind == "product":
        if args.grid:
            return G.product_torus_grid(args.k, args.nper, radius=args.radius)
        return G.product_torus(args.k, n=args.n * args.nper, radius=args.radius, seed=args.seed, noise=args.noise)
    if args.kind == "donut":
        if args.grid:
            return G.donut_grid(args.n, args.nper, R=args.radius, r=args.minor)
        return G.donut(args.n * args.nper, R=args.radius, r=args.minor, seed=args.seed, noise=args.noise)
    if args.kind == "sphere":
        return G.sphere(args.n, dim=args.k, radius=args.radius, seed=args.seed, noise=args.noise)
    raise SystemExit(f"unknown kind {args.kind!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["circle", "product", "donut", "sphere"], required=True)
    p.add_argument("--k", type=int, default=2, help="ambient torus dimension (product/sphere)")
    p.add_argument("--n", type=int, default=24, help="points (or nu for grids)")
    p.add_argument("--nper", type=int, default=8, help="points per circle (grid) / nv (donut)")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--minor", type=float, default=0.35, help="donut minor radius")
    p.add_argument("--grid", action="store_true", help="use deterministic grid sampling")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    data = build(args)
    ps = PointSet(data, name=args.kind)
    ps.to_csv(args.out)
    print(f"wrote {args.out}: n={ps.n} dim={ps.dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
