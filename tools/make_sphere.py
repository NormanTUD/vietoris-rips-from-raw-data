# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""Generate a ground-truth point cloud on the k-sphere S^k for TDA validation.

S^k lives in R^{k+1} and has beta_0 = 1, beta_k = 1, and all other beta = 0.
`--verify` checks that topology against the exact simplicial sphere (the boundary
of a (k+1)-simplex), and -- for the 1-sphere -- also recovers it from a
Vietoris-Rips complex on the actual cloud.
"""
import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import PointSet, generators as G
from vrtda import pairwise_distances, build_rips, persistent_homology, betti_at
from vrtda.complexes import make_simplicial_sphere
from vrtda.beartype_guard import beartype_module


def build(args: argparse.Namespace) -> np.ndarray:
    return G.sphere(args.n, dim=args.k, radius=args.radius, seed=args.seed, noise=args.noise)


def _expected_betti(k: int) -> list[int]:
    b = [0] * (k + 1)
    b[0] = 1
    b[k] = 1
    return b


def _rips_betti(X: np.ndarray, max_dim: int) -> list[int]:
    D = pairwise_distances(X)
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    eps = 1.6 * float(d.min(axis=1).mean())
    C = build_rips(X, D, eps, max_dim=max_dim)
    return persistent_homology(C).betti_at(eps)


def verify(k: int, cloud: np.ndarray) -> None:
    expected = _expected_betti(k)
    print(f"verify: S^{k} expected beta = {expected}")
    C = make_simplicial_sphere(k)
    exact = betti_at(C, float(C.values.max()))
    print(f"  exact complex beta = {exact}")
    assert exact == expected, f"exact complex beta {exact} != expected {expected}"
    if k == 1:
        rb = _rips_betti(cloud, max_dim=2)
        print(f"  rips-on-cloud beta = {rb}")
        assert (list(rb) + [0])[:2] == [1, 1], f"rips beta {rb} != [1,1]"
    else:
        print("  (rips-on-cloud not asserted: a uniform S^k cloud is scale-sensitive for k>=2)")
    print(f"  OK: S^{k} topology verified")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=2, help="sphere dimension (S^k embedded in R^{k+1})")
    p.add_argument("--n", type=int, default=256, help="number of points on the sphere")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="check ground-truth Betti numbers")
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    data = build(args)
    ps = PointSet(data, name="sphere")
    ps.to_csv(args.out)
    print(f"wrote {args.out}: n={ps.n} dim={ps.dim}")
    if args.verify:
        verify(args.k, data)
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
