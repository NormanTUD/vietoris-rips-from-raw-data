# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import vrtda as V
from vrtda.geometry import min_enclosing_ball_radius


def main():
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(15, 2))
    D = V.pairwise_distances(pts)
    r = 0.5
    TOL = 1e-9 * max(1.0, r)
    A = (D <= 2.0 * r + 2.0 * TOL).copy()
    np.fill_diagonal(A, False)
    # enumerate triangles the same way
    tris = []
    for i in range(14):
        S = np.where(A[i, i+1:])[0] + (i + 1)
        if S.size < 2:
            continue
        sub = A[np.ix_(S, S)]
        jj, kk = np.triu_indices(S.size, 1)
        m = sub[jj, kk]
        for a, b in zip(S[jj[m]], S[kk[m]]):
            tris.append((i, a, b))
    print("edges with d near 2r:", sorted([round(float(D[i, j]), 6) for i in range(15) for j in range(i+1,15) if D[i,j] <= 2*r + 2*TOL]))
    print("edge (9,11) d =", D[9, 11], " MEB(edge) =", min_enclosing_ball_radius(pts[[9,11]]))
    print("in A[9,11]?", bool(A[9,11]))
    print("\nTriangles whose MEB <= r+TOL but have an edge with d/2 > r+TOL (i.e. edge would be dropped):")
    for (i, a, b) in tris:
        s = pts[[i, a, b]]
        rad = min_enclosing_ball_radius(s)
        if rad <= r + TOL:
            edges = [(i, a), (i, b), (a, b)]
            for (u, v) in edges:
                d = D[u, v]
                if d / 2.0 > r + TOL:
                    print(f"  tri {(i,a,b)} MEB={rad:.6f}  edge ({u},{v}) d={d:.6f} d/2={d/2:.6f}  inA={bool(A[u,v])}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
