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
from vrtda.persistence import Barcode, Interval
from vrtda import distance as D

def bcd(intervals, dim=1):
    return Barcode(intervals=[Interval(b, d, dim, i) for i, (b, d) in enumerate(intervals)])

a = bcd([(0.0, 1.0), (0.5, 4.0), (2.0, 3.0)])
b = bcd([(0.2, 2.0), (1.0, 5.0)])
for p in [1, 2]:
    print(f"W_{p} =", D.p_wasserstein(a, b, dim=1, p=p))

# independent brute force: enumerate matchings of P to (Q union diagonal)
# W_p^p = min over matchings of sum matched ||p-q||^p + unmatchedP ||p-diag||^p + unmatchedQ ||q-diag||^p
P = np.array([(0.0,1.0),(0.5,4.0),(2.0,3.0)])
Q = np.array([(0.2,2.0),(1.0,5.0)])
def h(v, p):
    return (2.0*((v[1]-v[0])/2.0)**p)**(1/p)
def lp(u,v,p):
    return np.sum(np.abs(u-v)**p)**(1/p)
import itertools
def brute(p):
    # assign each P to a distinct Q or diagonal
    best = np.inf
    for assign in itertools.product(range(len(Q)+1), repeat=len(P)):  # +1 = diagonal
        used = set(j for j in assign if j != len(Q))
        if len(used) != len([j for j in assign if j != len(Q)]):
            continue  # two P to same Q
        cost = 0.0
        for i, j in enumerate(assign):
            if j == len(Q):
                cost += h(P[i], p)**p
            else:
                cost += lp(P[i], Q[j], p)**p
        for j in range(len(Q)):
            if j not in used:
                cost += h(Q[j], p)**p
        best = min(best, cost)
    return best**(1/p)
for p in [1,2]:
    print(f"brute W_{p} =", brute(p))
