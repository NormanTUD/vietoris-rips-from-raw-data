# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "matplotlib>=3.7"]
# ///
import sys, traceback
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
from vrtda import plotting
pts = np.random.default_rng(0).normal(size=(12, 2))
try:
    plotting.plot_attractor_overlay(pts, [[0, 1, 2, 3]], "/tmp/_ov.png", title="t")
    print("OK")
except Exception:
    traceback.print_exc()
