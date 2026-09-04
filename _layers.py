# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
import sys, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vrtda import datasets

t0=time.time()
layers = datasets.list_layers()
ref = [0, 21, 42, 63]
stack = np.vstack([datasets.load_token_cloud(layer=L).data for L in ref])
Xc = stack - stack.mean(0)
_u,_s,Vt = np.linalg.svd(Xc, full_matrices=False)
frame = Vt[:3]
ntok = stack.shape[0]//len(ref)
out = np.zeros((len(layers), ntok, 3))
for i,L in enumerate(layers):
    X = datasets.load_token_cloud(layer=L).data
    out[i] = (X - stack.mean(0)) @ frame.T
t2=time.time()
print(f"all {len(layers)} layers -> {out.shape} in {t2-t0:.2f}s total")
print("finite:", bool(np.all(np.isfinite(out))), " token0 L0:", out[0,0].round(3), " L64:", out[-1,0].round(3))
# est. json size
import json
s=len(json.dumps(out.tolist()))
print(f"traj JSON ~ {s/1024:.0f} KB")
