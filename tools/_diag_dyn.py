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
from vrtda import dynamics

conv = dynamics.convergence()
print("layers:", conv.layers[:5], "...", conv.layers[-3:])
print("mean_pairwise:", np.round(conv.mean_pairwise[[0,1,5,10,20,40,50,64]],2))
print("centroid_norm:", np.round(conv.centroid_norm[[0,1,5,10,20,40,50,64]],2))
print("mean_dist:", np.round(conv.mean_dist[[0,1,5,10,20,40,50,64]],2))
print("summary:", dynamics.convergence_summary(conv))

layers, mat, prompts = dynamics.per_language_final_token_distance(layers=[0,8,16,32,48,64])
print("\nper-language final-token dist to centroid:")
print("       ", "  ".join(f"L{L:>3d}" for L in layers))
for i in range(mat.shape[0]):
    print(f"  {prompts[i][:22]:22s} " + "  ".join(f"{mat[i,j]:6.1f}" for j in range(mat.shape[1])))

comp, var, ls = dynamics.flow_svd(layers=list(range(0,65,4)))
print("\nflow_svd explained variance:", np.round(var,4), "shape", comp.shape)

lyr, curve, peak = dynamics.attention_over_depth(metric="to_self")
print("\nattention to_self over depth (answ tokens): peak layer", peak)
print("  values@ [0,4,8,16,24,32,40,48,56,63]:", np.round(curve[[0,4,8,16,24,32,40,48,56,63]],3))
