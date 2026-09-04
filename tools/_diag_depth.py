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
from vrtda.depth_persistence import layer_barcodes, depth_chains, betti_heatmap, depth_profile

layers = [0, 16, 32, 48, 64]
print(f"layers: {layers}")
lr = layer_barcodes(layers=layers, eps_cap_frac=2.5, max_dim=2, texts=True)

prof = depth_profile(lr, dim=1)
print("\nlayer | nn | #H1int | totalPERSIST | beta1_peak@frac")
for L in layers:
    p = prof[L]
    print(f"  {L:3d} | {p['nn']:7.1f} | {p['n_intervals']:6d} | {p['total_persistence']:8.1f} | {p['beta_peak']:2d}@{p['beta_peak_frac']:.2f}")

H, fracs, Ls = betti_heatmap(lr, scale_fracs=np.linspace(0.5, 2.5, 9), dim=1, metric="betti")
print("\nbeta1 heatmap (rows=scale-frac x nn, cols=layer):")
print("        " + "".join(f"{L:>5d}" for L in Ls))
for s, f in enumerate(fracs):
    print(f"{f:5.2f}   " + "".join(f"{H[s,t]:5.0f}" for t in range(len(Ls))))

chains = depth_chains(lr, min_overlap=0.2, max_gap=2, top_k=10)
print(f"\n# significant-loop chains (top10/layer): {len(chains)}")
print("  (span, length, ntokens, sample token texts)")
for c in chains[:12]:
    txts = set()
    for L in c.layers():
        for t in c.per_layer_tokens[L]:
            i = lr[L].labels.index(t)
            if lr[L].texts:
                txts.add(lr[L].texts[i])
    print(f"  {c.span} len={c.length} ntok={len(c.tokens)} {sorted(txts)[:5]}")

long = [c for c in chains if c.length >= 3]
print(f"\n# chains spanning >=3 layers: {len(long)}")
for c in long[:10]:
    txts = set()
    for L in c.layers():
        for t in c.per_layer_tokens[L]:
            i = lr[L].labels.index(t)
            if lr[L].texts:
                txts.add(lr[L].texts[i])
    print(f"  {c.span} len={c.length} ntok={len(c.tokens)} {sorted(txts)[:5]}")
