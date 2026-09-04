# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from vrtda.depth_persistence import layer_barcodes, depth_chains, depth_profile

layers = [16, 17, 18, 19, 20]
lr = layer_barcodes(layers=layers, eps_cap_frac=2.5, max_dim=2, texts=True)
prof = depth_profile(lr, dim=1)
for L in layers:
    p = prof[L]
    print(f"  L{L:3d} nn={p['nn']:7.1f} #H1={p['n_intervals']:5d} totP={p['total_persistence']:7.1f} peak={p['beta_peak']}@{p['beta_peak_frac']:.2f}")

for gap in [1, 2]:
    chains = depth_chains(lr, min_overlap=0.2, max_gap=gap, top_k=10)
    ge2 = [c for c in chains if c.length >= 2]
    ge3 = [c for c in chains if c.length >= 3]
    print(f"\nmax_gap={gap}: chains={len(chains)}  len>=2: {len(ge2)}  len>=3: {len(ge3)}")
    for c in sorted(ge2, key=lambda c: -c.length)[:8]:
        txts = set()
        for L in c.layers():
            for t in c.per_layer_tokens[L]:
                i = lr[L].labels.index(t)
                if lr[L].texts:
                    txts.add(lr[L].texts[i])
        print(f"   {c.span} len={c.length} ntok={len(c.tokens)} {sorted(txts)[:5]}")
