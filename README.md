# vrtda — Vietoris–Rips Topological Data Analysis

A small, modular, dependency-light Python toolkit for **Topological Data Analysis (TDA)**:
it turns a cloud of points into a Vietoris–Rips (or strict Vietoris) simplicial complex,
computes **persistent homology**, **Betti numbers**, **barcodes**, and **attractor
features** — and works on *any* point cloud you throw at it (your own CSV, or the bundled
transformer-activation dataset).

No `venv`, no `GUDHI`. Just `uv` + `numpy`.

---

## What it does

| Capability | Tool |
|---|---|
| **One-command full analysis** of a CSV (barcode + Betti + attractors + report + plots) | `tools/analyze.py` |
| Betti numbers β_k(ε) over a range of ε | `tools/betti_sweep.py` |
| Persistence barcode → CSV | `tools/barcodes.py` |
| Feature selection + dimensionality reduction (PCA / UMAP / t-SNE) | `tools/project.py` |
| Compare persistent "attractor" features across many clouds | `tools/attractors.py` |
| Plots (Betti function, persistence summary = barcode + diagram, 2D cloud) with **matplotlib** | `tools/plot.py` |
| **Interactive HTML** — drag an ε-slider / ▶ Play to watch the Rips complex grow with **live β₀/β₁/β₂**, a persistence diagram and the Betti curve (3D, drag-to-rotate) | `tools/interactive.py` |
| Generate synthetic ground-truth (circle, torus T^k, donut, sphere) | `tools/make_torus.py` |
| Load the bundled transformer data + run TDA | `tools/load_smoke.py`, `examples/attractor_analysis.py` |
| Run the test suite (170 tests) | `tools/run_tests.py` |

Under the hood (`vrtda/` package):
`pointset`, `metrics`, `distances`, `geometry` (min-enclosing-ball), `complexes`
(Rips + Vietoris), `persistence` (GF(2) row-echelon), `homology`, `cohomology`,
`barcodes`, `generators`, `datasets`, `reduction`, `attractors`, `persistence_metrics`,
`distance` (bottleneck/Wasserstein), `cocycles`, `depth_persistence`, `mapper`,
`dynamics`, `reports`, `plotting` — every public function fully type-hinted and
runtime-checked (see **Runtime type checking** below).

See **`docs/MATH.md`** for the formal math and references, and **`docs/PLAN.md`** for the roadmap.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python ≥ 3.10 (uv handles this automatically)
- `numpy` (pulled automatically by the scripts)
- **Optional:** `matplotlib` (for plots), `umap-learn` / `scikit-learn` (for UMAP/t-SNE)

Every script is a [PEP 723](https://peps.python.org/pep-0723/) file: it declares its own
dependencies in the `# /// script` header, so `uv run` builds a cached, isolated
environment for it — no manual setup, no reinstall between runs.

---

## Runtime type checking (beartype, always on)

Every function in the library **and** in the tools/examples carries full type
hints, and **every one of them is checked at runtime** by
[beartype](https://beartype.readthedocs.io/). This is a hard contract, not an
opt-in feature:

- Each `vrtda` module wraps itself at load time
  (`vrtda/beartype_guard.py` → `beartype_module(__name__)`), covering all
  module-level functions, class methods **and** `@property` getters.
- There is **no flag, env var or argument to disable it** — type checking is
  part of the package's behaviour.
- A mis-typed call fails fast with a `BeartypeCallHint…Violation` pointing at the
  offending parameter/return, instead of producing wrong numbers silently.
- The one deliberate relaxation is the **PEP 484 numeric tower**: an `int` is
  accepted wherever a `float` is expected (e.g. `p_wasserstein(..., p=2)`),
  matching how the numeric API is actually called. `int` returns stay strict, and
  non-numeric types are always rejected.
- `beartype` is therefore a declared dependency in every PEP 723 header
  (`numpy`, `pytest`, `beartype`), so it is always present when the code runs.

You can see it working:

```bash
uv run tools/run_tests.py   # the suite exercises the checked paths
# any call that violates a hint raises immediately:
#   BeartypeCallHintParamViolation: … parameter X=… violates type hint …
```

---

## Quick start (3 lines, any data)

```bash
# 1. make a synthetic 2-torus (or skip this and use YOUR OWN csv)
uv run tools/make_torus.py --kind product --k 2 --nper 8 --grid --out torus2.csv

# 2. full analysis in one command -> ./out/  (report.md, barcode.csv, betti_function.csv, attractors.csv)
uv run tools/analyze.py --points torus2.csv --out-dir out/

# 3. with plots (pulls matplotlib automatically)
uv run --with matplotlib tools/analyze.py --points torus2.csv --out-dir out/ --plots
```

Open `out/report.md` — for the 2-torus you should see **β_0 = 1, β_1 = 2** (the two
torus loops) as *essential* features.

---

## Your own data: the CSV format

A point cloud is a plain CSV, **one point per row**:

```
x,y,z
0.1,0.2,0.3
1.4,-0.7,2.2
...
```

Rules:
- Row 1 = column names.
- Each following row = one point.
- **All columns are treated as coordinates by default** (so keep the file all-numeric, or specify columns).
- If your CSV has label/text columns, tell the tool which are coordinates:
  - `--value-cols x y z`  → treat *exactly* these as coordinates, ignore the rest.
  - `--index-cols id lang` → treat *exactly* these as labels, use the rest as coordinates.

That's it. Any of the tools below accepts `--points file.csv --value-cols ... --index-cols ...`.

> Tip: `pd.to_csv(df, index=False)` from pandas produces a compatible file.

---

## The tools (with examples)

### `analyze.py` — one command, full report  ⭐

```bash
# basic
uv run tools/analyze.py --points mydata.csv --out-dir out/

# pick coordinate columns, cosine metric, go up to triangles
uv run tools/analyze.py --points mydata.csv --value-cols a b c --metric cosine --max-dim 2 --out-dir out/

# with plots (matplotlib)
uv run --with matplotlib tools/analyze.py --points mydata.csv --out-dir out/ --plots

# larger epsilon window (1.6x nearest-neighbour by default)
uv run tools/analyze.py --points mydata.csv --frac 2.5 --out-dir out/
```

Writes: `report.md`, `barcode.csv`, `betti_function.csv`, `attractors.csv` (+ `*.png` with `--plots`).

### `make_torus.py` — synthetic ground truth

```bash
uv run tools/make_torus.py --kind circle  --grid --n 30 --out circle.csv
uv run tools/make_torus.py --kind product --k 2 --nper 8 --grid --out torus2.csv   # (S^1)^2
uv run tools/make_torus.py --kind product --k 3 --nper 4 --grid --out torus3.csv   # (S^1)^3
uv run tools/make_torus.py --kind donut   --n 20 --nper 10 --grid --out donut.csv
uv run tools/make_torus.py --kind sphere  --k 2 --n 120 --out sphere.csv
```

### `betti_sweep.py` — β_k(ε) table

```bash
uv run tools/betti_sweep.py --points torus2.csv --n 20
uv run tools/betti_sweep.py --points mydata.csv --value-cols x y --metric euclidean --max-dim 2 --n 30 --out sweep.csv
```

### `barcodes.py` — persistence barcode

```bash
uv run tools/barcodes.py --points torus2.csv --frac 2.0 --out barcode.csv
uv run tools/barcodes.py --points mydata.csv --value-cols x y --out barcode.csv --summary-out summary.csv
```

### `project.py` — feature selection + dimensionality reduction

```bash
# keep the 64 most variable dims, then PCA to 3D
uv run tools/project.py --points mydata.csv --top-k 64 --method pca --components 3 --out reduced.csv

# explicit dims, or use UMAP / t-SNE (add the dep)
uv run tools/project.py --points mydata.csv --dims 0,5,10,20 --method pca --components 2 --out reduced.csv
uv run --with umap-learn tools/project.py --points mydata.csv --method umap --components 2 --out reduced_umap.csv
uv run --with scikit-learn tools/project.py --points mydata.csv --method tsne --components 2 --out reduced_tsne.csv
```

### `attractors.py` — compare persistent features across many clouds

```bash
# compare your own CSVs side by side
uv run tools/attractors.py --csvs groupA.csv groupB.csv --max-dim 2 --out compare.csv

# ...or across transformer layers (bundled data)
uv run tools/attractors.py --layers 0 16 32 48 64 --max-dim 2
```

### `plot.py` — matplotlib plots (Betti function, persistence summary, 2D cloud)

```bash
# plots are ready out of the box (matplotlib is declared in plot.py's header)
uv run tools/plot.py --points mydata.csv --value-cols x y --out-dir plots/

# cosine metric, custom title
uv run tools/plot.py --points torus2.csv --metric cosine --title "2-torus" --out-dir plots/

# drop dim-0 noise and short-lived bars; show more top features per dimension
uv run tools/plot.py --points mydata.csv --min-dim 1 --min-persistence 0.08 --max-bars 16 --out-dir plots/
```

 Produces `plots/betti.png`, `plots/barcode.png`, `plots/cloud.png`. `barcode.png` is a
**two-panel persistence summary**: a compact barcode (dims ≥ `--min-dim`, sorted by
persistence, noise filtered by `--min-persistence`, at most `--max-bars` per dim) on the
left, and the full birth–death **persistence diagram** on the right. This keeps the figure
legible even for high-dim clouds that yield hundreds of short-lived dim-0 components.

### `interactive.py` — a live, draggable ε-slider in your browser  ⭐

Generates a **single self-contained `interactive.html`** (no internet, no build step) you
open in any browser. Drag the **ε slider** (or press **▶ Play**) and watch the
Vietoris–Rips complex grow — points → edges → faces — while the Betti numbers
**β₀ (H₀ components), β₁ (H₁ loops/holes), β₂ (H₂ voids)** update live next to a
persistence diagram and the Betti curve. **Drag the 3D view to rotate.**

```bash
# the exact 2-torus T² (target β = [1,2,1]) as a rotatable 3D donut
uv run tools/interactive.py --out interactive.html

# a circle (β₁ = 1), a blob cloud, or the 2-torus as a point cloud
uv run tools/interactive.py --shape circle --n 40 --out circle.html
uv run tools/interactive.py --shape blobs --out blobs.html
uv run tools/interactive.py --shape product --k 2 --nper 10 --out product.html

# your own data, any dimension:
uv run tools/interactive.py --points mydata.csv --out my.html
uv run tools/interactive.py --points mydata.csv --metric cosine --max-dim 2 --out my.html
```

**How it handles different dimensionalities** (the whole point of "really seeing" the
filtration, not just data points):
- **2D / 3D** clouds are drawn directly (3D is rotatable).
- **Higher-dim (≥4D)** clouds are displayed as a **3D PCA projection** — but the Rips
  complex and the Betti numbers are still computed in the *original* D-dim space, so the
  topology readout stays exact; the view just shows where the mass lives.
- **`torus-grid`** (default) uses the *exact* torus cell complex (not a noisy Rips cloud),
  so the torus settles cleanly on **β = [1, 2, 1]**: β₀ collapses to 1 as the surface
  connects, β₁ rises to 2 (the two loops), β₂ settles to 1 (the void).
- **`donut`** is the same exact T² complex shown as a 3D bagel — the reliable way to
  *see* a clean torus. **`donut-rips`** is the honest Vietoris–Rips version over the full
  ε range, kept to demonstrate Rips in action (and why it can't resolve a dense bagel).

```bash
# the clean bagel (exact T^2, instant, β = [1,2,1])
uv run tools/interactive.py --shape donut --nper 24 --out donut.html
# honest full-range Rips on a bagel (capped at 8x8 so it stays feasible)
uv run tools/interactive.py --shape donut-rips --nper 8 --out donut-rips.html
```

> **⚠️ Why a dense bagel point cloud never reads a clean torus under Rips**
> (reproduced: `make_torus --kind donut --nper 64 --grid` → 1536-pt bagel →
> `--points that.csv` reads **β₁ = 25**, **β₂ ≈ 62,064**, not the true **[1, 2, 1]**).
> Vietoris–Rips on a *dense* 2-manifold keeps far more triangles per vertex than a clean
> triangulation (~1.5), so its 2-skeleton **fills the torus void** and shreds β₁ into the
> grid's many short loops. There is no ε at which a dense Rips bagel reads β₁ = 2 — the
> surface-completion ε is in the infeasible (>~300k simplex) range.
>
> **Two distinct failure modes** of `--points <dense bagel>.csv` (both are Rips artifacts, not the true topology):
> 1. **Wrong Betti numbers** — β₁ stuck at the grid's loop count, β₂ explodes (over-filling).
> 2. **Incomplete 3D view** — the outer bulge is missing, because the outer surface's
>    triangles form at a larger ε than the inner surface, and the slider is capped below that.
>
> **Mitigations / safeguards (kept in sync across the codebase):**
> 1. **To *see* a clean, complete torus**, use `--shape donut` (exact T² cell complex) —
>    instant, β = [1,2,1], slider runs to the full surface. *This is the reliable path.*
> 2. **To watch Rips** on a bagel, use `--shape donut-rips` (capped at `nper ≤ 8`).
> 3. **`--points` auto-raises the slider** past the connectivity ε up to the largest
>    *feasible* ε (so the view is as complete as the browser/homology can handle) and prints
>    a yellow **over-filling NOTE** (triangles/vertex ≫ 1.5).
> 4. **`--eps-max <v>`** lets you push the slider higher toward the surface-completion scale;
>    it is capped at the feasibility wall (reported, never a crash).
>
> ```bash
> uv run tools/interactive.py --points bagel.csv --eps-max 0.28   # push toward the outer surface
> ```
>
> Code guards: the `IMPORTANT` block at the top of `tools/interactive.py`, `build_rips` in
> `vrtda/complexes.py`, `betti_function` in `vrtda/persistence.py`, the generation-time
> warning in `make_torus.py`, and the regression tests in
> `tests/test_interactive_display.py` (`test_donut_rips_full_range_overfills`,
> `test_points_dense_bagel_detects_and_auto_raises`, `test_max_feasible_eps_caps_at_wall`,
> `test_overfill_note_message`, `test_make_torus_dense_donut_warns`).

### `methods.py` — the selectable attractor-method suite  ⭐

One command to run **any** of the six analysis methods; every parameter is optional.
`--list` shows what's available.

```bash
uv run tools/methods.py --list                 # show the methods

# persistence metrics (entropy / landscape / image) for a barcode
uv run tools/methods.py metrics --a-layer 16 --dim 1 --plot out/metrics

# bottleneck + p-Wasserstein between two barcodes (top-k salient points)
uv run --with scipy tools/methods.py distance --a-layer 16 --b-layer 32 --top-k 15

# cross-LAYER attractors: profile + (scale x depth) heatmap + loop chains + stable core
uv run tools/methods.py depth --layers 16 17 18 19 20 --eps-cap-frac 2.0 --top-k 8
uv run --with matplotlib tools/methods.py depth --layers 16 17 18 --plot out/depth

# 1D Mapper (lens = residual norm by default)
uv run tools/methods.py mapper --layer 16 --n-bins 6 --plot out/mapper

# dynamical attractors: convergence, per-language, flow-SVD, attention-over-depth
uv run tools/methods.py dynamics --which convergence per_language flow attention
```

The `depth` method is the key one for the "attractors live across many layers"
question: because all 81 tokens share a fixed identity across layers, a loop that
reappears in consecutive layers (matched by token-set overlap) is an attractor
persisting in **depth**, reported as a chain with a `(layer_start, layer_end)` span.

---

## Working with matplotlib (plots)

Two ways, both automatic — **you never edit a venv**:

1. **Use a tool that already declares matplotlib** (simplest):
   ```bash
   uv run tools/plot.py --points mydata.csv --out-dir plots/
   ```
2. **Add matplotlib on the fly to any other script** with `uv run --with`:
   ```bash
   uv run --with matplotlib tools/analyze.py --points mydata.csv --out-dir out/ --plots
   ```

`uv` installs matplotlib into the script's cached environment once, then reuses it.
`project.py` works the same way for UMAP (`--with umap-learn`) and t-SNE (`--with scikit-learn`).

---

## Running the bundled transformer data

The repo ships `capital_berlin_multilingual/` — 5120-dim token activations (81 tokens ×
65 layers) of a 32B LLM answering "the capital of Germany" in 12 languages.

```bash
# quick smoke: load layer 0, run a small Rips + betti sweep
uv run tools/load_smoke.py

# full end-to-end: token clouds across layers -> attractor report (examples/report.md)
uv run examples/attractor_analysis.py --layers 0 16 32 48 64 --out examples/report.md

# plots for one layer's cloud
uv run --with matplotlib examples/attractor_analysis.py --layers 0 64 --out examples/report.md --plot-dir examples/plots/
```

Data loaders live in `vrtda/datasets.py` (`load_token_cloud`, `load_layer_points`,
`load_residual_matrix`). Point semantics: (a) the 81-token cloud per layer, and
(b) token×layer "layer-points".

---

## Running the tests

 ```bash
 uv run tools/run_tests.py            # all 170 tests
 uv run tools/run_tests.py -k torus   # only tests matching "torus"
 uv run tools/run_tests.py -k depth   # cross-layer attractor tests
 uv run tools/run_tests.py -k reduction -v
 ```

The suite validates the whole algebra on exact ground truth: abstract complexes
(disk, sphere, solid tetrahedron, T², T³) give exact Betti numbers with
homology = cohomology = barcode, and point clouds (S¹, T², T³) recover the correct
loop counts.

---

## Continuous integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the **full test suite on
every push and pull request**. It installs `uv`, then runs the suite in two
environments:

1. the base environment (`numpy` + `pytest` + `beartype`), and
2. a plotting environment (`+ matplotlib`) so the matplotlib-gated tests run too.

Both must pass for the check to go green. Locally the exact same command is:

```bash
uv run tools/run_tests.py
```

---

## Project structure

```
vrtda/                  # the library
  pointset.py           # PointSet: CSV in/out, dim/row selection, normalize, concat
  metrics.py            # metric registry: euclidean, squared, manhattan, cosine, normalized_euclidean
  distances.py          # pairwise distance matrix (validated: symmetric, finite, 0-diagonal)
  geometry.py           # minimum-enclosing-ball (strict Vietoris)
  complexes.py          # FilteredComplex, build_rips, build_vietoris, make_torus_grid_complex
  persistence.py        # persistent_homology -> Barcode / Interval (GF(2) row-echelon)
  homology.py           # gf2_rank, betti_at, betti_function, euler_characteristic
  cohomology.py         # cohomology_at (cross-checks homology)
  barcodes.py           # Barcode <-> CSV
  generators.py         # synthetic data: circle, (S^1)^k, donut, sphere, blobs, grids
  datasets.py           # loaders for the bundled transformer data
  reduction.py          # PCA (numpy), top-variance feature selection, UMAP/t-SNE (optional)
   attractors.py         # essential / long-lived / total-persistence metrics
   persistence_metrics.py # entropy / landscape / persistence image
   distance.py           # bottleneck + p-Wasserstein between barcodes
   cocycles.py           # extract the concrete 1-cycles (loops) behind H_1 classes
   depth_persistence.py  # cross-LAYER attractors: heatmap, chains, stable core, profile
   mapper.py             # 1D Mapper (lens -> Rips per bin -> betti)
   dynamics.py           # convergence, per-language, flow-SVD, attention-over-depth
   reports.py            # markdown / text report builder
   plotting.py           # optional matplotlib plotting (diagrams, heatmaps, overlay, ...)
   errors.py, debug.py   # error hierarchy, opt-in --debug / VR_DEBUG logging
tools/                  # PEP 723 command-line tools
    analyze.py  make_torus.py  betti_sweep.py  barcodes.py  project.py
    attractors.py  plot.py  interactive.py  methods.py  load_smoke.py  run_tests.py
tests/                  # pytest suite (170 tests)
examples/               # end-to-end example + generated report.md
docs/                   # MATH.md (formal math + refs), PLAN.md (roadmap)
capital_berlin_multilingual/   # bundled data (git-ignored)
```

---

## Key options

- **Metric** (`--metric`): `euclidean` (default), `squared`, `manhattan`, `cosine`,
  `normalized_euclidean`.
- **`--max-dim`**: highest simplex dimension (0=components, 1=loops, 2=cavities, …). Default 2.
- **`--frac`**: the largest ε as a multiple of the mean nearest-neighbour distance
  (default ~1.6). Raise it to see larger-scale topology.
- **`--min-fraction`** (attractors): "long-lived" threshold as a fraction of `eps_max`.
- **`VR_DEBUG=1`** (env var) or `--debug`: verbose, opt-in debug logging (off by default).

---

## Methodology (short)

- **Vietoris–Rips:** a simplex appears when all its edges have distance ≤ ε; its
  filtration value is the simplex *diameter* (max pairwise distance).
- **Strict Vietoris / Čech:** a simplex appears when its points fit in a ball of
  radius `r` (min-enclosing-ball radius ≤ r).
- **Persistent homology:** computed over `GF(2)` by column reduction of the boundary
  maps; each topological feature is an interval `[birth, death)`. *Essential* features
  (never die) are the stable ones; **attractors** = essential + long-lived features.
 - **Betti numbers:** `β_k(ε)` counts features alive at scale ε.
 - **Persistence metrics:** entropy (`-Σpᵢ log pᵢ`), the Cohen–Stein *landscape*
   (tents over sorted persistence values), and Bubenik's *persistence image*.
 - **Diagram distances:** *bottleneck* (min over matchings of the max L∞ move,
   unmatched → diagonal) and *p-Wasserstein* (optimal transport to the diagonal).
 - **Cross-layer (depth) persistence:** the 81 tokens keep a fixed identity across
   layers, so a loop recurring in consecutive layers (matched by token-set overlap)
   is an attractor persisting in *depth*; reported as chains with a layer span, plus
   a (relative-scale × layer) Betti heatmap and a stable core.
 - **Mapper** (Carlsson): lens → overlapping bins → Rips per bin → β₁ per node;
   nodes linked when their overlap is connected.
 - **Dynamical attractors:** answer-token convergence over depth, per-language
   centroid distance, SVD of the centroid trajectory (≈rank-1 flow), self-attention
   on the answer token over depth.

 Full derivations, the MEB construction, and nLab references: **`docs/MATH.md`**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown metric 'x'` | use one of the 5 metrics above |
| `non-numeric value at …` | your CSV has a non-numeric column — pass `--value-cols` or `--index-cols` |
| `simplices exceeded max_simplices` | lower `--max-dim` or `--frac` (denser/`max_dim` explodes simplex count) |
| `umap-learn / scikit-learn / matplotlib not installed` | prefix with `uv run --with <pkg>` (see "Working with matplotlib") |
| `uv: Failed to parse .../pyproject.toml` | a malformed `pyproject.toml` higher up the tree; run the script *from inside this repo* (all scripts here are PEP 723) |
