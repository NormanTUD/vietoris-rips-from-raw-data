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
| Plots (Betti function, barcode, 2D cloud) with **matplotlib** | `tools/plot.py` |
| Generate synthetic ground-truth (circle, torus T^k, donut, sphere) | `tools/make_torus.py` |
| Load the bundled transformer data + run TDA | `tools/load_smoke.py`, `examples/attractor_analysis.py` |
| Run the test suite (121 tests) | `tools/run_tests.py` |

Under the hood (`vrtda/` package):
`pointset`, `metrics`, `distances`, `geometry` (min-enclosing-ball), `complexes`
(Rips + Vietoris), `persistence` (GF(2) row-echelon), `homology`, `cohomology`,
`barcodes`, `generators`, `datasets`, `reduction`, `attractors`, `reports`, `plotting`.

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

### `plot.py` — matplotlib plots (Betti function, barcode, 2D cloud)

```bash
# plots are ready out of the box (matplotlib is declared in plot.py's header)
uv run tools/plot.py --points mydata.csv --value-cols x y --out-dir plots/

# cosine metric, custom title
uv run tools/plot.py --points torus2.csv --metric cosine --title "2-torus" --out-dir plots/
```

Produces `plots/betti.png`, `plots/barcode.png`, `plots/cloud.png`.

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
uv run tools/run_tests.py            # all 121 tests
uv run tools/run_tests.py -k torus   # only tests matching "torus"
uv run tools/run_tests.py -k reduction -v
```

The suite validates the whole algebra on exact ground truth: abstract complexes
(disk, sphere, solid tetrahedron, T², T³) give exact Betti numbers with
homology = cohomology = barcode, and point clouds (S¹, T², T³) recover the correct
loop counts.

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
  reports.py            # markdown / text report builder
  plotting.py           # optional matplotlib plotting
  errors.py, debug.py   # error hierarchy, opt-in --debug / VR_DEBUG logging
tools/                  # PEP 723 command-line tools
  analyze.py  make_torus.py  betti_sweep.py  barcodes.py  project.py
  attractors.py  plot.py  load_smoke.py  run_tests.py
tests/                  # pytest suite (121 tests)
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
