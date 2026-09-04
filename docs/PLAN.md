# vrtda — Bauplan

Ziel: Ein extrem modularer Python-Stack für Topologische Datenanalyse (TDA) auf
Rohtoken-Embeddings eines Transformer-Modells (hier: `capital_berlin_multilingual`),
um **Attraktoren** / persistente topologische Merkmale zu finden.

- Sprache: Deutsch (Doku/Kommentare), Code ohne Kommentare.
- Toolchain: `uv`, PEP-723-Skripte (kein venv, keine Re-Installation dank Cache).
- Keine GUDHI; eigene Algebra. Nur `numpy` (DR-Phasen ggf. `scikit-learn`/`umap-learn`).

## Architektur (Pakete)

```
vrtda/
  errors.py        Fehlerhierarchie
  debug.py         opt-in VR_DEBUG / --debug (default aus)
  pointset.py      PointSet: CSV, select_dims/rows, normalize, concat, from/to CSV
  metrics.py       Metrik-Registry (euclidean, squared, manhattan, cosine, normalized)
  distances.py     pairwise_distances (gechunkt, Symmetrie/Endlichkeit-Checks)
  geometry.py      min_enclosing_ball(_radius) — MEB-Orakel (Umkreis, affine Hülle)
  complexes.py     FilteredComplex, build_rips, build_vietoris, make_torus_grid_complex
  persistence.py   persistent_homology -> Barcode/Interval
  homology.py      gf2_rank, betti_at, betti_function, euler_characteristic
  cohomology.py    cohomology_at, cohomology_function, Match-Helper
  barcodes.py      Barcode <-> CSV (save/load/summary)
  generators.py    Ground-Truth-Sampler (Kreis, T^k, Donut, Sphere, Blobs, Binads, Grids)
tools/             PEP-723-CLI/Skripte (make_torus, betti_sweep, barcodes, project, run_tests)
tests/             pytest-Suite
docs/              MATH.md, PLAN.md
capital_berlin_multilingual/   Rohdaten
```

## Daten: `capital_berlin_multilingual/`

Modell: `DeepSeek-R1-Distill-Qwen-32B` (Qwen2ForCausalLM), `d_model=5120`,
`n_layers=64`, `n_heads=40`, `vocab=152064`.

- `all_token_streams/layer_XXX.csv` — 65 Dateien (Layer 000..064), je **81 Zeilen**
  (Tokens über 12 multilinguale Prompts), Spalten `prompt_idx, token_pos, token_text,
  dim_0000 .. dim_5119`.
- `final_token_streams/` — Final-Schicht-Streams (12 Prompts).
- `residual_norms/{norms,cosines,deltas}_all.csv` — 81 Zeilen × 65 Layer (Norm/Cosine/
  Delta pro Layer pro Token).
- `attention/*.csv` — 81 Zeilen, 64 Layer × 40 Heads.
- `predictions/*`, `prompts_meta.csv`, `group_info.json`, `model_info.json`.

### Punkt-Semantik (beide, konfigurierbar)
1. **Token-Cloud pro Layer:** die 81 Token-Embeddings (5120-dim) eines Layers als Punkte.
2. **Layer-Punkte (token × layer):** 81·65 = 5265 Punkte, jeder Token in jedem Layer
   (z.B. über `residual_norms`-Projektion oder ausgewählte Dimensionen) — zeigt, wie
   sich die Token-Geometrie über die Tiefe entwickelt.

### Dimensionalität
Rohdimension 5120 ist für Rips zu hoch → **Feature-Selection** (konkrete Dimensionen,
z.B. via Varianz/Komponenten) **und/oder** **DR**:
- `PCA` (deterministisch, `scipy`/numpy eigen)
- `UMAP`, `t-SNE` (nutzerwahl, optional Deps)
Kombinationen: konkrete Dimensionen direkt; konkrete + DR; alle + DR.

## Phasen

### Phase 0 — Scaffold + Smoke ✅
Verzeichnisstruktur, PEP-723 + `sys.path`-Muster verifiziert (`uv run tools/_smoke.py`).

### Phase 1 — Math Core ✅
Libs (siehe Architektur) + pytest-Suite (87 Tests, grün). **Validiert:**
- exakte Betti-Zahlen abstrakter Komplexe (Disk, S², Tetra, T², T³);
- `homology = cohomology = barcode`;
- Punkt-Cloud-Loop-Erkennung (S¹, T², T³ → β₁ = 1,2,3);
- MEB-Orakel + Vietoris-Geschlossenheit (zwei echte Bugs behoben).

**Rest Phase 1 (offen):**
- [ ] Phase-1-CLIs als PEP-723-Tools: `make_torus` (Ground-Truth-CSV), `betti_sweep`
      (β_k(ε) über ε), `barcodes` (Barcode-CSV aus Punkt-Cloud), `project` (DR/Feature-Selection).

### Phase 2 — Daten-Loading ✅
- [x] `vrtda/datasets.py`: `list_layers`, `load_token_cloud` (81×5120/Layer),
      `load_layer_points` (token×layer, Labels `p_t_L{L}`), `load_residual_matrix`
      (norms/cosines/deltas), `token_texts`.
- [x] `PointSet`-Pipeline: Metrik-Wahl, Feature-Selection (`select_dims`), Normalisierung.
- [x] Smoke `tools/load_smoke.py` (echte Daten): Token-Cloud Full-5120d, Feature-Selection,
      Residual-Norm-Trajektorien, layer_points. **Beobachtung:** Residual-Norm-Trajektorien
      zeigen β₀≈12 bei kleinem ε (die 12 multilingualen Prompts als Cluster); β₁ taucht in
      reduzierten Dimensionen auf.
- [x] `tests/test_datasets.py` (9 Tests, mit Skip-Guard).

### Phase 3 — Dimensionalitätsreduktion ✅
- [x] `vrtda/reduction.py`: `pca` (rein numpy via SVD: scores/components/evr/mean),
      `variance_of`, `top_variance_dims`, `reduce`-Dispatch; `umap_2d`/`tsne_2d` optional
      (lazy import, klare Fehlermeldung ohne Dep).
- [x] `tools/project.py`: Punkt-Cloud (CSV oder Dataset) → Feature-Selection
      (`--dims`/`--top-k`) → DR (`--method pca|umap|tsne`) → reduzierte CSV + Text-Report.
- [x] `tests/test_reduction.py` (9 Tests).
- [x] End-to-end: `project` → `betti_sweep` auf echten Daten (token-cloud, residual-norms).
      **Beobachtung:** Residual-Norm-Trajektorien sind ≈ rank-1 (evr[0]≈1); Token-Cloud
      Top-64-Dims → 3 PCs erklären ~58%.

### Phase 4 — Attraktor-Analyse 🕐
- [ ] Persistente, langlebige Features auf den reduzierten Token-Clouds.
- [ ] Vergleich über Layer (Tiefe): wie entstehen/verlassen sich Merkmale?
- [ ] Vergleich über Prompts/Sprachen (multilingual).
- [ ] Metriken: Persistenz-Fläche, Anzahl essentieller Intervalle, ε-Banden.

### Phase 5 — Reports & Beispiel 🕐
- [ ] End-to-End-Beispiel-Skript (`examples/`).
- [ ] Reports (CSV/Text) + optional 2D/3D-Visualisierung der Clouds & Barcodes.

## Laufbefehle

```
uv run tools/run_tests.py            # ganze pytest-Suite
uv run tools/run_tests.py -k torus   # Filter
VR_DEBUG=1 uv run tools/<x>.py       # Debug-Logging
```
