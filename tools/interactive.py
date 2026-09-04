# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "beartype>=0.18"]
# ///
"""Build a self-contained interactive HTML/JS file that shows a TDA filtration.

Drag the epsilon slider (or press Play) and watch the Vietoris-Rips complex grow:
points -> edges -> faces, while the Betti numbers beta_0 / beta_1 / beta_2
(H0 components, H1 loops/holes, H2 voids) update live, next to a live
persistence diagram and the Betti function. Drag the 3D view to rotate.

Lower-dim clouds (2D/3D) are drawn directly; higher-dim clouds are shown as a
3D PCA projection while the topology is still computed in the original D-dim
space (so the Betti numbers stay exact).

Examples:
    # the clean, exact 2-torus (target beta = [1,2,1]), 3D donut you can rotate
    uv run tools/interactive.py --out interactive.html

    # a circle (beta_1 = 1) and a blob cloud
    uv run tools/interactive.py --shape circle --n 40 --out circle.html
    uv run tools/interactive.py --shape blobs --out blobs.html

    # your own data (any dimension; PCA-projected to 3D for display)
    uv run tools/interactive.py --points mydata.csv --value-cols x y z --out my.html
    uv run tools/interactive.py --points mydata.csv --metric cosine --max-dim 2 --out my.html

    # the 2-torus as a point cloud (product of two circles) via Rips
    uv run tools/interactive.py --shape product --k 2 --nper 10 --out product.html
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from vrtda import PointSet, pairwise_distances
from vrtda.complexes import FilteredComplex, build_rips, make_torus_grid_complex
from vrtda.persistence import persistent_homology
from vrtda import generators as G
from vrtda.beartype_guard import beartype_module


def _nn(D: np.ndarray) -> float:
    d = D.copy(); np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def _pca3(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _u, _s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def _to3d(X: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    """Return 3D display coordinates + a projection label."""
    if X.shape[1] == 2:
        return np.column_stack([X, np.zeros(X.shape[0])]), "identity (2D, z=0)"
    if X.shape[1] == 3:
        return X, "identity (3D)"
    return _pca3(X), f"PCA 3D projection (data is {X.shape[1]}D)"


def _torus_surface(nu: int, nv: int, R: float = 1.0, r: float = 0.35) -> np.ndarray:
    """3D donut-surface coordinates for torus-grid vertex index = i + j*nu."""
    idx = np.arange(nu * nv)
    i = idx % nu
    j = idx // nu
    u1 = 2.0 * np.pi * i / nu
    v1 = 2.0 * np.pi * j / nv
    x = (R + r * np.cos(v1)) * np.cos(u1)
    y = (R + r * np.cos(v1)) * np.sin(u1)
    z = r * np.sin(v1)
    return np.column_stack([x, y, z])


def build_source(args: argparse.Namespace) -> tuple[FilteredComplex, np.ndarray, str, list[int] | None]:
    """Return (complex, display_points_3d, projection_label, target_betti_or_None)."""
    if args.shape == "torus-grid":
        nu = args.n
        C = make_torus_grid_complex(2, (nu, nu))
        simps = list(C.simplexes)

        def i_of(k: int) -> int:
            return k % nu

        vals = []
        for s in simps:
            d = len(s) - 1
            if d == 0:
                vals.append(0.0)
            elif d == 1:
                ph = (i_of(s[0]) + i_of(s[1])) / 2.0 / nu
                vals.append(1.0 + 0.8 * ph)
            else:
                ph = (i_of(s[0]) + i_of(s[1]) + i_of(s[2])) / 3.0 / nu
                vals.append(2.0 + 0.8 * ph)
        C = FilteredComplex(
            simps, np.array(vals, dtype=np.float64),
            np.array([len(s) - 1 for s in simps], dtype=np.int64), "torus_grad",
            {"nu": nu, "nv": nu},
        )
        pts = _torus_surface(nu, nu)
        return C, pts, "torus-grid surface (exact T^2 cell complex)", [1, 2, 1]

    # Rips-based sources
    if args.points is not None:
        X = PointSet.from_csv(args.points, value_cols=args.value_cols, index_cols=args.index_cols).data
        src = Path(args.points).stem
    elif args.shape == "circle":
        X = G.circle_grid(args.n, radius=1.0); src = f"circle ({args.n} pts)"
    elif args.shape == "donut":
        X = G.donut_grid(args.nper, args.nper); src = f"donut surface ({args.nper}x{args.nper})"
    elif args.shape == "product":
        X = G.product_torus_grid(args.k, args.nper); src = f"{args.k}-torus cloud ({args.nper}^{args.k})"
    elif args.shape == "sphere":
        X = G.sphere(args.n, dim=args.k - 1, radius=1.0); src = f"S^{args.k-1} ({args.n} pts)"
    elif args.shape == "blobs":
        X = G.gmm(3, args.n, 2); src = "3 gaussian blobs"
    else:
        raise SystemExit(f"unknown shape {args.shape!r}")

    D = pairwise_distances(X, args.metric)
    nn = _nn(D)
    eps_max = args.frac * nn
    C = build_rips(X, D, eps_max, max_dim=args.max_dim)
    pts, proj = _to3d(X, "rips")
    target = [1, 1] if args.shape == "circle" else None
    return C, pts, proj, target


def _round_list(x: list[float], nd: int = 5) -> list[float]:
    return [round(float(v), nd) for v in x]


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    C, pts, proj, target = build_source(args)
    bc = persistent_homology(C)
    md = max(args.max_dim, C.max_dim()) if args.shape != "torus-grid" else C.max_dim()
    eps_max = float(C.values.max())
    n_grid = args.n_grid
    grid = np.linspace(0.0, eps_max, n_grid)
    table = bc.betti_function(grid)
    maxdim = int(table.shape[1] - 1)

    n = len(pts)
    edges = [[int(a), int(b), round(float(C.values[C.index_of((int(a), int(b)))]), 5)]
             for (a, b) in (s for s in C.simplexes if len(s) == 2)]
    faces = [[int(a), int(b), int(c), round(float(C.values[C.index_of((int(a), int(b), int(c)))]), 5)]
             for (a, b, c) in (s for s in C.simplexes if len(s) == 3)]
    intervals = []
    for iv in bc.intervals:
        death = eps_max if not np.isfinite(iv.death) else float(iv.death)
        intervals.append([int(iv.dim), round(float(iv.birth), 5), round(death, 5)])

    title = args.title or C.kind
    ndim = int(pts.shape[1])
    extra = ""
    if "PCA" in proj:
        extra = f" · topology computed in the original high-dim space"
    sub = (f"{proj}{extra}  ·  {n} points  ·  {len(edges)} edges / {len(faces)} faces  ·  "
           f"ε_max = {eps_max:.3f}" + (f"  ·  target β = {target}" if target else ""))

    return {
        "title": title,
        "sub": sub,
        "metric": args.metric,
        "eps_max": round(eps_max, 5),
        "projection": proj,
        "target": target,
        "maxdim": maxdim,
        "points": [[round(float(v), 5) for v in row] for row in pts],
        "edges": edges,
        "faces": faces,
        "betti": {"grid": _round_list(list(grid)), "table": table.tolist(), "maxdim": maxdim},
        "intervals": intervals,
    }


def render_html(data: dict[str, object]) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data))


# The HTML/JS is a plain (non-f) template; only the __DATA__ token is substituted.
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#2b3340; --txt:#e6edf3; --mut:#8b98a9; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: var(--bg); color: var(--txt); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         display: flex; flex-direction: column; overflow: hidden; }
  #top { padding: 10px 16px 8px; border-bottom: 1px solid var(--line); }
  #top h1 { font-size: 17px; margin: 0 0 3px; font-weight: 650; }
  #top #sub { font-size: 12px; color: var(--mut); }
  #main { flex: 1; display: flex; min-height: 0; }
  #left { flex: 1.25; position: relative; min-width: 0; }
  #scene { width: 100%; height: 100%; display: block; cursor: grab; }
  #scene:active { cursor: grabbing; }
  #hint { position: absolute; left: 12px; bottom: 10px; font-size: 11px; color: var(--mut); pointer-events: none; }
  #right { flex: 1; min-width: 340px; max-width: 460px; border-left: 1px solid var(--line);
           display: flex; flex-direction: column; overflow-y: auto; }
  #cards { display: flex; gap: 8px; padding: 12px; }
  .card { flex: 1; background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 10px 8px; text-align: center; transition: border-color .2s, box-shadow .2s; }
  .card.match { border-color: #2ea043; box-shadow: 0 0 0 1px #2ea043 inset; }
  .card .lbl { font-size: 11.5px; color: var(--mut); line-height: 1.25; }
  .card .num { font-size: 34px; font-weight: 750; line-height: 1.1; margin-top: 4px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
  .panel { padding: 8px 14px 12px; border-bottom: 1px solid var(--line); }
  .panel h3 { margin: 6px 0 4px; font-size: 12.5px; font-weight: 650; color: #cdd6e0; letter-spacing:.2px; }
  .panel canvas { width: 100%; display: block; }
  #controls { border-top: 1px solid var(--line); padding: 10px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  button { background:#1f2937; color: var(--txt); border: 1px solid #3a4553; border-radius: 8px;
           padding: 7px 14px; cursor: pointer; font-size: 13px; }
  button:hover { background:#283241; }
  input[type=range] { flex: 1; min-width: 160px; accent-color: #4ea1ff; }
  #eps-readout { font-variant-numeric: tabular-nums; font-size: 13px; color: var(--mut); min-width: 118px; }
  .toggle { display:flex; align-items:center; gap:5px; font-size: 12.5px; color: var(--mut); cursor:pointer; user-select:none; }
  #badge { font-size: 12px; color:#2ea043; font-weight:650; }
</style>
</head>
<body>
  <div id="top">
    <h1 id="title"></h1>
    <div id="sub"></div>
  </div>
  <div id="main">
    <div id="left">
      <canvas id="scene"></canvas>
      <div id="hint">drag to rotate · slider / ▶ Play to step the filtration ε</div>
    </div>
    <div id="right">
      <div id="cards"></div>
      <div class="panel">
        <h3>Betti numbers over ε &nbsp;<span id="badge"></span></h3>
        <canvas id="bfun" height="150"></canvas>
      </div>
      <div class="panel">
        <h3>Persistence diagram (birth → death)</h3>
        <canvas id="diag" height="230"></canvas>
      </div>
      <div class="panel">
        <div class="toggle"><input type="checkbox" id="t-points" checked><label for="t-points">Points</label></div>
        <div class="toggle"><input type="checkbox" id="t-edges" checked><label for="t-edges">Edges (H¹)</label></div>
        <div class="toggle"><input type="checkbox" id="t-faces" checked><label for="t-faces">Faces (fill, H²)</label></div>
      </div>
    </div>
  </div>
  <div id="controls">
    <button id="play">▶ Play</button>
    <button id="reset">Reset</button>
    <button id="resetview">Reset view</button>
    <input type="range" id="slider" min="0" max="1000" value="0">
    <div id="eps-readout"></div>
  </div>

<script>
const DATA = __DATA__;
const DIM_NAME = {0:"H₀ components", 1:"H₁ loops / holes", 2:"H₂ voids", 3:"H₃", 4:"H₄"};
const DIM_COLOR = {0:"#4ea1ff", 1:"#3fd07a", 2:"#ff9f45", 3:"#e060c0", 4:"#c9d16a"};
const EMAX = DATA.eps_max;
const MD = DATA.maxdim;
const P = DATA.points, E = DATA.edges, F = DATA.faces, IV = DATA.intervals;
const GRID = DATA.betti.grid, TABLE = DATA.betti.table;

let rx = -0.45, ry = 0.7, eps = 0, playing = false, raf = null, lastT = 0;
let showPoints = true, showEdges = true, showFaces = true;
let fitR = 1.0;

const scene = document.getElementById("scene"), sctx = scene.getContext("2d");
const bfun  = document.getElementById("bfun"),  bctx = bfun.getContext("2d");
const diag  = document.getElementById("diag"),  dctx = diag.getContext("2d");

document.getElementById("title").textContent = DATA.title;
document.getElementById("sub").textContent = DATA.sub;

// ---- Betti cards ---------------------------------------------------------
const cardsEl = document.getElementById("cards");
const cardNums = [];
for (let d = 0; d <= MD; d++) {
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="lbl"><span class="dot" style="background:${DIM_COLOR[d]}"></span>${DIM_NAME[d] || ("H"+d)}<br>β<sub>${d}</sub></div><div class="num" id="b${d}">0</div>`;
  cardsEl.appendChild(el);
  cardNums.push(document.getElementById("b" + d));
}

// ---- canvas sizing -------------------------------------------------------
function fit(c){
  const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  c.width = Math.max(2, Math.round(r.width * dpr));
  c.height = Math.max(2, Math.round(r.height * dpr));
  return dpr;
}
function fitAll(){ fit(scene); fit(bfun); fit(diag); }
window.addEventListener("resize", () => { fitAll(); render(); });

// ---- 3D -> 2D projection -------------------------------------------------
function project(p){
  let x=p[0], y=p[1], z=p[2];
  let cy=Math.cos(ry), sy=Math.sin(ry);
  let x1=x*cy+z*sy, z1=-x*sy+z*cy;
  let cx=Math.cos(rx), sx=Math.sin(rx);
  let y1=y*cx-z1*sx, z2=y*sx+z1*cx;
  return [x1, y1, z2];
}
function computeFit(){
  let mx=0;
  for (const p of P){ const q=project(p); const r=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]; if(r>mx) mx=r; }
  fitR = Math.sqrt(mx) || 1;
}
function toScreen(q, w, h){
  const s = Math.min(w, h) * 0.40 / fitR;
  return [ w/2 + q[0]*s, h/2 - q[1]*s, q[2] ];
}

// ---- color by value ------------------------------------------------------
const STOPS = [[0,[30,80,220]],[0.25,[20,180,220]],[0.5,[45,200,95]],[0.75,[240,210,45]],[1,[232,64,64]]];
function colormap(t){
  t = Math.max(0, Math.min(1, t));
  for (let i=0;i<STOPS.length-1;i++){
    if (t <= STOPS[i+1][0]){
      const f=(t-STOPS[i][0])/(STOPS[i+1][0]-STOPS[i][0]);
      const a=STOPS[i][1], b=STOPS[i+1][1];
      return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
    }
  }
  return STOPS[STOPS.length-1][1];
}
const rgba=(c,a)=>`rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a==null?1:a})`;

// ---- betti lookup --------------------------------------------------------
function bettiAt(e){
  let idx = Math.round(e/EMAX*(GRID.length-1));
  idx = Math.max(0, Math.min(GRID.length-1, idx));
  return TABLE[idx];
}

// ---- scene render --------------------------------------------------------
function renderScene(){
  const dpr = window.devicePixelRatio || 1;
  const w = scene.width/dpr, h = scene.height/dpr;
  sctx.save(); sctx.scale(dpr, dpr);
  sctx.clearRect(0,0,w,h);
  const proj = P.map(project);
  const scr = proj.map(q => toScreen(q, w, h));

  // faces (triangles) with value <= eps, far first
  if (showFaces){
    const act = [];
    for (const t of F){ if (t[3] <= eps) act.push(t); }
    act.sort((a,b)=>{
      const za=(proj[a[0]][2]+proj[a[1]][2]+proj[a[2]][2])/3;
      const zb=(proj[b[0]][2]+proj[b[1]][2]+proj[b[2]][2])/3;
      return za-zb;
    });
    for (const t of act){
      sctx.beginPath();
      sctx.moveTo(scr[t[0]][0], scr[t[0]][1]);
      sctx.lineTo(scr[t[1]][0], scr[t[1]][1]);
      sctx.lineTo(scr[t[2]][0], scr[t[2]][1]);
      sctx.closePath();
      sctx.fillStyle = rgba(colormap(t[3]/EMAX), 0.16);
      sctx.fill();
    }
  }
  // edges with value <= eps
  if (showEdges){
    sctx.lineWidth = 1.2;
    for (const e of E){
      if (e[2] > eps) continue;
      sctx.strokeStyle = rgba(colormap(e[2]/EMAX), 0.9);
      sctx.beginPath();
      sctx.moveTo(scr[e[0]][0], scr[e[0]][1]);
      sctx.lineTo(scr[e[1]][0], scr[e[1]][1]);
      sctx.stroke();
    }
  }
  // points
  if (showPoints){
    sctx.fillStyle = "#e6edf3";
    for (let i=0;i<P.length;i++){
      sctx.beginPath();
      sctx.arc(scr[i][0], scr[i][1], 2.0, 0, Math.PI*2);
      sctx.fill();
    }
  }
  sctx.restore();
}

// ---- persistence diagram -------------------------------------------------
function renderDiagram(){
  const dpr = window.devicePixelRatio || 1;
  const w = diag.width/dpr, h = diag.height/dpr;
  dctx.save(); dctx.scale(dpr, dpr);
  dctx.clearRect(0,0,w,h);
  const m = 26, pw = w-m-8, ph = h-m-8;
  const X = v => m + (v/EMAX)*pw, Y = v => h-m - (v/EMAX)*ph;
  // axes + diagonal
  dctx.strokeStyle = "#3a4553"; dctx.lineWidth = 1;
  dctx.strokeRect(m, 8, pw, ph);
  dctx.strokeStyle = "#5a6675"; dctx.setLineDash([4,4]);
  dctx.beginPath(); dctx.moveTo(X(0),Y(0)); dctx.lineTo(X(EMAX),Y(EMAX)); dctx.stroke(); dctx.setLineDash([]);
  // crosshair at eps
  dctx.strokeStyle = "#e6edf3"; dctx.globalAlpha=0.5; dctx.lineWidth=1;
  dctx.beginPath(); dctx.moveTo(X(eps),8); dctx.lineTo(X(eps),h-m); dctx.stroke();
  dctx.beginPath(); dctx.moveTo(m,Y(eps)); dctx.lineTo(w-8,Y(eps)); dctx.stroke(); dctx.globalAlpha=1;
  // points
  for (const iv of IV){
    const alive = iv[1] <= eps && eps < iv[2];
    const dead = eps >= iv[2];
    dctx.globalAlpha = alive ? 1 : (dead ? 0.18 : 0.4);
    dctx.fillStyle = DIM_COLOR[iv[0]];
    dctx.beginPath(); dctx.arc(X(iv[1]), Y(iv[2]), alive?3.4:2.4, 0, Math.PI*2); dctx.fill();
  }
  dctx.globalAlpha=1;
  // labels
  dctx.fillStyle = "#8b98a9"; dctx.font = "10px system-ui";
  dctx.fillText("birth →", m, h-4);
  dctx.save(); dctx.translate(9, m+ph/2); dctx.rotate(-Math.PI/2); dctx.fillText("death ↑", 0, 0); dctx.restore();
  dctx.restore();
}

// ---- betti function ------------------------------------------------------
function renderBfun(){
  const dpr = window.devicePixelRatio || 1;
  const w = bfun.width/dpr, h = bfun.height/dpr;
  bctx.save(); bctx.scale(dpr, dpr);
  bctx.clearRect(0,0,w,h);
  const m = 22, pw = w-m-8, ph = h-14-8;
  let maxB = 1; for (const row of TABLE) for (const v of row) if (v>maxB) maxB=v;
  const X = i => m + (i/(GRID.length-1))*pw, Y = v => 8 + ph - (v/maxB)*ph;
  bctx.strokeStyle="#3a4553"; bctx.strokeRect(m,8,pw,ph);
  for (let d=0; d<=MD; d++){
    bctx.strokeStyle = DIM_COLOR[d]; bctx.lineWidth=1.6; bctx.beginPath();
    for (let i=0;i<GRID.length;i++){
      const x=X(i), y=Y(TABLE[i][d]);
      if (i===0) bctx.moveTo(x,y); else bctx.lineTo(x,y);
    }
    bctx.stroke();
  }
  // cursor
  const cx = m + (eps/EMAX)*pw;
  bctx.strokeStyle="#e6edf3"; bctx.globalAlpha=0.6; bctx.beginPath();
  bctx.moveTo(cx,8); bctx.lineTo(cx,8+ph); bctx.stroke(); bctx.globalAlpha=1;
  // legend
  let lx = m+6;
  for (let d=0; d<=MD; d++){
    bctx.fillStyle = DIM_COLOR[d]; bctx.fillRect(lx, h-6, 9, 3);
    bctx.fillStyle="#8b98a9"; bctx.font="10px system-ui";
    bctx.fillText("β"+d, lx+11, h-1); lx += 34;
  }
  bctx.restore();
}

// ---- Betti cards update --------------------------------------------------
function updateCards(){
  const b = bettiAt(eps);
  let match = true;
  for (let d=0; d<=MD; d++){
    cardNums[d].textContent = b[d];
    cardNums[d].style.color = DIM_COLOR[d];
    if (DATA.target && DATA.target[d] !== undefined && b[d] !== DATA.target[d]) match = false;
  }
  const badge = document.getElementById("badge");
  if (DATA.target){
    const m = b.slice(0, DATA.target.length).every((v,i)=>v===DATA.target[i]);
    badge.textContent = m ? "✓ matches expected topology" : "";
    for (let d=0; d<=MD; d++) document.querySelectorAll(".card")[d].classList.toggle("match", m);
  } else badge.textContent = "";
}

function render(){
  computeFit();
  renderScene();
  renderDiagram();
  renderBfun();
  updateCards();
  document.getElementById("eps-readout").textContent = `ε = ${eps.toFixed(3)}  /  ${EMAX.toFixed(3)}`;
}

// ---- controls ------------------------------------------------------------
const slider = document.getElementById("slider");
slider.addEventListener("input", () => { stopPlay(); eps = slider.value/1000*EMAX; render(); });
function setSlider(){ slider.value = Math.round(eps/EMAX*1000); }

const playBtn = document.getElementById("play");
function stopPlay(){ playing=false; if(raf) cancelAnimationFrame(raf); playBtn.textContent="▶ Play"; }
function tick(t){
  if(!playing) return;
  if(!lastT) lastT=t;
  const dt=(t-lastT)/1000; lastT=t;
  eps += dt * (EMAX/8);            // full sweep in ~8 s
  if (eps >= EMAX){ eps = EMAX; stopPlay(); }
  setSlider(); render();
  if (playing) raf = requestAnimationFrame(tick);
}
playBtn.addEventListener("click", () => {
  if (playing){ stopPlay(); return; }
  if (eps >= EMAX) eps = 0;
  playing=true; lastT=0; playBtn.textContent="⏸ Pause";
  raf = requestAnimationFrame(tick);
});
document.getElementById("reset").addEventListener("click", () => { stopPlay(); eps=0; setSlider(); render(); });
document.getElementById("resetview").addEventListener("click", () => { rx=-0.45; ry=0.7; render(); });

document.getElementById("t-points").addEventListener("change", e=>{showPoints=e.target.checked; render();});
document.getElementById("t-edges").addEventListener("change", e=>{showEdges=e.target.checked; render();});
document.getElementById("t-faces").addEventListener("change", e=>{showFaces=e.target.checked; render();});

// drag to rotate
let dragging=false, lx=0, ly=0;
scene.addEventListener("mousedown", e=>{dragging=true; lx=e.clientX; ly=e.clientY;});
window.addEventListener("mouseup", ()=>dragging=false);
window.addEventListener("mousemove", e=>{
  if(!dragging) return;
  ry += (e.clientX-lx)*0.01;
  rx += (e.clientY-ly)*0.01;
  rx = Math.max(-1.5, Math.min(1.5, rx));
  lx=e.clientX; ly=e.clientY;
  render();
});
// touch
scene.addEventListener("touchstart", e=>{const t=e.touches[0];dragging=true;lx=t.clientX;ly=t.clientY;},{passive:true});
scene.addEventListener("touchend", ()=>dragging=false);
scene.addEventListener("touchmove", e=>{const t=e.touches[0];ry+=(t.clientX-lx)*0.01;rx+=(t.clientY-ly)*0.01;
  rx=Math.max(-1.5,Math.min(1.5,rx)); lx=t.clientX; ly=t.clientY; render();},{passive:true});

// ---- go ------------------------------------------------------------------
fitAll();
setSlider();
render();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shape", default="torus-grid",
                   choices=["torus-grid", "circle", "donut", "product", "sphere", "blobs"],
                   help="synthetic source (default: torus-grid = exact T^2)")
    p.add_argument("--points", default=None, help="point-cloud CSV (overrides --shape)")
    p.add_argument("--value-cols", nargs="*", default=None)
    p.add_argument("--index-cols", nargs="*", default=None)
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "squared", "manhattan", "cosine", "normalized_euclidean"])
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--n", type=int, default=8, help="torus-grid per-axis cells / circle points / sphere points")
    p.add_argument("--nper", type=int, default=10, help="points per circle (donut/product grids)")
    p.add_argument("--k", type=int, default=2, help="ambient dim for product/sphere")
    p.add_argument("--frac", type=float, default=1.6, help="Rips: eps_max as a fraction of mean nearest-neighbour distance")
    p.add_argument("--n-grid", type=int, default=140, help="epsilon grid resolution for Betti curve / slider")
    p.add_argument("--title", default="")
    p.add_argument("--out", default="interactive.html", help="output HTML file")
    args = p.parse_args()

    data = build_payload(args)
    html = render_html(data).replace("__TITLE__", data["title"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_pts = len(data["points"])
    print(f"wrote {out}  ({n_pts} points, {len(data['edges'])} edges, {len(data['faces'])} faces)")
    print(f"   open it in a browser:  file://{out.resolve()}")
    return 0


beartype_module(__name__)


if __name__ == "__main__":
    raise SystemExit(main())
