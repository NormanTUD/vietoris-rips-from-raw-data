# vrtda — Mathematik

Formale Grundlagen des Pakets. Alle Konstruktoren sind deterministisch und über `GF(2)`
(definiert über dem Körper mit zwei Elementen). Die Persistenz ist **stabil** in dem
Sinne, dass sie von der Wahl des Koeffizientenkörpers unabhängig ist, solange das
Filtrations-Diagramm über einem Körper berechnet wird (wir nutzen `GF(2)`).

Referenzen (nLab):
- Vietoris-Komplex / Vietoris–Rips: https://ncatlab.org/nlab/show/Vietoris+complex
- Čech-Komplex (Vergleich): https://ncatlab.org/nlab/show/Cech+complex
- Simplicial complex: https://ncatlab.org/nlab/show/simplicial+complex
- Simplicial homology: https://ncatlab.org/nlab/show/simplicial+homology
- Persistent homology: https://ncatlab.org/nlab/show/persistent+homology
- Cohomology: https://ncatlab.org/nlab/show/cohomology
- Betti numbers: https://ncatlab.org/nlab/show/Betti+number
- Euler characteristic: https://ncatlab.org/nlab/show/Euler+characteristic

---

## 1. Punktmenge und Metrik

Eine endliche Punktmenge ist `X = {x_1, ..., x_n}` in einem metrischen Raum `(M, d)`.
Wir arbeiten mit Vektoren `x_i ∈ R^d` und Metriken `d` aus einem Registry
(`vrtda.metrics`):

- `euclidean`: `‖a - b‖₂`
- `squared`: `‖a - b‖₂²`
- `manhattan`: `‖a - b‖₁`
- `cosine`: `1 - <â, b̂>` mit `â = a/‖a‖₂`
- `normalized_euclidean`: `‖â - b̂‖₂` (Punkte werden auf die Einheitskugel geprojiziert)

Die Abstandsmatrix `D ∈ R^{n×n}` ist symmetrisch mit Null-Diagonale; `pairwise_distances`
prüft beides explizit.

---

## 2. Vietoris–Rips-Komplex

Für ein Schwellenwert `ε ≥ 0` definieren wir den **Rips-Komplex** `VR_ε(X)`:

- **Ecken:** alle `x_i`.
- **Kante** `{i, j}` genau dann, wenn `d(x_i, x_j) ≤ ε`.
- **Simplex** `[i_0 ... i_k]` genau dann, wenn alle Kanten existieren (das ist ein
  *Klique* in der ε-Nachbarschafts-Adjazenz).

```
VR_ε(X) = { [i_0 ... i_k] : max_{a<b} d(x_{i_a}, x_{i_b}) ≤ ε }
```

Der **Filtrationswert** eines Simplexes ist sein *Durchmesser*:

```
val([i_0 ... i_k]) = max_{a<b} d(x_{i_a}, x_{i_b})
```

Ecken haben Wert `0`. Das ergibt eine **filtrierte** simplicial complex: wenn ein
Simplex erscheint, sind alle seine Facetten bereits vorhanden (Facetten haben kleineren
oder gleichen Durchmesser).

> **Anmerkung (Rips vs. Čech).** Der Čech-Komplex `Cech_ε(X)` enthält genau die
> Simplexe, deren *gemeinsame Umkugel* Radius `≤ ε` hat. Es gilt
> `VR_ε ⊆ Cech_ε ⊆ VR_{2ε}` (1-Lipschitz-Äquivalenz in der Topologie). Wir implementieren
> Rips (Kliken) und — als strenge Variante — den Vietoris/Čech-Typ via Minimum-Enclosing-Ball.

---

## 3. Strikter Vietoris-Komplex (via Minimum Enclosing Ball)

Der **strikte Vietoris-Komplex** (äquivalent zum Čech-Komplex bei Radius `r`) enthält den
Simplex `[i_0 ... i_k]` genau dann, wenn die zugehörigen Punkte in einer Kugel mit
Radius `≤ r` liegen:

```
V_r(X) = { [i_0 ... i_k] : MEB_radius({x_{i_a}}) ≤ r }
```

wobei `MEB_radius` der Radius der **kleinsten einschließenden Kugel** (minimum enclosing
ball, MEB) ist. Für Kanten ist `MEB_radius({i,j}) = d(x_i, x_j)/2`, also
`V_r`-Kanten genau für `d ≤ 2r`. Die Umkehrung zur Adjazenz ist entscheidend:

- `MEB_radius(S) ≤ r  =>  max_{a<b} d(a,b) ≤ 2 r`.

Deshalb verwenden wir für die Kanten-Adjazenz `d ≤ 2r + 2·TOL` und für die Beibehaltung
`MEB ≤ r + TOL` (konsistente Toleranzen), um **Geschlossenheit unter Facetten** zu
garantieren (siehe `complexes.build_vietoris`).

### 3.1 Minimum Enclosing Ball (Kleinstkreis-Problem)

Die MEB einer Punktmenge `P` minimiert

```
min_{c, r}  r   s.t.   ‖c - x‖₂ ≤ r  für alle x ∈ P
```

Satz: Der Zentrum `c*` der MEB ist der **Umkreismittelpunkt** einer Teilmenge `S ⊆ P`
mit `|S| ≤ d+1`, die *affin unabhängig* ist. Wir enumerieren alle affininabhängigen
Teilmenge von Größe `m ∈ [2, min(|P|, d+1)]`, berechnen ihren (eindeutigen in der affinen
Hülle) Umkreiball und behalten den kleinsten, der alle Punkte enthält.

Umkreismittelpunkt für `m` affininabhängige Punkte `t_1,...,t_m`: Löse das
min-Norm-Problem im affinen Unterraum. Setze `base = t_1`, `M_j = t_j - base`
(`j = 2..m`). Das Gleichabstands-Kriterium reduziert sich auf

```
M_j · z = ‖M_j‖²/2      (j = 2..m)
c* = base + z*,   r* = ‖z*‖,   z* = M^T (M M^T)^{-1} b,   b_j = ‖M_j‖²/2
```

Für `m = 2` ergibt sich `z* = M_2/2`, also der **Mittelpunkt** und `r* = ‖M_2‖/2` —
der korrekte MEB-Radius einer Kante. (Eine naive `lstsq`-Min-Norm-Lösung des
unterbestimmten Systems würde hier *falsch* sein; das ist der behobene Bug in
`geometry._circumcenter`.)

---

## 4. Simpliciale Homologie über GF(2)

Sei `K` ein (filtrierter) simplicialer Komplex. Die **Kettengruppen** sind freie
`GF(2)`-Moduln, erzeugt von den `k`-Simplexen:

```
C_k(K) = ⨁_{[σ] ∈ K, dim σ = k} GF(2)·[σ]
```

Die **Randabbildung** `∂_k : C_k → C_{k-1}` ist die GF(2)-Summe der Facetten
(Ordnungs-Vorzeichen fallen über `GF(2)` weg):

```
∂[i_0 ... i_k] = ⊕_{j=0}^{k} [i_0 ... î_j ... i_k]
```

Die **`k`-te Homologie** ist `H_k = ker(∂_k) / im(∂_{k+1})`, und die **`k`-te Betti-Zahl**

```
β_k = dim ker ∂_k − dim im ∂_{k+1}
    = (n_k − rank ∂_k) − rank ∂_{k+1}
```

wobei `n_k` die Anzahl der `k`-Simplexe ist. `betti_at(K, ε)` berechnet dies am
Unterkomplex `{val ≤ ε}` über Matrixrang (`gf2_rank` mit bitmaske, O(nz) Spalten).

- `β_0` = Anzahl Zusammenhangskomponenten.
- `β_1` = Anzahl (unabhängiger) Schleifen/Loops.
- `β_k` für `k ≥ 2` = höhere Löcher.

**Euler-Charakteristik:** `χ = Σ_k (−1)^k n_k = Σ_k (−1)^k β_k`.

---

## 5. Persistent Homology

Sei `K` die Filtration `∅ = K_{t_0} ⊆ K_{t_1} ⊆ ...` (indexiert durch steigende
`val`). Für jedes `k` erhalten wir durch die Randabbildungen eine Kette von
`GF(2)`-linearen Abbildungen und damit für jedes Intervall `i ≤ j` eine Abbildung

```
H_k(K_{t_i}) → H_k(K_{t_j})
```

Die **persistenten Homologie-Klassen** sind Paare `(i, j)` (oder `(i, ∞)` für
*essentielle* Klassen, die nie sterben), die die Lebensdauer einer topologischen
Eigenschaft von `t_i` bis `t_j` kodieren.

### 5.1 Berechnung (Zeilenechelonform, pivots)

Wir reduzieren die (gestückelte) Randmatrix spaltenweise. Für die Spalte `j` (Simplex `j`):

1. Solange die aktuelle Menge `c` (Facetten von `j`, GF(2)-Kombination bereits
   reduzierter Spalten) nicht leer ist: `i = max(c)`. Falls Spalte `i` bereits einen
   Pivot hat (`pivot_col[i] = p`), setze `c = c Δ cols[p]`.
2. Ist `c` nicht leer, ist `i` der **Pivot** von Spalte `j` → persistentes Paar
   `(i, j)` (Geburt `i`, Tod `j`), d.h. `t_i ≤ t_j`.
3. Ist `c` leer, erzeugt Spalte `j` einen **neuen Zyklus** (Geburt `j`); ist `j` später
   als Geburtspart einer anderen Spalte verwendet, wird das Intervall geschlossen,
   sonst ist es essentiell (`∞`).

Dies ist das Standard-Algorithmus aus der Persistent-Homology-Literatur
(EDS, "column" / "persistence pairing"). Die Laufzeit ist `O(nz · α)`, wobei `nz` die
Anzahl Nicht-Nulleinträge (Facettenkanten) ist; wir speichern Spalten als `set`
(dünne Spalten, XOR = symmetrische Differenz).

### 5.2 Barcode und Betti-Funktion

Ein **Barcode** ist die Multimenge aller Intervalle `[t_birth, t_death)` pro Dimension.
Die **Betti-Funktion** an `ε` ist

```
β_k(ε) = |{ Intervalle in Dim k : t_birth ≤ ε < t_death }|
```

`Barcode.betti_at(ε)` zählt genau das. **Essentielle** Intervalle (`t_death = ∞`)
tragen zur *essentialen* Homologie bei — das sind die Merkmale, die für alle größeren
Skalen überleben. Für unsere Torus-Tests gilt: die Anzahl essentieller Intervalle in
Dimension `k` gleicht exakt der Betti-Zahl `β_k` des vollen Komplexes.

---

## 6. Cohomologie

Die **`k`-te Cohomologie** ist `H^k = coker(δ_{k-1}) / ker(δ_k)` mit der
**Ko-Randabbildung** `δ_k : C^k → C^{k+1}`, der Transponierten des Randes. Über `GF(2)`
(gilt der Satz von de Rham / universelle Koeffizienten für Körper) gilt:

```
dim H^k = dim H_k      (H^k ≅ H_k über einem Körper)
```

Wir implementieren `cohomology_at` über die Ko-Randabbildung (verteile jede `(k+1)`-Spalte
auf ihre `k`-Facetten — das sind die *Cofacetten*, nicht die Facetten; ein früherer
Bug vertauschte beides) und verifizieren in Tests `cohomology_at == betti_at` für
Mehrfach-Komplexe.

---

## 7. Validierungsergebnisse

Abstrakte Komplexe (exakte Betti-Zahlen, `betti_at` = `cohomology_at` = Barcode):

| Komplex            | β                          | Status |
|--------------------|----------------------------|--------|
| Disk (Voll-Triangle) | (1, 0, 0)                | OK     |
| S^2 (Tetra-Boundary) | (1, 0, 1)                | OK     |
| Voll-Tetraeder      | (1, 0, 0, 0)               | OK     |
| T^2 Grid 3×3        | (1, 2, 1)                  | OK     |
| T^2 Grid 4×5        | (1, 2, 1)                  | OK     |
| T^3 Grid 3×3×3      | (1, 3, 3, 1)               | OK     |

Punkt-Clouds (robuste Loop-Erkennung, `∃ ε : β_0=1 ∧ β_1=Ziel`):

| Punktmenge                | Ziel β_1 | Status |
|---------------------------|----------|--------|
| Kreis (S^1)               | 1        | OK     |
| T^2 (S^1×S^1)             | 2        | OK     |
| T^3 (S^1)^3               | 3        | OK     |

> **Ehrliche Einschränkung:** Der Rips-Komplex einer *dicht* abgetasteten 2D/3D-
> Oberfläche erzeugt auf mittleren Skalen viel *spurioses* Top-Homologie (viele kurze
> H_2/H_3-Intervalle). Die **essentialen**/langlebigen Features und `β_1` bleiben
> robust; die exakten Top-Betti-Zahlen (`β_k`) validieren wir deshalb über **abstrakte**
> Torus-Grids, nicht über dichte Punkt-Clouds. Das ist Standard-TDA-Verhalten, kein Bug.
