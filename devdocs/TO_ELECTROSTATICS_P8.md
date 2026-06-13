# P8 — Fast summation for the BEM matvec (breaking the O(N²) ceiling)

Status: **design, pre-implementation.** This is the one open piece of the NESSie
BEM port (`TO_ELECTROSTATICS.md` §6/P8). Everything below the matvec — kernels,
assembly, GMRES, post-processing, mesh gates, Python/CLI — is shipped and gated.
The remaining ceiling is the dense O(N²) matvec; this plan replaces it with an
approximate **treecode** so a useful bounded protein-scale regime becomes
reachable without abandoning the reference-tier accuracy story.

## 1. The bottleneck, precisely

The local solve's operator is `y = K·x` (double-layer); the nonlocal solve adds
`V`, `Vy`, `Ky` matvecs (`system.rs::NonlocalOperator`). Each dense operator is

```
y_i = Σ_j  M[i][j] · x_j ,   M[i][j] = laplace_collocation(kind, ξ_i, tri_j)
```

with the collocation point `ξ_i = centroid(element_i)` and `tri_j` the source
triangle (`laplace_matrices_cpu`). Building `M` is O(N²) memory; each matvec is
O(N²) time. The matrix-free `LocalOperator`/`NonlocalOperator` already drop the
memory to O(N) by re-deriving entries, but the **time** per matvec stays O(N²).
GMRES does tens of matvecs, so the whole solve is O(iters · N²).

## 2. Far-field reduction (why a particle method applies)

`laplace_collocation` is the **exact Rjasanow analytic integral** of the kernel
over `tri_j`. When `ξ_i` is far from `tri_j` relative to the triangle's size, that
integral converges to the point kernel at the triangle centroid `c_j` times area:

- Single layer:  `V[i][j] → Area_j / |ξ_i − c_j|`
- Double layer:  `K[i][j] → Area_j · (ξ_i − c_j)·n_j / |ξ_i − c_j|³`

So the matvec splits into:

```
y_i =  Σ_{j near i}  M_exact[i][j] · x_j        (NEAR: exact analytic collocation)
     + Σ_{j far  i}  kernel(ξ_i, c_j) · w_j      (FAR: point kernel, w_j folds in Area_j/x_j)
```

The far sum is a Laplace N-body sum. **But the centroid collapse alone is not
enough for a reference tier** — see §3.1.

## 3. Method choice

**Treecode (Barnes–Hut), not FMM, for v1.** Treecode is O(N log N) — enough to
lift the ceiling substantially (the actual reach depends on measured constants,
§3.3) — and needs only *particle/panel→cluster* expansions (no M2M/M2L/L2L). It is
far simpler to implement and to **gate** against the dense matvec. FMM (O(N)) is a
later phase only if benchmarks demand it, and only via a mature library — rolling
our own is disproportionate.

### 3.1 Panel-aware moments (the correctness fix — review catch)

Collapsing each panel to its centroid gives the monopole/dipole far field

```
∫_{T_j} G(ξ,y) dS_y  =  A_j·G(ξ,c_j) + O(A_j · h_j² / r³)
```

whose `O(h_j²/r³)` error is a **fixed floor that interpolation degree p cannot
reduce** — disqualifying for a reference tier. Instead carry **panel-aware
Chebyshev moments**: integrate the Lagrange basis over the panel,

```
single layer:   Q_k = Σ_j  x_j  ∫_{T_j} L_k(y) dS_y          (scalar moment)
double layer:   Q_k = Σ_j  x_j n_j  ∫_{T_j} L_k(y) dS_y       (vector moment)
```

with the panel integrals `∫_{T_j} L_k dS` precomputed accurately (a low-order
triangle cubature of a polynomial — exact to cubature order). A far cluster's
contribution to target ξ is then `Σ_k G(ξ,s_k)·Q_k` (single) /
`Σ_k ∇_y G(ξ,s_k)·Q_k` (double), where `s_k` are the cluster's Chebyshev proxy
points. This preserves finite-panel geometry while keeping barycentric
interpolation, so accuracy is genuinely controlled by `(p,θ)`.

The **dipole** is an explicit data-model decision: interpolate the three
components of `∇_y G` against the vector moment `Q_k` (chosen here), *not* a scalar
"carry n_j into the weight". Sign / normal orientation / derivative accuracy are
tested in isolation against `laplace_collocation(Double, …)` before any wiring.

### 3.2 Barycentric-Lagrange vs Cartesian multipole — bake-off result

BLTC (Wang–Krasny–Tlupova 2020) is kernel-independent and uniform across `V/K/
Vy/Ky`, but pays a `(p+1)³` constant per accepted interaction. Cartesian
multipole uses `C(p+3,3) = (p+1)(p+2)(p+3)/6` total-degree terms — fewer — at the
price of a Coulomb-specific Taylor recurrence (Lindsay–Krasny).

**Bake-off (P8.1, `tests/p8_bakeoff.rs`):** for a `1e-6` far-field accuracy target
across separations `{3,5,8}` and aspect ratios `{1,2,4}`, the minimal expansion
order and resulting term count (= kernel evals per accepted interaction):

| sep | aspect | p_bltc | p_cart | terms BLTC | terms Cartesian |
|----:|-------:|-------:|-------:|-----------:|----------------:|
| 3   | 1      | 4      | 4      | 125        | **35** |
| 5   | 2      | 3      | 3      | 64         | **20** |
| 8   | 2      | 3      | 2      | 64         | **10** |

**Decision: Cartesian for the operator.** It costs ~2–6× fewer terms at matched
accuracy in *every* tested config (even where it needs a higher `p`), and the term
count is the dominant per-interaction traversal cost. BLTC stays in the tree as the
**fallback for the Yukawa regular kernels** (`Vy/Ky`) if the screened-Coulomb
recurrence proves painful — its kernel-independence is the hedge. Both panel-aware
data models (scalar + dipole, BLTC; scalar, Cartesian) are gated in P8.1; the
Cartesian dipole recurrence lands with the operator (P8.2).

### 3.3 Honest complexity

Direct moment rebuild touches every ancestor cluster of each source — one cluster
per level — so it is **O(N·p³·depth)**, not O(N·p³). True linear rebuild needs
upward translations (M2M), which reintroduces FMM machinery; v1 accepts the
`·depth` factor and **measures** it. Per matvec the costs are, separately:
tree build (once per solve, x-independent), **moment rebuild** (per iteration,
O(N·p³·depth)), **traversal** (O(N·log N · p³) with `(p+1)³` kernel evals per
accepted pair), and the **exact near field** (O(N · near-list-size)). No reach
claim (10⁵ etc.) is made until these constants are measured.

## 4. The pieces to build

1. **Octree** whose cluster bounds **enclose triangle vertices** (not just
   centroids): a large / high-aspect panel can reach a target while its centroid
   looks far, so admissibility must see the true panel extent. Subdivide until
   ≤ `n_leaf` panels per leaf; store bounds, center, radius.
2. **Two independent length-scale tests** (review catch — these are *different*,
   neither derived from the other):
   - **Tree MAC** (is a cluster separated enough to expand?):
     `cluster_radius / dist(ξ, center) ≤ θ`.
   - **Rjasanow near-singular** (is a target close enough to an *individual* panel
     to need the exact/near-singular closed form?): the existing element-size
     criterion, unchanged.
   A cluster is expandable only if the MAC holds **and** every represented panel
   passes the panel-separation test `panel_radius / dist(ξ, panel) ≤ η`; otherwise
   recurse; at leaves use **exact** `laplace_collocation` per panel (this is where
   self / near-singular entries stay exact, preserving reference accuracy).
3. **Panel-aware moments** (§3.1): precompute `∫_{T_j} L_k dS` once; refresh the
   `x`-dependent moments each matvec.
4. **`TreecodeOperator: LinearOperator`** — drop-in for `DenseOperator` inside
   `LocalOperator`/`NonlocalOperator`. Octree + proxy geometry + panel integrals
   built once per solve; only the moments refresh per matvec.
5. **Wiring**: `solve_*_auto` selects the treecode past a triangle threshold
   (mirrors the GPU-vs-dense `DENSE_MATRIX_BUDGET` switch); `SurfaceSolveOptions`
   gains an accuracy knob. **No auto-enable** until §5 establishes calibrated
   `(p,θ,η)` presets; default stays exact dense for small meshes.

## 5. Gating (the reference-tier bar)

- **Isolated kernel sweeps** (before any operator): single-panel far-field error
  vs distance, triangle size, **aspect ratio**, and orientation — for both scalar
  (single) and dipole (double), the latter sign-checked against `laplace_collocation`.
- **Matvec approximation tolerance** (not "bit-error"): on real meshes (sphere
  ladder + a protein SES) and **several `x`** — random, constant, localized, and a
  physically representative right-hand side (relative error alone is unstable when
  `‖y‖` is small) — assert `‖y_tree − y_dense‖/‖y_dense‖ ≤ tol(p,θ,η)` for a ladder
  of `(p,θ,η)`, with **monotone** error decrease in `p` and with tightening `θ`.
- **Near/far continuity**: error stays smooth across the MAC boundary; adversarial
  non-uniform / mixed-scale meshes don't blow the near list or the error.
- **GMRES true-dense residual**: convergence judged on the **dense** operator
  residual, not the tree operator's — plus iteration-count regression and
  **charge-density (solution) error**, not energy alone.
- **End-to-end energy**: treecode rfenergy vs dense and vs analytic Born/Xie within
  a stated tolerance; treecode error must not dominate mesh discretization error on
  a refinement ladder (else it silently caps accuracy).
- **Invariance**: rotation, translation, scaling, normal-reversal.
- **Scaling**: per-stage timing/memory (tree build, moment rebuild, traversal,
  exact near field, each nonlocal kernel) vs N — show the cross-over N where the
  treecode beats dense O(N²). Determinism: parallelize over targets, serial
  per-target traversal, fixed reduction order.

### 5.1 Measured scaling (P8.3, `examples/p8_scaling.rs`)

Dense vs treecode double-layer matvec on a sphere ladder (`p=6, θ=0.5`, best of 5):

| N | dense build | dense matvec | tree build | tree matvec | matvec rel-L2 | speedup |
|------:|------------:|-------------:|-----------:|------------:|--------------:|--------:|
| 320   | 5.7 ms   | 0.031 ms | 0.1 ms | 2.8 ms   | 5.0e-4 | 0.01× |
| 1280  | 68 ms    | 0.134 ms | 0.3 ms | 15.8 ms  | 8.6e-4 | 0.01× |
| 5120  | 1019 ms  | 5.36 ms  | 0.7 ms | 69.6 ms  | 2.8e-3 | 0.08× |
| 20480 | (6.7 GiB — over cap) | — | 3.0 ms | 350 ms | — | — |

**Honest read:** the v1 treecode matvec is **slower** than the tight dense O(N²) loop
at these sizes — the direct `O(N·depth·p³)` moment rebuild has a large constant. But:
- the **trend** confirms the asymptotics — tree matvec grows ~linearly (15.8→69.6→350,
  ≈4× per 4×N) while dense grows ~quadratically (0.134→5.36, ≈40× per 4×N once the
  matrix leaves cache), so the speedup climbs `0.01→0.08×`;
- the **realized win today is O(N) memory**: dense `K` is `2·N²·8` B and cannot even be
  built past ~12k triangles (6.7 GiB at 20k), where the treecode runs at 350 ms/matvec;
- dense *build* is already 1 s at 5k and quadratic — the treecode builds in ms.

**Conclusion:** v1 treecode = a *memory* unlock (solve meshes dense can't hold), not a
*time* win. Accuracy is in hand throughout (rel-L2 ≤ 3e-3 at p=6; tighter with higher p).

### 5.2 M2M upward pass — and the measured bottleneck

The **M2M upward pass** (Cartesian translation, gated bit-exact vs the direct rebuild)
makes the moment build linear (`O(N·p³)` leaf cubature + `O(N·p⁴)` translations) instead
of `O(N·depth·p³·cub)`. But instrumenting the matvec shows the rebuild was **not** the
bottleneck:

| N | matvec | moment rebuild | traversal |
|------:|-------:|---------------:|----------:|
| 1280  | 11.8 ms | 1.6 ms (14%) | 10.2 ms |
| 5120  | 58 ms   | 8.0 ms (14%) | 50 ms |
| 20480 | 300 ms  | 37 ms (12%)  | 263 ms |

The matvec is **traversal-bound (~86%)**: per-target far-field Taylor evaluations (a `p³`
recurrence per accepted cluster) plus exact near-field analytic collocations — each pair
far costlier than dense's single FMA. So dense wins on *speed* wherever its `O(N²)` matrix
still fits; the treecode's realized value is and stays **O(N) memory**.

### 5.3 Near vs far cost split (decides whether an FMM can help)

An FMM accelerates only the **far** field, so it is worth building **only if the far
field dominates the traversal cost**. Measured (`examples/p8_scaling.rs`, timing
near-only vs far-only traversal):

| N | rebuild | near | far | **far cost share** |
|------:|--------:|------:|------:|-------:|
| 1280  | 1.6 ms | 5.9 ms  | 4.6 ms  | 44% |
| 5120  | 8.3 ms | 19.6 ms | 36.7 ms | 65% |
| 20480 | 38 ms  | 73 ms   | 215 ms  | **75%** |

Crucially the **count** is near-dominated (near-collocations are 60–84% of the work
items) but the **cost** is **far-dominated and growing** (75% at 20k): each far
cluster-eval recomputes the `O(p³)` Coulomb Taylor recurrence per target-cluster pair,
far costlier than one near collocation. So the matvec is **far-field-cost-bound** — and
that is exactly what an FMM amortizes (one M2L per source/target-cluster pair, shared by
all targets in the cluster, then a cheap L2P per particle), turning the per-target `p³`
recurrence work into per-cluster work. **The measurement justifies the FMM downward
pass** (M2L + L2L + L2P), for which the M2M (§5.2) is the prerequisite. Near-field
exact collocation is untouched (it carries the reference accuracy), so the achievable
speedup is bounded by the far share (~75% at scale).

### 5.4 FMM building blocks landed; the M2L cost wall (measured)

The two new FMM operators are implemented and gated bit-true:
- **M2L** (`cartesian::m2l_single`): source multipole → target **local** expansion,
  `L_m = (−1)^|m| R_t^|m| Σ_k R_s^|k| C(k+m,m) a_{k+m}(D) M̂_k` (Coulomb Taylor coeffs to
  order `2p`). Gate: M2L + L2P reproduces the direct multipole eval to `< 1e-9` — the
  intricate translation landed on the first try.
- **L2P** (`cartesian::eval_local_single`): evaluate the local expansion at a particle.

But the **dense Cartesian M2L is `O(p⁶)`**, and measured it costs **20× / 36× / 64×** a
single treecode far-eval at `p = 4 / 6 / 8`. An FMM amortizes one M2L over the targets in
a cluster, so the break-even cluster size is exactly that ratio — and at `p ≥ 6` the
default `n_leaf = 32` is **below** break-even. This is the classic result: a basic
Cartesian FMM with dense M2L does **not** beat the treecode; a real speedup needs
**accelerated M2L** — FFT-convolution (`O(p³ log p)`), spherical-harmonic rotations
(`O(p³)`), or plane-wave/exponential expansions (`O(p²)`) — each a major build on top of
this. Decision: land M2L/L2P as correct, reusable building blocks (the hard math, done
and gated); defer the full downward pass (L2L + interaction lists) and the M2L
acceleration it needs. No speed claim — the treecode stays the O(N)-memory tool. The
remaining genuine speed lever is accelerated-M2L FMM, scoped here for a funded follow-up.

## 6. Phasing

- **P8.1** Isolated summation harness: octree (vertex-enclosing) + **panel-aware
  scalar AND dipole** expansions, BLTC vs Cartesian bake-off, single-panel error
  sweeps. (A single-layer-only start would not unlock the local solve and risks
  validating the wrong dipole data model.)
- **P8.2** `TreecodeOperator` for `K` (and `V`) → the local solve runs on the
  treecode; matvec-tolerance + true-residual + energy/Born gates.
- **P8.3** Wire into `LocalOperator` + `solve_local_*` threshold + `SurfaceSolve`
  knob (opt-in, **no** auto-enable yet); scaling benchmark; calibrated presets;
  CLI/Python knob.
- **P8.4** Nonlocal `Vy`/`Ky`. **Not** truly optional if large-mesh *nonlocal*
  scaling matters: leaving them dense keeps the nonlocal solve O(N²). Optional only
  if nonlocal-at-scale is explicitly out of scope. FMM only if O(N log N) proves
  insufficient *and* via a mature library.

## 7. Risks / watch

- **Panel-collapse floor** — the headline correctness risk; addressed by §3.1
  panel-aware moments. Guard with the single-panel aspect-ratio sweep.
- **Two length scales conflated** — MAC (cluster separation) ≠ Rjasanow
  near-singular (panel proximity). Kept independent (§4.2); octree bounds enclose
  vertices so panel extent is visible to admissibility.
- **Rebuild cost** — `O(N·p³·depth)` per matvec, every GMRES iteration; measured,
  not assumed. Octree/proxy/panel-integrals built once per solve; only moments
  refresh.
- **Dipole sign/orientation** — invisible until the energy gate; tested in
  isolation first (§3.1).
- **Determinism vs rayon** — parallelize over targets, serial per-target traversal,
  fixed reduction order.
- **Scope honesty (§0)** — stays reference/research tier; GB-style energy-component
  wiring is out of scope until a bounded regime is demonstrated with measured
  constants.
