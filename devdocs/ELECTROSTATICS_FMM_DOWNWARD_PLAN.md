# FMM downward pass for the BEM treecode — L2L + interaction lists + traversal

## Motivation & honest scope

The `proteon-electrostatics` treecode (`src/fastsum/`) is today a **Barnes–Hut
M2P** scheme: each target centroid walks the octree independently, and a
well-separated cluster contributes via a direct multipole evaluation
(`eval_single_layer`/`eval_double_layer`, the M2P). The **FMM building blocks
already exist and are bit-exact-gated but unused in production**: `m2l_single`
(`cartesian.rs:361`, multipole→local) and `eval_local_single`
(`cartesian.rs:442`, L2P). What's missing to make the **FMM downward pass** real:

1. **L2L** — local→local downward translation (parent local expansion → child).
2. **Interaction-list builder** — per target node, the well-separated source
   nodes handled at this level (the FMM "V-list").
3. **The dual/single-tree downward traversal** wiring M2L accumulation + L2L
   push-down + L2P leaf evaluation, replacing the per-target M2P walk.

**Honest framing (do not oversell).** `TO_ELECTROSTATICS_P8.md` §0/§5.4 deferred
this deliberately: with the current **dense O(p⁶) M2L**, the FMM does *not* beat
Barnes–Hut M2P on speed — the win needs *accelerated* M2L (FFT/spherical/
plane-wave), a separate larger effort. So this work is an **algorithmic
completion**, not a speed PR: it makes the FMM exist and be correct, exercises
the proven M2L/L2P blocks in production, and is the exact infrastructure any
future accelerated M2L drops into. The plan states this in the code + PR.

## Scope

**In scope**

- `l2l_single(parent_local, s, t0, p) -> Vec<f64>` and `l2l_double` in
  `cartesian.rs`: the downward local translation. `s = R_child/R_parent`,
  `t0 = (c_child − c_parent)/R_parent`. Separable per axis, O(p⁴), using the
  transpose-of-M2M 1-D matrix `T[n][m] = C(m,n)·s^n·t0^{m-n}` (n ≤ m). Bit-exact
  unit test vs a direct re-evaluation (translate a known local, evaluate at
  several points, compare to the parent expansion at the same points).
- An interaction-list / admissibility module: a node pair `(A_target, B_source)`
  is **admissible** (M2L-able) when well-separated by the multipole MAC
  (`(r_A + r_B) ≤ θ·|c_A − c_B|`, the same `θ`), and its parent pair was *not*
  admissible (handled coarser). Near (inadmissible leaf) pairs go to exact P2P.
- The downward traversal in `operator.rs` as a new path (e.g.
  `CollocationTreecode { fmm: true }` or a sibling `FmmTreecode`): one upward
  M2M pass (reuse the existing moment build), then per target node accumulate
  `local += M2L(B)` over its interaction list, `L2L` the inherited parent local
  down, recurse; at leaves `L2P` each centroid + exact `laplace_collocation`
  P2P over the near list. Single tree (targets = source centroids share the
  octree), symmetric.
- Parity gate: the FMM matvec/solve must match the **dense** operator (and Born/
  Kirkwood/Xie analytics) to the *same* tolerance the existing Barnes–Hut
  treecode does — i.e. reuse `p8_local_solve.rs` style gates with the FMM path
  enabled; add an FMM-vs-dense residual test and an FMM-vs-Barnes–Hut
  agreement test (both approximate the same dense matrix, so they agree to ~the
  truncation error, not bit-for-bit).

**Out of scope (deferred, stated)**

- **Accelerated M2L** (FFT O(p³ log p) / spherical-harmonic / plane-wave) — the
  actual speed unlock; this PR keeps dense M2L.
- The M2L scaling-hardening (nondimensional rewrite, `cartesian.rs:351` note) —
  orthogonal; only matters at extreme coordinate scales.
- GPU FMM, multi-region/nonlocal-only FMM tuning beyond making the existing
  `Vy`/`Ky` operators downward-pass-capable if cheap.

## Design notes

### L2L correctness (the one new operator)

A local expansion about `c_parent` in normalized coords `v_p = (t−c_parent)/R_p`
is `Σ_m L^p_m v_p^m`. Substituting `t = c_child + R_c v_c`:
`v_p = t0 + s·v_c` per axis, so
`L^c_n = Σ_{m≥n} L^p_m · Π_axis C(m_a, n_a) t0_a^{m_a−n_a} s^{n_a}`.
This is M2M's separable structure transposed (sum *higher* parent indices into
*lower* child indices). The `local`/L2P path uses total-degree `cidx` indexing,
so L2L is written in `cidx` to match (the moment/M2M path uses the full cube —
keep the two indexings straight; this is the easiest place to introduce a bug).

### Interaction lists (the correctness-critical piece)

The classic FMM V-list: for target node A, the source nodes B at A's level that
are admissible with A but whose parents were inadmissible. Near/inadmissible
leaf pairs fall to P2P. Edge cases: the root (no parent), leaf-vs-internal
asymmetry, and a node admissible with itself's ancestor must not be double
counted (a source's contribution is delivered at exactly one level — either via
M2L into some ancestor of A, then L2L'd down, or via P2P at the leaf). The
single-tree symmetry must not double-count or drop.

### Equivalence gate

FMM and Barnes–Hut both approximate the dense matrix; they will *not* be
bit-identical (different admissibility partitions). The gate is: FMM solve
recovers Born/dense to the existing p8 tolerance, and FMM-vs-dense matvec
residual ≤ the Barnes–Hut residual at the same `(p, θ)` (the FMM, delivering
each far cluster once at the coarsest admissible level, should be *no worse*).

## Test plan

- **L2L unit (bit-exact):** build a local expansion about a parent box, L2L it to
  a child box, and assert `L2P(child_local, t) == L2P(parent_local, t)` for
  points inside the child (the translation is exact for polynomials ≤ p).
  Double-layer variant.
- **Interaction list:** small hand-built 2-level tree; assert each source is
  delivered to each target exactly once (M2L-at-a-level XOR P2P-at-leaf), and the
  union of M2L + near lists covers all sources with no overlap.
- **FMM-vs-dense matvec:** on a sphere mesh, `‖M_fmm x − M_dense x‖ / ‖M_dense x‖`
  ≤ the Barnes–Hut residual at the same `(p, θ)`.
- **FMM-vs-Barnes–Hut solve:** both treecode paths solve to Born within the
  existing p8 tolerance; their energies agree to truncation order.
- **Reuse p8 gates** with the FMM path on (`p8_local_solve`, `p8_nonlocal_solve`
  if cheap), and the NESSie post-parity unchanged (the solve still matches).

## Claudex review outcome (adopted)

1. **Dual-node traversal, NOT a uniform-tree V-list.** The octree is adaptive
   (variable-depth leaves, sparse children) so the "same-level, parent-
   inadmissible" V-list is unsafe across leaf/internal pairs. Use a dual-tree
   recursion over the single shared tree (Greengard–Rokhlin / adaptive-FMM,
   Engblom): start `(target_root, source_root)`; if admissible emit one M2L
   `(A←B)`; if both leaves emit the Cartesian product as P2P; else split the
   **larger** (box/radius) side and recurse on every resulting pair. This
   partitions every (target-leaf, source-leaf) pair **exactly once** — the
   structural guard against drop/double-count. Do **not** exploit `(A,B)=(B,A)`
   symmetry (the collocation + double-layer kernels are asymmetric). M2L
   accumulates into `local[A]`; a separate top-down L2L sweep pushes each node's
   local to its children; leaves do L2P + the P2P near list.
2. **`m2l_double` is the missing operator, not `l2l_double`.** The repo has
   `m2l_single` + scalar `eval_local_single` but no double-layer M2L. After M2L
   both layers carry **scalar** local coefficients, so a single scalar `l2l`
   serves both — I do NOT need `l2l_double`. I DO need `m2l_double`: contract the
   `Vec3` double-layer moments + source derivatives consistently with the
   existing `eval_double_layer`. Load-bearing because `K` (double layer) is in
   the matvec.
3. **L2L math confirmed correct** (`s=R_c/R_p`, `t0=(c_c−c_p)/R_p`, sum `m≥n`, no
   extra sign — polynomial recentering, not kernel differentiation). Indexing
   correction: `cidx` is a **single dense `(p+1)³` address** used by both moments
   and locals; they differ only in which total-degree entries are populated.
   Enforce `|m|≤p`, `|n|≤p`; keep `|·|>p` entries zero. The L2L unit test is
   **not** bit-exact (contraction order) — use a tight relative/ULP tolerance and
   also compare coefficients to a direct multinomial reference.
4. **Same `θ` is convergence-safe but NOT accuracy-equivalent.** The pair MAC
   `(r_s+r_t)≤θ|D|` implies `r_s/(D−r_t)≤θ` so it's at least as conservative as
   the point-target BH MAC — reusing `θ` is fine. But FMM adds target-local
   truncation on top of source-multipole truncation, so do NOT assert FMM error ≤
   BH error. **Calibrate** against dense.
5. **Correctness contract is ABSOLUTE vs dense, not BH-relative.** Gates:
   FMM-vs-dense relative matvec tol over several deterministic random vectors for
   **both V and K**; max/componentwise error (catches localized interaction
   drops); dense true residual for solved systems; analytic Born/Kirkwood
   end-to-end. FMM-vs-BH is a **diagnostic**, not a pass/fail ordering.
6. **Traversal coverage test on (target-centroid, source-panel) pairs**, compared
   to the exhaustive product — not just source IDs per node. Cover unbalanced
   trees, empty octants, degenerate unsplit leaves, self interactions, unequal
   radii, one-side-leaf. Verify interaction-list growth empirically (adaptive
   trees don't auto-bound interactions without balancing).

## Open questions for review

1. **Single-tree vs dual-tree.** Targets are the source centroids, so one octree
   suffices and the matvec is (near-)symmetric. Is the single-tree downward
   traversal with a V-list the right structure, or does the collocation (centroid
   targets, panel sources, non-symmetric near field) force a dual tree? My read:
   single tree, but the near-field P2P uses `laplace_collocation(centroid, panel)`
   which is asymmetric — confirm the interaction-list admissibility (built on box
   geometry) is still valid for the centroid-target / panel-source kernel.
2. **Admissibility constant.** Barnes–Hut uses `r_B ≤ θ·d(ξ, c_B)` per target;
   the FMM V-list uses box-pair `(r_A + r_B) ≤ θ·|c_A − c_B|`. Same `θ`, or does
   the pair criterion need a different constant to preserve the existing accuracy
   gate? (The pair criterion is stricter — likely fine, maybe over-refines.)
3. **Reuse moments or recompute.** The upward M2M pass already builds per-node
   moments each matvec. The downward pass adds per-node local-expansion arrays
   (allocate once, zero per matvec). Confirm the memory model stays O(N).
4. **Worth shipping with dense M2L?** Given the explicit "no speedup" caveat, is
   the algorithmic completion + infrastructure value sufficient to merge now, or
   should it wait behind accelerated M2L? (My recommendation: ship it — it's
   correctness-complete, gated, and unblocks the accelerated follow-up; the PR is
   explicit that it's not yet a speed win.)
