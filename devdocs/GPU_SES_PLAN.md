# GPU acceleration of the SES mesher — design plan

## Status (2026-06-13)

**K1 (seed) — DONE.** The profiled bottleneck (the seed stage, 77–90% of
`ses_mesh_sdf`) now runs on the GPU. `surface/seed_gpu.rs` caches a
NVRTC-compiled `seed_brute` kernel (`OnceLock`); `volume::distance_field`
compacts the boundary nodes and seeds them on the GPU behind the `cuda`
feature, with silent CPU fallback — the same auto-dispatch contract as the
force-field/SASA/OBC kernels. Measured **4.5–6.4× over the 16-core CPU
seed** on crambin (h=0.4/0.2/0.15), matching the CPU seed's nearest
*distance* (0 feature mismatches, 0.000 Å on crambin — equidistant ties,
which can pick a different but equally-near point, don't occur on real
coordinates) on an RTX 2070; gated by
`gpu_seed_matches_cpu_seed_on_boundary_nodes` plus the area/volume-vs-BALL
test under `--features cuda`. The brute kernel is O(boundary·atoms); a
spatial-hash kernel (codex correction #2) is the next perf step for very
large receptors.

**Still open:** K2 (jump-flood distance transform) and K3 (dual contour)
remain on the CPU — porting them keeps the whole field on-device and
avoids the host↔device seed round-trip.

---

## Goal & thesis

The SES mesher splits into a GPU-hostile half (the exact analytic Connolly graph:
reduced surface → arrangement/DCEL → CDT → weld; irregular, branch-divergent,
pointer-chasing — leave on CPU) and a GPU-ideal half (the **SDF/grid path**,
`surface/volume.rs`, built entirely from GPU-native primitives).

We accelerate **only the SDF path**. It is already proteon's robust hybrid
fallback (`ses_mesh` → `NumericalGrid` when the analytic path can't mesh), with
100% real-protein coverage; its only weakness is accuracy (area ~3.5% low at
h=0.4, converging to <0.5% at h=0.2). GPU gives two wins at once:

1. **Speed** (target 10–50× on the field stage) — matters at *scale* (batch over
   thousands of structures; single-protein latency is already seconds).
2. **Accuracy for free** — finer grids at the same wall-time push the SDF area to
   BALL (<0.35% at h=0.15). So GPU turns the robust-but-coarse path into
   robust + accurate + fast, potentially the new hybrid *default* (the
   `SesMethod` provenance flag is the switch).

Non-goal: GPU the analytic arrangement/CDT/weld. Out of scope; low ROI.

## Current CPU pipeline (what we're porting), `surface/volume.rs`

`ses_mesh_sdf(atoms, probe, spacing)`:
1. `Grid::enclosing` — regular node grid (dims, origin, spacing) padded so the
   field is negative on the boundary (closed iso-surface).
2. `Grid::distance_field(atoms, probe)` → `Vec<f64>` signed distance:
   - `AtomGrid::build` — uniform spatial hash over inflated (r+probe) atoms.
   - per-node: occupancy (inside any inflated atom?) + **seed** boundary-adjacent
     nodes with `AtomGrid::nearest_surface_point` (an *analytic* nearest point on
     an inflated atom — no voxel staircase).
   - `jump_flood` — vector (feature-point) jump-flooding transform: each node's
     nearest seeded surface point propagates over `log2(reach)` passes; signed
     distance = ±|node − nearest_surface|.
3. `manifold_dual_contour(grid, f)` — dual contouring: one vertex per sign-change
   cell (QEF-ish placement), quads across sign-change edges; a 12-way union-find
   (`uf_find`/`uf_union`) splits non-manifold cells so the output is manifold.
4. `orient_consistently` + flip to outward.

## GPU mapping (the three kernels)

All three stages are standard data-parallel GPU primitives; the port mirrors
proteon's existing CUDA kernels (`cuda` feature, cudarc + NVRTC, `gpu/*.cu`,
silent CPU-fallback auto-dispatch — same pattern as CHARMM/SASA/OBC/SW).

- **K1 — occupancy + seed.** One thread per grid node. Spatial-hash the inflated
  atoms into a GPU uniform grid (sorted by cell, prefix-sum bucket offsets — the
  standard particle-grid build); each node loops its 27 neighbour buckets for
  occupancy and the nearest inflated-atom surface point. Output: occupancy bits +
  seed feature-point array. (The `AtomGrid` build is itself a parallel sort +
  scan.)
- **K2 — jump-flooding distance transform.** JFA is a GPU-native algorithm:
  `ceil(log2(maxdim))` ping-pong passes, one thread per node, each sampling 26
  neighbours at offset `step` (halving) and keeping the closest feature point.
  Pure SIMT, coalesced, no divergence. This is the dominant cost and the biggest
  win.
- **K3 — dual contouring.** Per-cell: classify 12 edges by sign change, emit one
  vertex (compacted via prefix-sum/atomic counter), emit quads. The manifold
  union-find split is per-cell-local (≤12 corners) → fits in registers/thread.
  Output vertex + index buffers copied back; `orient_consistently`/flip stay CPU
  (cheap, or a final parallel pass).

Batch mode: a grid per structure; either one kernel launch per structure on a
stream, or a fused batched launch (structures packed with per-structure grid
offsets) for throughput over a corpus.

## Parity & correctness (proteon's standing rule)

Cross-path parity test (the established requirement for every accelerated path):
GPU vs CPU SDF mesh must agree within grid tolerance on a fixed fixture set
(single atom = 4πr²; two-atom toric; crambin) — area/volume within a tight
epsilon and identical watertight/euler topology at a given `spacing`. JFA and
dual-contour are deterministic, so this can be near-exact (bitwise feature points
if the float reductions are ordered identically; otherwise ≤1e-6). Gate it in CI
behind the `cuda` feature like the other kernels, plus a CPU-only correctness
test that already exists (`ses_area_and_volume_match_ball_and_converge`).

## Effort, sequencing, risk

- **Spike first:** port **K2 (JFA)** alone, feeding it the CPU-built seed grid and
  reading back the field for `manifold_dual_contour` on CPU. This isolates the
  dominant-cost kernel, measures the real speedup, and de-risks the cudarc/NVRTC
  plumbing before touching K1/K3. ~1 kernel of effort.
- Then K1 (spatial-hash grid build + seed) and K3 (dual contour) to keep the
  whole field on-device (avoid host↔device grid copies).
- **Risks:** (a) the `nearest_surface_point` analytic seed is the accuracy
  keystone — must port its exact math, not a voxel-snapped approximation, or the
  crease bias returns; (b) memory — a fine grid on a big protein is large
  (dims³·{occupancy,feature×3,dist}); tile/stream if it exceeds VRAM; (c) the
  manifold union-find split must reproduce the CPU topology exactly or parity
  fails. (d) float-reduction ordering for bitwise parity (acceptable to relax to
  ≤1e-6 + identical topology).

## Tier 2 (separate, also GPU-ideal): the self-intersection audit

Not production meshing, but the real 300s hog in validation (the EVIDENT/50K
gating pipeline). All-pairs triangle–segment test with a spatial hash → trivially
GPU-parallel (one thread per candidate triangle pair from the broad phase).
Independent of the meshing kernels; high value for corpus-scale gating.

## Open questions for review

1. Is K1's analytic seed (`nearest_surface_point`) the right thing to keep on the
   *exact* path, or should the GPU seed differently (e.g. a brute exact distance
   for boundary cells, since GPU makes it cheap) to kill the crease bias entirely?
2. JFA is approximate (rare errors vs exact EDT). At SES grid sizes is JFA's error
   below the dual-contour discretization, or should we use JFA+1/JFA+2 (extra
   cleanup passes) or a banded exact transform near the surface?
3. Batched-corpus launch shape: one stream per structure vs one fused kernel over
   packed grids — which wins given heterogeneous protein sizes?
4. Should GPU-SDF (fast+accurate) become the hybrid **default**, with analytic as
   opt-in exact refinement, or stay the fallback?

---

## Codex review — verdict & corrections (APPLIED 2026-06-07)

Verdict: **the CPU-analytic / GPU-SDF split is sound** — the exact Connolly
arrangement/DCEL/CDT/weld are poor CUDA targets, and "GPU contour-buildup" would
retain the hard synchronization/degeneracy/topology problems while accelerating a
stage that won't dominate corpus throughput. Keep exact meshing CPU-side. But
several kernel specifics were wrong (read the actual code), and the correctness
bar must be reframed from *area* to *sign/topology*. Corrections, all adopted:

1. **The correctness bar is sign-flips, not area.** A small distance error can flip
   `f = dist − probe` at a node → change sign patterns, asymptotic-decider
   outcomes, Euler χ, or create/delete a sub-grid neck. **Area parity is
   insufficient.** Build an **exact narrow-band reference** (brute-force / CPU
   KD-tree nearest sampled-feature distance in the surface band) and measure:
   max/percentile distance error, **sign disagreements**, ambiguous-face
   decisions, and resulting topology — for vanilla JFA / JFA+1 / JFA+2. This is
   the spike's real exit gate, not area.

2. **K1 is atom-centric, not node-centric.** The CPU code *rasterizes* each
   inflated sphere over its node bounding box for occupancy (scatter), and seeds
   with `NEAR_REACH = 2` (up to 125 buckets) + nested exposure queries — not a
   27-bucket per-node gather. Port: keep atom-centric occupancy rasterization
   (atomic bit-set / dense byte grid); **compact boundary nodes first**, then run
   the expensive projection/exposure kernel only on them — else K1 dominates.

3. **The seed is analytic but NOT exact.** `nearest_surface_point` is the closest
   point on an *individual* sphere → biased near intersection creases (documented).
   "Brute exact distance" is not just looping spheres: exact union-boundary
   distance needs exposed patches + pair-intersection circles + triple-sphere
   vertices = rebuilding the analytic geometry we're avoiding. So: **keep the
   current seed for v1**, quantify its error vs the narrow-band oracle, add
   exposed pair-circle candidates for boundary nodes only if measurement demands.

4. **K3 is mean-of-crossings, not QEF.** The CPU places each sheet vertex at the
   arithmetic mean of edge crossings. **Port that first**; do not introduce QEF
   (mixes a geometry change with the backend port and invalidates parity).

5. **JFA schedule must match CPU exactly:** it already is **JFA+1** (halving + a
   final unit pass) and floods only a **narrow band** (~`probe/spacing + 4`), NOT
   across `maxdim`. Reproduce that. Don't assume JFA error < discretization;
   choose JFA+1 unless the oracle shows a topology-relevant residual (then a
   narrow-band exact cleanup, not blanket JFA+2).

6. **Manifold split — the UF is thread-local-safe; the real risks are global:**
   adjacent cells must make *identical* asymptotic-decider + `denom<1e-12`
   decisions (CPU-f64 vs GPU divergence), variable sheet-vertex count per cell,
   and a deterministic `(cell, local-edge) → sheet-vertex` map. Use **≥2 passes**
   (classify+count sheets → exclusive-scan → emit vertices + a dense 12-entry
   edge→vertex table → emit quads). **No atomics for identity** (nondeterministic
   ordering). Compare topology+geometry **order-independently**; bitwise mesh
   parity is not a realistic requirement.

7. **Sequencing: profile the CPU stages FIRST.** K1's nested exposed-projection may
   rival or exceed JFA, so confirm K2 is actually the bottleneck before spiking
   it. K2 (JFA) spike with the exit criteria in (1) + timing split into
   kernel / PCIe transfer / end-to-end (the 3-component f64 feature round-trip can
   hide the production speedup). Then K1; **K3 last** (lowest arithmetic
   intensity, highest topology-sensitive complexity).

8. **Open-Q answers:** (1) radial seeds for v1, pair-circles only from measured
   error; (2) JFA+1, escalate only empirically; (3) **size-binned batches + a few
   streams** — not one fused packed launch (wastes work under heterogeneous dims),
   not one-stream-per-structure (scales poorly); (4) **do not default to GPU-SDF
   yet** — first establish accuracy/topology/determinism/memory/fallback on a
   representative corpus; long-term make fine-grid GPU-SDF the robust default and
   keep analytic CPU meshing as an explicitly **exact** method (not "refinement").

**Net:** plan is ready to implement. First action is **CPU profiling** (which of
K1/K2/K3 dominates) — no GPU needed — then the **K2 JFA spike** gated on
sign/topology parity against a narrow-band exact oracle, not area.

---

## CPU profiling results (2026-06-07, crambin 327 atoms, `bin/sdf_prof`)

Per-stage wall time of `ses_mesh_sdf` (single-threaded), `SES_SDF_PROF=1`:

| h | seed | jump_flood | dual_contour | occupancy | nodes |
|------|-----------|------------|------|------|------|
| 0.4  | **3263 ms (90%)** | 339 ms | 18 ms | 3 ms | 0.75 M |
| 0.2  | **13126 ms (80%)** | 2982 ms | 106 ms | 17 ms | 5.4 M |
| 0.15 | **23051 ms (77%)** | 6277 ms | 243 ms | 39 ms | 12.5 M |

**The SEED stage dominates (77–90%), NOT JFA** — codex's "K1 may rival or exceed
JFA" was an understatement (seed is 3–4× JFA). So the GPU/parallel target is
**K1 (`nearest_surface_point`: NEAR_REACH=2 nested exposed-projection per
boundary node)**, not K2. Seed scales ~`1/h²` (surface band), JFA ~`1/h³·log` —
JFA only overtakes at much finer grids than the practical range.

**Key finding — the mesher is fully single-threaded** (no `par_iter` in
`volume.rs`), yet `rayon` is already a `proteon-core` dep and the seed is
per-node-independent over an *immutable* `AtomGrid`. **Revised sequencing:**

1. **CPU rayon parallelization of seed (+ JFA passes) FIRST** — near-zero risk,
   potentially ~Ncores× on a many-core box, and it (a) may make the SDF fast
   enough that GPU is only for extreme scale, and (b) de-risks the GPU port by
   proving the parallel decomposition + giving a fast multi-core baseline to beat.
2. **Then GPU K1 (seed)** — the real bottleneck — gated on the sign/topology
   oracle. K2 (JFA) and K3 (dual-contour) follow only if profiling after K1 shows
   they then dominate.

Tooling: `bin/sdf_prof` (this profiler) + the `SES_SDF_PROF` env timers in
`volume.rs::distance_field`/`ses_mesh_sdf` are kept for measuring the parallel/GPU
speedups against this CPU baseline.
