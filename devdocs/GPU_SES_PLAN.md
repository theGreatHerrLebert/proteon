# GPU acceleration of the SES mesher — design plan

## Status (2026-06-13)

The SDF/grid path of the SES mesher (`surface/volume.rs`) is being moved onto
the GPU stage by stage; the exact analytic Connolly path stays CPU. Auto-dispatch
behind the `cuda` feature with silent CPU fallback — same contract as the
CHARMM/SASA/OBC kernels.

- **K1 (seed) — SHIPPED (PR #119).** The profiled bottleneck (seed stage, 77–90%
  of `ses_mesh_sdf`) runs on the GPU. `surface/seed_gpu.rs` caches an NVRTC
  `seed_brute` kernel (`OnceLock`); `volume::distance_field` compacts boundary
  nodes and seeds them on-device. **4.5–6.4× over the 16-core CPU seed** on
  crambin (RTX 2070), exact-distance parity. Gated by
  `gpu_seed_matches_cpu_seed_on_boundary_nodes` + area/volume-vs-BALL under
  `--features cuda`. The brute kernel is O(boundary·atoms); a spatial-hash kernel
  is the next perf step for very large receptors.
- **K2 (jump-flood) — SHIPPED (PR #121).** `jfa_kernel.cu` +
  `seed_gpu::jump_flood_gpu` reproduce `volume::jump_flood` exactly (JFA+1
  halving, 27-neighbour nearest-by-squared-distance, ping-pong buffers). Gated by
  `gpu_jump_flood_matches_cpu`. Standalone it's only ~on par with the 16-core CPU
  JFA (memory-bound + a full-grid host↔device round-trip per call) — the win
  needs the K1→K2 fusion, now shipped.
- **K1→K2 fusion — SHIPPED (PR #124).** `seed_gpu::seed_and_flood_gpu` runs seed
  and flood entirely on-device: a `fill_nan` kernel inits the full-grid feature
  buffer, the `seed_scatter` kernel writes each boundary node's nearest-exposed
  point straight at its grid index, and the JFA passes chain on that buffer —
  dropping the boundary-feature download, the host scatter, and the full-grid
  re-upload. `distance_field` calls this path. Gated by
  `gpu_fused_seed_flood_matches_unfused` (bit-identical to the unfused all-GPU
  pipeline). PR #124 also fixed an NVRTC `INFINITY`-undefined bug in
  `jfa_kernel.cu` that had silently disabled the *entire* SES GPU path (init
  compiles both modules) since #121.

### Still open

- **K3 (dual contour)** — still CPU; porting keeps field + contouring on-device.
  Guidance below. This is now the only SES SDF stage off the GPU.
- **Spatial-hash seed kernel** — the seed kernel is still O(boundary·atoms) brute
  force; a spatial hash is the next perf step for very large receptors.
- One-time NVRTC compile (~2 s for both kernels) is `OnceLock`-amortised, so the
  GPU path helps at batch scale (thousands of structures), not single-mesh
  latency.

---

## Goal & thesis

The SES mesher splits into a GPU-hostile half (the exact analytic Connolly graph:
reduced surface → arrangement/DCEL → CDT → weld; irregular, branch-divergent,
pointer-chasing — leave on CPU) and a GPU-ideal half (the **SDF/grid path**,
`surface/volume.rs`, built entirely from GPU-native primitives). We accelerate
**only the SDF path** — already proteon's robust hybrid fallback with 100%
real-protein coverage; its only weakness is accuracy (area ~3.5% low at h=0.4,
<0.5% at h=0.2). GPU gives both speed (matters at corpus scale) and accuracy for
free (finer grids at the same wall-time → <0.35% at h=0.15), potentially making
GPU-SDF the hybrid default. Non-goal: GPU the analytic arrangement/CDT/weld.

## K3 (dual contour) — porting guidance

`manifold_dual_contour(grid, f)` places one vertex per sign-change cell, quads
across sign-change edges, with a 12-way union-find that splits non-manifold cells.
GPU port notes (from the codex design review, still binding):

- **Vertex placement is mean-of-crossings, not QEF.** Port the arithmetic mean of
  edge crossings first; do not introduce QEF (mixing a geometry change with the
  backend port invalidates parity).
- **The manifold split UF is thread-local-safe (≤12 corners/cell); the real risks
  are global** — adjacent cells must make *identical* asymptotic-decider +
  `denom<1e-12` decisions (CPU-f64 vs GPU divergence), the sheet-vertex count
  varies per cell, and the `(cell, local-edge) → sheet-vertex` map must be
  deterministic. Use **≥2 passes** (classify+count sheets → exclusive-scan → emit
  vertices + a dense 12-entry edge→vertex table → emit quads). **No atomics for
  identity** (nondeterministic ordering). `orient_consistently`/flip stay CPU.

## Parity & correctness

The correctness bar is **sign/topology, not area** — a small distance error can
flip `f = dist − probe` at a node and change sign patterns, asymptotic-decider
outcomes, Euler χ, or create/delete a sub-grid neck. Each accelerated stage is
gated by a cross-path parity test against the CPU path on a fixed fixture set
(single atom = 4πr²; two-atom toric; crambin) plus the CPU-only
`ses_area_and_volume_match_ball_and_converge`. Compare mesh topology + geometry
order-independently; bitwise parity is not required.

## Tier 2 (separate, also GPU-ideal): the self-intersection audit

Not production meshing, but the real 300 s hog in validation (the EVIDENT/50K
gating pipeline). All-pairs triangle–segment test with a spatial hash → trivially
GPU-parallel (one thread per broad-phase candidate pair). Independent of the
meshing kernels; high value for corpus-scale gating.

---

_History (decided, kept in git): the full codex design review (verdict: CPU-analytic
/ GPU-SDF split is sound; correctness bar reframed area → sign/topology) and the
2026-06-07 CPU profiling that identified the seed stage (77–90%) as the bottleneck
and drove the K1-first sequencing. See the pre-2026-06-13 revisions of this file._
