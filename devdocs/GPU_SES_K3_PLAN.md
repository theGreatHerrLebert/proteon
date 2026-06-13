# GPU SES K3 — manifold dual contouring on-device (design)

Implementation plan for porting `volume.rs::manifold_dual_contour` (the last
SES-SDF stage still on CPU) to CUDA, so the whole field→mesh pipeline stays
on-device. Extends the K3 guidance in `GPU_SES_PLAN.md`.

## Status / decision (2026-06-14)

**Milestone shipped; full K3 deferred.** Per the codex plan review
(`GPU_SES_K3_PLAN.codex-review.md`) and a measurement, we did the cheaper
"compute the field on-device, download `f`" milestone first: GPU **finalize**
kernel (`jfa_kernel.cu::finalize_field`) consuming the resident flooded `feat` +
the host occupancy, so `distance_field` downloads **`n` f64 instead of `3n`**
(`seed_gpu::field_gpu`). Measured **1.32×** on the field stage (245→186 ms on a
314 K-node grid; `bench_field_vs_feat_download`), parity-gated by
`gpu_field_finalize_matches_cpu` + the end-to-end BALL area/volume test.

That captured most of the available transfer win (the `f` download is already ⅓
the original bytes). **Full K3** would only remove that remaining ⅓ download (and
the small CPU dual-contour compute) while taking on the highest topology risk in
the pipeline for a low-single-digit end-to-end gain — so it is **not worth
pursuing** unless keeping fields fully on-device becomes a strategic requirement.
The full design + the codex fixes (u8 local-sheet table, canonical UF roots,
round-to-nearest decider, per-node tri scan, the gate list) are kept below for
that contingency.

## Why full fusion (not an `f`-upload)

Dual contouring is only ~1% of SES-SDF runtime (243 ms at h=0.15 on crambin), so
porting it *in isolation* — upload `f`, contour on GPU, download mesh — is net
**slower** than the CPU pass (the `f` upload alone is an n·f64 transfer). K3 only
pays off if it removes the **full-field download** that currently happens between
flood and contouring: keep the flooded `feat` on the GPU, compute `f` there, run
dual contouring there, and return **only the mesh** (verts + tris, far smaller
than the field). So K3 requires also porting two cheap precursors:

1. **Occupancy** (`inside[]`) — currently a CPU sphere-rasterization. On GPU:
   per-node "inside any inflated atom?" via the **same `CellGrid`** the seed
   uses (a boolean spatial-hash query), into a `u8` grid. No new host data.
2. **Finalize** (`f[node]`) — `f = (inside ? dist : -dist) − probe`, with the
   `UNREACHED → large finite` and exact-zero nudge rules. A trivial per-node
   kernel over `feat` + `inside`.

Then dual contouring consumes `f` (device) and emits the mesh.

## Current CPU algorithm (what we port), `manifold_dual_contour`

Two passes over the `(nx-1)³` cell grid and the `nx³` node grid:

**Pass A (per cell):**
- Load 8 corner `f`; skip if all same sign.
- 12 cube-edge crossings: `crossing[e]`, `cpos[e] = lerp` at `t = fa/(fa-fb)`.
- Join crossings into **sheets** via a 12-element union-find, fed by per-face
  marching-squares: a face with 2 crossed edges unions them; a face with 4
  crossed edges uses the **asymptotic decider** (bilinear, from the 4 shared
  corner f's — deterministic so both cells across the face agree).
- One **vertex per sheet** = mean of its crossings' positions.
- Record `edge_vert[(cell, edge)] = vid` for each crossing edge.

**Pass B (per interior sign-changing grid edge):** the 4 incident cells (cyclic)
each contribute the sheet vertex of *their* copy of that edge → one quad (2
tris), winding from the sign of `f` at the edge's low node.

Tables (fixed): `CUBE_EDGES[12]`, `CORNER[8]`, `FACE_EDGES[6][4]`,
`FACE_CORNERS[6][4]`. Orientation (`orient_consistently` + volume-sign flip)
stays on the CPU (cheap, on the small mesh).

## GPU decomposition

All `f64` (the asymptotic decider and `t` must bit-match the CPU branch — a
divergent decider flips a sheet and tears the mesh). Cell linear index
`c = i + cx*(j + cy*k)`, `cx,cy,cz = dims-1`.

**K3a — classify + count (per cell).** Recompute corners/crossings/union-find;
output `sheet_count[c]` (0 if all-same-sign). Deterministic **local sheet index**:
scan `e = 0..12`; the first crossing edge whose UF root is new is sheet 0, the
next new root sheet 1, … (a 12-entry `root→localsheet` map in registers). This
fixes a per-cell ordering independent of thread scheduling.

**Scan.** Exclusive prefix sum of `sheet_count` over all cells → `vert_offset[c]`
and `total_verts`. Also need a per-cell **tri count** for Pass B output offsets;
compute `quad_count` per cell in a companion array (or fold into a second scan).
*Open: GPU scan vs a host round-trip of the two count arrays.* A host round-trip
is `ncells·u32` (~50 MB at h=0.15) up+down — large but simple and one-time; a
device scan (per-block scan + block-sum scan) keeps it on-device. Lean device
scan; fall back to host scan if it risks correctness for v1.

**K3b — emit vertices + edge_vert (per cell).** Recompute the same
classification (cheap, avoids storing per-cell state), write each sheet's mean
vertex at `vert_offset[c] + localsheet`, and fill the **dense** `edge_vert`
table: `edge_vert[c*12 + e] = vid` for crossing edges, `-1` otherwise. Size
`ncells·12·i32` (~600 MB at h=0.15 — **the memory risk**; see open questions).

**K3c — emit triangles (per grid edge / per node×3 dir).** One thread per node;
for each of the 3 axis directions test the interior sign-changing edge, look up
the 4 incident cells' `edge_vert`, write 2 tris at the scanned tri offset. The
`debug_assert(all four present)` becomes a guarded skip + an error flag the host
checks (a missing corner = a hole → fall back to CPU contour for that mesh).

Download `verts[total_verts]` + `tris[total_tris]`; CPU does
`orient_consistently` + volume flip.

## Parity strategy

The bar is **topology + geometry, order-independent** (not bitwise): the GPU may
emit vertices/tris in a different order, but the mesh must be the *same surface*.
Gates:
- New `gpu_dual_contour_matches_cpu` on the fixtures (single atom, two-atom toric,
  a saddle cell that exercises the 4-crossing decider): equal vertex count, equal
  tri count, equal area, equal volume, equal Euler χ / watertightness, and a
  Hausdorff-ish max vertex-set distance ≈ 0.
- The existing `ses_area_and_volume_match_ball_and_converge` now runs the *whole*
  pipeline on GPU → BALL parity is the end-to-end guard.
- A targeted **saddle/ambiguous-face fixture** so the asymptotic decider path is
  covered (CPU vs GPU identical sheet assignment).

## Open questions (for review)

1. **`edge_vert` memory** (600 MB at h=0.15). Options: (a) accept it (RTX 2070
   has 8 GB; the field buffers are already ~300 MB); (b) store only crossing
   edges via a compacted (cell,edge)→vid map (needs a hash or sort — complex);
   (c) cap the on-device contour to grids under a node threshold and CPU-contour
   the rest. Lean (a) with a guard that falls back to CPU when
   `ncells·12·4 > budget`.
2. **Scan**: device multi-block scan vs host round-trip of the count arrays. Does
   the host round-trip negate the K3 win (it's u32, not the f64 field — ~1/6 the
   bytes, and one-way each)? Probably acceptable; device scan is the clean answer.
3. **Decider determinism**: is replicating the exact `((f0*f2 − f1*f3)/denom <
   0.0) == (f0 < 0.0)` branch in CUDA f64 enough to guarantee both cells across a
   face agree, given no `-ffast-math` on NVRTC? (CPU already relies on this being
   exact between adjacent cells.)
4. **Worth it?** Given K3 is ~1% of runtime, the *only* benefit is dropping the
   full-field download. Confirm the saved transfer (n·3·f64 down) outweighs the
   added kernel/scan complexity + the `edge_vert` allocation — or whether a
   cheaper partial win (compute `f` on GPU, download `f` instead of `feat`, keep
   CPU contour) captures most of it at a fraction of the risk.

## Sequencing

1. Occupancy + finalize kernels (cheap, low-risk) → `f` on device; parity vs CPU
   `f` (exact).
2. K3a + scan + K3b + K3c behind a `dual_contour_gpu` entry; CPU fallback on any
   failure/over-budget. Parity gates above.
3. Wire into `ses_mesh_sdf`: when the GPU field path ran, continue on-device
   through the mesh; else CPU.
4. Fold the result into `GPU_SES_PLAN.md`.
