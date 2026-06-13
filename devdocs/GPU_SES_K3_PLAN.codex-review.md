# Codex plan review — GPU_SES_K3_PLAN.md (2026-06-13)

Independent review of the K3 dual-contour GPU port plan.

## Headline recommendation
**Stop at the `f`-download milestone first.** Implement GPU occupancy + finalize,
download one f64/node (instead of three feature f64s — ~2/3 transfer cut), keep
the validated CPU contour. Benchmark wall time / PCIe / peak memory. Only do full
K3 if the measured saving justifies the topology-sensitive complexity (contouring
is ~1% of runtime, so the upside is low-single-digit end-to-end).

## Major findings (if full K3 proceeds)
1. Don't require CPU/GPU bit-identical decider. CUDA may FMA-contract
   `f0*f2 - f1*f3`; near zero topology can diverge. Use round-to-nearest
   intrinsics (`__dmul_rn`/`__dsub_rn`) mirrored on CPU, or a division-free
   determinant-sign form. Add near-zero adversarial fixtures (saddle alone is
   insufficient).
2. Local-sheet scheme is sound (1 thread/cell), but canonicalize UF roots to the
   min crossed-edge index and enumerate in edge order; share ONE device helper
   between K3a and K3b. Adjacent cells need not match local-sheet numbers — only
   each shared grid edge → correct cell-local sheet.
3. Store `edge_sheet[cell][edge] = local_sheet | 0xff` as **u8** (not i32 vid):
   600 MB → 150 MB; `vid = vert_offset[cell] + edge_sheet`. Budget peak LIVE
   allocation (f, feat, occupancy, scan scratch, offsets, verts, tris), fallback
   on alloc/overflow.
4. Use a device scan (CUB/established primitive if available), not a bespoke one;
   host round-trip (~50 MB each way + sync) weakens K3's whole point. Tri scan is
   **per-node** (0/2/4/6), not per-cell.
5. Economic case weak until measured — prefer the f-download milestone.

## Missing gates
- u32 total/offset overflow → u64 or reject.
- Define exact-zero, fa==fb, nonfinite, UNREACHED, decider-tie, boundary-touch
  behavior.
- Missing sheet vertex must invalidate the WHOLE GPU result (no partial mesh).
- CPU fallback must reconstruct/download f after a late K3 failure.
- Test PRE-orientation winding (orient_consistently hides bad quad order).
- Compare canonical connectivity (each global edge's 4-cell sheet tuple + 2
  tris), not just area/volume/Euler/vertex-distance.
- Randomized cell-level differential tests; exhaustive sign configs; near-zero
  ambiguous faces; opposite face orientations; multi-sheet cells; boundary edges;
  degenerate interpolation.
