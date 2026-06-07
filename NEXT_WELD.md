# NEXT_WELD — making the cleaned SES watertight

> **STATUS: DONE (2026-06-06).** The cleaned SES is watertight on crambin —
> `watertight=TRUE, open=0, nonmanifold=0, euler=4` (= 2 components, matching the
> independent SDF mesher), area 2319.899 (+0.04 % vs BALL), self-intersection-free.
> Path taken: a **tolerance weld** (`Mesh::welded_within`, user-chosen over
> bit-identity) + great-circle seam unification (spheric edge sampled with the
> toric θ-end's `toric_column_curve`) + a **same-circle arc merge** that removed
> the `arrange_loops` plane-basis wrap-split (the actual dominant residual — NOT
> the neighbour-exclusion mismatch the earlier analysis feared). Commits
> `9451983`, `42b0358`. The 3-seam-type / registry plan below was superseded by
> this simpler route; kept for history. Remaining polish (not blocking): confirm
> watertightness across a corpus, tune the default `weld_eps`, wire
> `ses_mesh_cleaned_welded` into any public SES entry point.

Pick-up doc for finishing the SES singularity-cleaner **weld**. The geometry is
done and oracle-validated; what remains is the watertight topology.

## Where we are (branch `feat/ses-general-n`, not pushed)

- `assemble::ses_mesh_cleaned` builds the cleaned SES: toric faces via
  `cleaner::toric_trim_mesh` (neighbour burial + spindle cap), spheric faces via
  `cleaner::clip_spheric_face`, contact caps unchanged. It returns **concatenated
  patches** (no weld) so `surface_area()` is rigorous but the mesh is **open**.
- **Validated:** crambin analytic 2344.415 (+1.13 % vs BALL ≈2318.9) → cleaned
  **2319.825 (~+0.04 %)**, 0 errors on every face/arc. The cleaner removes 24.9 Å²
  (spheric 12.9 + toric 12.0). This already refutes "BALL's triangulation isn't
  portable."
- 92 surface tests, clippy clean, 9 codex reviews applied.

## The weld = 3 seam types, each needs bit-identical shared samples

`Mesh::welded()` dedups by `f64::to_bits()`, so two patches share an edge only if
they sample it at the **exact same** `Vec3`s. The existing (non-cleaned)
`ses_mesh_analytic` is watertight because every shared boundary is sampled once
and passed verbatim to both patches. The cleaner must preserve that for each seam.

### 1. Collision seams (spheric_i ↔ spheric_j on `C_ij`) — ✅ DONE
Both colliding faces' burial boundaries lie on the **same** canonical circle
`C_ij`. `clip_spheric_face` now samples burial arcs via
`cleaner::sample_circle_arc` on `canonical_burial_circle(i,j,…)` — bit-identical
between faces. (Commit `cb19f58`; test
`burial_seam_samples_are_frame_independent_for_both_faces`.)

### 2. Great-circle seams (toric θ-end ↔ spheric edge) — ⏳ TODO (the hard one)
The spheric↔toric boundary is the great circle on the probe sphere through the two
contacts (a θ-end column of the toric face). Today:
- `toric_face_mesh` / the cleaned toric trim sample the θ-end reentrant arc one way
  (`toric_column_curve` / `arc_on_sphere` over the probe-centre chain).
- `clip_spheric_face` samples great-circle arcs via `cap.rim_point(θ)` in **its own
  frame** — *not* bit-identical to the toric θ-end.

**To do:**
1. In `clip_spheric_face`, sample great-circle arcs via `arc_on_sphere(p, probe,
   corner_x, corner_y, n)` in **canonical low→high atom order** (exactly as
   `ses_mesh_analytic`'s spheric block does, assemble.rs:573-585), NOT via
   `cap.rim_point`. This requires threading the **atom indices** of each great
   circle into `clip_spheric_face` (currently it only has the 3 contact points
   `cs`; pass `f.atoms` alongside).
2. Where a burial cap **clips** a great-circle edge, the new corner is
   `C_ij ∩ great_circle` — must come from `SingularVertices::intern_corner`
   (registry, keyed `(i, j, {a,b}, branch)`), so the spheric clip and the toric
   trim resolve the **same** corner. So `clip_spheric_face` must take a
   `&mut SingularVertices` + the atom centres.
3. The toric trim's θ-end column must clip at the **same** registry corner when a
   neighbour buries that end. Today `toric_trim_mesh` clips per-column by φ but
   does not snap the θ-end great-circle boundary to the registry corner. Add that.

The trap: `arrange_loops` reconstructs corner positions independently and clusters
by tolerance — they are **not** bit-identical (codex). So every great-circle/burial
corner consumed by both a spheric and a toric patch MUST be replaced by the
registry-interned point before sampling.

### 3. Contact-cap ↔ toric φ-rim — ⏳ TODO
Contact caps (`fill_spherical_region` over `walk_cap_loops`) use the **untrimmed**
toric φ-rims as their boundary loops. When a neighbour trims the toric face near a
contact, the cap boundary must follow the **trimmed** rim.

**To do:** have `toric_trim_mesh` (or a sibling) emit the trimmed φ-rim boundary
per atom, and feed those to the contact-cap fill instead of the raw `ContactArc`
points. The φ-rims are already bit-identical shared samples in the analytic path;
keep that property through the trim (sample the trimmed rim once, share verbatim).

## Suggested order
1. **Great-circle seam (#2)** first — it's the dominant open boundary and the
   coupled one. Sub-steps: (a) thread `f.atoms` + `atom_centers` + `&mut
   SingularVertices` into `clip_spheric_face`; (b) switch great-circle sampling to
   `arc_on_sphere` canonical order; (c) intern + use the registry corner where a
   burial cuts a great-circle edge; (d) make the toric θ-end snap to the same
   corner.
2. **Contact seam (#3)** — feed trimmed rims to the caps.
3. `mesh.welded()` at the end of `ses_mesh_cleaned`; then
   `orient_consistently()` + flip.

## Validation / gate
- **Watertight:** `ses_mesh_cleaned(crambin).is_watertight()` == true (currently
  open). Track `num_nonmanifold_edges()` / open-edge count as it drops.
- **Self-intersection gate:** `intersect::self_intersections == 0` (transversal;
  note it misses the spindle's coplanar fold, so it's necessary-not-sufficient).
- **Area still 2319.8** (~+0.04 % vs BALL) — the weld must not move it.
- **Euler / components / signed volume > 0** vs BALL (codex Q6 richer gate).
- **Regression:** `cleaned_assembler_is_inert_without_collisions` stays green
  (tetra/chain unchanged).
- Run a **codex review** of the great-circle weld before/after (it touches the
  working analytic weld — easy to regress tri/tetra/chain).

## Key files
- `proteon-core/src/surface/cleaner.rs` — `clip_spheric_face`, `toric_trim_mesh`,
  `sample_circle_arc`, `SingularVertices` (registry), `singular_edges`,
  `sample_singular_edge`, `spindle_cap`.
- `proteon-core/src/surface/assemble.rs` — `ses_mesh_cleaned` (the assembler),
  `ses_mesh_analytic` (the watertight reference to mirror), `walk_cap_loops`,
  `toric_face_mesh`.
- `proteon-bin/src/bin/ses_singular_diag.rs` (tracked) — per-face removed-area
  diagnostic. `ses_protein.rs` (untracked) — ANALYTIC vs CLEANED area on a pdb.
- Plan docs: `TO_SES_WIRING.md`, `TO_SES_NONRADIAL_INTEGRATION.md`.

## Constraints (carry forward)
- `main` is PR-gated (ruleset 15659866); branch + `gh pr create` only.
- MSRV 1.75 — no `is_none_or`; use `map_or`. CI lint = `cargo clippy --workspace
  --all-targets -- -D warnings` + `cargo fmt --all -- --check`.
- Deploy to remote via commit+push+pull through GitHub, never rsync/scp.
- `ball-py` import name is `ball`, not `ball_py`.
