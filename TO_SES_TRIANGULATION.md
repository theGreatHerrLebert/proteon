# Plan: port BALL's Solvent-Excluded Surface (SES) triangulation to proteon

## 0. Goal & verdict

Port BALL's analytic **Solvent-Excluded Surface** + **triangulated mesh** pipeline
(Connolly / Sanner reduced-surface algorithm) into pure Rust in proteon, validated
the proteon way: **port, then gate every layer against BALL as an oracle** via
EVIDENT claims.

**Portability verdict (from reading the source): portable.** The ~12.6k LOC in
`ball/source/STRUCTURE/{reducedSurface,solventExcludedSurface,triangulatedSES,
triangulatedSAS,triangulatedSurface, RS*, SES*, SAS*, analyticalSES}.C` use **no
CGAL / GSL / Eigen / boost / threads / OpenMP** — only STL containers and BALL's
own small geometry kernel (`TVector3`, `TSphere3`, `TCircle3`, `TPlane3`,
`TAngle`, `TSimpleBox3`) + `HashMap`/`HashSet` (→ std/Rust hash maps). Input is a
plain `vector<TSphere3<double>>` + probe radius; output is an **index-based mesh**
(`MATHS/surface.h`: `Triangle{Index v1,v2,v3}` + vertex positions + normals) via
`TriangulatedSurface::exportSurface()`. No entanglement with BALL's `Atom`/
`Composite` hierarchy at the algorithmic level.

**The hard part is not dependencies — it's numerical robustness.** SES
singularity handling (`solventExcludedSurface.C` is 2,648 LOC with ~91 "singular"
references, plus a dedicated `SESSingularityCleaner`) and the RS ambiguity
perturbation (`RSComputer::correct()` shrinks an atom radius by `10*EPSILON` to
break 4-coplanar-atom probe ambiguity) must be ported **faithfully, same
epsilons**, or the mesh develops cracks/holes. The whole plan is built to isolate
and gate that risk.

## 1. The pipeline (as it exists in BALL)

```
spheres + probe_radius
  └─ ReducedSurface / RSComputer            reducedSurface.C
        rolling-probe combinatorics → RSVertex (atom), RSEdge (atom pair + torus),
        RSFace (atom triple + probe). Neighbor lookup keyed by sorted atom-index
        tuples (SortedPosition2/3). Start-finding by extremal atom; treatFace /
        treatEdge / thirdAtom; correct() for ambiguous probes.
  └─ SolventExcludedSurface / SESComputer   solventExcludedSurface.C
        RS → SES elements: contact faces (convex, on atoms), toric reentrant
        faces (probe rolling over 2 atoms), spheric reentrant faces (probe fixed
        on 3 atoms), convex/concave edges, singular vertices/edges.
  └─ SESSingularityCleaner                  solventExcludedSurface.C
        resolve probe-probe self-intersection: split singular toric faces,
        delete small (singular) toric faces, clean vertices/edges/faces.
  └─ TriangulatedSES / SESTriangulator      triangulatedSES.C
        phase-decomposed: triangulateToricFaces → partitionSingularEdges →
        triangulateContactFaces → triangulateSphericFaces. Spheric faces use
        precomputed template-sphere triangulations at a target `density`
        (buildTemplateSpheres / partitionOfCircle / buildAmbiguous|Unambiguous).
  └─ exportSurface()                        triangulatedSurface.C → MATHS/surface.h
        flatten pointer-linked mesh → index-based TSurface (verts, normals, tris)
```
(There is a parallel SAS path — `solventAccessibleSurface` / `triangulatedSAS` —
reusing the same RS. Out of scope for v1; add after SES lands.)

## 2. Where it goes in proteon

New module tree under **`proteon-core`** (pyo3-free; the geometry kernel lives
here too and is reusable):

```
proteon-core/src/surface/
  geom.rs        Vector3, Sphere3, Circle3, Plane3, Angle + intersections (probe-of-3,
                 circle-of-2, sphere-sphere, line-sphere). The numeric foundation.
  mesh.rs        index-based Mesh { verts: Vec<[f64;3]>, normals, tris: Vec<[u32;3]> }
                 + invariants (area, closed?, Euler χ, enclosed volume).
  rs.rs          ReducedSurface (RSVertex/RSEdge/RSFace as index-keyed arena, NOT
                 raw pointers) + the rolling-probe computer.
  ses.rs         SolventExcludedSurface (contact/toric/spheric faces, edges).
  singular.rs    singularity cleaner (the fragile layer, isolated on purpose).
  triangulate.rs SESTriangulator (the 4 phases + template spheres).
  mod.rs         public API: ses_mesh(spheres, probe, density) -> Mesh + the
                 per-layer entry points (reduced_surface(), ses(), ...) so each
                 layer is independently callable AND testable.
```
Python exposure for proteon's own users via `proteon-connector/src/py_surface.rs`
(mirrors the oracle binding), re-exported like the rest of Option B.

Pointer→arena translation: BALL's mesh and RS/SES graphs are `T*`-linked with
`friend`-class mutation. In Rust use **index arenas** (`Vec<Vertex>` + `u32`
handles) — cleaner, no `Rc<RefCell>`, and it makes the combinatorial invariants
(Euler characteristic, manifoldness) cheap to assert.

## 3. The oracle: expand ball-py to emit the SES mesh

ball-py source: `/scratch/TMAlign/ball-zomball/python/src/module.cpp` (pybind11,
`theGreatHerrLebert/ball`, on PyPI as `ball-py`). Today it exposes 11 functions
(sasa, *_energy, rmsd, hbonds, …) — **no surface mesh**. We add the oracle hooks:

- `ses_mesh(spheres, probe, density) -> dict` — `vertices (V,3)`, `normals (V,3)`,
  `triangles (T,3)` from `ReducedSurface → SES → SESSingularityCleaner →
  TriangulatedSES → exportSurface`. **Must also return metadata + stats**
  (codex), not just arrays: effective density / template-sphere resolution +
  vertex/tri counts, probe radius, BALL version+git sha, and correction/
  singular-cleaning counts + any warnings. For **pinned EVIDENT artifacts** store
  a **canonical summary** (invariants + a tolerance-quantized triangle hash after
  spatial sort), not the raw arrays — BALL's mesh emission order is not
  guaranteed stable run-to-run.
- `ses_area(spheres, probe) -> float` via `analyticalSES` — mesh-free analytic
  area. **Strong scalar cross-check, not independent ground truth** (it may share
  RS/SES intermediate state with the construction path).
- `reduced_surface_stats(spheres, probe) -> dict` — the full RS **graph** (L1
  gate): vertex/edge/face counts, face atom-triples + probe centers, edge
  atom-pairs, adjacency, component ids, probe root signs, correction events.
- `ses_graph(spheres, probe, cleaned: bool) -> dict` — the SES element graph
  **before and after** singular cleaning (L2 + L3 gates): face counts by type,
  edge/vertex counts by type, adjacency, ownership ids, per-face-type areas,
  Euler χ of the SES complex.

Use the low-level `ReducedSurface(vector<TSphere3<double>>, probe)` constructor so
the oracle takes **raw spheres** — avoid building a BALL `System`/`Atom` tree
unless an analytic path requires it (and if so, verify both paths agree). Publish
as a `ball-py` bump (PyPI publish-chain checklist memory: Trusted-Publisher +
pyproject version + unique-vs-history). These bindings are the substrate for all
claim-based gating below.

## 4. Decomposition into independently-testable layers (the core ask)

Each layer is callable and gated on its own. Where a mesh is non-unique, gate on
**invariants**, not vertex-exact equality.

**L0 — Geometry kernel (`geom.rs`).** No BALL needed.
- Unit + property tests: probe-touching-3-spheres center solves both roots;
  circle-of-2-spheres radius/normal; degenerate/no-solution cases return None.
- Cross-check a handful of predicates against BALL `MATHS` values if convenient.
- *Independently testable: fully.*

**L1 — Reduced Surface (`rs.rs`).** Oracle: `reduced_surface_stats`.
- Gate on the RS **graph**, not just face triples (codex: triples can match while
  the embedding differs): vertex/edge/face counts; the order-independent set of
  face atom-index **triples** + probe centers; **edge atom-pairs**; face↔edge and
  edge↔vertex **adjacency**; per-component ids; probe **root sign/orientation**;
  and the list of **ambiguity-correction events** (which atoms `correct()` fired
  on). RS is essentially unique modulo the perturbation, so this is near-exact.
- Watch: `correct()` perturbation must match BALL's `10*EPSILON` exactly. Keep a
  separate test of known 4-coplanar cases.
- *Independently testable: yes, against RS graph stats — no triangulation.*

**L2 — raw SES construction (`ses.rs`), BEFORE singular resolution.** Oracle:
SES element-graph parity (primary) + `ses_area` (cross-check).
- Gate primarily on the **SES element graph** (codex: area alone is too lossy —
  compensating errors pass): contact/toric/spheric **face counts**,
  convex/concave/singular **edge counts**, **vertex counts by type**, face↔edge
  and edge↔vertex adjacency, atom/probe **ownership ids**, and **per-face-type
  area totals** (ideally per-element areas sorted by stable ownership key).
- `ses_area` (analytic) is a strong **scalar cross-check**, NOT sole ground truth
  — `analyticalSES` may reuse the same RS/SES intermediate state, so a
  construction bug could hide in both. Gate the graph; cross-check the scalar.
- *Independently testable: yes, on the SES graph.*

**L3 — singularity cleaner (`singular.rs`) as a graph rewrite.** Oracle: pre-clean
**and** post-clean SES exposed from BALL.
- Gate L3 directly as a **before→after topology delta** (codex: not "watertight
  downstream"): singular toric faces split/removed, singular edges before/after,
  deleted vs new faces/edges/vertices, component + boundary-loop counts, and the
  **Euler characteristic of the SES element complex** before triangulation.
- **Metamorphic, not fixed-fixture-only**: for each singular case also test under
  permuted atom order, translation far from origin, uniform scaling, and
  ε-perturbation of radii/probe across the singular threshold. (Fixed fixtures
  pass while order/scale-dependent cleaner bugs survive.)
- Downstream watertightness remains an *integration* gate, not L3's primary one.
- *Independently testable: yes, as a graph rewrite against pre/post SES.*

**L4 — Triangulation (`triangulate.rs` + `mesh.rs`).** Oracle: `ses_mesh` (consumes
the cleaned SES only). Triangulations are NOT unique → never gate vertex-exact;
gate on invariants. **The trap (codex): "topologically closed but geometrically
wrong"** — watertight yet self-intersecting, inside-out, or overlapping patches
that preserve area. So the invariant set must catch geometry, not just topology:
  1. **Mesh area → analytic SES area** as density rises (convergence claim).
  2. **Manifold + watertight**: every edge shared by exactly 2 triangles,
     **boundary-loop count = 0**, no non-manifold edges; per-**component** Euler
     characteristic / genus (not just global) consistent with BALL.
  3. **Orientation**: **signed** enclosed volume > 0 (catches inside-out); face
     normals point outward vs local atom/probe geometry.
  4. **No self-intersections** (closed ≠ embedded).
  5. **Enclosed volume** + **bounding box** + **connected-component count** vs BALL.
  6. **Sampled distance to the analytic SES** (each mesh vertex lies on its
     owning patch within ε) — a *measured diagnostic*, not a tight gate.
  7. **Vertex/triangle counts** within a band of BALL's at the same density —
     **measured**, not gated (density discretization differs).

This mirrors the EVIDENT "gated scientific claim vs measured stat" split: graph
parity / area-convergence / watertightness / orientation / volume are **gated**;
raw counts and Hausdorff-to-BALL-mesh are **measured**. To make L4 diagnostics
sharp, carry **face-origin (ownership) labels** through triangulation so an error
can be localized to contact/toric/spheric and to the owning atom/probe.

## 5. Phasing (each phase = mergeable PR + its EVIDENT claim)

- **P0** ball-py oracle additions (`ses_mesh`, `ses_area`, `reduced_surface_stats`)
  + publish. Nothing to port yet — just stand up the oracle. *(unblocks everything)*
- **P1** `geom.rs` + `mesh.rs` + invariants. L0 tests. No oracle dep.
- **P2** `rs.rs` → L1 gated on `reduced_surface_stats`. EVIDENT claim:
  RS-combinatorics-parity-vs-BALL.
- **P3** `ses.rs` → L2 gated on analytic `ses_area` + face counts. EVIDENT claim:
  ses-area-vs-BALL-analytical (the headline scientific claim).
- **P4** `singular.rs` → L3. Micro-corpus of singular cases; watertightness gate.
- **P5** `triangulate.rs` → L4. EVIDENT claims: mesh-area-converges-to-analytic-SES,
  mesh-watertight, mesh-volume-vs-BALL.
- **P6** Python exposure (`py_surface.rs`) + `proteon ses` CLI subcommand (writes
  the mesh as OBJ/PLY/Parquet) — same pattern as the analysis CLI.
- **P7** (optional) SAS path; GPU later if profiling warrants.

## 6. Risks / things to watch

- **Count-parity on clean cases ≠ validation (Codex code review, 2026-06-04).**
  The oracle corpus must include **deliberately singular / degenerate** configs,
  not just well-spaced ones: tangent probes, zero-radius roll-circles, atoms
  whose roll-circle is nearly fully covered (a clear arc narrower than the
  sample spacing), tiny exposed caps, duplicate probe centers, torus
  self-intersection, multi-loop contact faces. The L1 edge/vertex predicates are
  currently **sampling-based** (`CIRCLE_SAMPLES`/`SPHERE_SAMPLES` in `rs.rs`) — a
  stopgap that can false-neg narrow arcs / tiny caps and false-pos near
  tangencies. Replace with exact blocked-arc / cap-coverage analysis before this
  runs on real molecules; until then, treat large-molecule RS as unvalidated.
- **Singularity epsilons** — the whole correctness story. Port `correct()` and the
  cleaner with identical constants; gate with a dedicated singular micro-corpus.
- **Determinism** — BALL's start-finding uses extremal atoms + iteration order over
  STL containers; replicate ordering so RS is reproducible (and parity-comparable).
- **`density` semantics** — what BALL's density parameter means (triangles per Å²?
  template subdivision level?) must be matched so mesh-count bands are comparable;
  pin it from `triangulatedSES.C` `buildTemplateSpheres`.
- **Non-manifold inputs** — disconnected molecules produce multiple RS components
  (`getRSComponent` loops); handle multi-component meshes and per-component Euler χ.
- **Float reproducibility** — BALL is `double` throughout; use `f64`, preserve op
  order in the geometry predicates (same discipline as the TM-align port).
- **Oracle cost** — BALL mesh computation on large proteins is slow; keep the
  oracle corpus small + curated (singular cases, a few real proteins, synthetic
  2–4 sphere configs with analytic answers).

- **"Closed but wrong" (codex)** — the central L4 trap: watertight yet
  self-intersecting / inside-out / overlapping. Mitigated by the orientation +
  self-intersection + per-component-genus invariants above, not edge-count alone.
- **Input domain** — define accepted inputs and behavior on degeneracies
  (duplicate atoms, coincident centers, zero/negative radii, huge coordinate
  magnitudes, fully-buried atoms, disconnected molecules). Decide explicitly
  whether "parity with BALL" includes reproducing BALL's behavior on *invalid*
  input or whether proteon validates and rejects. Test at the singular thresholds.

## 8. Review-resolved decisions (codex round)

- Keep RS and SES as **separate** gated layers; split SES into raw-construction
  (L2) and cleaned (L3), each with a graph contract.
- Gate L2/L3 on **element-graph parity**, not analytic area alone; area is a
  scalar cross-check. `analyticalSES` is not treated as independent ground truth.
- L4 gates on invariants incl. **orientation/signed-volume, self-intersection,
  per-component genus, boundary loops** — not just edge-shared-by-2.
- Hausdorff/sampled-distance to BALL's mesh is a **measured diagnostic**, never a
  tight gate. Distance-to-analytic-surface is the better sampled check.
- Geometry kernel stays **in `proteon-core`** until a second consumer exists.
- Synthetic fixtures (1 sphere, separated, 2 equal/unequal overlapping, tangent,
  3-sphere pocket, tetrahedral 4-sphere ambiguity) anchor **topology/symmetry**,
  not exact area (closed-form area for 3+ spheres is impractical).
- Raw BALL mesh is **not** stable enough for byte-exact pinned artifacts → pin
  canonical invariants + tolerance bands.
- Carry **face-origin ownership labels** through triangulation so L4 errors
  localize to face type + owning atom/probe.

## 8b. Knowledge gap CLOSED (BALL source read 2026-06-03)

- **Q1 — pre/post-clean SES exposure.** Singularity cleaning is **in-place inside
  `ses->compute()`**: `SESComputer::run()` = `preProcessing(); get();` (raw SES)
  then `while(!SESSingularityCleaner.run()){…}`. The raw (pre-clean) SES is only
  transient, so the L3 "before" snapshot needs a **small BALL-side helper** that
  runs `preProcessing()+get()` and stops (friend access from the binding) — not
  free, deferred to a follow-up binding. Also: the canonical driver wraps compute
  in a **`check()` + probe-perturbation retry** (±0.01, ≤10×); the oracle should
  replicate + report it.
- **Q2 — density.** Default **4.5**; `sqrt_density = sqrt(density)` drives circle/
  sphere subdivision. Note: `ses->clean(density)` runs at triangulation start, so
  the *cleaned* SES is density-dependent (real L3↔L4 coupling).
- **Q5 — ownership labels.** Fully available: `SESFace` carries `Type{contact,
  toric,spheric}` + the originating `RSVertex*`/`RSEdge*`/`RSFace*` → atom index /
  pair / triple. Exposable per face (and, with bookkeeping, per triangle).
- **analyticalSES independence.** `calculateSESArea(AtomContainer, probe)` is
  "the algorithm by Michael L. Connolly" — a **separate code lineage** from the
  RS/SES/triangulation classes, so it is a genuine cross-check (stronger than the
  draft feared). Confirmed at runtime: 1-sphere r=2 → 50.265 = exactly 4πr².
- **P0 STATUS: oracle hooks DONE** (`theGreatHerrLebert/ball` branch
  `feat/ses-mesh-oracle`, pushed). All four gateable-layer hooks built +
  runtime-smoke-tested on BALL v1.6:
  - `ses_mesh(spheres,probe,density)` — L4 mesh (verts/normals/tris + RS counts)
  - `ses_area(spheres,probe)` — Connolly analytic area+vol (independent x-check;
    1-sphere = 4πr² exactly)
  - `reduced_surface_stats(spheres,probe)` — L1 RS graph (counts, sorted atom
    triples + probe centers, atom pairs)
  - `ses_graph(spheres,probe)` — L2/L3 post-clean SES element graph (face-type
    counts + singular-edge count + atom ownership)
  Smoke tests (7) assert closed-form + combinatorial invariants; wheel CI runs
  them. **Build verified Qt5 5.15** (`FIND_PACKAGE(Qt5)`, no Qt6 anywhere; CI
  installs `qt5-qtbase-devel`/`qtbase5-dev`); the SES code is pure libBALL-core
  so the PyPI wheel build needs no new dep.
  Remaining: the **L3 pre-clean** `ses_graph` snapshot (cleaner runs in-place →
  needs a BALL-side `preProcessing()+get()` helper) + a `ball-py` **PyPI bump**
  so proteon's 3.12 venv can `pip install` it (open the branch as a PR + tag).

## 9. Open questions still to settle

(Questions 1–6 from the first draft were resolved in §8.) Remaining:

1. **Pre/post-clean SES exposure feasibility** — can ball-py expose the SES
   element graph *before* `SESSingularityCleaner` runs (L3's "before" snapshot)
   without re-running the pipeline, or does BALL clean in-place? If in-place, we
   compute the cleaner separately on a copied SES, or snapshot inside the binding.
2. **`density` exact meaning** — confirm from `buildTemplateSpheres` /
   `partitionOfCircle` what `density` controls (template subdivision level vs
   triangles/Å²) so proteon's parameter maps 1:1 and count-bands are comparable.
3. **Self-intersection test cost** — exact mesh self-intersection is O(T²) naive;
   is a spatial-hash/BVH check fast enough at corpus scale, or do we sample?
4. **Where does the analytic-area cross-check come from for L2** if `analyticalSES`
   shares state with construction — do we need a *truly* independent area (e.g.
   numerical SASA-style Monte-Carlo / Gauss-Bonnet on the mesh) as the tiebreak?
5. **Ownership-label stability** — are BALL's atom/probe ownership ids stable and
   exposable per SES element + per triangle, so cross-tool label comparison is
   meaningful, or only counts?
