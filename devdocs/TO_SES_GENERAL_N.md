# Plan: general-N analytic SES + singularity cleaner (finish the port)

## Where we are

`two_atom_ses` and `triangle3_ses` (proteon-core `surface/assemble.rs`) produce a
**watertight analytic SES gated vs `ball-py ses_area`** (<0.1% on triangle3, all 3
patch types). But the orchestration is a hardcoded special case: 2 probes, one
free arc per atom pair, one toric per pair, a 2-arc single-loop per contact cap.
Goal: a **general-N assembler** (any atom count, non-singular) → then the **L3
singularity cleaner** (faithful BALL port) → then **expose** (`proteon ses` CLI +
connector). All gated against BALL.

Built + tested already, reused: `geom`, `rs` (RS graph, oracle-gated), `ses`
(element graph), `elements` (ses_vertex / contact_circle / buried_cap / arc
samplers), `arrangement` (`boundary_loops`/`sample_loop`, multi-loop capable),
`chart` (`fill_spherical_region`), `cdt`, `registry` (shared-boundary backbone),
the patch meshers (`contact_cap_mesh`, `toric_face_mesh`, spheric via chart).

## Decisions from review (claudex)

- **A2 (index-based registry), not A1.** The built `registry` is already A2 — finish
  it; patches emit triangles indexing `registry.verts`, no weld. The hard part
  isn't welding, it's **curve identity** across discovery paths.
- **Curve key = oriented toric-interval identity** `(rs_edge, interval_id, atom)`,
  not endpoint pairs (full rings have no endpoints; distinct arcs can share them).
  Endpoints are validation, not primary identity.
- **Roll-circle splitting = analytic blocked-interval UNION**, NOT "split at RS
  faces + alternate" (fails at tangency/nested blockers). For each other atom,
  compute its blocked angular interval analytically; union all; complement = free
  intervals; classify endpoints as RS faces afterward. **This replaces the sampled
  RS-edge detection** (`rs.rs` admits missing narrow arcs) as the authority — the
  analytic interval computation determines edge existence itself.
- **`boundary_loops` is NOT general-N ready** (greedy tolerance linking, rejects
  multivalent vertices). The contact boundary must be built from the **interval/
  half-edge graph** (annotate each arrangement arc with its owning RS-edge/interval
  during extraction), not matched by coordinate afterward.
- **The real core = one globally consistent half-edge decomposition** shared by the
  roll-circle interval arrangement and the per-atom cap arrangement; both agree on
  curve identity, endpoints, orientation, event-merging. That, not patch
  triangulation, is the triangle3→N trap.
- **DEFER B (faithful singularity cleaner).** A pre/post graph oracle is
  insufficient to match BALL's cleaned graph (needs full incidence/ordering/
  ancestry; distinct rewrites give equal stats). Ship "analytic **non-singular**
  Connolly SES; singular rolling-probe configs **rejected explicitly**" — defensible,
  lands C sooner. Faithful cleaner = separate project.
- **Phasing before tetra:** (1) chain w/ a 2-neighbour middle atom + zero-face free
  rings, (2) a pair with two disjoint free intervals, (3) an atom with a multi-loop
  exposed region, (4) a cap loop alternating arcs from ≥3 RS edges, (5) tetra.
- **Gates:** incidence consistency, per-component Euler χ, **no unused registry
  curves**, watertight, area/vol/on-patch vs BALL; failure classification separating
  *unsupported singularity* from *numerical failure*.

## A. General-N assembler

The triangle3 assumptions and how each generalizes (codex-review):

1. **Roll-circle splitting.** A roll circle (RS edge i,j) is split by every RS face
   `{i,j,*}` into arcs; the **free** arcs (probe clears all other atoms) are the
   bounded toric faces — there can be **several** per pair, or **one full ring**
   (free toric) if the pair has no RS faces. Each free arc's two ends are RS faces
   = SES vertices. *(triangle3 = exactly one free arc, two ends.)*
2. **Contact caps multi-loop.** `boundary_loops` already returns ≥1 loops (annulus
   / pair-of-pants) for an atom's buried-cap arrangement. The cap mesher already
   takes multiple loops. The new part: the loop arcs must be the **same samples**
   as the incident toric φ-rims.
3. **Spheric faces.** One per RS face; 3 concave-arc edges shared with the 3 toric
   θ-ends. Unchanged from triangle3 except keyed generally.

**Watertightness = a registry that samples each shared curve ONCE.** Two design
options for how patches consume it:
- **(A1) Vec-sharing + exact weld** (what triangle3 does, but routed through the
  registry): the registry stores the sampled `Vec<Vec3>` per curve (keyed by its
  two SES-vertex endpoints + discriminator); every patch that touches the curve
  gets the *same* `Vec` → bit-identical → `mesh.welded()` fuses. Reuses the
  current point-based patch meshers. Risk: relies on exact weld + every patch
  asking with the identical key.
- **(A2) Index-based registry** (codex's preferred): patches emit triangles
  indexing directly into `registry.verts`; no weld, truly shared indices. Cleaner
  topology guarantee, but the patch meshers must be refactored to index-based
  (boundary indices in, interior appended to the registry, triangles out as
  indices).

**The shared curves + keys** (the registry already has the key shapes):
- contact-circle arc: keyed `(atom, edge, the two endpoint SES vertices)`; the
  toric φ-rim and the contact cap consume it. Sampled by **probe position** along
  the free arc (so toric and cap agree by construction).
- concave arc: keyed `(rs_face, pair)`; the toric θ-end and the spheric edge
  consume it. Sampled `arc_on_sphere` in canonical low→high atom order.
- SES vertices: keyed `(rs_face, atom)`.

**Phasing:** (a) roll-circle free-interval splitter (generalize the triangle3 free
arc; ≥1 intervals; tested on a chain/tetra RS) → (b) general toric faces from the
intervals → (c) general contact caps consuming registry arcs (multi-loop) → (d)
spheric faces → (e) the general assembler + gate vs `ball ses_area` on **tetra**
(4 atoms, the smallest case with >2 neighbours per atom) and a **chain** → (f)
real-protein gate (crambin) vs `ball.ses_mesh`/`ses_area`.

## B. Singularity handling — GEOMETRIC RESOLVER (decided)

> **DECISION:** not a faithful BALL-cleaner port. Detect probe self-intersection
> and resolve it geometrically via the arrangement/clipping we already have,
> producing a *valid watertight* SES on singular configs, **gated on area/volume/
> topology/residuals vs BALL** (`ses_mesh`/`ses_area`), NOT BALL's exact cleaned
> graph. This makes the exposed mesher work on real proteins. The faithful
> graph-identical cleaner is a separate project (claudex: oracle insufficient,
> bakes in BALL's tolerances/event-order) — explicitly out of scope here.
>
> Approach: detect a singular toric face (the two reentrant arcs of a probe sweep
> self-cross / probe-probe intersection within a free interval), split/clip it
> geometrically so the reentrant surface stays embedded, gate the result on
> area/volume/topology + self-intersection vs BALL on a singular corpus + crambin.
> Metamorphic tests (permuted order, translation, scale, ε across the threshold).

## B-faithful (deferred, separate project)

When atoms are close enough that the rolling probe **self-intersects**, BALL's
`SESSingularityCleaner` splits/deletes singular toric faces and rewrites the SES
graph. This is the chosen faithful-port piece (specification risk; the part the
boss might be "right" about). Plan:
- **Oracle:** `ball-py ses_graph(spheres, probe, cleaned)` exposing the SES element
  graph **before and after** cleaning — likely a `ball-py` binding bump (the
  current 0.1.0a6 exposes `ses_graph` but verify the cleaned pre/post split).
- **Detect** singular configs geometrically (self-intersecting toric: the two
  reentrant arcs of a probe sweep cross; probe-probe intersection). Port BALL's
  thresholds.
- **Rewrite** the SES element graph to match BALL's cleaned output (graph delta
  gated vs `ses_graph(cleaned=true)`), then feed the general-N assembler.
- **Gate metamorphically** (permuted order, translation, scale, ε across the
  singular threshold) — fixed fixtures hide order/scale-dependent bugs.

## C. Expose

`proteon ses <pdb> -o mesh.obj` CLI subcommand + a connector binding
(`py_surface.rs`), once A (and B for real proteins) land. Mirrors the grid mesher's
exposure path.

## Open questions for review
1. **A1 vs A2** — is the index-based registry worth the patch-mesher refactor, or
   is Vec-sharing + exact weld (proven on triangle3) sufficient for general-N
   watertightness? Where does Vec-sharing+weld break that the index registry fixes?
2. **Roll-circle interval ↔ contact-arc consistency** — the toric φ-rim is sampled
   by probe position along a free arc; the contact cap's boundary arc is the *same*
   contact-circle arc. How to guarantee identical samples when a contact circle is
   split by SES vertices from *different* toric free-arcs (multiple incident edges
   contributing arcs to one cap loop)?
3. **Singularity detection** — can self-intersection be detected purely
   geometrically (probe-probe distance) and resolved by the existing arrangement/
   clipping, or does matching BALL's *output* force porting its graph-rewrite
   thresholds verbatim? Is the ball-py cleaned-graph oracle sufficient to gate it?
4. **Phasing risk** — is tetra the right first general-N gate, or is there a
   simpler >2-neighbour case? What's the biggest correctness trap going from
   triangle3 to general-N?
5. **Effort honesty** — given the grid mesher already meshes proteins <1% vs BALL,
   how much of B (singularities) is needed for the *portability claim* vs a
   production mesher? Could the claim be made with general-N non-singular + an
   explicit "singular configs deferred" statement?
