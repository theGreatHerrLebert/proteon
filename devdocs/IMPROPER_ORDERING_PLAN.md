# IMPROPER_ORDERING_PLAN — canonical AMBER improper 4-tuple ordering

Status: **proposed**, not started. Opened 2026-08-27 out of the Molly.jl
oracle work (PR #211).

## 1. The observation

Three independent engines, the same `amber96.xml`, the same committed
structure (`tests/oracle/data/1crn_prepped_amber96.pdb`), the same **125
impropers** — three different energies:

| engine | improper torsion (kJ/mol) |
|---|---|
| proteon | 128.939 |
| OpenMM | 124.495 |
| Molly.jl | 116.827 |

proteon vs OpenMM 3.57%, proteon vs Molly 10.37%, OpenMM vs Molly 6.56%.

The spread is **geometry-dependent**. On a differently-hydrogenated crambin
the same three engines gave 81.08 / 75.72 / 75.61 — there OpenMM and Molly
agreed to 0.144%. A constant parameter-set offset cannot produce a gap that
changes rank and magnitude with hydrogen placement.

Every other force-field component is triangulated at 0.000% (bond, angle,
proper torsion, vdW) or 0.019% (electrostatic). Impropers are the only
outlier, which localizes the problem to improper handling specifically
rather than to typing, charges, or parameter parsing.

## 2. Leading hypothesis (NOT yet a diagnosis)

**The AMBER improper dihedral is order-sensitive, AMBER's wildcard
templates leave the order ambiguous, and the three engines disambiguate
differently.**

This is the *leading* hypothesis, not an established cause. Identical
improper counts and nominally identical parameters do **not** establish
ordering as the culprit. Competing explanations that Phase 1 must rule out:

| # | Alternative | How Phase 1 distinguishes it |
|---|---|---|
| A | Central atom in tuple position 2 vs 3 | Central-role field in the dump |
| B | Dihedral sign convention (φ vs −φ) | Common-evaluator replay of each engine's own dump |
| C | Phase convention, where phases are not only 0 or π | Dump `phi0` per term; check invariance explicitly |
| D | Multiple Fourier terms per improper, or duplicate/merged matches | Dump the full term **multiset**, not one term |
| E | Template-precedence rules yielding different term sets despite matching k/phase/periodicity | Compare term multisets per group |
| F | Evaluator bug in one engine | Common-evaluator replay (see below) |

**F is strongly disfavored before we start, and cheaply so.** In all three
engines the improper term is evaluated by the *same code* as the proper
torsion: proteon uses one `compute_dihedral` and one
`(v/div)*(1 + cos(f·φ − φ0))` loop for both
(`proteon-core/src/forcefield/energy.rs:171-210`); Molly stores both as
`PeriodicTorsion`; OpenMM puts both in `PeriodicTorsionForce`. Proper
torsions agree across all three to **0.000%**. An evaluator, sign or units
bug would have to be invisible on propers and visible only on impropers.

That argument is strong but not airtight: propers and impropers may not
span the same set of `φ0`/periodicity values, so a phase-specific sign
error could still hide. The common-evaluator replay in §3 confirms it for
a few kJ of effort, so we do it rather than assume it.

Note also that proteon's evaluator already iterates a term *list*
(`for term in terms`), so multi-term impropers are handled in the math —
the gap flagged as (D) is in the Phase 1 **dump schema**, not in proteon.

An improper is a dihedral over four atoms with one central atom bonded to
the other three. Its value depends on which neighbour occupies which
position. AMBER defines most impropers with double wildcards (e.g.
`* * N H`), so a template match does not determine the ordering — each
implementation must pick, and the picks differ.

### What proteon does today

`proteon-core/src/forcefield/topology.rs:960-985` — the AMBER heuristic
path enumerates all six permutations of the three neighbours and takes the
**first one that finds a parameter**, then breaks:

```rust
for &(i, k, l) in &[(n0,n1,n2), (n0,n2,n1), (n1,n0,n2),
                    (n1,n2,n0), (n2,n0,n1), (n2,n1,n0)] {
    ...
    if has_cosine || has_harmonic {
        improper_torsions.push(Torsion { i, j: k, k: center_idx, l });
        break;
    }
}
```

There is no canonicalization. The chosen tuple depends on the iteration
order of `neighbors[center_idx]`, which is a build artifact of bond
insertion order — not a force-field convention. Two structures that differ
only in atom ordering could in principle get different improper energies.

### What OpenMM does

`openmm/app/forcefield.py:_matchImproper` implements an explicit tie-break
for `ordering='default'`, with a comment conceding it is reverse-engineered
from AMBER ("it then follows some bizarre rules to pick the order"):

```python
a1, a2 = torsion[t2[1]], torsion[t3[1]]
e1, e2 = element(a1), element(a2)
if e1 == e2 and a1 > a2:
    a1, a2 = a2, a1                      # same element -> lower index first
elif e1 != carbon and (e2 == carbon or e1.mass < e2.mass):
    a1, a2 = a2, a1                      # carbon/heavier goes second
match = (a1, a2, torsion[0], torsion[t4[1]], tordef)
```

Central atom lands in **position 3**; the two leading atoms are ordered by
element identity, then mass, then index. It also prefers non-wildcard
template definitions over wildcard ones. OpenMM additionally supports
`ordering='amber'` and `ordering='charmm'` variants — which of these
`amber96.xml` actually requests must be read off the XML, not assumed.

### What Molly does

Unknown, and this is the first thing to establish. Molly builds impropers
in `Molly/src/force_field.jl` from the same XML. Since Molly agrees with
OpenMM exactly on every other component, and agrees with OpenMM on the
improper *count* (125), its divergence is most likely a different
tie-break — but it could also be a sign or phase convention in the
`PeriodicTorsion` evaluation. Do not assume.

**Why "same four atoms" is not enough.** Reversing a dihedral maps φ to
−φ, and `cos(nφ − φ0)` is invariant under that only for compatible phases
(φ0 = 0 or π); AMBER impropers are commonly π, but this must be checked per
term rather than assumed. Swapping which neighbour occupies the middle
non-central position changes the geometric improper outright. So a tuple
diff alone cannot tell "different order, same physics" from "different
order, different physics" — hence the replay design below.

**Falsifiable prediction.** If the hypothesis is right, the three engines
emit the same 125 *unordered* {center, neighbour-set} groups with the same
term multisets, but different *ordered* 4-tuples — and substituting
OpenMM's tuple ordering into proteon's terms reproduces OpenMM's energy to
floating-point noise. If the unordered groups or the term multisets differ,
the hypothesis is wrong and this plan needs rewriting.

## 3. Phase 1 — measure, do not fix (contract-first)

Mirrors the OBC GB Phase A pattern: land a deliberately-failing oracle that
pins the contract before touching any math. **No proteon behaviour changes
in this phase.**

The core instrument is a **cross-replay matrix**, not a tuple diff. A tuple
diff can show that engines chose different orderings; only a replay shows
whether that choice is what moves the energy.

### 3.1 Dump

One JSONL record per improper, per engine:

```json
{
  "engine": "proteon|openmm|molly",
  "center_role": 2,                    // which tuple slot the engine treats as central
  "tuple_serials": [12, 14, 13, 27],   // ordered, stable PDB serials (NOT load indices)
  "tuple_names":   ["CA","C","N","O"],
  "types":         ["CT","C","N","O"],
  "terms": [                            // a MULTISET, never a single term
    {"periodicity": 2, "phase_rad": 3.141592653589793, "k_kj": 4.6024}
  ],
  "phi_rad": 3.1039,
  "energy_kj": 0.0021
}
```

Two schema rules, both from review:

- **`terms` is a list.** AMBER torsions may carry several Fourier terms, and
  duplicate or merged matches are one of the competing explanations (§2 D/E).
  A single `k/phase/periodicity` field would silently hide that class.
- **Identity is the PDB serial**, not the engine's internal index. Index
  identity differs across engines and is itself a failure mode
  (`mapping mismatch` below).

Sources: proteon — extend the existing `dump_topology_obc` debug entry point
(`tests/test_dump_topology_obc.py`) rather than inventing a new one. OpenMM —
the improper-only force already split out by
`validation/amber96_molly_triangulate.py:_split_torsion_force`. Molly —
extend `molly_energy_oracle.jl` to emit `sils[4].is/js/ks/ls` with terms.

### 3.2 Replay

A single small reference evaluator, written independently of all three
engines (~30 lines: `compute_dihedral` + the cosine series), used twice:

1. **Self-replay.** For each engine, evaluate *its own* dump. If an engine's
   reported per-improper energy does not reproduce, the problem is that
   engine's evaluator, sign, units or phase handling — not ordering. This is
   the cheap test that settles §2 hypothesis F.
2. **Cross-substitution.** Evaluate proteon's term multisets against OpenMM's
   orderings and vice versa. If each engine self-replays but cross-substitution
   moves the energy as predicted, ordering is confirmed as the cause.

Also compare **forces**, not only energies, on the ~10 highest-contributing
impropers. Energy agreement can conceal a sign error that forces expose, and
forces are what actually matter for the minimizer.

### 3.3 Classify

Key every improper by a center derived **independently from connectivity**
(the atom bonded to the other three), *not* by any engine's asserted center —
keying on the engine's center would presume the very agreement under test.
Each engine's claimed `center_role` is retained as data.

| bucket | meaning |
|---|---|
| `identical` | same ordered tuple and same term multiset everywhere |
| `equivalent permutation` | different tuple, but provably identical φ-energy under the actual phases/periodicities |
| `permutation` | same atom set, different order, different energy (the expected bucket) |
| `central-role mismatch` | same unordered quartet, different atom treated as central |
| `set mismatch` | different atoms chosen — would falsify §2 |
| `term-set mismatch` | same tuple, different number/multiset of Fourier terms |
| `evaluation mismatch` | same tuple and same terms, different reported energy |
| `mapping mismatch` | serial-to-atom identity differs across dumps |

`equivalent permutation` matters: without it, a tuple diff reports
disagreements that are physically null and inflates the apparent problem.

### 3.4 Invariance test (proteon-internal only)

Shuffle ATOM record order *within each residue* of the fixture, reload,
recompute. proteon's improper energy is already bit-stable across repeated
loads of the same bytes (§9), but the chosen permutation depends on
`neighbors[center_idx]` insertion order, so order-invariance is the property
actually at risk.

Two implementation cautions from review:

- Preserve atom identity, coordinates, residue identity and bonding; compare
  physical atoms by stable identity after loading, or the test measures
  parser behaviour instead of topology behaviour.
- **Keep this proteon-internal.** OpenMM's tie-break reads atom *index*
  (`if e1 == e2 and a1 > a2`), so shuffling changes OpenMM's answer too. A
  cross-engine shuffle comparison is not interpretable.

If proteon's improper energy moves under a physically identical reordering,
that is a correctness bug in its own right and the §8 severity assessment
changes.

### 3.5 Exit criterion

Per group: every engine's ordered tuple, its full term multiset, its dihedral
angle, its energy, its bucket, and a passing self-replay. A count invariant
of 125 is **necessary but weak** — it cannot detect one parameterized term
being swapped for another, duplicate-term handling, or reassignment among the
same 125 structural groups. The exit criterion is the term multiset plus
replay, not counts and tuples.

Ship a failing/xfail test asserting the contract we want (proteon improper
== OpenMM improper within 0.1%), pointing at this document.

## 4. Phase 2 — decide the canonical convention

Only after Phase 1 data exists. The decision is *which* convention proteon
should adopt, and it is not automatically OpenMM's.

Inputs to the decision:

- What `amber96.xml` actually declares (`ordering` attribute per improper).
- What the AMBER spec and `parmed`/`tleap` do — the true reference, since
  OpenMM's own comment admits it is emulating observed AMBER behaviour.
  Consider adding `parmed` as a fourth read-only opinion here; it is
  pip-installable and reads AMBER prmtop directly.
- Whether the convention must differ per force field. proteon supports
  CHARMM19 too, and CHARMM impropers are harmonic and explicitly ordered by
  residue templates (`improper_paths_for_residue`), which is a *different*
  code path (`topology.rs:910`). **This plan must not disturb the CHARMM
  path**; CHARMM19+EEF1 is the production default.

### Decision rule (adopted from review)

**"Document and do not change" is the DEFAULT outcome of Phase 2.** proteon
changes only if BOTH conditions hold:

1. AmberTools / ParmEd confirms the intended convention **unambiguously** —
   not OpenMM's reverse-engineered emulation of it, which its own source
   comment describes as "some bizarre rules"; and
2. Either proteon demonstrably changes energy under a physically identical
   atom reordering (§3.4), **or** the corrected convention materially
   improves broad oracle results across the 1000-PDB set — not just the
   crambin fixture.

Rationale: the evidence does not currently establish which engine is right.
OpenMM and Molly disagree by 6.56% on this fixture while agreeing to 0.144%
on another, so "match OpenMM" is not self-evidently correct. A topology-wide
change adjacent to the production CHARMM path is not justified to move
4.4 kJ/mol on one structure.

Deliverable: a short decision note appended to this document — chosen
convention (possibly "none, documented"), why, and the expected numerical
effect.

## 5. Phase 3 — implement

Replace the break-on-first-match loop with an explicit canonicalization:
match the template first, then order the tuple by the chosen rule.

Constraints:

- The AMBER heuristic path and the CHARMM explicit-path branch stay
  separate. Gate any new ordering behind the same params-trait mechanism
  used by `excludes_aromatic_ring_diagonals()` rather than branching on
  force-field name at the call site.
- Improper **count** must not change (125 on the fixture). A count change
  means the template matching moved, which is out of scope here.
- Cross-path parity: NBL, SIMD, GPU and rayon paths all consume the same
  topology, so this should be transparent to them — but per
  `devdocs/ORACLE.md` the parity tests must be re-run, not assumed.
- **"Same file, different branch" is not sufficient CHARMM protection.**
  Shared tuple construction, canonicalization helpers, trait defaults,
  serialization/debug dumps and deduplication can all reach the CHARMM path
  even when the branch does not. Land a CHARMM topology-level regression
  *before* Phase 3 that pins the ordered improper tuples **and** the
  energies/forces on a representative fixture, and require it unchanged.

## 6. Phase 4 — re-measure and tighten

- Regenerate the frozen Molly reference (`tests/oracle/test_molly_energy.py`).
- Flip the Phase 1 xfail to a real assertion at whatever band the data
  supports.
- Re-run the 1000-PDB OpenMM oracle: the improper term is small (~1.3% of
  crambin's total) but the fix is topology-wide and could move totals on
  structures with more branched side chains than crambin.
- Update `devdocs/ORACLE.md`: the tolerance-matrix row currently reads
  "Improper (3-way) ... Open convention gap, not a pass."

## 7. Non-goals

- Changing which impropers are *found*. The BALL single-wildcard gap
  (10 vs 125) is a separate, already-documented issue.
- Touching CHARMM19 improper handling.
- Making Molly agree. If Phase 1 shows Molly is the outlier against both
  OpenMM and AMBER, the correct outcome is a documented Molly gap and
  possibly an upstream issue — not bending proteon toward it.
- Performance work.

## 8. Risks

- **The energy delta is small in the total, but not in the component.**
  ~4.4 kJ/mol vs OpenMM on a 9153 kJ/mol total is 0.05% — but it is
  3.6–10.4% *of the improper term*, and per-improper **forces** may be
  affected more than the scalar energy suggests. Forces are what the
  minimizer consumes, so the fold-preservation numbers are the place a real
  effect would show up, not the single-point energy. Phase 1 measures forces
  for this reason. It remains legitimate to conclude "document, do not
  change" — see the §4 decision rule.
- **Ordering may be load-bearing elsewhere.** If any golden fixture
  encodes the current tuple order, changing it will look like a
  regression. Grep for stored improper tuples before Phase 3.
- **PDBFixer non-determinism** (PR #211) means Phase 1 must run on the
  committed fixture, not a fresh prep, or the classification will be noisy.

## 9. Open questions

**Answered 2026-08-27, before review:**

1. ~~Does `amber96.xml` set an explicit `ordering` attribute?~~ **No.**
   `grep -o 'ordering="[a-z]*"' amber96.xml` returns nothing, so OpenMM's
   `ordering='default'` branch applies — the element/mass/index tie-break
   quoted in §2 is the one actually in play. Confirmed against OpenMM 8.5.2.

2. ~~Is proteon's improper energy stable across loads?~~ **Yes, for a fixed
   file.** Three loads of the committed fixture gave bit-identical
   `128.93855087084657` and `n_impropers=125`. So this is a wrong-convention
   problem, not a non-determinism problem, and the priority stays where §8
   puts it.

   Note the limit of that check: it shows stability across repeated loads of
   the *same byte sequence*. It does **not** show that the result is
   invariant under reordering the ATOM records of the same structure, which
   is the property actually at risk given that the chosen permutation
   depends on `neighbors[center_idx]` insertion order. Phase 1 should add
   that test explicitly: shuffle ATOM record order within each residue,
   reload, and compare improper energy. If that moves, the severity
   assessment in §8 changes and this work should be reprioritised.

**Still open:**

3. Should `parmed` join as a fourth opinion for Phase 2, or is that
   over-triangulating a 0.05% term?
4. Is the right outcome possibly "document and do not change"? §8 admits
   this. Reviewers should push on whether a 0.05%-of-total convention fix
   earns a topology-wide change to a code path that CHARMM19 also
   traverses, given CHARMM19+EEF1 is the production default.

## 10. Review log

### claudex (Codex, codex-cli 0.144.6) — 2026-08-27, plan v1

Full review: `devdocs/IMPROPER_ORDERING_PLAN.codex-review.md`.

**Adopted:**

| # | Point | Where |
|---|---|---|
| 1 | §2 overstated a hypothesis as a diagnosis | §2 retitled; six competing explanations tabulated with the test that separates each |
| 2 | Cross-replay matrix beats a tuple diff — self-replay separates evaluator bugs from ordering | §3.2, now the core instrument |
| 3 | Dump must carry a term **multiset**, not one `k/phase/periodicity` | §3.1 schema |
| 4 | Keying on `(center, frozenset(neighbours))` presumes the central-atom agreement under test | §3.3 — center now derived independently from connectivity |
| 5 | Classification buckets incomplete | §3.3 — added `equivalent permutation`, `central-role mismatch`, `term-set mismatch`, `evaluation mismatch`, `mapping mismatch` |
| 6 | Atom reordering is not universally energy-invariant; `cos(nφ−φ0)` reversal-invariant only for φ0 ∈ {0, π} | §2 |
| 7 | Shuffle test must preserve atom identity, and must stay proteon-internal because OpenMM's tie-break reads atom index | §3.4 |
| 8 | "Same file, different branch" is weak CHARMM protection | §5 — CHARMM topology regression required *before* Phase 3 |
| 9 | Count invariant of 125 is necessary but weak | §3.5 |
| 10 | 0.05%-of-total understates it: 3.6–10.4% of the component, and forces matter more than energy | §8; force comparison added to §3.2 |
| 11 | Make "document and do not change" the explicit default outcome, gated on two conditions | §4 decision rule |

**Contributed back (not in the review):** hypothesis F (evaluator bug) is
strongly disfavored a priori. All three engines evaluate impropers with the
*same code* as proper torsions — proteon shares one `compute_dihedral` and
one cosine loop between them (`energy.rs:171-210`), Molly stores both as
`PeriodicTorsion`, OpenMM puts both in `PeriodicTorsionForce` — and proper
torsions agree across all three to 0.000%. A sign/units/evaluator bug would
have to be invisible on propers and visible only on impropers. The argument
is not airtight (propers and impropers may not span the same φ0 set), so the
self-replay in §3.2 still runs it to ground rather than assuming it.
Recorded in §2.

**Net effect on scope:** Phase 1 grew (replay + forces + invariance test);
Phase 3 became conditional rather than assumed. The plan is now more likely
to end in "document, do not change" — which review considered the correct
default and this plan now adopts.
