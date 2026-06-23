# Multi-chain coverage — label-safe complex substrate

> Status: **design** (claudex-reviewed). Connects the assembly gate (#190,
> `label_safe_interface`) to the per-residue masking path (#192–195).
>
> **Honest scope (claudex):** v1 produces a *label-safe complex substrate* — the
> verified, masked per-chain coordinate inputs of a confirmed biological assembly.
> It is NOT a full interface-label format: it emits no contact map / neighbor
> edges and no multimer tensor packing. A consumer that computes pair labels from
> the complex coordinates can use it directly; a turnkey interface-training format
> is a follow-on.

## The gap

We can already *detect* that a deposited ASU is the biological assembly
(`assembly_is_asu is True`), and `label_safe_interface` gates on it. But the
coverage/masking export path is **single-chain v0**: `structure_coverage` raises
on a multi-chain input, and `prepare_for_supervision(min_coverage=...)` can't
produce the per-chain examples of a *verified complex*. So interface / contact /
SASA / neighbor-graph labels — the entire reason the assembly gate exists — are
**detectable but not exportable**. This closes that.

## What already works (verified, not assumed)

- **The clash scan is whole-complex.** `prepare` builds topology over all model-0
  chains, so `clash_residue_indices` already spans every chain (4hhb: 47/46/52/47
  across A/B/C/D). A chain-A residue that clashes *into* chain B is already in the
  set, so the existing per-chain `residue_clash_mask` already masks it. **Inter-
  chain clash masking needs no new code** — it falls out of the whole-complex scan.
- **Per-chain export already exists.** `build_structure_supervision_example(...,
  chain_id=X)` produces a chain's example with masking. A complex is N of these.
- **The assembly gate already answers "are inter-chain contacts real?"** A genuine
  0.4 Å vdW *overlap* across an interface is bad geometry (a real clash, correctly
  counted); but artificially tight *crystal-packing* contacts are exactly what
  `assembly_is_asu is False` rejects. So we do not need a separate "interface
  contact vs clash" heuristic — the assembly verification gates it.

So the work is **only the gate/coverage side**, not the masking side.

## Design

### 1. `structure_coverage` — per-chain over a complex

Today `structure_coverage(structure)` raises on >1 protein chain. Add a
complex-level entry that returns per-chain coverages:

```python
@dataclass
class ComplexCoverage:
    chains: dict[str, ResidueCoverage]   # chain_id -> its ResidueCoverage
    @property
    def min_coverage(self) -> float      # the weakest chain (the gate value)
    @property
    def total_coverage(self) -> float    # valid residues / all residues (reporting)

def complex_coverage(structure, *, profile="heavy_coords", report=None) -> ComplexCoverage:
    # one structure_coverage(chain_id=ch) per protein chain; report shared
    # (clash_residue_indices are whole-complex, already aligned by all-residue idx).
```

`min_coverage` (weakest chain) is the gate value: an interface label needs BOTH
partners usable, so one bad chain should drop the pair — not be averaged away by a
pristine partner. `total_coverage` is for reporting only.

**Known limitation (claudex) — chain-level is conservative.** Coverage says "this
chain is generally usable," but interface labels are *local*: a missing loop FAR
from the interface should not invalidate it, while one AT the interface absolutely
should. v1 gates on chain-min coverage, so it conservatively drops a complex for
an off-interface defect. The correct long-term answer is per-interface / per-
contact masks derived from the residue masks on both sides of each contact — an
explicit follow-on (it needs an inter-chain contact computation, which is the
*consumer's* concern in v1). v1 errs toward dropping, never toward a silent bad
interface label.

**Clash-index alignment (claudex) — proven, not assumed.** `clash_residue_indices`
is the global all-model-0-residue index namespace. Each per-chain
`structure_coverage` call maps its chain-local residues back to that SAME global
index (the `residue_clash_mask` re-walk), never reinterprets them as chain-local.
A test asserts the per-chain clash masks partition the global set exactly (4hhb:
per-chain clashing counts sum to `len(clash_residue_indices)`), so the mapping
provably survives per-chain slicing.

### 2. The interface gate — assembly AND coverage

`prepare_for_supervision` gains an opt-in interface mode. A complex is kept iff:

- `report.assembly_is_asu is True` (the deposited chains ARE the assembly), AND
- it has **≥ 2 protein chains** (a verified monomer is not an interface example —
  claudex), AND
- the prepared protein chains **match the assembly's chain list** — already
  GUARANTEED: `assembly_is_asu is True` is defined as "REMARK 350 chain list equals
  the present chains exactly," so codex's chain-set check is subsumed (no extra
  code; a coords-only extra chain or a missing one makes `is_asu` False), AND
- `complex_coverage(...).min_coverage >= floor` (every chain usable).

```python
prepare_for_supervision(paths, interface=True, min_coverage=0.8)
# res.coverage_info: ComplexCoverage; res kept iff verified-assembly + ≥2 chains + all pass
```

**Distinct drop reasons (claudex)** — not a flat "unsafe":

- `assembly_is_asu is False` → `requires_assembly_expansion`. This is NOT
  "biologically invalid": a monomeric ASU that expands to the assembly via BIOMT
  is valid — this pipeline just hasn't *built* the assembly coordinates (the
  assembly-builder follow-on). Logged distinctly so it's recoverable later.
- `assembly_is_asu is None` (no REMARK 350 / no path) → `assembly_unverified`,
  dropped: an interface label on an unverified oligomeric state is the crystal-
  artifact hazard the gate exists to prevent.
- `< 2 protein chains` → `not_a_complex`.
- chains present in coords but absent from the assembly chain list → those chains
  excluded; if that leaves `< 2`, the complex drops.

### 3. Export — a complex WRAPPER, not a loose list (claudex)

Per-chain examples emitted independently lose the chain relationships interface
labels need. So interface mode yields a complex-level wrapper that preserves
identity, ordering, and the shared report:

```python
@dataclass
class ComplexSupervisionExamples:
    record_id: str                                   # pdb/source id
    chain_order: list[str]                           # deposited chain order
    chain_examples: dict[str, StructureSupervisionExample]  # each masked, cross-chain-correct
    coverage: ComplexCoverage
    # the assembly + report context the consumer needs to compute pair labels
    assembly_is_asu: bool
    prep_report: PrepReport

def build_complex_supervision_examples(
    structure, *, prep_report, min_coverage, mask_untrustworthy_coords=True
) -> Optional[ComplexSupervisionExamples]:
    # None (with a logged drop reason) if the complex fails the interface gate;
    # else the wrapper with one masked example per verified chain.
```

The per-chain `StructureSupervisionExample`s are unchanged in format — each is the
same single-chain tensor bundle the export already produces, just masked
cross-chain-correctly. The multimer packing (asym_id/entity_id/sym_id
concatenation, AF-multimer relative-position features) and the inter-chain
contact-label emitter are deliberate follow-ons; the consumer computes pair labels
from `chain_examples`' coordinates in v1.

## What is explicitly NOT in v1

- **No new multimer tensor format** (concatenated residues, asym_id/entity_id/
  sym_id, AF-multimer relative-position features). v1 delivers verified, masked
  *per-chain* examples; the multimer packing is a follow-on the caller or a later
  PR owns.
- **No inter-chain contact/edge labels** (the actual interface contact map) — that
  is the *consumer* of this gate, not part of it. v1 makes the inputs trustworthy.
- **No symmetry expansion** — only the case where the deposited ASU already IS the
  assembly (`assembly_is_asu is True`); building the oligomer from BIOMT is the
  assembly-builder follow-on noted in `ASSEMBLY_HAZARD_DESIGN.md`.

## Test plan

- `complex_coverage` on 4hhb (4 chains): per-chain coverages present; `min` is the
  weakest; a chain with an injected missing loop drops `min` but not the others.
- **Clash-index alignment (claudex):** per-chain clash masks partition the global
  `clash_residue_indices` exactly (4hhb: per-chain counts sum to the report total)
  — proves the global→chain-local mapping survives slicing.
- Inter-chain clash masking: a residue clashing only across the interface is masked
  in its own chain's example (already true — pin it as a regression).
- Interface gate drop reasons: 4hhb (`True`, 4 chains, good coverage) → kept; 1ake
  (`False`) → `requires_assembly_expansion`; no-REMARK-350 → `assembly_unverified`;
  a verified single protein chain → `not_a_complex`; a chain in coords but absent
  from the assembly list → excluded.
- `build_complex_supervision_examples` returns a `ComplexSupervisionExamples` with
  N masked chains for a kept complex (chain_order preserved), None for a failed one.

## Validation

Extend `eval_archive_scale.py` with an `--interface` mode: how many of the 9,422
are verified assemblies, and of those how many pass per-chain coverage — the
honest interface-label yield.
