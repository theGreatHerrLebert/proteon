# Codex review — IMPROPER_ORDERING_PLAN.md

Reviewed 2026-08-27 via `claudex` (codex-cli 0.144.6). Plan v1.
Disposition of each point is recorded in §10 of the plan itself.

---

The ordering hypothesis is plausible, but Section 2 states it too confidently.
It is not yet a diagnosis; it is the leading topology-generation hypothesis.

AMBER wildcard matching can genuinely be ambiguous, and proteon's "first
matching neighbour permutation" is a red flag: it makes output depend on
bond/atom insertion order rather than an explicit force-field convention. That
is unacceptable if it changes physical energy under mere atom relabelling.

But identical improper counts and nominal parameters do not establish ordering
as the cause. A 3.6–10.4% spread in the improper component is also consistent
with:

- different central-atom placement or interpretation (central atom in position
  2 vs 3);
- a sign convention mismatch for the dihedral angle;
- phase convention differences, especially if phases are not only 0 or π;
- different treatment of multiple Fourier terms / duplicate matches;
- different template-precedence rules yielding different parameter *term sets*
  despite superficially matching `k`, phase, and periodicity;
- an evaluator bug in one engine.

Atom reordering is not universally energy-invariant. Reversing a dihedral
changes φ to −φ; `cos(nφ−phase)` is invariant only for compatible phases (not
generally). Changing which neighbour occupies the "middle" noncentral position
can change the geometric improper entirely. Thus, "same four atoms" is not
enough to infer equivalence.

The cheapest decisive experiment is a cross-replay matrix, not merely a tuple
diff:

1. Dump every ordered tuple and the complete list of Fourier terms per improper.
2. Evaluate all three dumps with one small, independently written reference
   evaluator on identical coordinates.
3. Confirm each engine's reported per-term energy agrees with that evaluator
   using its own tuple and terms.
4. Substitute OpenMM tuples into proteon's terms, and vice versa.

If an engine's own dump fails the common evaluator, this is an
evaluator/sign/units/phase issue. If each self-replays but cross-substitution
moves energy as predicted, it is ordering. Also compare forces for a few
high-contributing impropers; energy agreement alone can conceal sign errors.

## Is it worth doing?

The denominator argument understates the issue. The discrepancy is 0.05% of
this one total, but 3.6–10.4% of a force-field component, potentially with
nontrivial local force effects during minimization or dynamics. More
importantly, oracle culture makes a known, reproducible convention gap worth
understanding.

Against changing it: the evidence currently does not show which engine is
correct; OpenMM and Molly disagree materially, and OpenMM's default behavior is
explicitly reverse-engineered. A topology-wide modification near the production
CHARMM path is not justified solely to shave 4.4 kJ/mol from one fixture.

My recommendation: do Phase 1, including cross-replay and atom-relabel
invariance, but make "document and do not change" the default Phase 2 outcome
unless both conditions hold:

- AMBER/AmberTools or ParmEd confirms the desired convention unambiguously; and
- proteon demonstrably changes energy under a physically identical atom
  reordering, or the correct convention materially improves broad oracle
  results.

If those hold, a narrowly isolated AMBER-only fix is justified. Otherwise
retain behavior, document the known convention gap, and avoid production-path
risk.

## Phase 1 classification

The proposed buckets are useful but incomplete. Add at least:

- `term-set mismatch`: same tuple, but different number or multiset of Fourier
  terms, including duplicates and merged terms;
- `central-role mismatch`: same unordered quartet but a different atom treated
  as central;
- `evaluation mismatch`: same ordered tuple and full term list, but different
  reported energy;
- `equivalent permutation`: different tuple strings that produce identical
  angle/energy under the actual phases and periodicities;
- `mapping mismatch`: serial/index identity differs across dumps.

A key of `(center, frozenset(neighbours))` presumes the very central-atom
agreement being tested. Derive the structural center independently from
connectivity, and preserve each engine's asserted central role separately.

Also ensure the JSON record supports multiple terms per improper rather than one
`phase/k/periodicity`. AMBER-style torsions may be represented as several
periodic terms.

## Hidden risks

The plan correctly separates the CHARMM explicit-template path, but "same file,
different branch" is not enough protection. Shared tuple construction,
canonicalization helpers, trait defaults, serialization/debug dumps, or
deduplication can still affect CHARMM. Add a CHARMM topology-level regression
asserting both ordered improper tuples and energy/forces remain
byte-for-byte/numerically unchanged on a representative fixture.

The proposed PDB shuffle test is important, but implement it carefully: preserve
atom identity, coordinates, residue identity, bonding, and serial-to-atom
mapping. A naïve record shuffle may instead exercise parser assumptions or alter
OpenMM's index-based default tie-break. Compare physical atoms by stable
identity after loading.

Finally, a count invariant of 125 is necessary but weak. It does not detect
replacement of one parameterized term with another, duplicate-term handling, or
altered assignment among the same 125 structural groups. The exit criterion
should require a complete per-group term multiset and common-evaluator replay,
not only counts and ordered tuples.
