# USAlign multi-chain on a 3-pair fixture set

Operational case writeup for the USAlign multi-chain CI claim in
`claims/usalign_mm.yaml`.

## Problem

The proteon manifest's `subsystem` vocabulary lists `align.usalign`
alongside `align.tmalign`, but no claim file uses that vocab slot —
multi-chain alignment is exercised only by unit-level Rust tests
under `proteon-align/src/ext/mmalign/`, with no Python-level oracle
comparison against a canonical reference. A regression in the
chain-assignment heuristic, the SOI-iteration loop, or the complex-
score accumulator would not surface in any structured claim today.

The single-chain `tmalign.yaml` claim covers the TM-align core; this
claim covers the multi-chain extension.

## Trust Strategy

Validation against the canonical C++ USAlign binary
(github.com/pylelab/USAlign) running in `-mm 1` mode. Same oracle
choice as the single-chain claim, with the same caveat documented in
`failure_modes`: USAlign and proteon's MM-align both descend from
Yang Zhang's MM-align, so this is an independent *implementation*,
not an independent oracle. A second comparator (DALI multi-chain,
FoldSeek-mm) would strengthen Rule 5 "Use Independent Validation"
and is documented as a future tightening.

- **Oracle**: `USalign -mm 1 -outfmt 2` parsed via the same tabular
  output format as the single-chain claim.
- **Engine**: `proteon_connector.mm_align_pair(pdb1, pdb2)` returning
  `PyMMAlignResult { total_score, chain_pairs, chain_assignments }`.
- **Fixture set** (3 pairs):
  - `1ake_self` — adenylate kinase homodimer aligned to itself.
    Trivial-numerical case; TM=1.0 from both tools. Catches
    multimer-assembly convention bugs.
  - `1cse_self` — subtilisin-eglin heterodimer self-aligned.
    Trivial-numerical case for a heterodimer; chain-assignment
    convention check.
  - `1ake_vs_1cse` — totally different multimers cross-aligned via
    best-chain-pair search. TM ~0.3; the only fixture in this set
    that exercises the numerical machinery non-trivially.

## Evidence

`tests/oracle/test_usalign_mm_oracle.py` runs four assertions per
fixture pair:

- `total_tm_score` — `|proteon.total_score - cpp.tm_complex| < 5e-4`
  absolute. Matches the single-chain claim's tolerance.
- `per_chain_rmsd` — RMSD per assigned chain pair within 0.05 A
  absolute. Asserted per chain pair so a failure points at a
  specific chain rather than aggregating.
- `n_aligned_per_chain` — within +/- 2 residues per chain pair.
- `chain_assignments` — set-equality of `(query, target)` tuples.

The test skips silently when the C++ USAlign binary is not on PATH
(via `pytest.skip()` after a `which USalign` probe), matching the
existing single-chain pattern in `test_tmscore_oracle.py`.

## Assumptions

- USAlign's `-outfmt 2` tabular output reports the per-complex TM
  score and the per-chain breakdown in a parseable, stable format.
  A change to the output schema would break the parser without
  changing the underlying agreement.
- The three fixture pairs cover the convention/identity case (self-
  alignments) and one cross-multimer numerical case
  (1ake-vs-1cse). Two of three are trivial; the corpus is small by
  design — CI-tier, not release-tier.
- proteon's chain-assignment heuristic and USAlign's both produce
  the global-optimum chain pairing on these fixtures. If either
  drops to a local optimum on edge-case multimers (large symmetry
  groups, near-degenerate scores), the assignment-set-equality
  assertion would fail spuriously; we have not seen this on the
  three fixtures.

## Failure Modes

- **Self-alignment is trivial.** Two of three pairs are
  self-alignments where TM=1.0 is algorithmically forced. Numerical
  drift in the SOI / chain-assignment heuristic only surfaces on
  the 1ake-vs-1cse pair. Tightening: add biologically meaningful
  multimer pairs (hemoglobin variant set, antibody Fab pairs).
- **Implementation-twin oracle.** USAlign and proteon's mm-align
  share algorithmic ancestry. A bug copied from the original
  MM-align C++ would pass this claim. A truly independent oracle
  (DALI, FoldSeek-mm) would close that gap.
- **No corpus_sha.** Fixture PDBs at `test-pdbs/` (mixed proteon
  source-tree and `/scratch/TMAlign/test-pdbs/` host paths) are
  byte-stable across runs but not pinned by a hash; a re-download
  with subtle differences would silently shift the assertions.
- **Set-equality on chain_assignments.** The assertion ignores
  pair-emission order. A consumer that depends on USAlign's
  specific reporting order would not be guarded by this claim.

## Lessons

- A subsystem listed in the vocabulary but bare in the claim set is
  a defect of the manifest itself — the vocab promises coverage
  that does not exist. Closing that gap is cheap and worth doing
  for every vocab term before the next release tag.
- Trivial-numerical fixtures (self-alignments) are useful as
  infrastructure / convention checks even when they do not test
  numerical agreement. The cost is one fixture line in the YAML;
  the benefit is catching multimer-assembly regressions early.
