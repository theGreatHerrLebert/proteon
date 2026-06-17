# HBOND_ORACLE_PLAN — backbone H-bond oracle vs DSSP (RELIABILITY §13)

Status: DRAFT (for claudex). proteon's `backbone_hbonds` uses the **same
Kabsch–Sander electrostatic criterion DSSP uses** (energy < −0.5 kcal/mol), and
DSSP *is* a backbone-H-bond detector — so canonical mkdssp is the natural oracle.
`tests/test_hbond.py` has unit tests but no external oracle. This adds one,
reusing the DSSP-oracle infrastructure (#169/#170).

## 1. What's compared
- **proteon:** `backbone_hbonds(s, energy_cutoff=-0.5)` → rows
  `[res_a, res_b, energy, distance]` over amino-acid residues (0-based).
- **DSSP (reference):** mkdssp via Biopython exposes, per residue, its two best
  `NH→O` and `O→NH` partners as `(relidx, energy)`. Build the H-bond set from the
  `NH_O_1/NH_O_2` fields (donor's N–H → acceptor at `position + relidx`) with
  energy < −0.5. mkdssp-only — `gmx dssp` emits SS but **not** H-bond energies,
  so there is no gmx fallback here (the test skips locally, runs in CI).

## 2. The convention wrinkle (measured)
proteon's `[res_a, res_b]` columns are the **opposite** order to DSSP's NH→O
direction: proteon pairs match DSSP at 26/26 only when swapped. An H-bond
between residues i,j is the same physical bond regardless of column, so compare
as **unordered residue-index pairs** (`frozenset{a, b}`). This sidesteps the
labeling difference without hiding anything — a missing or spurious bond still
shows. Measured (proteon vs mkdssp 4.2.2, unordered):

| struct | proteon | dssp | precision | recall |
|--------|---------|------|-----------|--------|
| 1crn   | 25      | 24   | 96.0%     | 100.0% |
| 1ubq   | 42      | 45   | 97.6%     | 91.1%  |
| 1enh   | 36      | 36   | 100.0%    | 100.0% |
| 1ake   | 311     | 294  | 93.6%     | 99.0%  |
| 4hhb   | 457     | 447  | 95.0%     | 97.1%  |

## 3. Index alignment
Both sides are 0-based positional over the same AA residues (mkdssp residue count
== proteon AA count, confirmed in the DSSP oracle: 46/76/54/428/574). DSSP's
`relidx` is in its sequential dssp-index space, which equals the positional
offset, so `acceptor_pos = donor_pos + relidx`. Assert residue-count parity
(coverage) before comparing — a mismatch is a real bug, not a wobble.

## 4. Assertions
- **precision ≥ 0.90** (floor 93.6%) — proteon H-bonds that DSSP also finds.
- **recall ≥ 0.88** (floor 91.1%) — DSSP H-bonds proteon also finds.
- **count parity:** `|n_proteon − n_dssp| / n_dssp ≤ 0.12` (catches systematic
  over/under-detection the set overlap alone could mask).
- **energy agreement on matched pairs:** median `|E_proteon − E_dssp| < 0.5`
  kcal/mol (both Kabsch–Sander; build pair→energy dicts on both sides).
- report the symmetric difference (proteon-only / DSSP-only pairs) in failures.

## 5. Backend reuse
Reuse the DSSP oracle's mkdssp machinery verbatim: binary discovery
(`PROTEON_MKDSSP` → `mkdssp` → `dssp`), the synthetic-`HEADER` prepend (mkdssp 4.x
mmCIF trap), and **present-but-broken ⇒ loud, not skipped** (no gmx fallback
here, so a broken mkdssp `pytest.fail`s). `@pytest.mark.oracle("hbond")`.

## 6. Test set + CI
Same five structures (in-repo after #170). CI already installs `dssp`; the test
runs there and self-skips locally (no mkdssp).

## 7. Validation
Container (ubuntu:24.04 + mkdssp 4.2.2): DSSP H-bond sets dumped for all five and
compared to proteon on the host (the table above) — already done. The shipped
reference-builder will be run verbatim in-container before the PR, same as #169.

## 8. Review log (claudex) — adopted
1. **Align by residue ID, not positional integers** (codex's strongest point):
   map BOTH proteon and DSSP H-bonds to `(chain, resseq, icode)` pairs (the same
   identity mapping the DSSP oracle uses), then compare. Count parity is
   necessary-not-sufficient (insertion codes / missing residues / chain order can
   match counts yet misalign). Re-validated id-based: aggregate **94.8%
   precision / 97.6% recall**; per-structure unchanged. `relidx` is an offset into
   the Biopython DSSP-row order (`acceptor_row = donor_row + relidx`, bounds-
   checked); inter-chain bonds are valid (measured 0 on these fixtures, but the
   id mapping handles them). Failure output carries `chain:resseq:icode`, not ints.
2. **Direction-convention test is NOT feasible — documented instead.** Codex
   suggested pinning proteon's donor/acceptor order. Measured: of 1crn's common
   bonds, **22 are opposite-order, 4 same-order** vs DSSP's NH→O — proteon has no
   stable column convention, and emits ~2 bonds reciprocally (27 rows → 25 unique
   unordered). So unordered-pair comparison is the *correct* oracle, not a
   shortcut; a strict direction assert would just be flaky. The test docstring
   documents this; the comparison dedups via `frozenset`.
3. **Energy:** median `|ΔE| < 0.5` AND **p90 `|ΔE| < 1.0`** (codex: median hides
   tails). Measured: median ≤ 0.42, p90 ≤ 0.70 kcal/mol — both hold with margin.
   Symmetric-difference (proteon-only / DSSP-only) pairs reported in failures.
4. **Tighter count parity ≤ 0.08** (was 0.12; measured max 6.7%) + an
   **aggregate** precision/recall assert across all five (a broad mild
   degradation shouldn't pass when every structure sits just above its floor).
5. **mkdssp-only** (no gmx H-bond energies) ⇒ present-but-broken `pytest.fail`s
   (no fallback). Validated in-container; mkdssp version recorded in the dump.
6. Duplicate-pair note: proteon lists a few bonds reciprocally; the oracle
   compares the `frozenset`-deduplicated set (documented).

## 9. Open questions (resolved during validation)
1. Unordered-pair comparison loses H-bond *direction* — acceptable, or should the
   oracle assert proteon's direction convention separately (once) so a future
   donor/acceptor swap regression is caught?
2. recall ≥ 0.88 vs precision ≥ 0.90 — the recall floor (1ubq 91.1%) is tighter
   to the threshold; is 0.88 the right margin, or per-structure?
3. Energy agreement: median `|Δ| < 0.5` — or a stricter percentile (the cutoff
   itself is −0.5, so near-threshold bonds are where proteon/DSSP disagree on
   membership; their energies there are ~−0.5 by definition)?
4. Count-parity 12% — 1ubq is the loosest (42 vs 45 = 6.7%); is 12% too loose?
