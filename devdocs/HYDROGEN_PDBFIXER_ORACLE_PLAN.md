# HYDROGEN_PDBFIXER_ORACLE_PLAN — second H-placement oracle vs PDBFixer (§ RELIABILITY)

Status: DRAFT (for claudex). proteon already has `test_reduce_hydrogen_oracle.py`
(vs Reduce, the gold standard) — but it is **dormant in CI** (Reduce isn't
apt-installable; CI lists it among "heavier oracles [that] skip"). So proteon's
H-placement has **no oracle running every PR**. This adds a second, independent,
**CI-runnable** oracle vs **PDBFixer** (OpenMM lineage; pip-installable), so the
H-placement path is gated on every PR. Two independent oracles, like pydssp +
mkdssp for DSSP.

## 1. Why this is NOT a tight per-atom-position oracle (measured)
Feasibility was measured in-container (proteon AMBER96 all-atom H vs PDBFixer
`addMissingHydrogens`, identical heavy atoms). Per-atom agreement is
**convention/rotamer-dependent**, not uniformly tight:
- **uniquely-determined H** (single `HA`, aromatic ring H): med **0.12–0.21 Å** — tight.
- backbone amide `H`: method-dependent (bisector vs peptide-plane); med 0.17–0.54, outliers ~1.2 Å.
- methyl / OH / NH3+ / glycine `HA2/HA3`: **rotamer/prochiral noise ~1.5 Å** even when chemically identical.
So a naive by-name position compare mostly measures rotamer labeling. The oracle
is therefore **three layers**, each asserting what's actually robust.

## 2. The three layers (thresholds from measured 1crn/1ubq/1enh)
- **L1 — completeness.** Match H by `(chain, resseq, atom_name)`.
  - **precision** = |proteon ∩ PDBFixer| / |proteon| ≥ **0.95** (measured 99.4–99.8%):
    proteon places no spurious/mislabeled H.
  - **recall** = |∩| / |PDBFixer| ≥ **0.80** (measured 87.7–99.7%): catches gross
    under-placement. The ~12% gap is the protonation-state convention difference
    (PDBFixer's pH-7 charged-group / terminus H proteon doesn't add) — DOCUMENTED,
    not a failure; the 0.80 floor tolerates it but trips if sidechain-H placement
    regresses wholesale.
- **L2 — tight position on uniquely-determined H** (single `HA` + aromatic ring H;
  EXCLUDES glycine `HA2/HA3` (prochiral) and all rotatable/methyl H): median
  < **0.30 Å**, p90 < **0.60 Å** (measured med 0.12–0.21, p90 ≤ 0.49).
- **L3 — loose sanity on all matched H**: every matched proteon H within
  **2.5 Å** of its PDBFixer namesake (measured max 1.95–2.15) — bounds rotamer
  noise, catches catastrophic misplacement.

## 3. Reference construction (PDBFixer)
- `PDBFixer(filename)` → `addMissingHydrogens(7.0)` ONLY (NOT
  `findMissingResidues`/`addMissingAtoms` — keep heavy atoms byte-identical to
  proteon's input so H parents match). Extract H atoms (element 'H') →
  `{(chain,resseq,name): xyz}`, positions nm→Å (×10).
- proteon side: `prepare(s, ff="amber96", hydrogens="all", minimize=False,
  strip_hydrogens=True)` — AMBER96 gives ALL-atom H (CHARMM19 is united-atom /
  polar-only, which would shrink the overlap). Then read H atoms by element.
- Naming: both use PDB v3 / IUPAC (`H`, `HA`, `HB1/2/3`, `HG`, `HD1`…) ⇒ direct
  `(chain,resseq,name)` match; measured coverage ~100% of proteon's H.

## 4. Backend / CI (mkdssp-style)
- pip-installable (`openmm` + `pdbfixer`); skip if absent, **present-but-broken
  ⇒ `pytest.fail`** (no silent CI skip). `@pytest.mark.oracle("pdbfixer")`.
- CI: add `pip install pdbfixer openmm` to the oracle job (openmm wheel ~150 MB —
  acceptable for the oracle job; runs every PR).
- Same five structures as the DSSP/H-bond oracles (single + multi-chain).

## 5. Validation
Container (ubuntu:24.04 + pip openmm/pdbfixer): the shipped reference builder run
verbatim, the comparison + thresholds validated on the host against real PDBFixer
1.12.0 / OpenMM 8.1.1 output (done — the L1/L2/L3 numbers above). Multi-chain
(1ake/4hhb) validated before merge.

## 6. Relationship to the Reduce oracle
Complementary, not redundant: **Reduce** = tight positions on everything with 3
documented convention gaps, gold standard, **off-CI**. **PDBFixer** = completeness
+ rigid-H position + sanity, independent lineage, **every PR**. The completeness
layer (right H atoms placed) is something Reduce parity doesn't directly give.

## 7. Review log (claudex + measurement) — adopted
1. **Restrict to PROTEIN residues (exclude water).** Measurement: the global
   recall gap (88%) was almost ENTIRELY water H — proteon doesn't H-ify HOH by
   default, PDBFixer does (1ubq: 80 of 82 misses = 40 HOH × H1/H2). Excluding
   water, **protein-only precision 99.4–99.8%, recall 99.7–99.8%** — proteon
   misses only 1–2 protein H per structure (a terminus + a His tautomer). So L1
   becomes a clean two-sided gate, and codex's stratified-titratable-recall
   bucketing is unnecessary — water exclusion already isolates the gap.
   - **L1: precision ≥ 0.95 AND recall ≥ 0.95** (both measured ~99.7%); the 1–2
     residual misses (N/C-termini, His tautomer) are documented and within margin.
2. **L2 minimum sample count** (codex): assert the rigid (HA+aromatic) set is
   ≥ 20 atoms per structure — a tight-position layer covering a handful of atoms
   could pass while covering nothing. Measured n = 55–86.
3. **L3 p99 + max** (codex): `p99 < 2.5 Å` AND `max < 3.0 Å` (not max<2.5 — too
   tight to the measured 2.15 for PDBFixer/OpenMM version headroom).
4. **L2 rigid set stays minimal** (codex): single `HA` + aromatic ring H only;
   do NOT expand to topology-inferred CH (maintenance sink / fragile).
5. **Print pdbfixer + openmm versions** into output/failure (codex) — PDBFixer
   behaviour drifts with templates/pH/OpenMM residue defs. In-repo fixtures only
   (the 5 test PDBs), no CI downloads.
6. **Incomplete-heavy residues** (codex): skipping `addMissingAtoms` keeps parents
   identical; residues with missing heavy atoms simply contribute fewer matched H
   (coverage-reported), not a hard fail. The 5 fixtures are complete, so moot here.

## 8. Open questions (resolved during measurement)
1. **L1 recall floor 0.80** — right level? The protonation gap is residue-type
   systematic (Lys/Arg/His/termini). Should recall instead be measured only on
   residue types where proteon and PDBFixer agree on protonation, to tighten it?
2. **L2 rigid set** — single `HA` + aromatic only. Worth adding other uniquely-
   determined H (e.g., `HB` on a CH with 3 distinct heavy neighbours), or keep
   minimal/robust?
3. **pH 7** fixed for PDBFixer. Document that the comparison is pH-7-convention;
   any value in a second pH? (Lean: no — one convention, documented.)
4. **Heavy-atom identity**: skipping `addMissingAtoms` keeps parents identical,
   but if a test structure has missing heavy atoms, PDBFixer may place fewer H
   for that residue. Acceptable (coverage-reported), or pre-filter such residues?
