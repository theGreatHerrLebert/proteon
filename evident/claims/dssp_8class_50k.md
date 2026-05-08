# DSSP 8-class secondary structure: proteon vs mkdssp at 50K corpus

Operational case writeup for the 50K-scale DSSP-vs-mkdssp release-tier
claim in `claims/dssp_8class_50k.yaml`. mkdssp is the wider biology
community's canonical Kabsch-Sander 1983 reference implementation;
proteon ports the same algorithm. Per-residue agreement at 50K scale
is the v0.2.0 trust pyramid's DSSP rung.

## Why this claim is at 50K

The CI claim `proteon-dssp-8class-vs-mkdssp-reference` exercises
per-PDB regression on a small fixture set. The release-tier 1k-class
DSSP claim (`claims/dssp.yaml`) compares against pydssp (a different
reimplementation). Neither covers the question "do proteon and the
canonical mkdssp agree across the production-relevant population?"

This 50K claim closes that. Same protein-only random sample the
CHARMM and AMBER96 50K oracles use (seed=42 reproducibility), so
the population is comparable across release-tier capabilities.

## Approach

1. **Source corpus**: the same `validation/protein_only_50k.txt`
   pre-filtered list the CHARMM 50K oracle uses (44,210 protein-only
   PDBs filtered from a 50,000-PDB random wwPDB sample).
2. **Per-PDB**: load via proteon → call `proteon.dssp` (8-class) on
   one side; spawn mkdssp via subprocess on the other side; parse
   mkdssp's text output for the per-residue SS column.
3. **Compare**: per-residue match rate. Loop is canonicalised to '-'
   on both sides (mkdssp emits ' ', proteon emits 'C') so the
   space-vs-character convention isn't a confounder.
4. **Skip path**: per-PDB length mismatch (proteon and mkdssp count
   residues differently when chain-break records are present)
   recorded as 'skipped', not 'error', so the JSONL distinguishes
   genuine SS disagreements from prep-side conventions.
5. **Pebble per-task isolation**: a single pathological PDB that
   segfaults mkdssp gets cleaned up as a per-task error rather than
   cascading the pool. Same pattern as the CHARMM oracle.

## Outcome

PENDING — first 50K run scheduled. Headline numbers (n_attempted,
n_ok, median agreement_rate, per-class composition deltas) will be
filled in at first lock.

Empirically the two implementations agree at 0.97-1.00 on canonical
structures. The 0.95 median tolerance leaves comfortable room for
boundary wobble (helix/strand boundary residues where H-bond energies
cross the -0.5 kcal/mol threshold and small geometry differences flip
the call) without letting a systematic regression slip through.

## What this claim isn't

- **Not a per-PDB regression test.** The CI claim
  `proteon-dssp-8class-vs-mkdssp-reference` covers that.
- **Not a parity claim against pydssp.** That's `claims/dssp.yaml` —
  different oracle (Python reimplementation by a separate group),
  different scope (1k-class collapsed to H/E/-).
- **Not a 3-class claim.** This compares full 8-class output. A 3-class
  collapse would hide the H↔G↔I helix-flavor agreement and
  E↔B strand/bridge boundary that the 8-class metric exposes.
