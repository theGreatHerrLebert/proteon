# AMBER96 fold preservation: proteon vs GROMACS

Operational case writeup for the release-tier
`proteon-amber96-fold-preservation-vs-gromacs-release-1k-pdbs` claim
(issue #37).

## Why a third oracle

Fold preservation under AMBER96 minimization is already checked against
OpenMM (`fold_preservation_amber`). GROMACS adds a **third, independent C
lineage** for the same parameter set — the strongest form of the
"triangulate with two oracles" rule: a shared bug in proteon and OpenMM
that both miss is unlikely to also exist in GROMACS, which has its own
energy code, minimizer (L-BFGS), and topology builder (`pdb2gmx`).

## Method

Each arm minimizes the same seeded 1k sample in vacuum and reports the
TM-score between the input CA trace and the post-minimization CA trace.
The metric is the per-PDB difference `gromacs.tm_score - proteon.tm_score`;
the claim bounds its median magnitude. This is a **relative agreement
between two minimizers**, not an absolute fold-preservation floor — both
could be wrong in the same direction and still agree.

GROMACS chain per structure: `pdb2gmx -ff amber96 -water none -ignh` →
`editconf` (30 Å vacuum box) → `grompp` (L-BFGS EM, 100 steps, 14 Å
cutoffs) → `mdrun -nt 1` → CA extract → `proteon.tm_score`.

## Population scoping

`pdb2gmx` accepts only complete, standard protein residues. Structures
with missing heavy atoms, incomplete rings, nonstandard residues, or
nucleic acid are **skipped** (out of population), matching the OpenMM
arm's skip-on-missing behaviour rather than running `addMissingAtoms`
(which hangs deterministically, PR #47). The comparison surface is
"well-resolved standard protein", consistent with the other fold and
corpus claims. NMR multi-model ensembles surface as a post-extraction
`CA shape mismatch` error and are likewise outside the single-structure
comparison surface.

Coverage (n_ok / n_attempted) is **documented, not gated** — it measures
how many structures both minimizers complete cleanly, not whether they
agree where both do.

## Runners

- proteon: `validation/tm_fold_preservation_amber.py`
- GROMACS: `validation/tm_fold_preservation_gromacs.py`
- join:    `validation/fold_preservation/join_gromacs_pair.py`
