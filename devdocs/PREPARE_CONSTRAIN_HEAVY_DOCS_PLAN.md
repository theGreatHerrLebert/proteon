# Plan: honest "when to use which `constrain_heavy`" docs

## Goal

After unifying `constrain_heavy` to an H-only default (PR-G), users need a
clear, honest decision guide for when to override it. The choice interacts with
three other axes, which is what makes it convoluted:

1. **What you'll do with the structure** — faithful analysis vs energy/MD.
2. **`reconstruct`** — if missing heavy atoms were added, H-only leaves them at
   their template guess (unrelaxed); a clean structure wants heavy relaxation.
3. **Force field** — AMBER96 (vacuum, all-atom) vs CHARMM19+EEF1 (implicit
   solvent). Full all-atom minimization is physical for CHARMM+EEF1 but
   *distorting* for AMBER in vacuum (unscreened electrostatics collapse the
   structure), which is the whole reason `None` is FF-aware.

## The three settings (recap)

- `constrain_heavy=True` (DEFAULT) — H-only. Freeze heavy atoms, relax only H.
  Preserves experimental coordinates exactly (CA-RMSD 0). `heavy_relaxed=False`.
- `constrain_heavy=False` — heavy relax. Move all atoms. Deeper energy minimum,
  removes clashes, settles reconstructed atoms; moves backbone ~0.5 Å.
  `heavy_relaxed=True`.
- `constrain_heavy=None` — FF-aware: heavy-relax for CHARMM19+EEF1, H-only for
  AMBER96.

## Proposed decision tree (to be reviewed for physics correctness)

```
What do you need from prepare()?

1. A faithful protonated structure for ANALYSIS
   (alignment, SASA, DSSP, contacts, ML features, supervision)
   → constrain_heavy=True   (the DEFAULT)
     Heavy atoms keep their deposited coordinates; only H are placed/relaxed.
     ⚠ If reconstruct=True added missing heavy atoms, those sit at their
        template positions (unrelaxed) and may clash. Usually fine for
        geometric analysis; if the rebuilt regions matter, prefer (2).

2. An energy-minimized / MD-ready structure
   (compute_energy, minimize, MD, anything that trusts final_energy)
   → CHARMM19+EEF1 : constrain_heavy=False   (heavy relax; implicit solvent
                     makes full minimization physical → deeper, clash-free min)
   → AMBER96       : constrain_heavy=True (H-only) — do NOT full-relax in
                     vacuum (unscreened electrostatics distort geometry); add
                     solvent/restraints in a real MD engine for production.
   → Don't want to think about it? constrain_heavy=None (FF-aware) picks the
     above per force field automatically.

3. Reconstructed a lot of missing heavy atoms and want them settled
   → constrain_heavy=False on CHARMM19+EEF1 (relaxes the rebuilt atoms into
     place). On AMBER96 prefer an external solvated minimization.
```

## Honesty signals to reference in the docs

- `report.heavy_relaxed` — did heavy atoms actually move? False ⇒ `final_energy`
  is NOT a heavy-atom minimum (carries crystal strain; rebuilt atoms/clashes
  unrelaxed). Gate energy trust on this.
- `report.converged` — did the minimizer reach the gradient tolerance.
- `report.ready` / `report.fully_typed` — usability / full FF coverage.

## Deliverables

1. Expand the `constrain_heavy` docstring in `prepare()` with a compact version
   of the tree (the authoritative inline reference).
2. A prose section + the tree in `docs/subsystems/prepare.md` ("Choosing how
   much to minimize"), cross-linked from the quickstart.
3. Keep it honest: state plainly that the default does NOT equilibrate the
   structure, and that reconstruct + H-only leaves rebuilt atoms unrelaxed.

## Open questions for claudex

1. Is the AMBER96-in-vacuum claim correct and correctly scoped — is H-only (or
   None) really the right guidance for AMBER, and is "full relax distorts in
   vacuum" accurate, or is it more nuanced (e.g. cutoff/dielectric settings)?
2. Is the reconstruct⇒heavy-relax guidance right, or are template-placed heavy
   atoms good enough that H-only is fine even after reconstruct?
3. Is there a fourth axis worth surfacing (e.g. NMR ensembles, membrane
   proteins, structures with many clashes) where the tree misleads?
4. Should the default itself be reconsidered when reconstruct adds many atoms
   (i.e. auto-switch to heavy relax), or is a documented warning the right call
   (keep behaviour predictable, don't surprise)?
```
