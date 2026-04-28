# Forcefield / MD

Energy evaluation, minimization, and short MD trajectories. Three force
fields are implemented:

- **CHARMM19 + EEF1** (default) — implicit solvent, fast, the workhorse for
  preparation / minimization.
- **AMBER96** — explicit-style; validated to ≤0.5% on all components vs OpenMM
  at NoCutoff (218/218 invariants pass).
- **OBC GB (Phase B)** — generalized Born implicit solvent, CPU + GPU paths.
  Validated to ≤5% GB / ≤1% total vs OpenMM on crambin; GPU matches CPU to
  1e-11.

GPU paths use CUDA kernels (feature `cuda`) and silently fall back to CPU when
no usable device is detected.

## Example

```python
import proteon

s = proteon.load("1crn.pdb")
proteon.place_all_hydrogens(s)   # backbone + sidechain, modifies in place

energy = proteon.compute_energy(s)
# energy is a dict with: total, bond_stretch, angle_bend, torsion,
# improper_torsion, vdw, electrostatic, solvation (+ topology counts)
print(energy["total"], energy["vdw"], energy["electrostatic"])

minimized = proteon.minimize_hydrogens(s)
```

For backbone-only H placement use `place_peptide_hydrogens(s)`; for ligands
or non-standard residues use `place_general_hydrogens(s)`.

## GPU availability

```python
proteon.gpu_available()   # bool
proteon.gpu_info()        # dict with device name, memory, compute capability
```

## API reference

::: proteon.forcefield
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
