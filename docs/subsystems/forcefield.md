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
s = proteon.add_hydrogens(s, mode="all")

energy = proteon.compute_energy(s)
print(energy.total, energy.bonded, energy.vdw, energy.electrostatic)

minimized = proteon.minimize_hydrogens(s)
```

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
