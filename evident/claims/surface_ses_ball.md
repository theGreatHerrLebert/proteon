# SES surface area: proteon vs BALL on crambin

Operational case writeup for the SES-vs-BALL CI claim in
`claims/surface_ses_ball.yaml`.

## Problem

proteon meshes the analytic solvent-excluded (Connolly) surface in
`proteon-core/src/surface/` and exposes it to Python through the
connector's `py_surface.ses_mesh_py`, returning a triangulated mesh
plus its area, volume, and watertightness. This surface drives
downstream analysis (cavity/pocket detection, surface-area features,
visualisation) and the area is a load-bearing scalar.

The mesher is a clean-room port of BALL's reduced-surface →
solvent-excluded-surface pipeline, and until now its only external
check was a hand-run "+0.05% vs BALL area" measurement on a couple of
inputs. The risk surface is real: the analytic SES is a stitch of
spheric, toric, and contact-cap patches whose boundaries must weld
watertight; a regression in patch geometry, the arrangement walk, or
the contact-cap CDT fill changes the area without necessarily crashing.
An independent-implementation oracle is the cheapest insurance.

## Trust Strategy

Validation. BALL's `calculateSESArea`/`calculateSESVolume`
(`analyticalSES.h`) is an independent C++ implementation of the same
Connolly construction — and, importantly, its *analytic* path is robust
to the near-tangency degeneracies that defeat its own *triangulation*
(`ball.ses_mesh` raises `DivisionByZero` on e.g. 1ijp). So
`ball.ses_area` is the stable reference; disagreement between it and
proteon's mesh area surfaces a real geometry bug in either pipeline.

## Inputs

Crambin (`test-pdbs/1crn.pdb`), 327 heavy atoms, single chain — the
surface pipeline's clean repro case (meshes on the analytic path with
no perturbation). The test takes proteon's atom coordinates and assigns
a fixed per-element vdW radius table (C 1.70, N 1.55, O 1.52, S 1.80,
H 1.20, default 1.70 Å), then feeds the **same** sphere set to both
sides: `ses_mesh_py(coords, radii=…)` and `ball.ses_area([[x,y,z,r]…])`.
Identical inputs ⇒ the comparison isolates the meshing/area algorithm,
not radius assignment (which the SASA oracle covers separately).

## What is checked

- **SES area** agrees within **0.5 %** relative. Measured ~0.037 % on
  crambin (proteon 2319.87 Å² vs BALL 2319.02 Å²); the 0.5 % band gates
  an algorithmic regression without tripping on the mesh-vs-analytic
  discretisation residual (≤0.32 % even on a degenerate, perturbation-
  recovered 22-protein subset).
- **SES volume** within **1.0 %** relative (looser — a mesh's enclosed
  volume integrates per-facet discretisation error). Measured ~0.043 %.
- **Watertight == true** and **method == "analytic"**. The precondition
  that makes the area well-defined: a non-closed mesh, or a silent grid
  fallback, would compare a different surface, so it gates the claim.

## Provenance & pins

Automatic (pytest, CI tier). `ball.ses_area` ships in **ball-py
0.1.0a6** (the SES bindings); the proteon side needs the connector built
from the surface branch (`py_surface.ses_mesh_py`). Both sides skip
cleanly when absent, so the suite stays green on older pins until the
CI `ball-py` pin is bumped to 0.1.0a6.

## Evidence

```
pytest tests/oracle/test_ses_ball_oracle.py::TestSesBallOracle -v
```
