# Local energy + potentials, and Born convergence (P5)

Operational case writeup for the local post-processing claim in
`claims/electrostatics_post.yaml`.

## Problem

P5 turns the solved surface Cauchy data into the scientific outputs: the
reaction-field energy `W*` (kJ/mol) and the electrostatic potential at points in
the interior (Ω), on the surface (Γ), and in the solvent (Σ)
(`proteon-electrostatics/src/{post,analytic}.rs`, ported from NESSie
`src/bem/post.jl` + `src/testmodel/born/`).

This is the first layer whose output is a *physical answer a user wants* rather than
an intermediate. It is also where every prior layer is exercised at once, with the
unit constants (`potprefactor = ec/4π/ε0`, `energy_factor = ec·Nₐ·1e-3/2`) finally
applied — a wrong constant or a wrong domain formula shows up here as a wrong energy.

## Trust Strategy

Validation, three angles of increasing independence:

1. **Born closed form vs NESSie.** The Born ion has an exact reaction-field energy
   (no BEM, no mesh). proteon's `born_rfenergy` (local + nonlocal) is gated against
   NESSie's `rfenergy(LocalES|NonlocalES, ion)` for all nine built-in ions. Same
   algebra, so this is a tight transcription check (1e-12).

2. **NESSie `post_dump` parity.** `rfenergy` and `espotential` over the Ω/Σ/Γ sample
   sets, on the 512-element Born sodium mesh, vs NESSie. `post_dump` omits `u`/`q`
   (it solves the `:blas` path internally), so proteon solves the local system first.

3. **The science gate — BEM → Born convergence.** The strongest, NESSie-independent
   result: solve the *local BEM* on a triangulated sphere with a central unit charge
   and read off `rfenergy`; it must converge to the closed-form Born energy as the
   icosphere refines. This runs the whole local stack (collocation → assembly →
   GMRES → post) against **analytic physics**, not against another implementation.

## Inputs

- `analytic.json` — NESSie's closed-form Born energies (nine ions, local + nonlocal).
- `post_local_na.json` — NESSie's `rfenergy` + `espotential` (Ω/Σ/Γ) on the Born
  sodium mesh. The test consumes its normals/charges/params verbatim and solves for
  `u`/`q`.
- The convergence test builds its own meshes: `icosphere(R=2 Å)` at subdivisions 2
  (320 triangles) and 3 (1280), unit charge at the centre.

## Tolerances

- Born closed form vs NESSie: **< 1e-12** (local + nonlocal, nine ions).
- `post_dump` parity: **< 1e-6** (rfenergy + Ω/Σ/Γ potentials).
- BEM → Born: **< 3%** at subdivision 3, strictly below the subdivision-2 error
  (refinement moves toward the analytic value), energy negative throughout. This is a
  *discretisation* band, not a parity tolerance — the gate asserts convergence, the
  load-bearing scientific claim.

## What this does NOT cover

- **Convergence rate.** Two subdivisions show the trend and a fine-mesh error, not a
  fitted order of convergence in a stated norm — the natural strengthening.
- **Charge geometry.** One central charge on a sphere. Off-centre / multiple charges,
  and the near-surface potential (where the jump-corrected Γ trace matters most), are
  only lightly covered; the Σ/Γ sample sets are scaled copies of the centroids.
- **An independent Born oracle.** The closed form is gated only against NESSie (same
  algebra); a CAS/mpmath high-precision Born fixture is the owed "oracle for the
  oracle".
- **Arbitrary / protein-scale meshes.** The 3% band is specific to this radius and
  subdivision range; general meshes need the P6.5 mesh-acceptance work.

## How to run

```bash
cargo test -p proteon-electrostatics      # lib + all parity/convergence test files
```

No Julia at gate time; `nalgebra` (the LU oracle used elsewhere) is a test-only
dev-dependency.
