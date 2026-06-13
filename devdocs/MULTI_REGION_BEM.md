# Multi-region BEM — cavities by orientation (codex-corrected)

**Status:** design (corrected after codex round 1). The original per-element-`frac` plan
below §6 is **superseded** — see this banner.

## CORRECTION (codex round 1) — cavities need orientation, not per-element ε

The per-element `frac_i = ε_in/ε_out` generalization (original §1) is **wrong** for the
ordinary cavity case and is not a valid general multi-region operator either. The real
picture:

- **Buried solvent cavities (same εΩ, same εΣ) need NO formula change.** Orient the whole
  solute boundary *outward-from-solute*: the outer molecular surface points into exterior
  solvent (geometrically outward, +volume); a cavity surface points into the pocket's
  solvent (geometrically **inward**, **−volume**). Then *every* panel has solute on the
  interior side and solvent on the exterior side, so `f = εΩ/εΣ` is **uniform** — cavity
  panels included. The existing scalar §5 solve is already correct; only the orientation
  must be right.
- **P6.5's `orient_outward` (every component → +volume) is therefore WRONG for cavities.**
  Inner boundary components of a multiply-connected solute domain MUST be negatively
  oriented (outward-from-solute = geometrically inward). Correct orientation is by
  **nesting parity**: a component at nesting depth `d` (number of components containing
  it) gets signed-volume sign `(−1)^d` (top-level body +, cavity −, island +, …). The
  current code only avoids producing wrong results because it *refuses* cavities; the fix
  is to orient-by-nesting and handle them.
- **Genuinely different dielectrics per body / per region are a separate, harder problem**
  — a real multi-domain (region-incidence) derivation, NOT per-row scaling. Out of scope
  here. This work covers same-εΩ/εΣ cavities + multi-body only.
- **Source terms / stage 2 are unchanged** (all solute regions share εΩ): `umol = Σ q/(εΩr)`,
  `qmol = ∂n umol`, `b2 = (2π+K)u`, `potprefactor` universal. A charge in body A seen by
  body B's surface is an ordinary harmonic cross contribution, not a new kernel.

### Corrected plan

- **Layer 1 (this session):** per-component **nesting depth**; `orient_by_nesting` (sign
  `(−1)^d`) replacing the all-outward flip for multiply-connected meshes; **lift the
  cavity refusal**; gate a cavity BEM solve against the **l=0 concentric-shell** analytic.
- **Layer 2+:** off-center Kirkwood-series gate (nonconstant surface data, cross-interface
  coupling); genuinely-different-dielectric multi-domain derivation (separate effort).

### Analytic oracle (codex) — l=0 concentric shells

A central charge `q` in concentric dielectric shells (interface radii `r_k`, dielectric
`ε` just inside/outside each `r_k`) has reaction potential at the centre

```
φ_reac(0) = q · Σ_k ( 1/ε_out(k) − 1/ε_in(k) ) / r_k
```

(the single-interface case is exactly Born). Energy `W* = ½ q · φ_reac · prefactor`. Test
geometry: **three concentric spheres** `a<b<c` — solute island (`r<a`, εΩ, central
charge), solvent cavity (`a<r<b`, εΣ), solute body (`b<r<c`, εΩ), exterior solvent
(`r>c`, εΣ). This exercises a full 3-level nest (island/cavity/body) and the orientation
`(−1)^d` (body +, cavity −, island +) in one shot, with no series transcription.

---

_(Original design below, superseded by the correction above. Kept for the record.)_

# Multi-region BEM — per-element dielectric (original, superseded)

**Status:** design (pre-implementation). Reviewed by: _pending codex_.

The single-region local/nonlocal solve (`ELECTROSTATICS_FORMULATION.md` §5/§6) assumes
one solute region Ω (εΩ) inside one closed surface Γ, one solvent Σ (εΣ) outside. Spec
§10 leaves **multiple components and buried cavities open**. This designs the extension.

## 0. What already works, and what doesn't

- **Multiple *separate* solute bodies, same dielectric** — already correct. Every element
  has solute (εΩ) inside, solvent (εΣ) outside, so the scalar `frac = εΩ/εΣ` is uniform
  and the existing single-system solve treats all of Γ together. (The P6.5 acceptance
  gate validates this case; cavities are currently refused.)
- **Buried cavities** (a solvent pocket inside solute) — NOT modelled. A cavity surface
  has its dielectrics **swapped**: solvent (εΣ) *inside* the pocket, solute (εΩ)
  *outside*. The transmission condition — hence the BEM coefficient — is the inverse
  ratio there. The scalar `frac` cannot express that.
- **Different dielectrics per body** — also unsupported (rare for proteins; same physics
  as cavities: a per-region dielectric).

## 1. The generalization: per-element dielectric ratio

Each closed component separates an *inside* region from an *outside* region; each region
carries a dielectric. For element `i` define

```
frac_i = ε_in(i) / ε_out(i)
```

where `ε_in`/`ε_out` are the dielectrics immediately inside/outside the component element
`i` belongs to. The single-region case is `frac_i ≡ εΩ/εΣ`.

**Local system (§5) with per-element frac.** The scalar `frac` enters the local operator
and RHS as a row-`i` (observation-element) coefficient, so the generalization replaces it
with `frac_i`:

```
M_ij = 2π(1 + frac_i)·δ_ij  +  (frac_i − 1)·K_ij           (was scalar frac)
b_i  = (K·umol)_i − 2π·umol_i − frac_i·(V·qmol)_i
```

`K`/`V` are pure geometry (unchanged). Claim to verify: this is the standard
piecewise-constant-dielectric collocation BEM — each interface carries its own
transmission-condition ratio — and it reduces **exactly** to §5 when `frac_i` is uniform.

### 1.1 Region → dielectric, by nesting parity

Regions alternate solvent/solute by nesting depth (the P6.5 winding machinery already
computes per-component nesting):

```
depth 0  (outside everything)     → solvent  εΣ
depth 1  (inside the outer shell) → solute   εΩ
depth 2  (inside a cavity)        → solvent  εΣ
depth 3  (island in the cavity)   → solute   εΩ
…              even → εΣ,  odd → εΩ
```

A component at nesting depth `d` separates region `d` (outside it) from region `d+1`
(inside it). So `ε_out = ε(parity d)`, `ε_in = ε(parity d+1)`, and `frac_i` follows from
the depth of element `i`'s component. (Different-dielectric *bodies* are the same
machinery with a per-region dielectric table instead of the εΩ/εΣ alternation.)

### 1.2 Source terms across regions

`umol`/`qmol` are the molecular-potential traces from the point charges. A charge sits in
a solute region; its Coulomb field is `1/ε_source` where `ε_source` is that region's
dielectric (εΩ). With cavities, a charge could be in any solute region (outer body or an
island); the trace each element sees still sums `q_c / |ξ−r_c|` — **but the `1/εΩ`
prefactor and the energy `potprefactor` must use the dielectric of the charge's own
region**, which is εΩ for any solute region (so unchanged when all solute regions share
εΩ; it matters only for different-dielectric bodies). *Open question for review: do
cross-region source contributions (a charge in body A seen by body B's surface) need a
region-aware Green's function, or does the uniform-εΩ assumption keep `umol` as-is?*

## 2. Layered plan

- **Layer 1 (this session): per-element-`frac` infrastructure, provably equivalent.**
  Generalize `LocalOperator` and `solve_local_elements` to take `frac: &[f64]` (length N).
  Keep a scalar-taking wrapper. Gate: a uniform `frac` reproduces the current scalar solve
  **bit-for-bit** (no physics change, pure refactor). This de-risks the plumbing before
  any cavity physics.
- **Layer 2: region determination + the cavity gate.** Compute per-element `frac_i` from
  nesting depth (§1.1); stop refusing cavities; gate the cavity solve against an analytic
  **concentric-sphere** model (§3). The real physics step.
- **Layer 3: nonlocal (§6) per-element ε, post-processing/energy per region.**

## 3. Oracle / gating (NESSie is single-component — no help)

- **Concentric-sphere analytic model.** A point charge in a layered dielectric sphere
  (solute shell + concentric solvent cavity, or two-dielectric sphere) has a closed-form
  reaction-field energy via the Kirkwood multipole series. Implement it (high-precision,
  independent of the BEM), gate the cavity BEM energy against it on analytic concentric
  meshes — the L4-style science gate, exactly as Born gated the single-region solve.
- **Uniform-frac equivalence** (Layer 1) is the cheap, exact foundation gate.
- **Reciprocity / self-consistency** invariants where an analytic form is unavailable.

## 4. Scope / non-goals

- **In (eventually):** per-element dielectric local + nonlocal solve; nesting-parity
  region assignment; concentric-sphere analytic gate; lift the cavity refusal.
- **This session:** Layer 1 only (the equivalent refactor + its gate).
- **Out:** non-spherical analytic oracles (none exist — rely on concentric-sphere +
  invariants); per-charge *different* εΩ within one body; ionic-strength region effects.

## 5. Open questions for review

1. **Is per-element `frac_i` (row-`i` coefficient) the correct generalization of the §5
   operator/RHS to piecewise-constant dielectrics, reducing exactly to §5 when uniform?**
   Or does a cavity require more than swapping the ratio (e.g. an off-diagonal
   region-coupling term, or a sign change in the double-layer beyond `K`'s geometry)?
2. **Source terms (§1.2):** with all solute regions at εΩ, does `umol`/`qmol` stay as-is,
   or does a cross-region charge contribution need a region-aware kernel?
3. **Orientation convention:** §5 assumes outward normals (P6.5 auto-flips to outward).
   For a cavity shell, "outward" (toward solvent in the pocket) points *inward*
   geometrically. Does `frac_i = ε_in/ε_out` with a consistently-outward (per the P6.5
   per-component orient) mesh get the double-layer sign right, or must cavity shells keep
   their physical (inward-geometric) normal?  **This is the subtlety most likely to bite.**
4. **Stage-2 (`V·q = b2`):** does per-element `frac` change stage 2, or only stage 1 / the
   energy? (`V` is geometry-only; `b2 = (2π+K)·u` has no `frac` — so stage 2 is unchanged?)
5. **Is the concentric-sphere Kirkwood series the right analytic oracle, and is it
   independent enough (re-deriving it in Rust re-introduces transcription risk)?**
