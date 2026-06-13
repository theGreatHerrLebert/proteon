# Electrostatics formulation & convention spec (P0.5)

The load-bearing deliverable of `TO_ELECTROSTATICS.md` §1b. It transcribes NESSie's
actual boundary-integral systems **algebraically**, from source, so the Rust port in
`proteon-electrostatics` reproduces them exactly — not by inference during coding. A
single convention bug (a sign, a `4π`, a dielectric factor) can hide behind close
NESSie parity while the Born energy compensates; pinning the equations here and
unit-testing them as a spec is what prevents that.

All references are to NESSie.jl (MIT, Thomas Kemmer); paths below are relative to the
NESSie checkout (`../NESSie.jl/`). Where a number is given it is copied from source,
not re-derived. **`§9` is the P0.5 acceptance checklist** — the spec is "done" when
those gates are green.

> Status: DRAFT. Independently reviewed by Codex (2026-06-09,
> `archive/ELECTROSTATICS_FORMULATION.codex-review.md`) — two critical transcription fixes
> applied (the nonlocal matrix-diagonal vs preconditioner-vector split in `§6`; the
> implicit-RHS dimensional caveat), the `§10` local-limit claim corrected, the unit
> chain resolved, and the nonlocal `:Σ` potential written out. **Still owed: an
> entrywise check against a running NESSie** on a 2–4 element mesh + the unit-chain
> test (`§9`). Until then, treat `assembly_kernels_dump` as kernel-block parity only,
> and do not ship energies.

---

## 1. Discretization

NESSie uses **piecewise-constant boundary elements with point collocation** — *not*
Galerkin:

- **Trial space:** one constant unknown per flat triangle.
- **Collocation (test):** Dirac functionals at the **triangle centroids**. The
  observation set is `Ξ = [e.center for e in model.elements]`
  (`bem/local.jl:87`, `bem/nonlocal.jl:78`). Every system row is "evaluate the
  integral equation at centroid `ξ_i`".
- **Unknowns**, each a length-`numelem` vector over elements in mesh order:
  - local: `u = γ₀int(φ*)`, `q = γ₁int(φ*)` (`bem/local.jl:5–8`).
  - nonlocal: adds `w = γ₀ext(Ψ)` (`bem/nonlocal.jl:5–9`).
  - All stored **premultiplied by `4π·ε0`** (deferred to post-processing; see `§7`).
- **Block/unknown ordering** in the nonlocal system vector: `[u; q; w]`, i.e. block
  `b` occupies indices `b·numelem .. (b+1)·numelem` (`bem/nonlocal.jl:44–52,
  145–147`). proteon's `BlockLayout` must match this exactly.

Charges are assigned to the solute interior; the source terms (`§4`) treat them as a
structureless Coulomb field divided by `εΩ`. Net charge, multiple components, and
buried cavities are **open** (`§10`).

---

## 2. Constants & units

From `base/constants.jl`:

| Symbol | Definition | Source |
|---|---|---|
| `σ` | `0.5` — solid-angle jump fraction for an a.e.-smooth surface | `:42` |
| `ε0` | `1 / (4π · 1e-7 · 299792458²)` (vacuum permittivity, F/m) | `:29` |
| `ec` | `1.602176e-9` = **10¹⁰ × elementary charge** (the `Å→m` factor `1e10` is folded in) | `:51` |
| `potprefactor(T)` | `ec / (4π · ε0)` ≈ `1.145 · 4π` | `:71` |
| `yukawa(opt)` | `√(εΣ/ε∞) / λ` = `1/Λ`, with `Λ = λ·√(ε∞/εΣ)` | `:151` |
| `defaultopt` | `εΩ=2, εΣ=78, ε∞=1.8, λ=20 Å` | `:131` |

**`σ` appears as `4π·σ = 2π`** everywhere, because every operator value is
premultiplied by `4π` (next section). So a diagonal "jump" term reads `2π`, not `0.5`.

---

## 3. Operators (collocation matrices) — all premultiplied by 4π

Four `numelem × numelem` matrices, each entry = the kernel integrated over element `j`
evaluated at centroid `ξ_i`. **Every value carries an implicit `4π` factor** (removed
once, at the end of post-processing).

| Matrix | Kernel | NESSie call |
|---|---|---|
| `V`  | single-layer Laplace | `Rjasanow.laplacecoll!(SingleLayer, …)` |
| `K`  | double-layer Laplace | `Rjasanow.laplacecoll!(DoubleLayer, …)` |
| `Vy` | **regular** single-layer Yukawa, `Vʸ − V` | `Radon.regularyukawacoll!(SingleLayer, …, yuk)` |
| `Ky` | **regular** double-layer Yukawa, `Kʸ − K` | `Radon.regularyukawacoll!(DoubleLayer, …, yuk)` |

**Critical naming caveat:** in NESSie's implicit `NonlocalSystemMatrix` the fields
named `Vy`/`Ky` hold the **regular** parts `Vʸ−V` / `Kʸ−K`, *not* the full Yukawa
operators — confirmed by matching the implicit `*` (`bem/nonlocal.jl:216–243`) to the
explicit assembly (`:80–138`), where the same quantities come from
`regularyukawacoll!`. proteon must use the regular parts. The full Yukawa operator, if
ever needed, is `Vʸ = Vy + V`.

**Normal orientation:** outward from the solute Ω (proteon must normalize/reject
inward meshes — orientation reversal is an operator-sign test, not a physical
invariant; `TO_ELECTROSTATICS.md` §4).

**Self / jump terms (hand-checkable):**
- Double-layer Laplace `K` **InPlane self term is exactly 0** (`Rjasanow.jl`,
  `_laplacepot(DoubleLayer, InPlane, …) = 0`). Hence `diag(K) = 0` for flat panels —
  this is why the implicit nonlocal `diag` carries only `2π − diag(Ky)` and `2π`
  (`bem/nonlocal.jl:210–214`), never a `diag(K)` term.
- The `2π` ( = `4π·σ`) diagonal is the half-jump; it is added explicitly to the
  system, not produced by the kernel.
- Single-layer `V` InPlane self term is the finite Rjasanow closed form (nonzero) —
  the per-element analytic integral over the element's own plane.

---

## 4. Source terms (RHS building blocks)

Observation at centroids. Both stored values are the raw geometric sums **divided by
`εΩ`** (`bem/local.jl:91–92`, `bem/nonlocal.jl:63–64`); the comment "initially
premultiplied by `4π·ε0·εΩ`" is unit bookkeeping — the numeric stored is the bare sum
over charges / `εΩ`, and the `4π·ε0` is applied at the end.

- **`umol`** = `(1/εΩ) · Σ_q  q.val / max(|ξ − q.pos|, tol)`
  (`base/potentials.jl:94–99`, `_molpotential`). γ₀ trace of the molecular potential.
- **`qmol`** = `(1/εΩ) · ( −Σ_q  q.val · (ξ.center − q.pos)·n / max(|ξ.center − q.pos|³, tol) )`
  (`base/potentials.jl:146–156`, `_molpotential_dn`; `n` = element outward normal).
  γ₁ trace (normal derivative). **Note the leading minus sign** inside `_molpotential_dn`.
- `tol = _etol(T)` for the implicit path (`1.45e-8` f64).

---

## 5. Local system (2-block) — `LocalES`

Canonical (implicit) form, `bem/local.jl:211–237`. Solved in **two stages**.

**Stage 1 — solve for `u`:**

```
M_u · u = b_u
M_u · x = 2π(1 + εΩ/εΣ)·x  +  (εΩ/εΣ − 1)·(K·x)        (bem/local.jl:189–195)
diag(M_u) = 2π(1 + εΩ/εΣ)                               (bem/local.jl:173–179)
b_u = K·umol  −  2π·umol  −  (εΩ/εΣ)·(V·qmol)           (bem/local.jl:228)
```

**Stage 2 — solve for `q`:**

```
V · q = b_q
b_q = 2π·u + K·u  =  (2π·I + K)·u                       (bem/local.jl:233)
⇒ q = V⁻¹ (2π·I + K) u                                  (bem/local.jl:234)
```

(The explicit `:blas` assembly `:82–151` is algebraically identical — useful as a
cross-check but the implicit form is what proteon ports.)

---

## 6. Nonlocal system (3-block) — `NonlocalES`

Implicit operator `A·[x1; x2; x3]` with `x1=u, x2=q, x3=w`, `bem/nonlocal.jl:216–243`.
With `V,K` = Laplace single/double and `Vy = Vʸ−V`, `Ky = Kʸ−K` (regular Yukawa):

```
row1 (u-block):
  2π·x1  −  K·x1  −  Ky·x1  +  (ε∞/εΣ)·(Ky·x3)
        +  (εΩ/ε∞ − εΩ/εΣ)·(Vy·x2)  +  (εΩ/ε∞)·(V·x2)

row2 (q-block):
  2π·x1  +  K·x1  −  V·x2

row3 (w-block):
  2π·x3  −  K·x3  +  (εΩ/ε∞)·(V·x2)
```

Equivalent 9-block matrix (explicit assembly `bem/nonlocal.jl:44–138`; `m23=m31=0`):

```
        | x1 (u)                  | x2 (q)                         | x3 (w)        |
  row1  | 2π·I − K − Ky           | (εΩ/ε∞ − εΩ/εΣ)·Vy + (εΩ/ε∞)·V | (ε∞/εΣ)·Ky    |
  row2  | 2π·I + K                | −V                              | 0             |
  row3  | 0                       | (εΩ/ε∞)·V                       | 2π·I − K      |
```

**⚠ Two different "diagonals" — do not conflate (Codex review, critical).** The
*algebraic* matrix diagonal follows from the blocks above: block (2,2) is `−V`, so
the middle block diagonal is **`−diag(V)`**. But NESSie's `diag(A)` method
(`bem/nonlocal.jl:210–214`) returns

```
diag_nessie(A) = [ 2π − diag(Ky) ;  +diag(V) ;  2π ]    (diag(K)=0 for flat panels, §3)
```

with a **`+diag(V)`** middle block. That method is **not** the true matrix diagonal —
it is the vector fed to `Preconditioners.DiagonalPreconditioner(A)` for the Jacobi
preconditioner (`bem/implicit.jl:130`). So:

- **Algebraic diagonal** (for explicit assembly + the entrywise assembly gate):
  middle block `−diag(V)`.
- **NESSie's GMRES preconditioner vector** (to reproduce NESSie's `:gmres` path):
  middle block `+diag(V)`.

Porting `+diag(V)` as the assembled matrix diagonal would silently corrupt the
explicit system; flipping it to `−diag(V)` inside a parity GMRES path would stop
reproducing NESSie. Keep them as separate quantities in `system.rs`. (The local
system has no such split: `diag(M_u) = 2π(1+εΩ/εΣ)` *is* both, since `diag(K)=0`.)

**RHS** — algebraically `[b1; 0; 0]` with only the first block nonzero (this matches
the explicit assembly, `bem/nonlocal.jl:66–138`):

```
b1 = K·umol  +  (1 − εΩ/εΣ)·(Ky·umol)  −  2π·umol
        −  (εΩ/ε∞)·(V·qmol)  +  (εΩ/εΣ − εΩ/ε∞)·(Vy·qmol)
b2 = 0
b3 = 0
```

**⚠ source caveat (Codex review, critical):** the *implicit* path builds only the
length-`n` `b1` and passes it to the `3n×3n` operator (`bem/nonlocal.jl:277–280`),
which is dimensionally inconsistent as written. The explicit path's `[b1;0;0]` is the
correct, dimensionally-sound form and is what proteon ports; treat the implicit
construction as an apparent NESSie source defect / version discrepancy and **resolve
it against a running oracle** before trusting the nonlocal RHS.

Solve `A · [u;q;w] = [b1;0;0]`, then split the solution into the three blocks
(`:282–289`).

---

## 7. Post-processing & the unit chain

### 7.1 Reaction-field energy (`rfenergy`, `bem/post.jl:17–43`)

Per charge, evaluate Laplace operators **at the charge positions** `qposs` (not
centroids):

```
wstar = −[K · u](qposs)  +  [V · q](qposs)     (K,V here = laplacecoll! at charge pts)
W* = (wstar · qvals) / 4π · potprefactor(T) · (ec · 6.022140857e10 / 2)
```

Unit chain, in order (`bem/post.jl:31–42`):
- `/ 4π` — undo the `4π` premultiplier on `K`, `V`.
- `potprefactor = ec/(4π·ε0)` — applies `4π·ε0` (for `u,q`), one elementary charge
  (`1.602e-19`), and `Å→m` (`1e10`), all folded into the `ec`/`ε0` constants.
- `· ec · 6.022140857e10 / 2` — second elementary charge, Avogadro × `J→kJ`, and the
  `½` for double-counting pairwise interactions.

**Literal constants — RESOLVED (Codex review), no inconsistency.** The chain is
dimensionally sound:

```
ec · 6.022140857e10  =  (e·10¹⁰) · (Nₐ·10⁻¹³)  =  e · Nₐ · 10⁻³
```

i.e. the trailing literal `6.022140857e10` is `Nₐ·10⁻¹³`; its `10⁻¹³` cancels the
`Å→m` `10¹⁰` baked into the *second* `ec`, leaving the second elementary charge,
Avogadro, and `J→kJ`. The *first* `ec` (inside `potprefactor`) supplies the sole
`Å→m`. **Reproduce NESSie's constants and operation order** for parity (note
`1.602176` is truncated vs the exact modern `e`), but gate physical equivalence with
the Born unit test (`§9.3`) on a **defined float type + evaluation order** — not
"bit-for-bit across compilers" (FP contraction/reordering makes that too strong).
Do not "tidy" the constants.

### 7.2 Electrostatic potential (`espotential(domain, Ξ, bem)`, `bem/post.jl:75–85`)

`espotential = rfpotential + molpotential`, dispatched by domain. On-Γ uses the
**limiting trace**, not a raw on-surface kernel evaluation:

- **`:Γ`** (`:253–285`): `φ = u[closest_element(ξ)]·potprefactor + molpotential(ξ)`.
- **`:Ω`** (`:305–356`): `φ_rf = (1/4π)·(−[K·u] + [V·q])·potprefactor`, plus
  `molpotential`. (`K,V` at the interior points.)
- **`:Σ`** local (`:373–400`):
  `φ = ( −εΩ/εΣ·[V·(q+qmol)] + [K·(u+umol)] ) · potprefactor/4π`.
- **`:Σ`** nonlocal (`:402–447`), written out (4 operator applications, then
  `× potprefactor/4π`):

```
φ_Σ =  −(εΩ/ε∞)·[ V · (q + qmol) ]
     +  εΩ(1/εΣ − 1/ε∞)·[ (Vʸ−V) · (q + qmol) ]
     +  [ K · (u + umol) ]
     +  [ (Kʸ−K) · ( u + (1 − εΩ/εΣ)·umol − (ε∞/εΣ)·w ) ]
   then × potprefactor / 4π
```

  (`Vʸ−V = Vy`, `Kʸ−K = Ky`, the regular parts.)

`molpotential(ξ) = _molpotential(ξ)/εΩ · potprefactor` (`base/potentials.jl:53–55`).

**Domain sampling (oracle):** centroids lie on Γ, so the harness must sample Ω/Σ at
genuinely interior/exterior points (done in `post_dump`).

---

## 8. Mapping to proteon

| Spec object | proteon-electrostatics | NESSie oracle dump |
|---|---|---|
| `V, K` | `laplace.rs` (×4π) | `collocation_dump` |
| `Vy, Ky` (regular) | `yukawa.rs` (×4π) | `yukawa_dump` |
| `umol, qmol` | `system.rs` RHS builder | (in `solve_dump`) |
| `M_u`, `diag`, local RHS | `system::assemble(Local)` | `assembly_kernels_dump` → **then** assembled system at P0.5 |
| 9-block `A`, `diag`, RHS | `system::assemble(Nonlocal)` | as above |
| two-stage / 3-block solve | `solve.rs` | `solve_dump` |
| `rfenergy`, `espotential` | `post.rs` | `post_dump`, `analytic` |
| `BlockLayout [u;q;w]` | `system::BlockLayout` | block ordering above |

This section is what upgrades `assembly_kernels_dump` into a true **entrywise assembly
oracle**: once the block matrices above are emitted (not just the kernel blocks), the
Rust `assemble()` is gated entrywise against them.

---

## 9. P0.5 acceptance (the gates that close this spec)

1. **Block matrices + RHS written out** — `§5`, `§6` (done above; verify against a
   running NESSie on a 2–4 element mesh, entrywise).
2. **Hand-computable single fixtures** (checked in; non-NESSie references):
   - `K[i,i] = 0` for every flat element (double-layer Laplace InPlane). Also check
     the **regular Yukawa** `Ky[i,i]` flat-panel self term (confirm whether it is also
     0 against the kernel) — pure hand/kernel check.
   - **Algebraic** diagonals: local `diag(M_u) = 2π(1 + εΩ/εΣ)`; nonlocal algebraic
     diagonal middle block `−diag(V)`. **Separately** pin NESSie's preconditioner
     vector `[2π − diag(Ky); +diag(V); 2π]` (they differ in the middle-block sign —
     see §6). Gate both, labelled distinctly.
   - One single-layer self entry `V[i,i]` for an equilateral triangle, observation at
     its centroid, from the Rjasanow InPlane closed form
     `Σ_edges h·log((1+sinφ₂)(1−sinφ₁)/((1−sinφ₂)(1+sinφ₁)))/2` — derive by hand, pin.
   - One-charge / one-panel **`qmol` sign** fixture (the leading minus + `(ξ−q)·n`).
3. **Unit-chain test (no kernels/assembly/solve)** — inject an analytic reaction
   potential `φ_rf`, run it through the `rfenergy` prefactor chain (`§7.1`), and
   recover the closed-form Born energy. This validates `§7`'s constant arithmetic in
   isolation and reconciles the literal-constant question. (The real Born **BEM**
   end-to-end test is P5, not here.)
4. **Dimensional analysis** of `§7.1`/`§7.2` written down (V = C/F; W* = kJ/mol).

---

## 10. Conventions still to confirm (do not code around silently)

- **Net charge / total charge** behavior of the source terms and energy.
- **Multiple solute components & buried cavities:** the per-component system and the
  topology-aware assignment of each charge to its dielectric region (point-in-
  component, not distance) — `TO_ELECTROSTATICS.md` §6. NESSie's bundled models are
  single-component; this is unexercised by the starter corpus.
- **Exterior decay / gauge:** the radiation/decay condition at infinity implied by the
  fundamental solutions, and any potential-gauge choice.
- **`εΣ = ε∞` + `λ→0` ⇒ local — likely NOT a clean operator-level identity (Codex
  review).** Substituting into `§6` does *not* reduce the operator to `§5`: the `w`
  block remains, and as `λ→0` the regular parts tend to `Vy→−V`, `Ky→−K` (not 0), so
  rows 1 and 3 become redundant/singular — a degenerate 3-field system, not `§5`.
  Physical *potentials* may still converge to the local model after **eliminating `w`
  and taking the limit carefully**. Do **not** gate "operator collapses to §5"; if
  used at all, gate the limiting physical potential, with the elimination written
  out first. (This replaces the optimistic claim in earlier drafts.)
- **Stage-2 well-posedness (`§5`).** `q = V⁻¹(2π·I + K)u` assumes `V` is invertible
  and decently conditioned. Continuous closed-surface single-layer theory supports
  this in the right trace spaces, but it does **not** guarantee invertibility/
  conditioning of the nonsymmetric centroid-collocation `V` on arbitrary meshes.
  **Require a closed, non-degenerate, consistently-oriented surface**; treat solver
  failure / ill-conditioning as a porting concern (surface it, don't silently
  return). Same caveat for the nonlocal solve.
- **`defaultopt` εΩ = 2**, but the README single-ion vacuum example sets `εΩ = 1`.
  Pin the parameter set used for each fixture explicitly (the harness already does).
