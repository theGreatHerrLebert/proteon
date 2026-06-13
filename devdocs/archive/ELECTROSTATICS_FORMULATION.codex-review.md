# Codex review (gpt-5.5, 2026-06-09) — ELECTROSTATICS_FORMULATION.md (DRAFT)

_Independent review of the transcription before any code is written against it._

**Findings**

1. **Critical: §6 mislabels NESSie’s nonlocal `diag(A)` as the matrix diagonal.**  
   The displayed 9-block matrix and row-form operator agree exactly, including `(2,2) = -V`. Therefore the actual algebraic diagonal is
   \[
   [\,2\pi-\operatorname{diag}(K_y);\;-\operatorname{diag}(V);\;2\pi\,].
   \]
   NESSie nevertheless returns `+diag(V)` in [`nonlocal.jl:210`](</scratch/TMAlign/NESSie.jl/src/bem/nonlocal.jl:210>). That method feeds `DiagonalPreconditioner(A)` through [`implicit.jl:130`](</scratch/TMAlign/NESSie.jl/src/bem/implicit.jl:130>); it is not consistent with the operator’s mathematical diagonal. §6 and §9.2 must distinguish:

   - **algebraic diagonal:** middle block `-diag(V)`;
   - **NESSie GMRES preconditioner vector:** middle block `+diag(V)`.

   Porting the stated vector as the matrix diagonal would silently alter an explicit assembly; changing it to negative in a parity GMRES path would cease reproducing NESSie.

2. **Critical: implicit nonlocal RHS is not literally equivalent to the explicit solve in the checked source.**  
   The explicit implementation allocates a `3n` RHS with only block one populated ([`nonlocal.jl:66`](</scratch/TMAlign/NESSie.jl/src/bem/nonlocal.jl:66>)). The implicit implementation constructs only the length-`n` expression `b1`, then passes it to a `3n × 3n` matrix ([`nonlocal.jl:277`](</scratch/TMAlign/NESSie.jl/src/bem/nonlocal.jl:277>)). The spec’s `[b1;0;0]` is algebraically correct and matches explicit assembly, but is not a literal transcription of that implicit call. Document this as an apparent NESSie source defect or version discrepancy and make the running-oracle test resolve it.

3. **High: §10’s local-limit invariant does not follow from §6 as stated.**  
   For `εΣ=ε∞`, several `Vy` coefficients vanish, but the `w` block remains. As `λ→0`, the Yukawa exponent tends to infinity; away from the diagonal, the regular parts tend toward `Vy→−V`, `Ky→−K`, rather than zero. Substitution makes rows one and three approach the same equation, producing a redundant/singular three-field representation, not directly the two-stage §5 system. Physical potentials may converge to the local model after eliminating auxiliary variables and taking the limit carefully, but “operator collapses to §5” is presently unsupported and likely false literally.

4. **Medium: stage-two well-posedness needs an explicit assumption.**  
   §5 correctly transcribes `Vq=(2πI+K)u`, but `q=V⁻¹…` assumes invertibility. Continuous 3-D closed-surface Laplace single-layer theory supports coercivity/invertibility in the appropriate trace spaces. That does not automatically prove invertibility or good conditioning of this nonsymmetric centroid-collocation matrix on arbitrary meshes. Require a closed, nondegenerate, consistently oriented surface and treat solver failure/conditioning as a porting concern.

5. **Medium: §7.1’s constants are unusual but dimensionally reconcilable.**  
   This is not a derivable inconsistency:
   \[
   ec\,(6.022140857\times10^{10})
   =(e\,10^{10})(N_A\,10^{-13})
   =eN_A10^{-3}.
   \]
   Thus the embedded Å-to-m factor in the second `ec` is cancelled by `10^{-10}` hidden in the trailing literal, leaving the required elementary-charge and J-to-kJ conversion. Reproduce NESSie’s constants and operation order for parity, but separately test the physical closed form. “Bit-for-bit” should be limited to a defined floating-point type/compiler behavior; a Born test establishes numerical/physical equivalence, not universal bit identity.

6. **Low: §3’s operator claims are sound, with naming caveats.**  
   **Source-verified:** both implicit `Vy/Ky` interaction functions call `regularyukawacoll`, and explicit assembly does likewise. Radon defines these as Yukawa minus Laplace and premultiplied by `4π` ([source](https://tkemmer.github.io/NESSie.jl/latest/intern/radon/)). Internal docstrings calling them full Yukawa matrices are misleading; the draft’s warning is warranted.

   **Source- and theory-verified:** `diag(K)=0` for this flat-panel collocation implementation. The observation centroid lies exactly in its own triangle plane, and the InPlane double-layer branch returns zero ([`Rjasanow.jl:227`](</scratch/TMAlign/NESSie.jl/src/Rjasanow.jl:227>)). This is implementation-specific, not a general statement that the boundary double-layer operator has no jump; the `2π` jump remains explicit.

7. **Porter traps to promote from notes into normative requirements.**  
   The `qmol` sign in §4 is correct: `ddot(center,pos,n)=(center-pos)·n`, followed by the source’s leading minus. Pin a one-charge, one-panel sign fixture. `rfenergy` evaluates at charge positions, not centroids. Nonlocal exterior `:Σ` requires all four Laplace/regular-Yukawa combinations in [`post.jl:402`](</scratch/TMAlign/NESSie.jl/src/bem/post.jl:402>), so “see source” is insufficient for a formulation spec; write that equation out.

**Verification Boundary**

Verified directly against source: §5 algebra and `2π(1+εΩ/εΣ)` diagonal; §6 row/block algebra; regular-part semantics; flat-panel `Kii=0`; RHS signs; charge-position energy evaluation; constants and unit chain. NESSie’s online docs independently confirm centroid-stored traces, premultiplication, regular Yukawa semantics, and Jacobi-preconditioned GMRES ([solvers](https://tkemmer.github.io/NESSie.jl/latest/lib/solvers/)).

Sanity-checked only: continuous BEM well-posedness as applied to this discrete collocation matrix, arbitrary topology/components, and the nonlocal-to-local physical limit. No running NESSie comparison was performed.
