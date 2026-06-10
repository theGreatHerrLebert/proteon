# Codex review (gpt-5.5, 2026-06-09) — TO_ELECTROSTATICS.md
**Independent Review**

**Critical correctness gaps**

1. **The actual boundary-integral equations are absent.** “2-block” and “3-block” are not enough to validate the physics. The plan must state every block algebraically, including:
   - Interior/exterior normal orientation.
   - Trace jump terms (`±½I`, or solid-angle equivalent).
   - Definitions of single- and double-layer operators.
   - Which dielectric factors multiply each block.
   - Whether `q` is `∂φ/∂n`, dielectric flux, or a scaled quantity.
   - Coulomb-source normalization and the exact `4π`, `ε₀`, Å, elementary-charge, and kJ/mol conversions.

   Without this specification, close NESSie parity can reproduce a convention mistake while the Born energy compensates for it in post-processing.

2. **Self interactions need explicit treatment.** Centroid collocation puts the observation point on its own triangle. The plan discusses analytic singular integrals but does not specify:
   - Diagonal/self terms for both operators.
   - The double-layer principal value.
   - Jump/solid-angle handling at a piecewise-flat surface.
   - Near-singular, non-self interactions between close triangles.

   The last item is especially important for SES re-entrant regions: ordinary seven-point cubature may be accurate for the regular Yukawa term but not necessarily for nearly touching elements.

3. **The nonlocal model needs a physics-level specification.** The 3-block formulation should be derived or transcribed in the plan, with boundary conditions and limiting cases. Required tests include:
   - Nonlocal model approaching local dielectric behavior in the appropriate parameter limit.
   - Yukawa parameter tending to zero.
   - `εΣ = ε∞`.
   - Zero-charge and equal-dielectric cases producing zero reaction energy.

   Otherwise the implementation is validated mainly as “the same three blocks as NESSie,” not as the intended Lorentz nonlocal model.

4. **Energy-only analytical convergence is insufficient.** Reaction-field energy is a scalar sum over charges and can hide spatially compensating potential errors, sign errors, and charge-order mistakes. Gate analytical:
   - Reaction potential at each charge.
   - Potential at interior/exterior radial samples.
   - Boundary traces and dielectric flux continuity.
   - Energy assembled independently as `½ Σ qᵢ φ_rf(rᵢ)`.

**Mesh and input assumptions**

5. **Monotone convergence should not be required.** P0 collocation BEM on independently generated meshes commonly converges non-monotonically. Gate an error envelope or observed convergence rate over shape-regular, nested refinements. Record energy extrapolation separately.

6. **A mesh-density ladder is meaningless without quality controls.** Add gated preconditions or measured diagnostics for:
   - Watertightness, manifold edges, consistent outward orientation.
   - Signed volume and connected components.
   - Minimum angle, aspect ratio, area distribution, and zero-area faces.
   - Self-intersections and near-contact separation.
   - Charge-to-surface distance relative to local element size.

   Refining a geometrically biased SES can converge to the wrong domain, exactly analogous to “closed but geometrically wrong.”

7. **Do not use the SES mesher to establish the initial sphere theorem.** Generate exact spherical triangulations directly, with vertices projected onto the analytic sphere. This separates BEM convergence from SES geometry convergence. Later run a second, explicitly coupled SES+BEM claim.

8. **Charges on or very near the boundary require a defined policy.** Reject them with a scale-aware tolerance or implement specialized near-singular evaluation. A fixed absolute epsilon is inadequate. Also specify behavior for charges outside the intended solute component and inside disconnected cavities.

**Solver concerns**

9. **“O(N) memory” is true, but each matvec remains O(N²).** Production matrix-free GMRES without FMM, treecode, or hierarchical matrices will become impractical quickly. This should be stated prominently.

10. **GMRES is more than ~150 lines if made trustworthy.** Restarting, modified Gram-Schmidt stability, happy breakdown, residual replacement, stagnation, left/right preconditioning, scaling, and diagnostics matter. Jacobi may be weak for refined or poor-quality meshes and for the nonlocal block system.

   Gate the true unpreconditioned residual and blockwise residuals, not only Cauchy-data parity. Add condition/iteration sweeps over mesh refinement and dielectric contrast. Consider block scaling or a block-diagonal preconditioner rather than scalar Jacobi.

11. **Direct-solve parity needs an independent factorization.** Comparing implicit and explicitly assembled operators tests indexing, but GMRES versus the same operator does not validate assembly. Use a vetted dense LU/QR implementation for small fixtures.

**Oracle design**

12. NESSie parity is necessary but not independent. NESSie’s analytical models are mathematically independent of its BEM path, but porting those formulas into Rust creates shared transcription and special-function risk. Better independent checks are:
   - High-precision analytical fixtures generated externally and checked in.
   - Local-Poisson comparison with APBS or another mature PB solver on matched spherical geometry.
   - Gauss-law/flux identities, reciprocity, dielectric-equality invariants, and manufactured harmonic solutions.
   - Mesh orientation reversal tests at full-system level.

13. Per-element parity should include randomized high-precision numerical quadrature as a third oracle, especially for near-singular configurations. “Regular Yukawa + Laplace” is definitional and may share correlated errors.

**Sequencing and effort**

P0-P5 is broadly sensible, but insert two phases:

- Before assembly: publish the equations/convention specification and test high-precision kernel references.
- Before production integration: establish mesh acceptance diagnostics and solver scaling limits.

P7 should not expose arbitrary protein runs until failure modes and convergence diagnostics are part of the API. Overall this is **large**, not medium, if “validated solver” includes robust SES inputs, nonlocal conditioning, and useful protein-scale performance. The kernel port alone may be medium.

**Open questions already substantially answered**

- **Q3:** The plan already implies PQR for parity and force-field charges for production.
- **Q5:** It already proposes a small checked-in corpus and on-demand dense convergence fixtures.
- **Q4:** The stated architecture and optional P8 effectively choose standalone/alternative tier first; only later force-field integration remains undecided.
- **Q1:** Partly answered by deferring SES-fed claims, but direct analytic sphere meshes are the cleaner resolution.
- **Q2 and Q6:** Still genuinely open.
tokens used
