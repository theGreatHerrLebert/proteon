# Codex review v2 (gpt-5.5, 2026-06-09) — TO_ELECTROSTATICS.md
**Overall verdict:** v2 closes most v1 gaps conceptually, but several gates are still underspecified or described as more independent than they actually are. I would approve exploration, not yet the full P0–P8 commitment.

1. **§1b is directionally right but not yet actionable enough.** “Transcribe the equations” needs an acceptance checklist: complete block matrices with RHS vectors, operator arguments and normal derivatives, basis/test spaces, area weighting, unknown ordering, source terms, exterior-at-infinity condition, and post-processing formulas. Include one hand-computable one-triangle/operator-entry fixture. Otherwise two implementations can use the same symbols while differing in discretization.

   P0.5 also contradicts itself: an “end-to-end” Born-ion BEM unit test cannot be green before kernels and assembly exist. At P0.5, test the dimensional/unit conversion chain using an injected analytic reaction potential. Move the actual Born BEM test to P5.

2. **The dense LU is not an independent assembly oracle.** LU applied to the Rust-assembled matrix independently validates the iterative solver, not assembly. Comparison with NESSie validates assembly by parity, but remains shared-lineage validation. A true assembly oracle must independently construct matrix entries from quadrature/formulas, or compare every dense block and RHS against dumped NESSie matrices.

3. **The high-precision quadrature oracle needs a concrete singular-integration design.** Generic “tanh-sinh/adaptive triangle rules” is insufficient for on-panel single-layer singularities and double-layer principal values. Specify Duffy transformations, panel subdivision, precision, error estimators, and how the jump term is separated from the principal-value integral. Otherwise P0 can become an open-ended numerical-analysis project.

4. **Several L4 gates are not presently gateable:**
   - “Observed convergence rate” needs a specified norm, expected minimum rate, refinement family, and fitting rule. Projected sphere refinements are connectivity-nested but not geometrically nested.
   - Boundary flux continuity may be imposed algebraically by the formulation, making it a consistency check rather than an independent physical validation.
   - Evaluating `φ` “at boundary traces” must distinguish limiting traces from direct kernel evaluation on Γ.
   - Reciprocity needs a precise discrete weighted inner product; a collocation matrix is generally not naively symmetric.
   - Orientation reversal is not a physical invariant. Production should normalize or reject inward orientation; deliberately reversing normals is an operator-sign test.
   - Confirm algebraically that `εΣ = ε∞` plus `κ→0` really produces the claimed local model. This is model-specific, not self-evident.
   - APBS cannot exactly match a triangulated boundary because it introduces grid and boundary-discretization error. Treat it as a loose cross-method benchmark, not a tight gate.

5. **New scope risk:** P0 now contains five Julia exporters, singular high-precision quadrature, external CAS fixtures, and a sphere mesher. P5 adds analytical models, GMRES, convergence studies, invariants, and APBS. That is already a large validated numerical project. “Kernel port is medium” remains fair; “P0–P5 medium core” does not. A more honest split is:
   - Medium: formulation spec plus L0–L2 kernel parity.
   - Large: reliable local solve and scientific validation.
   - Larger/experimental: SES-fed nonlocal protein solver.

6. **Highest failure risk:** near-singular integration on realistic SES meshes, coupled with conditioning. The plan identifies it but only promises to “document the accuracy floor.” That is insufficient for P6.5/P7: if realistic meshes exceed the floor, the solver has no valid production path. Define a mandatory adaptive subdivision or corrected quadrature fallback before SES-fed results can pass, rather than deferring it indefinitely.

7. **O(N²) changes the product recommendation.** It is not a reason to abandon BEM, but it is a reason not to position this as an alternative to GB for routine protein workloads. Ship it initially as a high-accuracy reference/research tier with explicit triangle and runtime limits. Nonlocal protein-scale BEM should proceed only if benchmarks establish a useful bounded regime or fast summation receives a funded phase. Do not wire it as a general force-field energy component at P8 without that result.

Still missing from a domain perspective: explicit approximation spaces and panel normalization, treatment of multiple connected components and cavities, exterior radiation/decay and potential gauge, net-charge behavior, topology-aware assignment of charges to dielectric regions, and convergence validation under geometric as well as density refinement.

The plan is substantially better, but the recommendation should be narrowed: proceed with formulation and kernel work; make the local solver a separately reviewed commitment; keep production nonlocal protein BEM conditional on near-singular remediation and scaling evidence.
