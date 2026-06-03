// OBC (Onufriev-Bashford-Case) Generalized Born CUDA kernels.
//
// Line-traceable port of proteon's CPU implementation in gb_obc.rs, which
// is itself a port of OpenMM's ReferenceObc.cpp. See that Rust file for
// the algebra / sign conventions.
//
// Pipeline (4 kernels, each launched once per single-point eval):
//   1. obc_born_radii:       one thread per atom i -> born_radii[i], obc_chain[i]
//   2. obc_energy_forces_direct: rowwise per-atom (first-loop:
//       energy + direct pair force + bornForces accumulator)
//   3. obc_chain_transform:  per-atom, bornForces[i] *= R_eff_i^2 · obc_chain[i]
//   4. obc_force_spread:     2D (i, j != i), spreads bornForces via HCT
//       integrand derivative; uses atomicAdd since multiple threads
//       write to each forces[*].
//
// Energy convention matches the CPU gb_obc_energy:
//   G_pol = -½ · τ · k_C · Σ_{i,j} q_i q_j / f_GB(r_ij, R_i, R_j)
// Rowwise thread-i accumulates 0.5·Gpol for each partner j (symmetric,
// so thread j doubles it on its own row); the self term (j==i) gets
// its own 0.5·Gpol once when include_self_term is nonzero.
//
// Force convention: forces[i] = -dE/dr_i (physical force), accumulated
// the same way as other CUDA kernels in this tree.

extern "C" __global__ void obc_born_radii(
    const double* __restrict__ coords,      // 3N
    const double* __restrict__ obc_radius,  // N, intrinsic ρ per atom (Å)
    const double* __restrict__ obc_scale,   // N, HCT scale per atom
    double offset,                           // ρ − ρ' = dielectric offset (Å)
    double alpha_obc,
    double beta_obc,
    double gamma_obc,
    int    n_atoms,
    double* __restrict__ born_radii,        // N (out)
    double* __restrict__ obc_chain          // N (out): (1-tanh²)·offset·(α−2βψ+3γψ²)/ρ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    double radius_i = obc_radius[i];
    double offset_radius_i = radius_i - offset;
    double radius_i_inverse = 1.0 / offset_radius_i;

    double xi = coords[i*3+0], yi = coords[i*3+1], zi = coords[i*3+2];

    double sum = 0.0;
    for (int j = 0; j < n_atoms; ++j) {
        if (j == i) continue;
        double radius_j = obc_radius[j];
        double offset_radius_j = radius_j - offset;
        double scaled_radius_j = obc_scale[j] * offset_radius_j;

        double dx = xi - coords[j*3+0];
        double dy = yi - coords[j*3+1];
        double dz = zi - coords[j*3+2];
        double r2 = dx*dx + dy*dy + dz*dz;
        // Zero-separation guard (matches CPU policy in gb_obc.rs).
        if (r2 < 0.01) continue;
        double r = sqrt(r2);
        double r_scaled_radius_j = r + scaled_radius_j;
        if (offset_radius_i >= r_scaled_radius_j) continue;

        double r_inverse = 1.0 / r;
        double diff = fabs(r - scaled_radius_j);
        double l_ij_denom = (offset_radius_i > diff) ? offset_radius_i : diff;
        double l_ij = 1.0 / l_ij_denom;
        double u_ij = 1.0 / r_scaled_radius_j;
        double l_ij2 = l_ij * l_ij;
        double u_ij2 = u_ij * u_ij;
        double ratio = log(u_ij / l_ij);
        double term = l_ij - u_ij
                    + 0.25 * r * (u_ij2 - l_ij2)
                    + 0.5 * r_inverse * ratio
                    + 0.25 * scaled_radius_j * scaled_radius_j * r_inverse * (l_ij2 - u_ij2);
        if (offset_radius_i < (scaled_radius_j - r)) {
            term += 2.0 * (radius_i_inverse - l_ij);
        }
        sum += term;
    }
    sum *= 0.5 * offset_radius_i;
    double sum2 = sum * sum;
    double sum3 = sum * sum2;
    double tanh_arg = alpha_obc * sum - beta_obc * sum2 + gamma_obc * sum3;
    double tanh_sum = tanh(tanh_arg);

    born_radii[i] = 1.0 / (1.0 / offset_radius_i - tanh_sum / radius_i);

    double chain = offset_radius_i * (alpha_obc - 2.0 * beta_obc * sum + 3.0 * gamma_obc * sum2);
    obc_chain[i] = (1.0 - tanh_sum * tanh_sum) * chain / radius_i;
}

// First OBC energy/force loop in a rowwise layout (one thread per atom i).
//
// Each thread iterates j over all atoms. For i != j, thread i attributes
// 0.5·Gpol(i,j) to its per-atom energy; thread j symmetrically attributes
// the other half in its own row, so the upper-triangular pair sum is
// replayed exactly without atomics on forces[i] or born_forces[i]. The
// symmetric pair force contribution is handled the same way:
//   force-on-i from (i,j) = +deltaR_ij · dGpol_dr
//   force-on-j from (i,j) = +deltaR_ji · dGpol_dr = -deltaR_ij · dGpol_dr
// Each thread writes the "force on self" contribution to its own row.
//
// Self term (j==i): only thread i sees it. include_self_term == 0 skips
// it (matches CPU's ObcGbParams::include_self_term flag).
extern "C" __global__ void obc_energy_forces_direct(
    const double* __restrict__ coords,      // 3N
    const double* __restrict__ charges,     // N
    const double* __restrict__ born_radii,  // N
    double pre_factor,                       // -τ·k_C (already negative)
    int    n_atoms,
    int    include_self_term,                // 1 to include i==j, 0 to skip
    double* __restrict__ per_atom_energy,   // N (out)
    double* __restrict__ forces,            // 3N (accumulated)
    double* __restrict__ born_forces        // N (accumulated, untransformed)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    double pqi = pre_factor * charges[i];
    double xi = coords[i*3+0], yi = coords[i*3+1], zi = coords[i*3+2];
    double ri_born = born_radii[i];

    double energy_i = 0.0;
    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    double bf_i = 0.0;

    for (int j = 0; j < n_atoms; ++j) {
        double alpha2, d_ij, exp_term, denom2, denom, gpol, d_gpol_dalpha2;
        if (j == i) {
            if (!include_self_term) continue;
            alpha2 = ri_born * ri_born;
            if (alpha2 <= 0.0) continue;
            // r = 0 at self; limit of d_ij = 0 -> expTerm = 1 -> denom2 = alpha2.
            d_ij = 0.0;
            exp_term = 1.0;
            denom2 = alpha2;
            denom = ri_born;
            gpol = pqi * charges[i] / denom;
            d_gpol_dalpha2 = -0.5 * gpol * exp_term * (1.0 + d_ij) / denom2;
            // Self term contributes 0.5·Gpol to energy; no force on self.
            energy_i += 0.5 * gpol;
            bf_i += d_gpol_dalpha2 * ri_born;
            continue;
        }

        double rj_born = born_radii[j];
        alpha2 = ri_born * rj_born;
        if (alpha2 <= 0.0) continue;

        double dx = coords[j*3+0] - xi;
        double dy = coords[j*3+1] - yi;
        double dz = coords[j*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        d_ij = r2 / (4.0 * alpha2);
        exp_term = exp(-d_ij);
        denom2 = r2 + alpha2 * exp_term;
        denom = sqrt(denom2);

        gpol = pqi * charges[j] / denom;
        double d_gpol_dr     = -gpol * (1.0 - 0.25 * exp_term) / denom2;
        d_gpol_dalpha2       = -0.5 * gpol * exp_term * (1.0 + d_ij) / denom2;

        // 0.5·Gpol per thread; partner thread j adds its half.
        energy_i += 0.5 * gpol;
        // Force on SELF: +deltaR_{self→partner} · dGpol_dr.
        // Thread i sees deltaR = r_j - r_i; thread j (visiting i) sees
        // deltaR = r_i - r_j = -above, which gives +force-on-j symmetrically.
        fxi += dx * d_gpol_dr;
        fyi += dy * d_gpol_dr;
        fzi += dz * d_gpol_dr;

        bf_i += d_gpol_dalpha2 * rj_born;
    }

    per_atom_energy[i] = energy_i;
    // Rowwise: direct assignment to own row, no atomics.
    forces[i*3+0] += fxi;
    forces[i*3+1] += fyi;
    forces[i*3+2] += fzi;
    born_forces[i] = bf_i;
}

// Multiply born_forces[i] by R_eff_i^2 · obc_chain[i]. Trivially parallel.
extern "C" __global__ void obc_chain_transform(
    const double* __restrict__ born_radii,
    const double* __restrict__ obc_chain,
    int    n_atoms,
    double* __restrict__ born_forces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    born_forces[i] *= born_radii[i] * born_radii[i] * obc_chain[i];
}

// Second OBC loop: spread (transformed) bornForces through the HCT
// integrand derivative. Asymmetric — force on j from (i->j) uses
// bornForces[i] AND t3 computed with atom i's offset radius as the
// "self" and atom j's scaled radius as the "partner". The symmetric
// (j->i) pair uses bornForces[j] and t3 with roles swapped; both are
// real and independent, so we launch one thread per (i, j) with j != i
// and atomicAdd into forces[i] and forces[j].
extern "C" __global__ void obc_force_spread(
    const double* __restrict__ coords,
    const double* __restrict__ obc_radius,
    const double* __restrict__ obc_scale,
    const double* __restrict__ born_forces,   // already transformed
    double offset,
    int    n_atoms,
    double* __restrict__ forces               // 3N accumulated
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_atoms || j >= n_atoms || i == j) return;

    double radius_i = obc_radius[i];
    double offset_radius_i = radius_i - offset;
    double radius_j = obc_radius[j];
    double offset_radius_j = radius_j - offset;
    double scaled_radius_j = obc_scale[j] * offset_radius_j;
    double scaled_radius_j2 = scaled_radius_j * scaled_radius_j;

    double dx = coords[j*3+0] - coords[i*3+0];
    double dy = coords[j*3+1] - coords[i*3+1];
    double dz = coords[j*3+2] - coords[i*3+2];
    double r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < 0.01) return;
    double r = sqrt(r2);
    double r_scaled_radius_j = r + scaled_radius_j;
    if (offset_radius_i >= r_scaled_radius_j) return;

    double diff = fabs(r - scaled_radius_j);
    double l_ij_denom = (offset_radius_i > diff) ? offset_radius_i : diff;
    double l_ij = 1.0 / l_ij_denom;
    double u_ij = 1.0 / r_scaled_radius_j;
    double l_ij2 = l_ij * l_ij;
    double u_ij2 = u_ij * u_ij;

    double r_inverse = 1.0 / r;
    double r2_inverse = r_inverse * r_inverse;

    double t3 = 0.125 * (1.0 + scaled_radius_j2 * r2_inverse) * (l_ij2 - u_ij2)
              + 0.25 * log(u_ij / l_ij) * r2_inverse;
    double de = born_forces[i] * t3 * r_inverse;

    double fx = dx * de;
    double fy = dy * de;
    double fz = dz * de;
    atomicAdd(&forces[i*3+0], -fx);
    atomicAdd(&forces[i*3+1], -fy);
    atomicAdd(&forces[i*3+2], -fz);
    atomicAdd(&forces[j*3+0],  fx);
    atomicAdd(&forces[j*3+1],  fy);
    atomicAdd(&forces[j*3+2],  fz);
}
