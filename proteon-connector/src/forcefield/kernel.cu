// Full nonbonded CUDA kernel: LJ 12-6 + Coulomb + cubic switching + EEF1 pair correction + forces.
// One thread per NBL pair. Energy output per-pair (reduced on host).
// Forces accumulated via f64 atomicAdd (requires CC 6.0+).

extern "C" __global__ void nonbonded_energy_forces(
    const double* __restrict__ coords,       // N*3 flat
    const int*    __restrict__ pair_i,        // M pairs
    const int*    __restrict__ pair_j,        // M pairs
    const int*    __restrict__ pair_is_14,    // M: 1 if 1-4 pair, 0 otherwise
    const double* __restrict__ lj_r,          // n_types: Rmin/2 per type
    const double* __restrict__ lj_eps,        // n_types: epsilon per type
    const int*    __restrict__ atom_types,     // N: type index per atom
    const double* __restrict__ charges,        // N: partial charge per atom
    const int*    __restrict__ is_hydrogen,    // N: 1 if H, 0 otherwise
    const double* __restrict__ eef1_dg_free,   // N
    const double* __restrict__ eef1_volume,    // N
    const double* __restrict__ eef1_sigma,     // N
    const double* __restrict__ eef1_r_min,     // N
    int n_pairs,
    double cutoff_sq,
    double cuton_sq,
    double eef1_cutoff_sq,
    double coulomb_factor,
    double scee_inv,
    double scnb_inv,
    double pi_sqrt_pi,
    double* __restrict__ pair_vdw,
    double* __restrict__ pair_elec,
    double* __restrict__ pair_solv,
    double* __restrict__ forces
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    int a = pair_i[tid], b = pair_j[tid];
    double dx = coords[a*3+0] - coords[b*3+0];
    double dy = coords[a*3+1] - coords[b*3+1];
    double dz = coords[a*3+2] - coords[b*3+2];
    double r2 = dx*dx + dy*dy + dz*dz;

    double e_vdw = 0.0, e_elec = 0.0, e_solv = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    if (r2 <= cutoff_sq && r2 >= 0.01) {
        double r = sqrt(r2);
        double inv_r = 1.0 / r;
        int is14 = pair_is_14[tid];
        double scale_vdw = is14 ? scnb_inv : 1.0;
        double scale_es  = is14 ? scee_inv : 1.0;

        // Cubic switching function
        double sw = 1.0, dsw_dr2 = 0.0;
        if (r2 > cuton_sq) {
            double diff_off = cutoff_sq - r2;
            double diff_on_off = cutoff_sq - cuton_sq;
            double inv3 = 1.0 / (diff_on_off * diff_on_off * diff_on_off);
            sw = diff_off * diff_off * (cutoff_sq + 2.0*r2 - 3.0*cuton_sq) * inv3;
            dsw_dr2 = 6.0 * diff_off * (cuton_sq - r2) * inv3;
        }

        // LJ 12-6
        int ti = atom_types[a], tj = atom_types[b];
        double eps = sqrt(lj_eps[ti] * lj_eps[tj]);
        double rmin = lj_r[ti] + lj_r[tj];
        if (eps > 1e-10 && rmin > 1e-10) {
            double sr = rmin * inv_r;
            double sr2 = sr * sr;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double e_lj = scale_vdw * eps * (sr12 - 2.0 * sr6);
            e_vdw = sw * e_lj;

            double de_dr = scale_vdw * eps * (-12.0*sr12 + 12.0*sr6) * inv_r;
            double total_de_dr = sw * de_dr + e_lj * dsw_dr2 * 2.0 * r;
            fx += total_de_dr * dx * inv_r;
            fy += total_de_dr * dy * inv_r;
            fz += total_de_dr * dz * inv_r;
        }

        // Coulomb
        double qi = charges[a], qj = charges[b];
        if (fabs(qi) > 1e-10 && fabs(qj) > 1e-10) {
            double e_es = scale_es * coulomb_factor * qi * qj * inv_r;
            e_elec = sw * e_es;

            double de_dr = -e_es * inv_r;
            double total_de_dr = sw * de_dr + e_es * dsw_dr2 * 2.0 * r;
            fx += total_de_dr * dx * inv_r;
            fy += total_de_dr * dy * inv_r;
            fz += total_de_dr * dz * inv_r;
        }

        // EEF1 solvation pair correction (9 Å cutoff)
        if (r2 <= eef1_cutoff_sq && is_hydrogen[a] == 0 && is_hydrogen[b] == 0) {
            double dg_free_a = eef1_dg_free[a], vol_b = eef1_volume[b];
            if (fabs(dg_free_a) > 1e-10 && vol_b > 1e-10) {
                double sig_a = eef1_sigma[a];
                double dr_a = (r - eef1_r_min[a]) / sig_a;
                double exp_a = exp(-dr_a * dr_a);
                double norm_a = sig_a * pi_sqrt_pi;
                double e_a = -0.5 * vol_b * dg_free_a * exp_a / (norm_a * r2);
                e_solv += e_a;
                double de_a = e_a * (-2.0 * dr_a / sig_a - 2.0 * inv_r);
                fx += de_a * dx * inv_r;
                fy += de_a * dy * inv_r;
                fz += de_a * dz * inv_r;
            }
            double dg_free_b = eef1_dg_free[b], vol_a = eef1_volume[a];
            if (fabs(dg_free_b) > 1e-10 && vol_a > 1e-10) {
                double sig_b = eef1_sigma[b];
                double dr_b = (r - eef1_r_min[b]) / sig_b;
                double exp_b = exp(-dr_b * dr_b);
                double norm_b = sig_b * pi_sqrt_pi;
                double e_b = -0.5 * vol_a * dg_free_b * exp_b / (norm_b * r2);
                e_solv += e_b;
                double de_b = e_b * (-2.0 * dr_b / sig_b - 2.0 * inv_r);
                fx += de_b * dx * inv_r;
                fy += de_b * dy * inv_r;
                fz += de_b * dz * inv_r;
            }
        }
    }

    pair_vdw[tid]  = e_vdw;
    pair_elec[tid] = e_elec;
    pair_solv[tid] = e_solv;

    if (fx != 0.0 || fy != 0.0 || fz != 0.0) {
        atomicAdd(&forces[a*3+0], -fx);
        atomicAdd(&forces[a*3+1], -fy);
        atomicAdd(&forces[a*3+2], -fz);
        atomicAdd(&forces[b*3+0],  fx);
        atomicAdd(&forces[b*3+1],  fy);
        atomicAdd(&forces[b*3+2],  fz);
    }
}
