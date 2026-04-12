// Bonded energy + forces kernels: bonds, angles, torsions.
// One thread per interaction. Forces via atomicAdd.
// Combined into one file for simplicity — production would split.

// ============================================================
// Bond stretch: E = k * (r - r0)^2
// ============================================================
extern "C" __global__ void bond_energy_forces(
    const double* __restrict__ coords,     // N*3
    const int*    __restrict__ bond_i,     // M_bonds
    const int*    __restrict__ bond_j,     // M_bonds
    const double* __restrict__ bond_k,     // M_bonds: spring constant
    const double* __restrict__ bond_r0,    // M_bonds: equilibrium distance
    int n_bonds,
    double* __restrict__ bond_energies,    // M_bonds output
    double* __restrict__ forces            // N*3 output (atomicAdd)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_bonds) return;

    int a = bond_i[tid], b = bond_j[tid];
    double dx = coords[b*3+0] - coords[a*3+0];
    double dy = coords[b*3+1] - coords[a*3+1];
    double dz = coords[b*3+2] - coords[a*3+2];
    double r = sqrt(dx*dx + dy*dy + dz*dz);

    if (r < 1e-10) {
        bond_energies[tid] = 0.0;
        return;
    }

    double dr = r - bond_r0[tid];
    double k = bond_k[tid];
    bond_energies[tid] = k * dr * dr;

    // Force: F = -dE/dr = -2k(r-r0) * unit_vector
    double f_mag = -2.0 * k * dr / r;
    double fx = f_mag * dx;
    double fy = f_mag * dy;
    double fz = f_mag * dz;

    atomicAdd(&forces[a*3+0], -fx);
    atomicAdd(&forces[a*3+1], -fy);
    atomicAdd(&forces[a*3+2], -fz);
    atomicAdd(&forces[b*3+0],  fx);
    atomicAdd(&forces[b*3+1],  fy);
    atomicAdd(&forces[b*3+2],  fz);
}

// ============================================================
// Angle bend: E = k * (theta - theta0)^2
// ============================================================
extern "C" __global__ void angle_energy_forces(
    const double* __restrict__ coords,
    const int*    __restrict__ angle_i,    // M_angles: atom i
    const int*    __restrict__ angle_j,    // M_angles: central atom j
    const int*    __restrict__ angle_k,    // M_angles: atom k
    const double* __restrict__ angle_kf,   // M_angles: force constant
    const double* __restrict__ angle_t0,   // M_angles: equilibrium angle (rad)
    int n_angles,
    double* __restrict__ angle_energies,
    double* __restrict__ forces
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_angles) return;

    int ai = angle_i[tid], aj = angle_j[tid], ak = angle_k[tid];

    // Vectors j→i and j→k
    double rji_x = coords[ai*3+0] - coords[aj*3+0];
    double rji_y = coords[ai*3+1] - coords[aj*3+1];
    double rji_z = coords[ai*3+2] - coords[aj*3+2];
    double rjk_x = coords[ak*3+0] - coords[aj*3+0];
    double rjk_y = coords[ak*3+1] - coords[aj*3+1];
    double rjk_z = coords[ak*3+2] - coords[aj*3+2];

    double rji_len = sqrt(rji_x*rji_x + rji_y*rji_y + rji_z*rji_z);
    double rjk_len = sqrt(rjk_x*rjk_x + rjk_y*rjk_y + rjk_z*rjk_z);

    if (rji_len < 1e-10 || rjk_len < 1e-10) {
        angle_energies[tid] = 0.0;
        return;
    }

    double cos_theta = (rji_x*rjk_x + rji_y*rjk_y + rji_z*rjk_z) / (rji_len * rjk_len);
    // Clamp to [-1, 1] for numerical safety
    cos_theta = fmax(-1.0, fmin(1.0, cos_theta));
    double theta = acos(cos_theta);
    double dtheta = theta - angle_t0[tid];
    double k = angle_kf[tid];
    angle_energies[tid] = k * dtheta * dtheta;

    // Force: same formula as ferritin's energy.rs (cross-product method)
    double sin_theta = sqrt(fmax(1e-12, 1.0 - cos_theta*cos_theta));
    double dv = 2.0 * k * dtheta / sin_theta;

    double inv_ji = 1.0 / rji_len;
    double inv_jk = 1.0 / rjk_len;
    double inv_ji2 = inv_ji * inv_ji;
    double inv_jk2 = inv_jk * inv_jk;
    double inv_ji_jk = inv_ji * inv_jk;

    // dtheta/dr_i
    double fi_x = dv * (rjk_x * inv_ji_jk - cos_theta * rji_x * inv_ji2);
    double fi_y = dv * (rjk_y * inv_ji_jk - cos_theta * rji_y * inv_ji2);
    double fi_z = dv * (rjk_z * inv_ji_jk - cos_theta * rji_z * inv_ji2);

    // dtheta/dr_k
    double fk_x = dv * (rji_x * inv_ji_jk - cos_theta * rjk_x * inv_jk2);
    double fk_y = dv * (rji_y * inv_ji_jk - cos_theta * rjk_y * inv_jk2);
    double fk_z = dv * (rji_z * inv_ji_jk - cos_theta * rjk_z * inv_jk2);

    atomicAdd(&forces[ai*3+0], fi_x);
    atomicAdd(&forces[ai*3+1], fi_y);
    atomicAdd(&forces[ai*3+2], fi_z);
    atomicAdd(&forces[ak*3+0], fk_x);
    atomicAdd(&forces[ak*3+1], fk_y);
    atomicAdd(&forces[ak*3+2], fk_z);
    atomicAdd(&forces[aj*3+0], -(fi_x + fk_x));
    atomicAdd(&forces[aj*3+1], -(fi_y + fk_y));
    atomicAdd(&forces[aj*3+2], -(fi_z + fk_z));
}

// ============================================================
// Torsion (proper + improper): E = (V/div) * (1 + cos(f*phi - phi0))
// One thread per torsion TERM (not per torsion — a single i-j-k-l
// torsion can have multiple Fourier terms).
// ============================================================
extern "C" __global__ void torsion_energy_forces(
    const double* __restrict__ coords,
    const int*    __restrict__ tor_i,      // M_terms
    const int*    __restrict__ tor_j,      // M_terms
    const int*    __restrict__ tor_k,      // M_terms
    const int*    __restrict__ tor_l,      // M_terms
    const double* __restrict__ tor_v,      // M_terms: V/div amplitude
    const double* __restrict__ tor_f,      // M_terms: periodicity (float)
    const double* __restrict__ tor_phi0,   // M_terms: phase
    int n_terms,
    double* __restrict__ tor_energies,
    double* __restrict__ forces
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_terms) return;

    int ai = tor_i[tid], aj = tor_j[tid], ak = tor_k[tid], al = tor_l[tid];

    // Vectors
    double b1_x = coords[aj*3+0] - coords[ai*3+0];
    double b1_y = coords[aj*3+1] - coords[ai*3+1];
    double b1_z = coords[aj*3+2] - coords[ai*3+2];
    double b2_x = coords[ak*3+0] - coords[aj*3+0];
    double b2_y = coords[ak*3+1] - coords[aj*3+1];
    double b2_z = coords[ak*3+2] - coords[aj*3+2];
    double b3_x = coords[al*3+0] - coords[ak*3+0];
    double b3_y = coords[al*3+1] - coords[ak*3+1];
    double b3_z = coords[al*3+2] - coords[ak*3+2];

    // Normal vectors n1 = b1 × b2, n2 = b2 × b3
    double n1_x = b1_y*b2_z - b1_z*b2_y;
    double n1_y = b1_z*b2_x - b1_x*b2_z;
    double n1_z = b1_x*b2_y - b1_y*b2_x;
    double n2_x = b2_y*b3_z - b2_z*b3_y;
    double n2_y = b2_z*b3_x - b2_x*b3_z;
    double n2_z = b2_x*b3_y - b2_y*b3_x;

    double n1_sq = n1_x*n1_x + n1_y*n1_y + n1_z*n1_z;
    double n2_sq = n2_x*n2_x + n2_y*n2_y + n2_z*n2_z;
    double b2_len = sqrt(b2_x*b2_x + b2_y*b2_y + b2_z*b2_z);

    if (n1_sq < 1e-20 || n2_sq < 1e-20 || b2_len < 1e-10) {
        tor_energies[tid] = 0.0;
        return;
    }

    // Dihedral angle via atan2
    double cos_phi = (n1_x*n2_x + n1_y*n2_y + n1_z*n2_z) / sqrt(n1_sq * n2_sq);
    // m = n1 × b2_unit
    double m_x = (n1_y*b2_z - n1_z*b2_y) / b2_len;
    double m_y = (n1_z*b2_x - n1_x*b2_z) / b2_len;
    double m_z = (n1_x*b2_y - n1_y*b2_x) / b2_len;
    double m_len = sqrt(m_x*m_x + m_y*m_y + m_z*m_z);
    double sin_phi = (m_x*n2_x + m_y*n2_y + m_z*n2_z);
    // Normalize by |m| * |n2| to match CPU's compute_dihedral
    double m_n2 = fmax(1e-10, m_len) * sqrt(n2_sq);
    sin_phi /= m_n2;

    double phi = atan2(sin_phi, cos_phi);

    // Energy: E = v * (1 + cos(f*phi - phi0))
    double v = tor_v[tid];
    double f = tor_f[tid];
    double phi0 = tor_phi0[tid];
    tor_energies[tid] = v * (1.0 + cos(f * phi - phi0));

    // Force: dE/dphi = -v * f * sin(f*phi - phi0)
    // Then project dE/dphi onto Cartesian forces using the BALL formula.
    double dEdphi = -v * f * sin(f * phi - phi0);

    double inv_n1_sq = 1.0 / n1_sq;
    double inv_n2_sq = 1.0 / n2_sq;

    // dE/dt = (dE/dphi) / (|n1|^2 * |b2|) * (n1 × b2)
    double scale_t = dEdphi / (n1_sq * b2_len);
    double dEdt_x = scale_t * (n1_y*b2_z - n1_z*b2_y);
    double dEdt_y = scale_t * (n1_z*b2_x - n1_x*b2_z);
    double dEdt_z = scale_t * (n1_x*b2_y - n1_y*b2_x);

    // dE/du = -(dE/dphi) / (|n2|^2 * |b2|) * (n2 × b2)
    double scale_u = -dEdphi / (n2_sq * b2_len);
    double dEdu_x = scale_u * (n2_y*b2_z - n2_z*b2_y);
    double dEdu_y = scale_u * (n2_z*b2_x - n2_x*b2_z);
    double dEdu_z = scale_u * (n2_x*b2_y - n2_y*b2_x);

    // F1 = dEdt × b2
    double f1_x = dEdt_y*b2_z - dEdt_z*b2_y;
    double f1_y = dEdt_z*b2_x - dEdt_x*b2_z;
    double f1_z = dEdt_x*b2_y - dEdt_y*b2_x;

    // F4 = dEdu × b2
    double f4_x = dEdu_y*b2_z - dEdu_z*b2_y;
    double f4_y = dEdu_z*b2_x - dEdu_x*b2_z;
    double f4_z = dEdu_x*b2_y - dEdu_y*b2_x;

    // r vectors for F2, F3
    double r13_x = coords[ak*3+0] - coords[ai*3+0];
    double r13_y = coords[ak*3+1] - coords[ai*3+1];
    double r13_z = coords[ak*3+2] - coords[ai*3+2];
    double r24_x = coords[al*3+0] - coords[aj*3+0];
    double r24_y = coords[al*3+1] - coords[aj*3+1];
    double r24_z = coords[al*3+2] - coords[aj*3+2];
    double r21_x = coords[ai*3+0] - coords[aj*3+0];
    double r21_y = coords[ai*3+1] - coords[aj*3+1];
    double r21_z = coords[ai*3+2] - coords[aj*3+2];
    double r34_x = coords[al*3+0] - coords[ak*3+0];
    double r34_y = coords[al*3+1] - coords[ak*3+1];
    double r34_z = coords[al*3+2] - coords[ak*3+2];

    // F2 = cross(r13, dEdt) + cross(dEdu, r34)
    double f2_x = (r13_y*dEdt_z - r13_z*dEdt_y) + (dEdu_y*r34_z - dEdu_z*r34_y);
    double f2_y = (r13_z*dEdt_x - r13_x*dEdt_z) + (dEdu_z*r34_x - dEdu_x*r34_z);
    double f2_z = (r13_x*dEdt_y - r13_y*dEdt_x) + (dEdu_x*r34_y - dEdu_y*r34_x);

    // F3 = cross(r21, dEdt) + cross(r24, dEdu)
    double f3_x = (r21_y*dEdt_z - r21_z*dEdt_y) + (r24_y*dEdu_z - r24_z*dEdu_y);
    double f3_y = (r21_z*dEdt_x - r21_x*dEdt_z) + (r24_z*dEdu_x - r24_x*dEdu_z);
    double f3_z = (r21_x*dEdt_y - r21_y*dEdt_x) + (r24_x*dEdu_y - r24_y*dEdu_x);

    atomicAdd(&forces[ai*3+0], f1_x); atomicAdd(&forces[ai*3+1], f1_y); atomicAdd(&forces[ai*3+2], f1_z);
    atomicAdd(&forces[aj*3+0], f2_x); atomicAdd(&forces[aj*3+1], f2_y); atomicAdd(&forces[aj*3+2], f2_z);
    atomicAdd(&forces[ak*3+0], f3_x); atomicAdd(&forces[ak*3+1], f3_y); atomicAdd(&forces[ak*3+2], f3_z);
    atomicAdd(&forces[al*3+0], f4_x); atomicAdd(&forces[al*3+1], f4_y); atomicAdd(&forces[al*3+2], f4_z);
}
