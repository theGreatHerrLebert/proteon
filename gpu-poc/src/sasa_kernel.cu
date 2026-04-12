// GPU Shrake-Rupley SASA: one thread per atom.
// Each thread tests N_POINTS surface points against all neighbors.
// O(N_atoms * N_points * N_atoms) brute force — fast enough on GPU
// for structures up to ~30k atoms. Production would use a cell list.

extern "C" __global__ void sasa_shrake_rupley(
    const double* __restrict__ coords,       // N*3
    const double* __restrict__ expanded,     // N: r_i + probe
    const double* __restrict__ expanded_sq,  // N: (r_i + probe)^2
    const double* __restrict__ unit_points,  // M*3 (golden spiral on unit sphere)
    int n_atoms,
    int n_points,
    double inv_n_points,
    double four_pi,
    double* __restrict__ sasa               // N output: per-atom SASA in A^2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    double xi = coords[i*3+0];
    double yi = coords[i*3+1];
    double zi = coords[i*3+2];
    double ri = expanded[i];
    double ri_sq = expanded_sq[i];

    // Pre-filter neighbors: atoms whose expanded spheres overlap with mine.
    // This avoids checking distant atoms for every test point.
    // Store neighbor indices in shared memory would be ideal, but for POC
    // we just re-check the distance per point (branch predictor helps).

    int exposed = 0;

    for (int p = 0; p < n_points; p++) {
        double px = xi + ri * unit_points[p*3+0];
        double py = yi + ri * unit_points[p*3+1];
        double pz = zi + ri * unit_points[p*3+2];

        int buried = 0;
        for (int j = 0; j < n_atoms && !buried; j++) {
            if (j == i) continue;

            // Quick atom-level distance check first
            double sum_r = ri + expanded[j];
            double dax = coords[j*3+0] - xi;
            double day = coords[j*3+1] - yi;
            double daz = coords[j*3+2] - zi;
            if (dax*dax + day*day + daz*daz >= sum_r * sum_r)
                continue;

            // Point inside neighbor's expanded sphere?
            double dx = px - coords[j*3+0];
            double dy = py - coords[j*3+1];
            double dz = pz - coords[j*3+2];
            if (dx*dx + dy*dy + dz*dz < expanded_sq[j]) {
                buried = 1;
            }
        }

        if (!buried) exposed++;
    }

    sasa[i] = (double)exposed * inv_n_points * four_pi * ri_sq;
}
