// GPU Shrake-Rupley SASA with neighbor prefilter.
// One thread per atom. Each thread only checks PRE-COMPUTED neighbors
// (uploaded as flat array with per-atom offsets) instead of all N atoms.
// Reduces O(N² × M) to O(N × k × M) where k = avg neighbors (~20-50).

extern "C" __global__ void sasa_shrake_rupley(
    const double* __restrict__ coords,          // N*3
    const double* __restrict__ expanded,        // N: r_i + probe
    const double* __restrict__ expanded_sq,     // N: (r_i + probe)^2
    const double* __restrict__ unit_points,     // M*3 (golden spiral)
    const int*    __restrict__ neighbor_offsets, // N: start index in neighbor_indices
    const int*    __restrict__ neighbor_counts,  // N: number of neighbors for atom i
    const int*    __restrict__ neighbor_indices, // flat: all neighbor indices
    int n_atoms,
    int n_points,
    double inv_n_points,
    double four_pi,
    double* __restrict__ sasa                   // N output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    double xi = coords[i*3+0];
    double yi = coords[i*3+1];
    double zi = coords[i*3+2];
    double ri = expanded[i];
    double ri_sq = expanded_sq[i];

    int nb_start = neighbor_offsets[i];
    int nb_count = neighbor_counts[i];

    int exposed = 0;

    for (int p = 0; p < n_points; p++) {
        double px = xi + ri * unit_points[p*3+0];
        double py = yi + ri * unit_points[p*3+1];
        double pz = zi + ri * unit_points[p*3+2];

        int buried = 0;
        for (int k = 0; k < nb_count && !buried; k++) {
            int j = neighbor_indices[nb_start + k];
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
