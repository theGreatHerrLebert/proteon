// GPU-K1: the SES SDF seed (nearest exposed surface point per boundary node),
// brute-force over all inflated atoms. One thread per boundary node.
// This mirrors `volume.rs::AtomGrid::nearest_surface_point` exactly (radial
// projection onto each inflated sphere, kept iff not strictly inside any other
// inflated sphere, nearest such projection wins) — minus the spatial hash, which
// a production kernel would add. Brute force is O(nodes * atoms^2); fine at
// spike scale, and it measures raw GPU throughput on the exposure test.
//
// Two entry points share one device helper:
//   - `seed_brute`   writes the feature compacted (feat[3*t] per input node t).
//   - `seed_scatter` writes into a full-grid buffer at feat[3*out_idx[t]], so the
//     fused seed->jump-flood path keeps the field on-device (no host scatter).
// `fill_nan` initialises a full-grid feature buffer to NaN ("no feature yet")
// before `seed_scatter` writes only the boundary nodes.

// Nearest exposed surface point on the union of inflated spheres, for one node
// `(px,py,pz)`. Writes `(bx,by,bz)` = the nearest exposed projection, or NaN if
// none. Identical math to the original `seed_brute` body.
__device__ void nearest_exposed(
    double px, double py, double pz,
    const double* atoms, // m*4 (cx, cy, cz, inflated_radius)
    int m,
    double* bx, double* by, double* bz)
{
    double bestd = 1e300;
    double rx = nan(""), ry = nan(""), rz = nan("");

    for (int i = 0; i < m; i++) {
        double cx = atoms[4 * i + 0];
        double cy = atoms[4 * i + 1];
        double cz = atoms[4 * i + 2];
        double r = atoms[4 * i + 3];
        double dx = px - cx, dy = py - cy, dz = pz - cz;
        double len = sqrt(dx * dx + dy * dy + dz * dz);
        if (len == 0.0) continue; // p at the centre — degenerate, skip
        double inv = r / len;
        double projx = cx + dx * inv;
        double projy = cy + dy * inv;
        double projz = cz + dz * inv;

        // Exposed = not strictly inside any OTHER inflated sphere.
        bool exposed = true;
        for (int j = 0; j < m; j++) {
            if (j == i) continue;
            double ox = atoms[4 * j + 0];
            double oy = atoms[4 * j + 1];
            double oz = atoms[4 * j + 2];
            double orr = atoms[4 * j + 3];
            double ex = projx - ox, ey = projy - oy, ez = projz - oz;
            if (ex * ex + ey * ey + ez * ez < orr * orr - 1e-9) { exposed = false; break; }
        }
        if (exposed) {
            double ddx = px - projx, ddy = py - projy, ddz = pz - projz;
            double d = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d < bestd) { bestd = d; rx = projx; ry = projy; rz = projz; }
        }
    }
    *bx = rx; *by = ry; *bz = rz;
}

extern "C" __global__ void seed_brute(
    const double* nodes, // n*3  (boundary-node positions)
    const double* atoms, // m*4  (cx, cy, cz, inflated_radius)
    int n,
    int m,
    double* feat)        // n*3  (nearest exposed surface point, or NaN)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    nearest_exposed(nodes[3 * t + 0], nodes[3 * t + 1], nodes[3 * t + 2],
                    atoms, m,
                    &feat[3 * t + 0], &feat[3 * t + 1], &feat[3 * t + 2]);
}

// Scatter variant for the fused path: the same per-node computation, but the
// result is written to the full-grid buffer at the node's grid index
// `out_idx[t]`. The buffer must be pre-filled with NaN (`fill_nan`) so the
// non-boundary nodes read as "no feature" for the jump-flood.
extern "C" __global__ void seed_scatter(
    const double* nodes,  // n*3  (boundary-node positions)
    const double* atoms,  // m*4  (cx, cy, cz, inflated_radius)
    const int* out_idx,   // n    (grid index of each boundary node)
    int n,
    int m,
    double* feat_full)    // (nx*ny*nz)*3, pre-filled NaN
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int g = out_idx[t];
    nearest_exposed(nodes[3 * t + 0], nodes[3 * t + 1], nodes[3 * t + 2],
                    atoms, m,
                    &feat_full[3 * g + 0], &feat_full[3 * g + 1], &feat_full[3 * g + 2]);
}

// Fill a flat buffer with NaN. One thread per element (length = grid_nodes*3).
extern "C" __global__ void fill_nan(double* buf, long long n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    buf[t] = nan("");
}
