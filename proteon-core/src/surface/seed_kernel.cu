// GPU-K1 spike: the SES SDF seed (nearest exposed surface point per boundary
// node), brute-force over all inflated atoms. One thread per boundary node.
// This mirrors `volume.rs::AtomGrid::nearest_surface_point` exactly (radial
// projection onto each inflated sphere, kept iff not strictly inside any other
// inflated sphere, nearest such projection wins) — minus the spatial hash, which
// a production kernel would add. Brute force is O(nodes * atoms^2); fine at
// spike scale, and it measures raw GPU throughput on the exposure test.
extern "C" __global__ void seed_brute(
    const double* nodes, // n*3  (boundary-node positions)
    const double* atoms, // m*4  (cx, cy, cz, inflated_radius)
    int n,
    int m,
    double* feat)        // n*3  (nearest exposed surface point, or NaN)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double px = nodes[3 * t + 0];
    double py = nodes[3 * t + 1];
    double pz = nodes[3 * t + 2];

    double bestd = 1e300;
    double bx = nan(""), by = nan(""), bz = nan("");

    for (int i = 0; i < m; i++) {
        double cx = atoms[4 * i + 0];
        double cy = atoms[4 * i + 1];
        double cz = atoms[4 * i + 2];
        double r = atoms[4 * i + 3];
        double dx = px - cx, dy = py - cy, dz = pz - cz;
        double len = sqrt(dx * dx + dy * dy + dz * dz);
        // Skip degenerate near-centre directions. Matches the CPU seed's
        // Vec3::normalized() guard (rejects |dir| < EPSILON = 1e-6) so the two
        // paths reject the same candidates.
        if (len < 1e-6) continue;
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
            if (d < bestd) { bestd = d; bx = projx; by = projy; bz = projz; }
        }
    }
    feat[3 * t + 0] = bx;
    feat[3 * t + 1] = by;
    feat[3 * t + 2] = bz;
}
