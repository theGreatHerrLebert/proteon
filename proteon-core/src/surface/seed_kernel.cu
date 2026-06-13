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
        // Match the CPU's `Vec3::normalized()` guard (geom::EPSILON = 1e-6): a
        // node within 1e-6 Å of a centre has no well-defined radial direction,
        // so the CPU rejects that projection — the GPU must too, or seeds (and
        // thus meshes) can diverge for tiny-radius / precisely-aligned inputs.
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

// ---------------------------------------------------------------------------
// Spatial-hash seed: same result as seed_scatter, but prunes the O(atoms²)
// brute exposure with a uniform cell grid — the production path for large
// receptors. Mirrors `volume.rs::AtomGrid::nearest_surface_point` EXACTLY:
//   - uniform grid, cell = max inflated radius, key = floor(coord / cell);
//   - expanding-ring search shell-by-shell with the SAME provable-termination
//     bound (atoms first seen at shell r are ≥ (r−2)·cell away);
//   - exposure via a fixed ±2-cell (NEAR_REACH) gather, strict-inside with the
//     same 1e-9 slack.
// Atoms are pre-sorted into cells on the host (counting sort): `atoms` is in
// cell order, `cell_start[c]..cell_start[c+1]` is cell c's atom range, and cell
// c = (kz·dimy + ky)·dimx + kx in (key − kmin) coordinates.

// Is `proj` strictly inside any inflated atom other than `self_atom`? (exposure
// test) — ±2-cell gather around proj's cell, matching the CPU's NEAR_REACH.
__device__ bool buried_hashed(
    double projx, double projy, double projz,
    int self_atom,
    const double* atoms,
    const int* cell_start,
    int kxmin, int kymin, int kzmin,
    int dimx, int dimy, int dimz,
    double cell)
{
    int kx = (int)floor(projx / cell);
    int ky = (int)floor(projy / cell);
    int kz = (int)floor(projz / cell);
    for (int dz = -2; dz <= 2; dz++) {
        int cz = kz + dz;
        if (cz < kzmin || cz >= kzmin + dimz) continue;
        for (int dy = -2; dy <= 2; dy++) {
            int cy = ky + dy;
            if (cy < kymin || cy >= kymin + dimy) continue;
            for (int dx = -2; dx <= 2; dx++) {
                int cx = kx + dx;
                if (cx < kxmin || cx >= kxmin + dimx) continue;
                long long c = ((long long)(cz - kzmin) * dimy + (cy - kymin)) * dimx + (cx - kxmin);
                int lo = cell_start[c], hi = cell_start[c + 1];
                for (int a = lo; a < hi; a++) {
                    if (a == self_atom) continue;
                    double ox = atoms[4 * a + 0], oy = atoms[4 * a + 1];
                    double oz = atoms[4 * a + 2], orr = atoms[4 * a + 3];
                    double ex = projx - ox, ey = projy - oy, ez = projz - oz;
                    if (ex * ex + ey * ey + ez * ez < orr * orr - 1e-9) return true;
                }
            }
        }
    }
    return false;
}

extern "C" __global__ void seed_hashed_scatter(
    const double* nodes,   // nb*3  (boundary-node positions)
    const int* out_idx,    // nb    (grid index of each boundary node)
    int nb,
    const double* atoms,   // natoms*4, sorted into cell order
    const int* cell_start, // (dimx*dimy*dimz)+1, prefix sum
    int kxmin, int kymin, int kzmin,
    int dimx, int dimy, int dimz,
    double cell,
    double* feat_full)     // (nx*ny*nz)*3, pre-filled NaN
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nb) return;
    double px = nodes[3 * t + 0], py = nodes[3 * t + 1], pz = nodes[3 * t + 2];

    int kx = (int)floor(px / cell);
    int ky = (int)floor(py / cell);
    int kz = (int)floor(pz / cell);
    int kxmax = kxmin + dimx - 1, kymax = kymin + dimy - 1, kzmax = kzmin + dimz - 1;
    // max_r = farthest shell that can hold an atom (Chebyshev span to the grid).
    int sx = max(abs(kx - kxmin), abs(kx - kxmax));
    int sy = max(abs(ky - kymin), abs(ky - kymax));
    int sz = max(abs(kz - kzmin), abs(kz - kzmax));
    int max_r = max(sx, max(sy, sz));

    double bestd = 1e300;
    double bx = nan(""), by = nan(""), bz = nan("");

    for (int r = 0; r <= max_r; r++) {
        // Provably done: atoms in shell ≥ r are ≥ (r−2)·cell from the node.
        if (bestd < 1e300) {
            double lb = (double)max(r - 2, 0) * cell;
            if (lb * lb > bestd) break;
        }
        for (int dz = -r; dz <= r; dz++) {
            int cz = kz + dz;
            if (cz < kzmin || cz > kzmax) continue;
            for (int dy = -r; dy <= r; dy++) {
                int cy = ky + dy;
                if (cy < kymin || cy > kymax) continue;
                for (int dx = -r; dx <= r; dx++) {
                    // Chebyshev-exactly-r shell (skip the interior already done).
                    if (max(abs(dx), max(abs(dy), abs(dz))) != r) continue;
                    int cx = kx + dx;
                    if (cx < kxmin || cx > kxmax) continue;
                    long long c =
                        ((long long)(cz - kzmin) * dimy + (cy - kymin)) * dimx + (cx - kxmin);
                    int lo = cell_start[c], hi = cell_start[c + 1];
                    for (int a = lo; a < hi; a++) {
                        double ax = atoms[4 * a + 0], ay = atoms[4 * a + 1];
                        double az = atoms[4 * a + 2], ar = atoms[4 * a + 3];
                        double ddx = px - ax, ddy = py - ay, ddz = pz - az;
                        double len = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                        if (len < 1e-6) continue; // matches geom::EPSILON guard
                        double inv = ar / len;
                        double projx = ax + ddx * inv;
                        double projy = ay + ddy * inv;
                        double projz = az + ddz * inv;
                        if (buried_hashed(projx, projy, projz, a, atoms, cell_start,
                                          kxmin, kymin, kzmin, dimx, dimy, dimz, cell)) {
                            continue;
                        }
                        double ex = px - projx, ey = py - projy, ez = pz - projz;
                        double d = ex * ex + ey * ey + ez * ez;
                        if (d < bestd) { bestd = d; bx = projx; by = projy; bz = projz; }
                    }
                }
            }
        }
    }
    int g = out_idx[t];
    feat_full[3 * g + 0] = bx;
    feat_full[3 * g + 1] = by;
    feat_full[3 * g + 2] = bz;
}

// Fill a flat buffer with NaN. One thread per element (length = grid_nodes*3).
extern "C" __global__ void fill_nan(double* buf, long long n)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    buf[t] = nan("");
}
