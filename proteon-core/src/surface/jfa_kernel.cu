// GPU-K2: one pass of the jump-flooding vector distance transform for the SES
// SDF field. One thread per grid node. Reproduces `volume.rs::jump_flood`'s
// inner pass EXACTLY: each node scans its 27 neighbours at offset `step`
// (di,dj,dk in {-step, 0, step}, in dk-dj-di order) and keeps the nearest
// non-empty feature point by SQUARED distance, strict `<`. Same scan order and
// rule as the CPU, so results match exactly except that an exact equal-distance
// tie may resolve to a different (equally-near) candidate if FMA contraction
// perturbs the squared distance — the nearest *distance* is unaffected. Empty
// cells carry NaN.
//
// The host ping-pongs `src`/`dst` over the halving schedule
// (next_pow2(reach) … 2, 1, 1 — the "JFA+1" variant), reading back the final
// `src`. Node position is recomputed from the index: origin + (i,j,k)*spacing.
extern "C" __global__ void jfa_pass(
    const double* src, // n*3  (current nearest feature per node, NaN = none)
    double* dst,       // n*3  (output)
    int nx,
    int ny,
    int nz,
    int step,
    double ox,
    double oy,
    double oz,
    double h)
{
    long long n = (long long)nx * ny * nz;
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n) return;

    int i = (int)(cell % nx);
    int j = (int)((cell / nx) % ny);
    int k = (int)(cell / ((long long)nx * ny));
    double hx = ox + i * h, hy = oy + j * h, hz = oz + k * h;

    // Start from this cell's current feature. NVRTC does not pull in <math.h>,
    // so `INFINITY` is undefined here; use the same large finite "no feature"
    // sentinel as seed_kernel.cu (any real squared distance is far below it).
    double bx = src[3 * cell + 0], by = src[3 * cell + 1], bz = src[3 * cell + 2];
    double bestd;
    if (isnan(bx)) {
        bestd = 1e300;
    } else {
        double dx = hx - bx, dy = hy - by, dz = hz - bz;
        bestd = dx * dx + dy * dy + dz * dz;
    }

    for (int dk = -step; dk <= step; dk += step) {
        int kk = k + dk;
        if (kk < 0 || kk >= nz) continue;
        for (int dj = -step; dj <= step; dj += step) {
            int jj = j + dj;
            if (jj < 0 || jj >= ny) continue;
            for (int di = -step; di <= step; di += step) {
                int ii = i + di;
                if (ii < 0 || ii >= nx) continue;
                long long nc = (long long)ii + (long long)nx * (jj + (long long)ny * kk);
                double fx = src[3 * nc + 0], fy = src[3 * nc + 1], fz = src[3 * nc + 2];
                if (isnan(fx)) continue;
                double dx = hx - fx, dy = hy - fy, dz = hz - fz;
                double d = dx * dx + dy * dy + dz * dz;
                if (d < bestd) {
                    bestd = d;
                    bx = fx;
                    by = fy;
                    bz = fz;
                }
            }
        }
    }
    dst[3 * cell + 0] = bx;
    dst[3 * cell + 1] = by;
    dst[3 * cell + 2] = bz;
    // step == 0 can't happen (schedule is >= 1); the di=dj=dk=0 self-read just
    // re-confirms the current best (d == bestd, not <), matching the CPU.
}

// Finalize: turn the flooded feature grid into the signed distance field, so the
// host downloads one f64 per node instead of the 3-feature grid. One thread per
// node. Mirrors `volume.rs::distance_field`'s finalize loop EXACTLY:
//   dist  = isnan(feat) ? UNREACHED(1e18) : |node − feat|
//   v     = (inside ? dist : −dist) − probe
//   f     = (v == 0) ? f64::EPSILON : v
// `inside[node]` (occupancy) is computed on the host (it already drives boundary
// detection) and uploaded; node position is recomputed from the index.
extern "C" __global__ void finalize_field(
    const double* feat,           // n*3 (NaN = unreached)
    const unsigned char* inside,  // n   (1 = node inside any inflated atom)
    int nx,
    int ny,
    int nz,
    double ox,
    double oy,
    double oz,
    double h,
    double probe,
    double* f)                    // n out
{
    long long n = (long long)nx * ny * nz;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i = (int)(t % nx);
    int j = (int)((t / nx) % ny);
    int k = (int)(t / ((long long)nx * ny));
    double px = ox + i * h, py = oy + j * h, pz = oz + k * h;

    double sx = feat[3 * t + 0], sy = feat[3 * t + 1], sz = feat[3 * t + 2];
    double dist;
    if (isnan(sx)) {
        dist = 1e18; // UNREACHED sentinel, matches the CPU constant
    } else {
        double dx = px - sx, dy = py - sy, dz = pz - sz;
        dist = sqrt(dx * dx + dy * dy + dz * dz);
    }
    double v = (inside[t] ? dist : -dist) - probe;
    // f64::EPSILON = 2^-52; nudge an exact zero consistently off the surface.
    f[t] = (v == 0.0) ? 2.2204460492503131e-16 : v;
}
