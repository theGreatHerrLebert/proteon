// GPU-K2: one pass of the jump-flooding vector distance transform for the SES
// SDF field. One thread per grid node. Reproduces `volume.rs::jump_flood`'s
// inner pass EXACTLY: each node scans its 27 neighbours at offset `step`
// (di,dj,dk in {-step, 0, step}, in dk-dj-di order) and keeps the nearest
// non-empty feature point by SQUARED distance, strict `<` (so equal-distance
// ties resolve to the same candidate as the CPU). Empty cells carry NaN.
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

    // Start from this cell's current feature.
    double bx = src[3 * cell + 0], by = src[3 * cell + 1], bz = src[3 * cell + 2];
    double bestd;
    if (isnan(bx)) {
        bestd = INFINITY;
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
