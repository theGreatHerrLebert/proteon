// GPU Rjasanow analytic Laplace collocation — fills the single (V) and double (K)
// layer collocation matrices, one thread per (observation i, element j) entry.
//
// Mirrors proteon-electrostatics/src/laplace.rs operation-for-operation (double
// precision, same branches/guards), so GPU and CPU agree to libm precision.

#define ETOL 1.45e-8

__device__ __forceinline__ double jsign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    if (x == 0.0) return 0.0;
    return nan("");
}
__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __forceinline__ double norm3(const double* a) { return sqrt(dot3(a, a)); }
__device__ __forceinline__ void sub3(const double* a, const double* b, double* o) {
    o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2];
}
__device__ __forceinline__ void cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
// f64::clamp semantics: NaN passes through.
__device__ __forceinline__ double clampd(double x, double lo, double hi) {
    if (isnan(x)) return x;
    return x < lo ? lo : (x > hi ? hi : x);
}
__device__ __forceinline__ double logterm(double chi2, double s) {
    double t1 = sqrt(1.0 - chi2 * s * s);
    double t2 = sqrt(1.0 - chi2) * s;
    return (t1 + t2) / (t1 - t2);
}

// kind: 0 = single layer, 1 = double layer.
__device__ double laplacepot_closed(int kind, bool in_plane, double s1, double s2,
                                     double h, double d) {
    if (kind == 0) {
        if (in_plane) {
            return h * log((1.0 + s2) * (1.0 - s1) / ((1.0 - s2) * (1.0 + s1))) / 2.0;
        }
        double da = fabs(d);
        double chi2 = da * da / (da * da + h * h);
        double chi = sqrt(chi2);
        double result = h * log(logterm(chi2, s2) / logterm(chi2, s1)) / 2.0;
        return result + da * (asin(chi * s2) - asin(s2) - asin(chi * s1) + asin(s1));
    }
    if (in_plane) return 0.0;
    double chi = fabs(d) / sqrt(d * d + h * h);
    return jsign(d) * (asin(chi * s1) - asin(s1) - asin(chi * s2) + asin(s2));
}

__device__ double laplacepot_edge(int kind, const double* xi, const double* x1,
                                  const double* x2, const double* normal, double dist) {
    double u1[3], u2[3], v[3];
    sub3(x1, xi, u1);
    sub3(x2, xi, u2);
    sub3(x2, x1, v);
    double u1n = norm3(u1), u2n = norm3(u2), vn = norm3(v);
    double s1 = clampd(dot3(u1, v) / (u1n * vn), -1.0, 1.0);
    double s2 = clampd(dot3(u2, v) / (u2n * vn), -1.0, 1.0);
    double h = sqrt(u1n * u1n * (1.0 - s1 * s1));
    if (h < ETOL || 1.0 - fabs(s1) < ETOL || 1.0 - fabs(s2) < ETOL || fabs(s1 - s2) < ETOL) {
        return 0.0;
    }
    bool in_plane = fabs(dist) < ETOL;
    double cr[3];
    cross3(u1, u2, cr);
    return jsign(dot3(cr, normal)) * laplacepot_closed(kind, in_plane, s1, s2, h, dist);
}

__device__ double laplace_collocation(int kind, const double* xi, const double* v1,
                                      const double* v2, const double* v3,
                                      const double* normal, double distorig) {
    double dist = dot3(xi, normal) - distorig;
    double xip[3];
    if (fabs(dist) >= ETOL) {
        xip[0] = xi[0] - normal[0] * dist;
        xip[1] = xi[1] - normal[1] * dist;
        xip[2] = xi[2] - normal[2] * dist;
    } else {
        xip[0] = xi[0]; xip[1] = xi[1]; xip[2] = xi[2];
    }
    return laplacepot_edge(kind, xip, v1, v2, normal, dist)
         + laplacepot_edge(kind, xip, v2, v3, normal, dist)
         + laplacepot_edge(kind, xip, v3, v1, normal, dist);
}

extern "C" __global__ void laplace_matrices(
    const double* __restrict__ verts,    // nf*9: v1,v2,v3 per element
    const double* __restrict__ normals,  // nf*3
    const double* __restrict__ distorig, // nf
    const double* __restrict__ cent,     // nf*3: observation centroids
    int nf,
    double* __restrict__ V,              // nf*nf, row-major
    double* __restrict__ K)              // nf*nf
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nf * nf;
    if (idx >= total) return;

    int i = (int)(idx / nf);  // observation point
    int j = (int)(idx % nf);  // element
    const double* xi = &cent[i * 3];
    const double* v1 = &verts[j * 9];
    const double* v2 = &verts[j * 9 + 3];
    const double* v3 = &verts[j * 9 + 6];
    const double* nrm = &normals[j * 3];
    double dorig = distorig[j];

    V[idx] = laplace_collocation(0, xi, v1, v2, v3, nrm, dorig);
    K[idx] = laplace_collocation(1, xi, v1, v2, v3, nrm, dorig);
}

// Matrix-free matvec: y[i] = Σ_j collocation(kind, ξ_i, elem_j) · x[j]. One thread per
// output row i, looping all elements j — the collocation is recomputed every call (no
// stored matrix → O(N) memory). `kind`: 0 = single (V·x), 1 = double (K·x).
extern "C" __global__ void laplace_matvec(
    const double* __restrict__ verts,    // nf*9
    const double* __restrict__ normals,  // nf*3
    const double* __restrict__ distorig, // nf
    const double* __restrict__ cent,     // nf*3
    const double* __restrict__ x,        // nf
    int nf,
    int kind,
    double* __restrict__ y)              // nf
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;
    const double* xi = &cent[i * 3];
    double acc = 0.0;
    for (int j = 0; j < nf; j++) {
        const double* v1 = &verts[j * 9];
        const double* v2 = &verts[j * 9 + 3];
        const double* v3 = &verts[j * 9 + 6];
        acc += laplace_collocation(kind, xi, v1, v2, v3, &normals[j * 3], distorig[j]) * x[j];
    }
    y[i] = acc;
}

// ---- regular Yukawa (Yukawa − Laplace) collocation, matrix-free -------------------
//
// Mirrors yukawa.rs operation-for-operation: the smooth (regular) part of the Yukawa
// potential integrated over each triangle with the 7-point Radon cubature, with the
// small-`yukawa·r` alternating-series guard against catastrophic cancellation and the
// r→0 limits. `kind`: 0 = single layer, 1 = double layer.

#define SERIES_THRESHOLD 0.1

// Regular single-layer Yukawa potential at quadrature point `x` for observation `xi`
// (×4π). Series for e^(−c) − 1 when c = yukawa·r is small.
__device__ double reg_yuk_single(const double* x, const double* xi, double yukawa) {
    double d[3];
    sub3(x, xi, d);
    double rnorm = norm3(d);
    if (rnorm <= ETOL) return -yukawa; // limit r → 0
    double scalednorm = yukawa * rnorm;
    if (scalednorm < SERIES_THRESHOLD) {
        double term = -scalednorm;
        double tolerance = ETOL * fabs(term);
        double tsum = 0.0;
        for (int i = 1; i <= 15; i++) {
            if (fabs(term) <= tolerance) break;
            tsum += term;
            term *= -scalednorm / ((double)i + 1.0);
        }
        return tsum / rnorm;
    }
    return (exp(-scalednorm) - 1.0) / rnorm;
}

// Regular double-layer Yukawa potential (normal derivative) at `x` for `xi` (×4π).
__device__ double reg_yuk_double(const double* x, const double* xi, double yukawa,
                                 const double* normal) {
    double d[3];
    sub3(x, xi, d);
    double rnorm = norm3(d);
    if (rnorm <= ETOL) return yukawa * yukawa / 2.0 / sqrt(3.0); // limit r → 0
    double cosovernorm2 = dot3(d, normal) / (rnorm * rnorm * rnorm);
    double scalednorm = yukawa * rnorm;
    if (scalednorm < SERIES_THRESHOLD) {
        double term = scalednorm * scalednorm / 2.0;
        double tolerance = ETOL * fabs(term);
        double tsum = 0.0;
        for (int i = 2; i <= 16; i++) {
            if (fabs(term) <= tolerance) break;
            tsum += term * ((double)i - 1.0);
            term *= -scalednorm / ((double)i + 1.0);
        }
        return tsum * cosovernorm2;
    }
    return (1.0 - (1.0 + scalednorm) * exp(-scalednorm)) * cosovernorm2;
}

// Regular-Yukawa collocation of one triangle at `xi`: 7-point Radon cubature × 2·area.
// `xb`/`yb`/`wb` are the precomputed barycentric rule (hoisted to one per thread).
__device__ double reg_yukawa_collocation(int kind, const double* xi, const double* v1,
                                         const double* v2, const double* v3,
                                         const double* normal, double area, double yukawa,
                                         const double* xb, const double* yb,
                                         const double* wb) {
    double e1[3], e2[3];
    sub3(v2, v1, e1);
    sub3(v3, v1, e2);
    double value = 0.0;
    for (int j = 0; j < 7; j++) {
        // NESSie map: point = x·e1 + y·e2 + v1 (left-assoc per component).
        double p[3] = {
            e1[0] * xb[j] + e2[0] * yb[j] + v1[0],
            e1[1] * xb[j] + e2[1] * yb[j] + v1[1],
            e1[2] * xb[j] + e2[2] * yb[j] + v1[2],
        };
        double pot = (kind == 0) ? reg_yuk_single(p, xi, yukawa)
                                 : reg_yuk_double(p, xi, yukawa, normal);
        value += pot * wb[j];
    }
    return value * 2.0 * area;
}

// Matrix-free regular-Yukawa matvec: y[i] = Σ_j regyukawa(kind, ξ_i, elem_j)·x[j].
extern "C" __global__ void yukawa_matvec(
    const double* __restrict__ verts,   // nf*9
    const double* __restrict__ normals, // nf*3
    const double* __restrict__ area,    // nf
    const double* __restrict__ cent,    // nf*3
    const double* __restrict__ x,       // nf
    int nf,
    int kind,
    double yukawa,
    double* __restrict__ y)             // nf
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;

    // Radon 7-point rule (barycentric x/y + weights), computed once per thread from
    // √15 with the same expressions as quadrature.rs (CUDA may still fuse arithmetic, so
    // results match to rounding, not necessarily bit-for-bit).
    double s = sqrt(15.0);
    double xb[7] = {1.0 / 3.0,      (6.0 + s) / 21.0,       (9.0 - 2.0 * s) / 21.0,
                    (6.0 + s) / 21.0, (6.0 - s) / 21.0,     (9.0 + 2.0 * s) / 21.0,
                    (6.0 - s) / 21.0};
    double yb[7] = {1.0 / 3.0,      (9.0 - 2.0 * s) / 21.0, (6.0 + s) / 21.0,
                    (6.0 + s) / 21.0, (9.0 + 2.0 * s) / 21.0, (6.0 - s) / 21.0,
                    (6.0 - s) / 21.0};
    double wb[7] = {9.0 / 80.0,         (155.0 + s) / 2400.0, (155.0 + s) / 2400.0,
                    (155.0 + s) / 2400.0, (155.0 - s) / 2400.0, (155.0 - s) / 2400.0,
                    (155.0 - s) / 2400.0};

    const double* xi = &cent[i * 3];
    double acc = 0.0;
    for (int j = 0; j < nf; j++) {
        const double* v1 = &verts[j * 9];
        const double* v2 = &verts[j * 9 + 3];
        const double* v3 = &verts[j * 9 + 6];
        acc += reg_yukawa_collocation(kind, xi, v1, v2, v3, &normals[j * 3], area[j], yukawa,
                                      xb, yb, wb)
             * x[j];
    }
    y[i] = acc;
}

