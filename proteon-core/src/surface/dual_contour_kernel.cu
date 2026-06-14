// GPU manifold dual contouring (SES-SDF K3) — mirrors the CPU
// `volume.rs::manifold_dual_contour` bit-for-bit modulo emission order.
//
// All-f64. The asymptotic decider and the `t` interpolation must make the SAME
// branch/value as the CPU plain-f64 left-associative evaluation — a divergent
// decider flips a sheet and tears the mesh. We use round-to-nearest intrinsics
// (__dmul_rn/__dsub_rn/__dadd_rn/__ddiv_rn) so NVRTC cannot FMA-contract
// `f0*f2 - f1*f3` (Rust without fast-math also evaluates these as separate
// rounded ops), matching the CPU.
//
// Pipeline (host orchestrates the exclusive scans between kernels):
//   k3a_sheet_count  (per cell)  -> sheet_count[ncells]
//   k3_node_tri_count(per node)  -> tri_count[nnodes]   (0/2/4/6)
//   [host exclusive-scan -> vert_offset, tri_offset, totals; upload]
//   k3b_emit_verts   (per cell)  -> verts[3*total], edge_sheet[ncells*12] (u8)
//   k3c_emit_tris    (per node)  -> tris[3*total], error_flag

extern "C" {

// Cube edges as corner-index pairs; corner c = bits (dx,dy,dz), dx low.
__device__ const int CUBE_EDGES[12][2] = {
    {0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7}, {0,4},{1,5},{2,6},{3,7}};
__device__ const int CORNER[8][3] = {
    {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
__device__ const int FACE_EDGES[6][4] = {
    {4,10,6,8},{5,11,7,9},{0,9,2,8},{1,11,3,10},{0,5,1,4},{2,7,3,6}};
__device__ const int FACE_CORNERS[6][4] = {
    {0,2,6,4},{1,3,7,5},{0,1,5,4},{2,3,7,6},{0,1,3,2},{4,5,7,6}};

__device__ inline int uf_find(int* p, int x) {
    int r = x;
    while (p[r] != r) r = p[r];
    int c = x;
    while (p[c] != c) { int n = p[c]; p[c] = r; c = n; }
    return r;
}
__device__ inline void uf_union(int* p, int a, int b) {
    int ra = uf_find(p, a), rb = uf_find(p, b);
    p[ra] = rb;
}

// Asymptotic decider, mirroring the CPU: denom = f0-f1+f2-f3 (left assoc);
// joined_02 iff |denom|<1e-12 OR ((f0*f2-f1*f3)/denom < 0) == (f0 < 0).
__device__ inline bool decide_joined_02(double f0, double f1, double f2, double f3) {
    double denom = __dsub_rn(__dadd_rn(__dsub_rn(f0, f1), f2), f3);
    if (fabs(denom) < 1e-12) return true;
    double num = __dsub_rn(__dmul_rn(f0, f2), __dmul_rn(f1, f3));
    double q = __ddiv_rn(num, denom);
    return (q < 0.0) == (f0 < 0.0);
}

// Classify a cell's 8 corner values into sheets. Fills edge_sheet[12] with the
// local sheet index of each crossing edge (255 = no crossing) and returns the
// number of sheets. Local sheet numbering is by first appearance scanning
// e=0..11 (== canonicalize each component to its minimum crossed-edge index),
// so it is deterministic and thread-independent.
__device__ int classify_cell(const double* cf, unsigned char* edge_sheet) {
    bool crossing[12];
    for (int e = 0; e < 12; ++e) {
        double fa = cf[CUBE_EDGES[e][0]];
        double fb = cf[CUBE_EDGES[e][1]];
        crossing[e] = (fa < 0.0) != (fb < 0.0);
    }
    int parent[12];
    for (int e = 0; e < 12; ++e) parent[e] = e;
    for (int fa = 0; fa < 6; ++fa) {
        const int* fe = FACE_EDGES[fa];
        int cnt = 0, idx[4];
        for (int t = 0; t < 4; ++t) if (crossing[fe[t]]) idx[cnt++] = t;
        if (cnt == 2) {
            uf_union(parent, fe[idx[0]], fe[idx[1]]);
        } else if (cnt == 4) {
            const int* fc = FACE_CORNERS[fa];
            bool j02 = decide_joined_02(cf[fc[0]], cf[fc[1]], cf[fc[2]], cf[fc[3]]);
            if (j02) {
                uf_union(parent, fe[0], fe[1]);
                uf_union(parent, fe[2], fe[3]);
            } else {
                uf_union(parent, fe[1], fe[2]);
                uf_union(parent, fe[3], fe[0]);
            }
        }
    }
    int root_sheet[12];
    for (int e = 0; e < 12; ++e) root_sheet[e] = -1;
    int ns = 0;
    for (int e = 0; e < 12; ++e) {
        if (crossing[e]) {
            int r = uf_find(parent, e);
            if (root_sheet[r] < 0) root_sheet[r] = ns++;
            edge_sheet[e] = (unsigned char)root_sheet[r];
        } else {
            edge_sheet[e] = 255;
        }
    }
    return ns;
}

// K3a: one thread per cell -> number of sheets (vertices) in that cell.
__global__ void k3a_sheet_count(const double* f, int nx, int ny, int nz, unsigned int* sheet_count) {
    long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long cx = nx - 1, cy = ny - 1, cz = nz - 1;
    long ncells = cx * cy * cz;
    if (c >= ncells) return;
    int i = (int)(c % cx);
    int j = (int)((c / cx) % cy);
    int k = (int)(c / (cx * cy));
    double cf[8];
    int neg = 0;
    for (int t = 0; t < 8; ++t) {
        int ni = i + CORNER[t][0], nj = j + CORNER[t][1], nk = k + CORNER[t][2];
        double v = f[ni + nx * (nj + (long)ny * nk)];
        cf[t] = v;
        if (v < 0.0) ++neg;
    }
    if (neg == 0 || neg == 8) { sheet_count[c] = 0; return; }
    unsigned char es[12];
    sheet_count[c] = (unsigned int)classify_cell(cf, es);
}

// Whether grid edge from node (i,j,k) along `dir` (0=x,1=y,2=z) is an interior
// sign-changing edge that emits a quad — matching the CPU Pass B conditions.
__device__ inline bool node_quad(const double* f, int nx, int ny, int nz,
                                 int i, int j, int k, int dir) {
    long here_idx = i + nx * (j + (long)ny * k);
    bool here = f[here_idx] < 0.0;
    if (dir == 0) {
        if (!(i + 1 < nx && j > 0 && k > 0)) return false;
        return here != (f[(i + 1) + nx * (j + (long)ny * k)] < 0.0);
    } else if (dir == 1) {
        if (!(j + 1 < ny && i > 0 && k > 0)) return false;
        return here != (f[i + nx * ((j + 1) + (long)ny * k)] < 0.0);
    } else {
        if (!(k + 1 < nz && i > 0 && j > 0)) return false;
        return here != (f[i + nx * (j + (long)ny * (k + 1))] < 0.0);
    }
}

// K3-node-count: one thread per node -> #tris it emits (2 per crossing dir).
__global__ void k3_node_tri_count(const double* f, int nx, int ny, int nz, unsigned int* tri_count) {
    long node = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nn = (long)nx * ny * nz;
    if (node >= nn) return;
    int i = (int)(node % nx);
    int j = (int)((node / nx) % ny);
    int k = (int)(node / ((long)nx * ny));
    unsigned int c = 0;
    for (int dir = 0; dir < 3; ++dir) if (node_quad(f, nx, ny, nz, i, j, k, dir)) c += 2;
    tri_count[node] = c;
}

// K3b: one thread per cell -> emit each sheet's mean vertex + the edge_sheet table.
__global__ void k3b_emit_verts(const double* f, int nx, int ny, int nz,
                               double ox, double oy, double oz, double spacing,
                               const unsigned int* vert_offset,
                               double* verts, unsigned char* edge_sheet_out) {
    long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long cx = nx - 1, cy = ny - 1, cz = nz - 1;
    long ncells = cx * cy * cz;
    if (c >= ncells) return;
    int i = (int)(c % cx);
    int j = (int)((c / cx) % cy);
    int k = (int)(c / (cx * cy));
    double cf[8];
    int neg = 0;
    for (int t = 0; t < 8; ++t) {
        int ni = i + CORNER[t][0], nj = j + CORNER[t][1], nk = k + CORNER[t][2];
        double v = f[ni + nx * (nj + (long)ny * nk)];
        cf[t] = v;
        if (v < 0.0) ++neg;
    }
    unsigned char* es = &edge_sheet_out[c * 12];
    if (neg == 0 || neg == 8) {
        for (int e = 0; e < 12; ++e) es[e] = 255;
        return;
    }
    unsigned char esheet[12];
    int ns = classify_cell(cf, esheet);
    // Accumulate the mean crossing position per sheet.
    double accx[8], accy[8], accz[8];
    double cnt[8];
    for (int s = 0; s < ns; ++s) { accx[s] = accy[s] = accz[s] = 0.0; cnt[s] = 0.0; }
    for (int e = 0; e < 12; ++e) {
        es[e] = esheet[e];
        if (esheet[e] == 255) continue;
        int a = CUBE_EDGES[e][0], b = CUBE_EDGES[e][1];
        double fa = cf[a], fb = cf[b];
        double t = fa / (fa - fb);
        double pax = ox + (i + CORNER[a][0]) * spacing;
        double pay = oy + (j + CORNER[a][1]) * spacing;
        double paz = oz + (k + CORNER[a][2]) * spacing;
        double pbx = ox + (i + CORNER[b][0]) * spacing;
        double pby = oy + (j + CORNER[b][1]) * spacing;
        double pbz = oz + (k + CORNER[b][2]) * spacing;
        int s = esheet[e];
        accx[s] += pax + (pbx - pax) * t;
        accy[s] += pay + (pby - pay) * t;
        accz[s] += paz + (pbz - paz) * t;
        cnt[s] += 1.0;
    }
    unsigned int base = vert_offset[c];
    for (int s = 0; s < ns; ++s) {
        long vid = base + s;
        verts[vid * 3 + 0] = accx[s] / cnt[s];
        verts[vid * 3 + 1] = accy[s] / cnt[s];
        verts[vid * 3 + 2] = accz[s] / cnt[s];
    }
}

// Vertex id for cell `c`'s copy of cube edge `e`. Returns -1 if the cell has no
// sheet there (a hole — the host falls back to the CPU contour).
__device__ inline long edge_vid(const unsigned int* vert_offset, const unsigned char* edge_sheet,
                                long c, int e) {
    unsigned char s = edge_sheet[c * 12 + e];
    if (s == 255) return -1;
    return (long)vert_offset[c] + s;
}

// K3c: one thread per node -> emit the quads (2 tris each) for its 3 grid edges.
__global__ void k3c_emit_tris(const double* f, int nx, int ny, int nz,
                              const unsigned int* vert_offset, const unsigned char* edge_sheet,
                              const unsigned int* tri_offset, unsigned int* tris, int* err) {
    long node = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nn = (long)nx * ny * nz;
    if (node >= nn) return;
    int i = (int)(node % nx);
    int j = (int)((node / nx) % ny);
    int k = (int)(node / ((long)nx * ny));
    long cx = nx - 1, cy = ny - 1;
    long off = tri_offset[node]; // base tri index for this node
    bool here = f[i + nx * (j + (long)ny * k)] < 0.0;
    // Per direction, the 4 incident cells (cyclic), their cube edge, and the flip.
    for (int dir = 0; dir < 3; ++dir) {
        if (!node_quad(f, nx, ny, nz, i, j, k, dir)) continue;
        long cells[4];
        int edges[4];
        bool flip;
        if (dir == 0) {
            cells[0] = i + cx * ((j - 1) + cy * (k - 1)); edges[0] = 3;
            cells[1] = i + cx * (j + cy * (k - 1));       edges[1] = 2;
            cells[2] = i + cx * (j + cy * k);             edges[2] = 0;
            cells[3] = i + cx * ((j - 1) + cy * k);       edges[3] = 1;
            flip = here;
        } else if (dir == 1) {
            cells[0] = (i - 1) + cx * (j + cy * (k - 1)); edges[0] = 7;
            cells[1] = i + cx * (j + cy * (k - 1));       edges[1] = 6;
            cells[2] = i + cx * (j + cy * k);             edges[2] = 4;
            cells[3] = (i - 1) + cx * (j + cy * k);       edges[3] = 5;
            flip = !here;
        } else {
            cells[0] = (i - 1) + cx * ((j - 1) + cy * k); edges[0] = 11;
            cells[1] = i + cx * ((j - 1) + cy * k);       edges[1] = 10;
            cells[2] = i + cx * (j + cy * k);             edges[2] = 8;
            cells[3] = (i - 1) + cx * (j + cy * k);       edges[3] = 9;
            flip = here;
        }
        long v[4];
        bool ok = true;
        for (int t = 0; t < 4; ++t) {
            v[t] = edge_vid(vert_offset, edge_sheet, cells[t], edges[t]);
            if (v[t] < 0) ok = false;
        }
        if (!ok) { atomicExch(err, 1); off += 2; continue; }
        long a = v[0], b = v[1], cc = v[2], d = v[3];
        unsigned int* t0 = &tris[off * 3];
        unsigned int* t1 = &tris[(off + 1) * 3];
        if (flip) {
            t0[0] = a; t0[1] = cc; t0[2] = b;
            t1[0] = a; t1[1] = d;  t1[2] = cc;
        } else {
            t0[0] = a; t0[1] = b;  t0[2] = cc;
            t1[0] = a; t1[1] = cc; t1[2] = d;
        }
        off += 2;
    }
}

} // extern "C"
