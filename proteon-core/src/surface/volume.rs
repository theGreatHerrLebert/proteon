//! SES by iso-surfacing a signed distance field — the sane mesher (chosen over
//! analytic patch stitching; see `TO_SES_STITCHING.md`).
//!
//! **Erosion identity.** The solvent-accessible solid `A_p` is the union of the
//! atoms inflated by the probe radius. The solvent-*excluded* solid is `A_p`
//! eroded by a probe-radius ball: `{ x : dist(x, complement(A_p)) ≥ probe }`.
//! Its boundary is the SES — contact caps, toric bridges, and reentrant pockets
//! all at once. So
//!
//! ```text
//!   f(x) = dist(x, complement(A_p)) − probe        (>0 inside the SES solid)
//! ```
//!
//! has the whole SES as `f = 0`. We sample `f` on a regular grid and extract the
//! iso-surface with **manifold dual contouring** (`manifold_dual_contour`: a dual
//! vertex per surface *sheet* in each cell, a quad per sign-changing grid edge) —
//! a 2-manifold by construction for non-degenerate fields (exact-zero samples are
//! nudged off the grid corners), on any surface topology, no patch stitching.
//!
//! `dist(x, complement(A_p))` is the distance to the SAS (the union boundary).
//! We compute it from an *analytically-seeded* vector distance transform: nodes
//! adjacent to the inside/outside boundary are seeded with the nearest point on
//! an inflated atom (`AtomGrid::nearest_surface_point`), then a **jump-flooding
//! vector transform** (`jump_flood`) propagates each node's nearest seeded
//! surface point; distance is `|node − nearest_surface|`, signed by occupancy.
//!
//! Because the seeds are *analytic* (not voxel-snapped) and densely cover the
//! surface (one per boundary cell), the field has no occupancy staircase and the
//! positional error is `O(h)` — including at concave creases, where the per-node
//! `nearest_surface_point` is biased (the true nearest SAS point is on an
//! intersection rim) but the dense band still carries near-rim seeds, so JFA
//! recovers it. Empirically area/volume converge smoothly to `ball-py ses_area`:
//! sub-0.5 % at `h = 0.2`, sub-0.35 % even on sharp-crease / reentrant cases at
//! `h = 0.15`. A uniform spatial hash (`AtomGrid`) keeps occupancy/seeding
//! `O(neighbours)`. Topology below the grid scale (necks/components thinner than
//! `h`) is resolution-limited, like any grid iso-surface.

use super::geom::{Sphere, Vec3};
use super::mesh::Mesh;
use rayon::prelude::*;

/// Triangulate the SES of `atoms` (van-der-Waals spheres) for the given `probe`
/// radius, sampling the distance field at `spacing` Å. Returns a closed,
/// outward-oriented mesh. Returns an empty mesh if `atoms` is empty.
pub fn ses_mesh_sdf(atoms: &[Sphere], probe: f64, spacing: f64) -> Mesh {
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "grid spacing must be finite and positive"
    );
    assert!(
        probe.is_finite() && probe >= 0.0,
        "probe must be finite ≥ 0"
    );
    if atoms.is_empty() {
        return Mesh::default();
    }
    let prof = std::env::var("SES_SDF_PROF").is_ok();
    let grid = Grid::enclosing(atoms, probe, spacing);
    let f = grid.distance_field(atoms, probe);
    let t = std::time::Instant::now();
    let mut mesh = manifold_dual_contour(&grid, &f);
    if prof {
        let [nx, ny, nz] = grid.dims;
        eprintln!(
            "  SDF dual_contour: {:.1}ms  (grid {nx}x{ny}x{nz} = {} nodes)",
            t.elapsed().as_secs_f64() * 1e3,
            nx * ny * nz
        );
    }
    // Guarantee outward (solvent-facing) orientation regardless of the seam
    // winding surface_nets emitted.
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip();
    }
    mesh
}

/// A regular node grid: `dims` nodes per axis, node `(i,j,k)` at
/// `origin + (i,j,k)·spacing`.
struct Grid {
    origin: Vec3,
    spacing: f64,
    dims: [usize; 3],
}

impl Grid {
    /// A grid enclosing every inflated atom with at least one `probe`+spacing of
    /// outside padding on all sides, so the field is negative on the boundary
    /// (the iso-surface is closed).
    fn enclosing(atoms: &[Sphere], probe: f64, spacing: f64) -> Self {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for a in atoms {
            let r = a.radius + probe;
            lo.x = lo.x.min(a.center.x - r);
            lo.y = lo.y.min(a.center.y - r);
            lo.z = lo.z.min(a.center.z - r);
            hi.x = hi.x.max(a.center.x + r);
            hi.y = hi.y.max(a.center.y + r);
            hi.z = hi.z.max(a.center.z + r);
        }
        // Pad by probe (the erosion depth) + 2 cells so the boundary nodes are
        // safely in the complement.
        let pad = probe + 2.0 * spacing;
        let origin = Vec3::new(lo.x - pad, lo.y - pad, lo.z - pad);
        let span = Vec3::new(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
        let dims = [
            (((span.x + 2.0 * pad) / spacing).ceil() as usize) + 1,
            (((span.y + 2.0 * pad) / spacing).ceil() as usize) + 1,
            (((span.z + 2.0 * pad) / spacing).ceil() as usize) + 1,
        ];
        Grid {
            origin,
            spacing,
            dims,
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    #[inline]
    fn pos(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + i as f64 * self.spacing,
            self.origin.y + j as f64 * self.spacing,
            self.origin.z + k as f64 * self.spacing,
        )
    }

    /// `f = signed_dist(x, SAS) − probe` at every node, with `signed_dist > 0`
    /// inside `A_p` (the union of probe-inflated atoms). The SES is `f = 0`: the
    /// points exactly `probe` deep inside the SAS, i.e. `A_p` eroded by the probe.
    ///
    /// `signed_dist` is an `O(h)`-accurate Euclidean distance to the SAS: every
    /// node adjacent to the inside/outside boundary is seeded with the *analytic*
    /// nearest point on the inflated-atom union (an exposed radial projection),
    /// then a jump-flooding vector transform propagates each node's nearest seeded
    /// surface point. `signed_dist = ±|node − nearest_surface|`, signed by
    /// occupancy. Analytic seeds (not voxel-snapped) remove the binary-occupancy
    /// staircase, so area/volume converge smoothly — see the module docs for the
    /// crease-bias argument and the measured convergence.
    fn distance_field(&self, atoms: &[Sphere], probe: f64) -> Vec<f64> {
        const UNREACHED: f64 = 1e18; // finite sentinel for nodes JFA never reached
        let [nx, ny, nz] = self.dims;
        let n = nx * ny * nz;
        let prof = std::env::var("SES_SDF_PROF").is_ok();
        let mut t = std::time::Instant::now();
        let lap = |name: &str, t: &mut std::time::Instant| {
            if prof {
                eprintln!("  SDF {name}: {:.1}ms", t.elapsed().as_secs_f64() * 1e3);
                *t = std::time::Instant::now();
            }
        };
        let grid = AtomGrid::build(atoms, probe);
        lap("atomgrid_build", &mut t);

        // Occupancy of A_p by rasterizing each inflated sphere into its node
        // bounding box — O(atoms · R³/h³), far cheaper than an O(neighbours)
        // query at every node.
        let mut inside = vec![false; n];
        let inv_h = 1.0 / self.spacing;
        for s in &grid.spheres {
            let r2 = s.radius * s.radius;
            let lo = |c: f64, o: f64| (((c - s.radius - o) * inv_h).floor() as isize).max(0);
            let hi = |c: f64, o: f64, m: usize| {
                (((c - o + s.radius) * inv_h).ceil() as isize).min(m as isize - 1)
            };
            let (i0, i1) = (
                lo(s.center.x, self.origin.x),
                hi(s.center.x, self.origin.x, nx),
            );
            let (j0, j1) = (
                lo(s.center.y, self.origin.y),
                hi(s.center.y, self.origin.y, ny),
            );
            let (k0, k1) = (
                lo(s.center.z, self.origin.z),
                hi(s.center.z, self.origin.z, nz),
            );
            for k in k0..=k1 {
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        let (i, j, k) = (i as usize, j as usize, k as usize);
                        if self.pos(i, j, k).square_distance(s.center) <= r2 {
                            inside[self.idx(i, j, k)] = true;
                        }
                    }
                }
            }
        }
        lap("occupancy", &mut t);

        // Seed: nodes adjacent (6-neighbour) to a sign change carry their analytic
        // nearest point on the SAS. NaN x marks "no feature yet".
        //
        // This is the profiled bottleneck (77–90% of `ses_mesh_sdf` — the per-node
        // exposed-projection in `nearest_surface_point`). Each node writes only its
        // own `feat[idx]` and reads only the *immutable* `inside`/`grid`, so it is
        // embarrassingly parallel and the result is identical to the serial loop
        // (every node computes a deterministic value regardless of thread).
        const NONE: [f64; 3] = [f64::NAN; 3];
        let nxy = nx * ny;
        // A node is on the boundary iff it differs in occupancy from a
        // 6-neighbour — exactly the band straddling f = 0 that needs a feature.
        let is_boundary = |idx: usize| -> bool {
            let i = idx % nx;
            let j = (idx / nx) % ny;
            let k = idx / nxy;
            let ins = inside[idx];
            [
                (i + 1 < nx).then(|| self.idx(i + 1, j, k)),
                (i > 0).then(|| self.idx(i - 1, j, k)),
                (j + 1 < ny).then(|| self.idx(i, j + 1, k)),
                (j > 0).then(|| self.idx(i, j - 1, k)),
                (k + 1 < nz).then(|| self.idx(i, j, k + 1)),
                (k > 0).then(|| self.idx(i, j, k - 1)),
            ]
            .into_iter()
            .flatten()
            .any(|nb| inside[nb] != ins)
        };

        let mut feat = vec![NONE; n];

        // Features only need to reach nodes within ~probe of the surface (that's
        // the band straddling f = 0); beyond it the sign alone is correct.
        let reach = (probe / self.spacing).ceil() as usize + 4;

        // Fused GPU seed + jump-flood: with the `cuda` feature and a usable
        // device, compact the boundary nodes and run seed→flood *entirely
        // on-device* — the seeded feature buffer never returns to the host
        // between the two stages (no boundary-feature download, host scatter, or
        // full-grid re-upload). Any GPU failure → `fused_on_gpu = false` and the
        // CPU seed + CPU flood below. Same nearest distance as the CPU seed
        // (ties aside) and the same JFA+1 schedule.
        #[cfg(feature = "cuda")]
        let fused_on_gpu = {
            let bidx: Vec<usize> = (0..n)
                .into_par_iter()
                .filter(|&idx| is_boundary(idx))
                .collect();
            let bpos: Vec<Vec3> = bidx
                .iter()
                .map(|&idx| self.pos(idx % nx, (idx / nx) % ny, idx / nxy))
                .collect();
            match super::seed_gpu::seed_and_flood_gpu(
                &bpos,
                &bidx,
                &grid.spheres,
                self.dims,
                reach,
                self.origin,
                self.spacing,
            ) {
                Some(f) => {
                    feat = f;
                    true
                }
                None => false,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let fused_on_gpu = false;

        // CPU fallback (the parity-validated path): seed each boundary node with
        // its analytic nearest point on the SAS, then jump-flood. Both write only
        // their own outputs from immutable inputs, so they are order-independent.
        if !fused_on_gpu {
            feat.par_iter_mut().enumerate().for_each(|(idx, f)| {
                if is_boundary(idx) {
                    let i = idx % nx;
                    let j = (idx / nx) % ny;
                    let k = idx / nxy;
                    if let Some(s) = grid.nearest_surface_point(self.pos(i, j, k)) {
                        *f = [s.x, s.y, s.z];
                    }
                }
            });
            jump_flood(&mut feat, self.dims, reach, &|i, j, k| self.pos(i, j, k));
        }
        lap("seed+flood", &mut t);

        // signed distance to the SAS, then erode by the probe.
        let mut f = vec![0.0f64; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = self.idx(i, j, k);
                    let s = feat[idx];
                    // Unreached nodes are farther than `reach` from any surface, so
                    // they are deep in the sign-correct interior/exterior and never
                    // border the f=0 band. Use a large *finite* sentinel (not ∞) so
                    // a stray sign-change interpolation can never produce NaN.
                    let dist = if s[0].is_nan() {
                        UNREACHED
                    } else {
                        self.pos(i, j, k).distance(Vec3::new(s[0], s[1], s[2]))
                    };
                    let v = if inside[idx] { dist } else { -dist } - probe;
                    // Nudge an exact zero off the surface (consistently, toward
                    // inside) so no edge crossing lands exactly on a grid corner —
                    // that would make a vertex coincident with a node and a
                    // degenerate triangle (codex-review). Negligible vs any real
                    // distance; for real coordinates this effectively never fires.
                    f[idx] = if v == 0.0 { f64::EPSILON } else { v };
                }
            }
        }
        lap("finalize", &mut t);
        f
    }
}

/// Uniform spatial hash over the probe-inflated atoms, so occupancy and
/// nearest-surface queries cost O(neighbours) rather than O(atoms).
struct AtomGrid {
    /// inflated spheres (centre, radius + probe)
    spheres: Vec<Sphere>,
    cell: f64,
    buckets: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
    /// inclusive bucket-key bounds, so a query knows the shell radius that covers
    /// every atom (the sound fallback for `nearest_surface_point`).
    kmin: (i64, i64, i64),
    kmax: (i64, i64, i64),
}

/// Bucket reach for `AtomGrid` *exposure* queries — sound because an atom that
/// can bury a point has its centre within one `cell` (= max inflated radius) of
/// it, i.e. ≤ 2 buckets after `floor` bucketing. The *nearest-surface* query does
/// NOT use a fixed reach (an exposed point can be arbitrarily far when nearer
/// projections are all buried); it expands until provably done.
const NEAR_REACH: i64 = 2;

impl AtomGrid {
    fn build(atoms: &[Sphere], probe: f64) -> Self {
        let spheres: Vec<Sphere> = atoms
            .iter()
            .map(|a| Sphere::new(a.center, a.radius + probe))
            .collect();
        // Cell ≈ the largest inflated radius, so any query touches O(1) buckets.
        let cell = spheres.iter().map(|s| s.radius).fold(1.0_f64, f64::max);
        let mut buckets: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
            std::collections::HashMap::new();
        let mut kmin = (i64::MAX, i64::MAX, i64::MAX);
        let mut kmax = (i64::MIN, i64::MIN, i64::MIN);
        for (idx, s) in spheres.iter().enumerate() {
            let k = Self::key(s.center, cell);
            kmin = (kmin.0.min(k.0), kmin.1.min(k.1), kmin.2.min(k.2));
            kmax = (kmax.0.max(k.0), kmax.1.max(k.1), kmax.2.max(k.2));
            buckets.entry(k).or_default().push(idx);
        }
        AtomGrid {
            spheres,
            cell,
            buckets,
            kmin,
            kmax,
        }
    }

    #[inline]
    fn key(p: Vec3, cell: f64) -> (i64, i64, i64) {
        (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        )
    }

    /// Visit every atom whose bucket is within `reach` cells of `p`. With
    /// `cell = max inflated radius`, any atom whose surface lies within one
    /// `cell` of `p` (the relevant range for both nearest-surface and burial
    /// queries) has its centre ≤ `2·cell` away in distance, i.e. ≤ 2 buckets
    /// after `floor` bucketing — hence callers use `reach = NEAR_REACH` (2). The
    /// per-atom distance test inside the callback rejects false positives.
    fn for_each_near(&self, p: Vec3, reach: i64, mut f: impl FnMut(usize)) {
        let (kx, ky, kz) = Self::key(p, self.cell);
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    if let Some(b) = self.buckets.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &i in b {
                            f(i);
                        }
                    }
                }
            }
        }
    }

    /// Visit every atom whose bucket is at Chebyshev distance **exactly** `r` from
    /// key `k` (the shell at radius `r`; `r = 0` is the centre bucket). Used by the
    /// expanding-ring nearest-surface search.
    fn for_each_in_shell(&self, k: (i64, i64, i64), r: i64, mut f: impl FnMut(usize)) {
        let visit = |kk: (i64, i64, i64), f: &mut dyn FnMut(usize)| {
            if let Some(b) = self.buckets.get(&kk) {
                for &i in b {
                    f(i);
                }
            }
        };
        if r == 0 {
            visit(k, &mut f);
            return;
        }
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs().max(dy.abs()).max(dz.abs()) == r {
                        visit((k.0 + dx, k.1 + dy, k.2 + dz), &mut f);
                    }
                }
            }
        }
    }

    /// The analytic nearest point on the SAS (the boundary of the union): the
    /// closest *exposed* radial projection onto an inflated sphere.
    ///
    /// **Expanding-ring search** (correctness fix — a fixed bucket reach is
    /// unsound here, codex/GPU-oracle review): the closest *exposed* projection can
    /// sit on an atom several buckets away when every nearer projection is buried,
    /// so we grow the search shell by shell and stop only once it is *provable* no
    /// farther atom can beat the current best. An atom first appearing at shell `r`
    /// has its centre ≥ `(r−1)·cell` from `p` (one axis bucket differs by `r`), so
    /// its nearest surface point is ≥ `(r−2)·cell` away (radius ≤ `cell`); once that
    /// lower bound exceeds the best exposed distance, shells ≥ `r` cannot improve
    /// it. Exposure still uses the sound fixed `NEAR_REACH` (a burying atom is
    /// always within one `cell`). Falls back to covering every atom (`max_r`) when
    /// no nearby projection is exposed. `None` only if the structure has no atoms.
    fn nearest_surface_point(&self, p: Vec3) -> Option<Vec3> {
        let (kx, ky, kz) = Self::key(p, self.cell);
        let span = |k: i64, lo: i64, hi: i64| (k - lo).abs().max((k - hi).abs());
        let max_r = span(kx, self.kmin.0, self.kmax.0)
            .max(span(ky, self.kmin.1, self.kmax.1))
            .max(span(kz, self.kmin.2, self.kmax.2));

        let mut best: Option<(f64, Vec3)> = None;
        let mut r = 0i64;
        while r <= max_r {
            // Provably done: atoms in shell ≥ r are ≥ (r−2)·cell from p.
            if let Some((d_best, _)) = best {
                let lb = (r - 2).max(0) as f64 * self.cell;
                if lb * lb > d_best {
                    break;
                }
            }
            self.for_each_in_shell((kx, ky, kz), r, |i| {
                let s = self.spheres[i];
                let Some(dir) = (p - s.center).normalized() else {
                    return; // p at the centre — degenerate, skip
                };
                let proj = s.center + dir * s.radius;
                // Exposed = not strictly inside any other inflated sphere.
                let mut exposed = true;
                self.for_each_near(proj, NEAR_REACH, |j| {
                    if j != i {
                        let o = self.spheres[j];
                        if proj.square_distance(o.center) < o.radius * o.radius - 1e-9 {
                            exposed = false;
                        }
                    }
                });
                if exposed {
                    let d = p.square_distance(proj);
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, proj));
                    }
                }
            });
            r += 1;
        }
        best.map(|(_, pt)| pt)
    }
}

/// Jump-flooding vector distance transform: each node adopts the nearest seeded
/// feature point. Halving passes from `next_pow2(reach)` down to 1 propagate each
/// seed up to ~`reach` nodes — enough to make the distance exact in the band
/// straddling `f = 0`; farther nodes stay NaN (sign-only, which is all they need).
/// `pos(i,j,k)` maps a node to world space.
fn jump_flood(
    feat: &mut [[f64; 3]],
    dims: [usize; 3],
    reach: usize,
    pos: &(impl Fn(usize, usize, usize) -> Vec3 + Sync),
) {
    let [nx, ny, nz] = dims;
    let nxy = nx * ny;
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    // Halving schedule next_pow2(reach) … 2, 1, then one extra unit pass — the
    // "JFA+1" variant, which cleans up the rare wrong-nearest cell vanilla JFA
    // leaves near Voronoi boundaries. (With dense band seeding the base error is
    // already tiny; this is cheap insurance.)
    let mut schedule: Vec<usize> = Vec::new();
    let mut step = reach.max(1).next_power_of_two();
    while step >= 1 {
        schedule.push(step);
        step /= 2;
    }
    schedule.push(1);

    let mut src = feat.to_vec();
    let mut dst = feat.to_vec();
    for step in schedule {
        // Each output node reads only the immutable `src` (double-buffered) and
        // writes only its own `dst[cell]` — independent across nodes, so the pass
        // is data-parallel with a result identical to the serial sweep.
        let src_ref = &src;
        dst.par_iter_mut().enumerate().for_each(|(cell, out)| {
            let i = cell % nx;
            let j = (cell / nx) % ny;
            let k = cell / nxy;
            let here = pos(i, j, k);
            let cur = src_ref[cell];
            let mut best = cur;
            let mut bestd = if cur[0].is_nan() {
                f64::INFINITY
            } else {
                here.square_distance(Vec3::new(cur[0], cur[1], cur[2]))
            };
            for dk in [-(step as isize), 0, step as isize] {
                let kk = k as isize + dk;
                if kk < 0 || kk as usize >= nz {
                    continue;
                }
                for dj in [-(step as isize), 0, step as isize] {
                    let jj = j as isize + dj;
                    if jj < 0 || jj as usize >= ny {
                        continue;
                    }
                    for di in [-(step as isize), 0, step as isize] {
                        let ii = i as isize + di;
                        if ii < 0 || ii as usize >= nx {
                            continue;
                        }
                        let cand = src_ref[idx(ii as usize, jj as usize, kk as usize)];
                        if cand[0].is_nan() {
                            continue;
                        }
                        let d = here.square_distance(Vec3::new(cand[0], cand[1], cand[2]));
                        if d < bestd {
                            bestd = d;
                            best = cand;
                        }
                    }
                }
            }
            *out = best;
        });
        std::mem::swap(&mut src, &mut dst);
    }
    feat.copy_from_slice(&src);
}

// The 12 edges of a cube, as pairs of corner indices (corner c = bit pattern
// (dx, dy, dz) with dx the low bit).
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // x-edges
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // y-edges
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // z-edges
];
// Corner offsets matching the bit pattern above.
const CORNER: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

// The 6 cube faces, each as its 4 cube-edge indices in cyclic order, and the 4
// corner indices in the matching cyclic order. `FACE_EDGES[f][t]` is the edge
// between `FACE_CORNERS[f][t]` and `FACE_CORNERS[f][(t+1)%4]`.
const FACE_EDGES: [[usize; 4]; 6] = [
    [4, 10, 6, 8],  // x=0
    [5, 11, 7, 9],  // x=1
    [0, 9, 2, 8],   // y=0
    [1, 11, 3, 10], // y=1
    [0, 5, 1, 4],   // z=0
    [2, 7, 3, 6],   // z=1
];
const FACE_CORNERS: [[usize; 4]; 6] = [
    [0, 2, 6, 4],
    [1, 3, 7, 5],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 1, 3, 2],
    [4, 5, 7, 6],
];

fn uf_find(parent: &mut [usize; 12], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != c {
        let n = parent[c];
        parent[c] = r;
        c = n;
    }
    r
}
fn uf_union(parent: &mut [usize; 12], a: usize, b: usize) {
    let (ra, rb) = (uf_find(parent, a), uf_find(parent, b));
    parent[ra] = rb;
}

/// **Manifold dual contouring.** Like surface nets — a dual vertex per cell, a
/// quad per sign-changing grid edge — but a cell whose surface passes through as
/// *several disconnected sheets* gets **one vertex per sheet**, and each quad
/// corner is routed to the sheet that owns its grid edge. That removes the
/// non-manifold pinches naive surface nets makes at saddle cells (which appear
/// on real protein surfaces), giving a 2-manifold by construction.
///
/// Sheets are the connected components of the cell's 12 edge-crossings, joined
/// by the marching-squares segments on each of the 6 faces; the lone ambiguous
/// face (all four edges crossing) is resolved by the bilinear **asymptotic
/// decider**, deterministic from the four shared corner values so the two cells
/// across that face always agree. Orientation is finalized by the caller.
fn manifold_dual_contour(grid: &Grid, f: &[f64]) -> Mesh {
    let [nx, ny, nz] = grid.dims;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let cell_index = |i: usize, j: usize, k: usize| i + cx * (j + cy * k);
    let mut verts: Vec<Vec3> = Vec::new();
    // (cell, cube-edge) → the vertex of the sheet that edge belongs to.
    let mut edge_vert: std::collections::HashMap<(usize, usize), u32> =
        std::collections::HashMap::new();

    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut cf = [0.0f64; 8];
                for (c, off) in CORNER.iter().enumerate() {
                    cf[c] = f[grid.idx(i + off[0], j + off[1], k + off[2])];
                }
                let neg = cf.iter().filter(|&&x| x < 0.0).count();
                if neg == 0 || neg == 8 {
                    continue;
                }
                // Crossing position per cube edge.
                let mut crossing = [false; 12];
                let mut cpos = [Vec3::new(0.0, 0.0, 0.0); 12];
                for (e, &(a, b)) in CUBE_EDGES.iter().enumerate() {
                    let (fa, fb) = (cf[a], cf[b]);
                    if (fa < 0.0) == (fb < 0.0) {
                        continue;
                    }
                    // A crossing endpoint must be a reached (finite) node; the
                    // UNREACHED sentinel never borders the surface, so this guards
                    // a NaN/∞ interpolation.
                    debug_assert!(
                        fa.abs() < 1e17 && fb.abs() < 1e17,
                        "sign-changing edge touches an unreached node"
                    );
                    let t = fa / (fa - fb);
                    let pa = grid.pos(i + CORNER[a][0], j + CORNER[a][1], k + CORNER[a][2]);
                    let pb = grid.pos(i + CORNER[b][0], j + CORNER[b][1], k + CORNER[b][2]);
                    crossing[e] = true;
                    cpos[e] = pa + (pb - pa) * t;
                }

                // Join crossings into sheets via per-face marching-squares segments.
                let mut parent: [usize; 12] = std::array::from_fn(|e| e);
                for fa in 0..6 {
                    let fe = FACE_EDGES[fa];
                    let crossed = [
                        crossing[fe[0]],
                        crossing[fe[1]],
                        crossing[fe[2]],
                        crossing[fe[3]],
                    ];
                    match crossed.iter().filter(|&&x| x).count() {
                        2 => {
                            let mut it = (0..4).filter(|&t| crossed[t]);
                            let (a, b) = (it.next().unwrap(), it.next().unwrap());
                            uf_union(&mut parent, fe[a], fe[b]);
                        }
                        4 => {
                            let fc = FACE_CORNERS[fa];
                            let (f0, f1, f2, f3) = (cf[fc[0]], cf[fc[1]], cf[fc[2]], cf[fc[3]]);
                            let denom = f0 - f1 + f2 - f3;
                            // Asymptotic decider: is the f0/f2 diagonal joined
                            // through the face interior? Then the surface cuts off
                            // corners c1 and c3 (segments {E0,E1} and {E2,E3}).
                            let joined_02 = if denom.abs() < 1e-12 {
                                true
                            } else {
                                ((f0 * f2 - f1 * f3) / denom < 0.0) == (f0 < 0.0)
                            };
                            if joined_02 {
                                uf_union(&mut parent, fe[0], fe[1]);
                                uf_union(&mut parent, fe[2], fe[3]);
                            } else {
                                uf_union(&mut parent, fe[1], fe[2]);
                                uf_union(&mut parent, fe[3], fe[0]);
                            }
                        }
                        _ => {}
                    }
                }

                // One vertex per sheet = mean of its crossings.
                let mut sheet: std::collections::HashMap<usize, (Vec3, f64, u32)> =
                    std::collections::HashMap::new();
                for e in 0..12 {
                    if crossing[e] {
                        let r = uf_find(&mut parent, e);
                        let entry = sheet.entry(r).or_insert_with(|| {
                            let vid = verts.len() as u32;
                            verts.push(Vec3::new(0.0, 0.0, 0.0));
                            (Vec3::new(0.0, 0.0, 0.0), 0.0, vid)
                        });
                        entry.0 = entry.0 + cpos[e];
                        entry.1 += 1.0;
                    }
                }
                for (acc, n, vid) in sheet.values() {
                    verts[*vid as usize] = *acc * (1.0 / *n);
                }
                for e in 0..12 {
                    if crossing[e] {
                        let r = uf_find(&mut parent, e);
                        edge_vert.insert((cell_index(i, j, k), e), sheet[&r].2);
                    }
                }
            }
        }
    }

    // One quad per interior sign-changing grid edge. The four incident cells, in
    // cyclic order, each contribute the sheet vertex of *their* copy of that grid
    // edge (cube-edge index given per direction).
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let quad = |slots: [(usize, usize); 4], flip: bool, tris: &mut Vec<[u32; 3]>| {
        let v: [Option<u32>; 4] = std::array::from_fn(|t| edge_vert.get(&slots[t]).copied());
        debug_assert!(
            v.iter().all(Option::is_some),
            "crossing grid edge with a cell missing its sheet vertex → hole"
        );
        if let [Some(a), Some(b), Some(c), Some(d)] = v {
            if flip {
                tris.push([a, c, b]);
                tris.push([a, d, c]);
            } else {
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let here = f[grid.idx(i, j, k)] < 0.0;
                if i + 1 < nx && j > 0 && k > 0 && here != (f[grid.idx(i + 1, j, k)] < 0.0) {
                    quad(
                        [
                            (cell_index(i, j - 1, k - 1), 3),
                            (cell_index(i, j, k - 1), 2),
                            (cell_index(i, j, k), 0),
                            (cell_index(i, j - 1, k), 1),
                        ],
                        here,
                        &mut tris,
                    );
                }
                if j + 1 < ny && i > 0 && k > 0 && here != (f[grid.idx(i, j + 1, k)] < 0.0) {
                    quad(
                        [
                            (cell_index(i - 1, j, k - 1), 7),
                            (cell_index(i, j, k - 1), 6),
                            (cell_index(i, j, k), 4),
                            (cell_index(i - 1, j, k), 5),
                        ],
                        !here,
                        &mut tris,
                    );
                }
                if k + 1 < nz && i > 0 && j > 0 && here != (f[grid.idx(i, j, k + 1)] < 0.0) {
                    quad(
                        [
                            (cell_index(i - 1, j - 1, k), 11),
                            (cell_index(i, j - 1, k), 10),
                            (cell_index(i, j, k), 8),
                            (cell_index(i - 1, j, k), 9),
                        ],
                        here,
                        &mut tris,
                    );
                }
            }
        }
    }

    Mesh {
        verts,
        normals: Vec::new(), // optional; area/volume/topology don't need them
        tris,
    }
}

// ---------------------------------------------------------------------------
// GPU-K1 spike: the seed stage (nearest exposed surface point per boundary
// node) on the GPU, vs serial / 16-core CPU. Brute-force kernel (no GPU spatial
// hash yet); measures raw throughput + parity. Behind the `cuda` feature.
// ---------------------------------------------------------------------------

/// Result of [`seed_bench`].
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct SeedBench {
    pub n_atoms: usize,
    pub n_boundary: usize,
    pub cpu_serial_ms: f64,
    pub cpu_parallel_ms: f64,
    pub gpu_kernel_ms: f64,
    pub gpu_total_ms: f64, // upload + kernel + download
    pub max_feature_diff: f64,
    pub mismatched: usize, // GPU vs CPU-spatial-hash, differ > 1e-6
    // Localization: does the GPU brute-force agree with a CPU brute-force (→ the
    // disagreement is the CPU spatial-hash prune, not the kernel)?
    pub gpu_vs_cpubrute_mismatch: usize,
    pub hash_vs_cpubrute_mismatch: usize,
    /// Of the hash-vs-brute mismatches, how many have a genuinely different
    /// nearest *distance* (a real prune bug) vs equal distance (benign tie).
    pub hash_vs_brute_distance_bug: usize,
    pub max_distance_error: f64,
}

/// CPU brute-force nearest exposed surface point (loops ALL inflated atoms — the
/// exact logic the GPU kernel runs, no spatial hash). The ground-truth reference
/// for the spike's parity localization.
#[cfg(feature = "cuda")]
fn nearest_surface_brute(p: Vec3, spheres: &[Sphere]) -> [f64; 3] {
    let mut best: Option<(f64, Vec3)> = None;
    for (i, s) in spheres.iter().enumerate() {
        let Some(dir) = (p - s.center).normalized() else {
            continue;
        };
        let proj = s.center + dir * s.radius;
        let mut exposed = true;
        for (j, o) in spheres.iter().enumerate() {
            if j != i && proj.square_distance(o.center) < o.radius * o.radius - 1e-9 {
                exposed = false;
                break;
            }
        }
        if exposed {
            let d = p.square_distance(proj);
            if best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, proj));
            }
        }
    }
    best.map_or([f64::NAN; 3], |(_, pt)| [pt.x, pt.y, pt.z])
}

/// The boundary nodes (6-neighbour sign-change cells) of the SES grid, as world
/// positions — the inputs the seed stage runs on. Shared by CPU and GPU paths.
#[cfg(feature = "cuda")]
fn boundary_nodes(grid: &Grid, atoms: &[Sphere], probe: f64) -> Vec<Vec3> {
    let [nx, ny, nz] = grid.dims;
    let ag = AtomGrid::build(atoms, probe);
    let n = nx * ny * nz;
    let mut inside = vec![false; n];
    let inv_h = 1.0 / grid.spacing;
    for s in &ag.spheres {
        let r2 = s.radius * s.radius;
        let lo = |c: f64, o: f64| (((c - s.radius - o) * inv_h).floor() as isize).max(0);
        let hi = |c: f64, o: f64, m: usize| {
            (((c - o + s.radius) * inv_h).ceil() as isize).min(m as isize - 1)
        };
        let (i0, i1) = (
            lo(s.center.x, grid.origin.x),
            hi(s.center.x, grid.origin.x, nx),
        );
        let (j0, j1) = (
            lo(s.center.y, grid.origin.y),
            hi(s.center.y, grid.origin.y, ny),
        );
        let (k0, k1) = (
            lo(s.center.z, grid.origin.z),
            hi(s.center.z, grid.origin.z, nz),
        );
        for k in k0..=k1 {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let (i, j, k) = (i as usize, j as usize, k as usize);
                    if grid.pos(i, j, k).square_distance(s.center) <= r2 {
                        inside[grid.idx(i, j, k)] = true;
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let ins = inside[grid.idx(i, j, k)];
                let boundary = [
                    (i + 1 < nx).then(|| grid.idx(i + 1, j, k)),
                    (i > 0).then(|| grid.idx(i - 1, j, k)),
                    (j + 1 < ny).then(|| grid.idx(i, j + 1, k)),
                    (j > 0).then(|| grid.idx(i, j - 1, k)),
                    (k + 1 < nz).then(|| grid.idx(i, j, k + 1)),
                    (k > 0).then(|| grid.idx(i, j, k - 1)),
                ]
                .into_iter()
                .flatten()
                .any(|nb| inside[nb] != ins);
                if boundary {
                    out.push(grid.pos(i, j, k));
                }
            }
        }
    }
    out
}

/// GPU-K1 spike: time the SES seed (nearest exposed surface point per boundary
/// node) on serial CPU, 16-core CPU, and the GPU brute-force kernel, and report
/// parity (max feature difference). The CPU `nearest_surface_point` uses the
/// spatial hash; the GPU kernel is brute-force over all inflated atoms — they
/// must agree (the hash only prunes provably-irrelevant atoms), and a production
/// GPU kernel would add the hash. This is a *learning* spike, not production.
#[cfg(feature = "cuda")]
pub fn seed_bench(
    atoms: &[Sphere],
    probe: f64,
    spacing: f64,
) -> Result<SeedBench, Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use std::time::Instant;

    let grid = Grid::enclosing(atoms, probe, spacing);
    let ag = AtomGrid::build(atoms, probe);
    let bnodes = boundary_nodes(&grid, atoms, probe);
    let nb = bnodes.len();

    // CPU serial.
    let t = Instant::now();
    let cpu_feat: Vec<[f64; 3]> = bnodes
        .iter()
        .map(|&p| {
            ag.nearest_surface_point(p)
                .map_or([f64::NAN; 3], |s| [s.x, s.y, s.z])
        })
        .collect();
    let cpu_serial_ms = t.elapsed().as_secs_f64() * 1e3;

    // CPU 16-core (the real baseline the GPU must beat).
    let t = Instant::now();
    let _cpu_par: Vec<[f64; 3]> = bnodes
        .par_iter()
        .map(|&p| {
            ag.nearest_surface_point(p)
                .map_or([f64::NAN; 3], |s| [s.x, s.y, s.z])
        })
        .collect();
    let cpu_parallel_ms = t.elapsed().as_secs_f64() * 1e3;

    // GPU brute-force.
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(include_str!("seed_kernel.cu"), opts)?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("seed_brute")?;
    let stream = ctx.default_stream();

    let nodes_flat: Vec<f64> = bnodes.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let atoms_flat: Vec<f64> = ag
        .spheres
        .iter()
        .flat_map(|s| [s.center.x, s.center.y, s.center.z, s.radius])
        .collect();
    let m_i32 = ag.spheres.len() as i32;
    let nb_i32 = nb as i32;

    let t = Instant::now();
    let d_nodes = stream.clone_htod(&nodes_flat)?;
    let d_atoms = stream.clone_htod(&atoms_flat)?;
    let mut d_feat = stream.alloc_zeros::<f64>(nb * 3)?;
    stream.synchronize()?;
    let tk = Instant::now();
    {
        let mut a = stream.launch_builder(&kernel);
        a.arg(&d_nodes);
        a.arg(&d_atoms);
        a.arg(&nb_i32);
        a.arg(&m_i32);
        a.arg(&mut d_feat);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(nb as u32))?;
        }
    }
    stream.synchronize()?;
    let gpu_kernel_ms = tk.elapsed().as_secs_f64() * 1e3;
    let gpu_flat = stream.clone_dtoh(&d_feat)?;
    let gpu_total_ms = t.elapsed().as_secs_f64() * 1e3;

    // CPU brute-force reference (same logic as the kernel) to localize any
    // disagreement: kernel bug (gpu ≠ cpu-brute) vs CPU spatial-hash prune
    // (cpu-hash ≠ cpu-brute).
    let cpu_brute: Vec<[f64; 3]> = bnodes
        .par_iter()
        .map(|&p| nearest_surface_brute(p, &ag.spheres))
        .collect();

    let differ = |a: [f64; 3], b: [f64; 3]| -> Option<f64> {
        if a[0].is_nan() && b[0].is_nan() {
            return None;
        }
        if a[0].is_nan() != b[0].is_nan() {
            return Some(f64::INFINITY);
        }
        let d = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
        (d > 1e-6).then_some(d)
    };

    let mut max_feature_diff = 0.0_f64;
    let mut mismatched = 0usize;
    let mut gpu_vs_cpubrute_mismatch = 0usize;
    let mut hash_vs_cpubrute_mismatch = 0usize;
    let mut hash_vs_brute_distance_bug = 0usize;
    let mut max_distance_error = 0.0_f64;
    for (i, c) in cpu_feat.iter().enumerate() {
        let g = [gpu_flat[3 * i], gpu_flat[3 * i + 1], gpu_flat[3 * i + 2]];
        if let Some(d) = differ(*c, g) {
            mismatched += 1;
            if d.is_finite() {
                max_feature_diff = max_feature_diff.max(d);
            }
        }
        if differ(g, cpu_brute[i]).is_some() {
            gpu_vs_cpubrute_mismatch += 1;
        }
        if differ(*c, cpu_brute[i]).is_some() {
            hash_vs_cpubrute_mismatch += 1;
            // Distance from the node to each candidate nearest-surface point: if
            // they differ, the hash genuinely missed a closer/farther point (a
            // prune bug); if equal, it's an equidistant tie (benign).
            let p = bnodes[i];
            let b = cpu_brute[i];
            if !c[0].is_nan() && !b[0].is_nan() {
                let dh = p.distance(Vec3::new(c[0], c[1], c[2]));
                let db = p.distance(Vec3::new(b[0], b[1], b[2]));
                let de = (dh - db).abs();
                if de > 1e-6 {
                    hash_vs_brute_distance_bug += 1;
                    max_distance_error = max_distance_error.max(de);
                }
            } else {
                // One found a point, the other NaN — a real search discrepancy.
                hash_vs_brute_distance_bug += 1;
            }
        }
    }

    Ok(SeedBench {
        n_atoms: atoms.len(),
        n_boundary: nb,
        cpu_serial_ms,
        cpu_parallel_ms,
        gpu_kernel_ms,
        gpu_total_ms,
        max_feature_diff,
        mismatched,
        gpu_vs_cpubrute_mismatch,
        hash_vs_cpubrute_mismatch,
        hash_vs_brute_distance_bug,
        max_distance_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// Parity gate for the GPU SDF seed. The invariant that matters for the
    /// distance field is the nearest-feature *distance* at each boundary node
    /// (what `jump_flood` propagates), not the exact feature point: when two
    /// exposed projections are equidistant, the CPU (spatial-hash order) and GPU
    /// (atom-array order) may pick different — but equally valid — points. So we
    /// assert equal exposed/NaN status and equal node→feature distance, and
    /// additionally that the points themselves agree on the (overwhelming)
    /// non-tie majority. Runs only with `cuda` *and* a usable device; skips
    /// otherwise (mirrors production's silent CPU fallback).
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_seed_matches_cpu_seed_on_boundary_nodes() {
        // A dense overlapping cluster — exactly where the exposure test (the
        // nearest *exposed* projection) is nontrivial.
        let atoms = vec![
            s(0.0, 0.0, 0.0, 1.7),
            s(1.5, 0.0, 0.0, 1.7),
            s(0.0, 1.5, 0.0, 1.5),
            s(0.6, 0.6, 1.4, 1.5),
            s(-1.4, 0.3, 0.2, 1.6),
        ];
        let probe = 1.4;
        let spacing = 0.4;
        let grid = Grid::enclosing(&atoms, probe, spacing);
        let ag = AtomGrid::build(&atoms, probe);
        let bnodes = boundary_nodes(&grid, &atoms, probe);
        assert!(!bnodes.is_empty(), "no boundary nodes to test");

        let Some(gpu) = crate::surface::seed_gpu::seed_boundary_gpu(&bnodes, &ag.spheres) else {
            eprintln!("skipping: no usable GPU");
            return;
        };
        let cpu: Vec<[f64; 3]> = bnodes
            .iter()
            .map(|&p| {
                ag.nearest_surface_point(p)
                    .map_or([f64::NAN; 3], |q| [q.x, q.y, q.z])
            })
            .collect();
        assert_eq!(cpu.len(), gpu.len());

        let dist = |p: Vec3, f: &[f64; 3]| (p - Vec3::new(f[0], f[1], f[2])).norm();
        let mut max_dist_err = 0.0_f64;
        let mut point_mismatches = 0usize;
        for (&p, (c, g)) in bnodes.iter().zip(cpu.iter().zip(&gpu)) {
            assert_eq!(c[0].is_nan(), g[0].is_nan(), "exposed/NaN status differs");
            if c[0].is_nan() {
                continue;
            }
            // Distance to the chosen feature must agree (the SDF invariant).
            max_dist_err = max_dist_err.max((dist(p, c) - dist(p, g)).abs());
            // Track exact-point disagreements (allowed only for equidistant ties).
            if (0..3).any(|k| (c[k] - g[k]).abs() > 1e-6) {
                point_mismatches += 1;
            }
        }
        assert!(
            max_dist_err < 1e-6,
            "GPU vs CPU nearest distance differs by {max_dist_err} Å"
        );
        // On real coordinates ties are vanishingly rare; this fixture has none,
        // so the points are bit-identical too. (A tie would still pass the
        // distance assert above.)
        assert_eq!(
            point_mismatches, 0,
            "unexpected feature-point disagreements"
        );
    }

    /// The GPU jump-flood must reproduce the CPU transform: same reached/unreached
    /// status and the same flooded *distance* at every node (the field input).
    /// Equidistant ties can pick a different but equally-near feature, so we
    /// compare distance, not the feature point. Cuda + device only; else skips.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_jump_flood_matches_cpu() {
        let dims = [24usize, 22, 20];
        let spacing = 0.4;
        let origin = Vec3::new(-1.0, 2.0, 0.5);
        let grid = Grid {
            origin,
            spacing,
            dims,
        };
        let [nx, ny, nz] = dims;
        let n = nx * ny * nz;

        // Scatter a handful of seeds (feature = the node's own world position),
        // the rest NaN — a clean input to flood.
        let mut seeded = vec![[f64::NAN; 3]; n];
        for &(i, j, k) in &[
            (2, 3, 4),
            (20, 5, 2),
            (7, 18, 15),
            (12, 12, 10),
            (0, 0, 0),
            (23, 21, 19),
            (5, 9, 3),
        ] {
            let p = grid.pos(i, j, k);
            seeded[i + nx * (j + ny * k)] = [p.x, p.y, p.z];
        }
        let reach = 12usize;

        let mut cpu = seeded.clone();
        jump_flood(&mut cpu, dims, reach, &|i, j, k| grid.pos(i, j, k));
        let Some(gpu) =
            crate::surface::seed_gpu::jump_flood_gpu(&seeded, dims, reach, origin, spacing)
        else {
            eprintln!("skipping: no usable GPU");
            return;
        };
        assert_eq!(cpu.len(), gpu.len());

        let mut max_dist_err = 0.0_f64;
        let mut nan_mismatch = 0usize;
        for idx in 0..n {
            let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
            let here = grid.pos(i, j, k);
            let dist = |f: &[f64; 3]| (here - Vec3::new(f[0], f[1], f[2])).norm();
            if cpu[idx][0].is_nan() != gpu[idx][0].is_nan() {
                nan_mismatch += 1;
                continue;
            }
            if !cpu[idx][0].is_nan() {
                max_dist_err = max_dist_err.max((dist(&cpu[idx]) - dist(&gpu[idx])).abs());
            }
        }
        assert_eq!(nan_mismatch, 0, "reached/unreached status differs");
        assert!(
            max_dist_err < 1e-9,
            "GPU vs CPU flooded distance differs by {max_dist_err}"
        );
    }

    /// The fused GPU seed+flood (`seed_and_flood_gpu`, the production path) must
    /// equal the *unfused all-GPU pipeline* — the GPU seed (`seed_boundary_gpu`)
    /// scattered on the host into the full grid, then the GPU flood
    /// (`jump_flood_gpu`). Both use the same seed and JFA kernels, so the only
    /// difference is that the fused path keeps the buffer on-device (NaN fill +
    /// scatter at each boundary index + JFA chained without a host round-trip)
    /// instead of downloading, scattering, and re-uploading. The values are
    /// therefore bit-identical.
    ///
    /// (We deliberately do *not* compare against a CPU flood here: on a real,
    /// tie-rich seed, GPU-JFA and CPU-JFA tie-break equidistant features
    /// differently, and a downstream node inherits a different — equally near at
    /// the seed, but not downstream — feature, so distances diverge at the ~1e-2
    /// level. That is correct behaviour, documented on `gpu_jump_flood_matches_
    /// cpu`, whose synthetic seed has no ties; it just isn't what *fusion*
    /// guards.) Cuda + device only.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_fused_seed_flood_matches_unfused() {
        // Same dense overlapping cluster as the seed-parity test — nontrivial
        // exposure plus real flooding to propagate.
        let atoms = vec![
            s(0.0, 0.0, 0.0, 1.7),
            s(1.5, 0.0, 0.0, 1.7),
            s(0.0, 1.5, 0.0, 1.5),
            s(0.6, 0.6, 1.4, 1.5),
            s(-1.4, 0.3, 0.2, 1.6),
        ];
        let probe = 1.4;
        let spacing = 0.4;
        let grid = Grid::enclosing(&atoms, probe, spacing);
        let [nx, ny, nz] = grid.dims;
        let n = nx * ny * nz;
        let ag = AtomGrid::build(&atoms, probe);

        // Boundary nodes + their flat grid indices (positions are exactly
        // origin + idx·spacing, so the inverse mapping rounds back exactly).
        let bnodes = boundary_nodes(&grid, &atoms, probe);
        assert!(!bnodes.is_empty(), "no boundary nodes to test");
        let bidx: Vec<usize> = bnodes
            .iter()
            .map(|p| {
                let i = ((p.x - grid.origin.x) / spacing).round() as usize;
                let j = ((p.y - grid.origin.y) / spacing).round() as usize;
                let k = ((p.z - grid.origin.z) / spacing).round() as usize;
                grid.idx(i, j, k)
            })
            .collect();
        let reach = (probe / spacing).ceil() as usize + 4;

        // Unfused reference: the SAME GPU seed, scattered on the host into the
        // full grid, then the GPU jump-flood (identical JFA kernel to the fused
        // path — so any difference would be the fusion's scatter/chaining).
        let Some(seed_feats) = crate::surface::seed_gpu::seed_boundary_gpu(&bnodes, &ag.spheres)
        else {
            eprintln!("skipping: no usable GPU");
            return;
        };
        let mut seeded = vec![[f64::NAN; 3]; n];
        for (b, &idx) in bidx.iter().enumerate() {
            seeded[idx] = seed_feats[b];
        }
        let Some(reference) = crate::surface::seed_gpu::jump_flood_gpu(
            &seeded,
            grid.dims,
            reach,
            grid.origin,
            spacing,
        ) else {
            eprintln!("skipping: no usable GPU");
            return;
        };

        let Some(gpu) = crate::surface::seed_gpu::seed_and_flood_gpu(
            &bnodes,
            &bidx,
            &ag.spheres,
            grid.dims,
            reach,
            grid.origin,
            spacing,
        ) else {
            eprintln!("skipping: no usable GPU");
            return;
        };
        assert_eq!(reference.len(), gpu.len());

        let mut max_dist_err = 0.0_f64;
        let mut nan_mismatch = 0usize;
        for idx in 0..n {
            let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
            let here = grid.pos(i, j, k);
            let dist = |f: &[f64; 3]| (here - Vec3::new(f[0], f[1], f[2])).norm();
            if reference[idx][0].is_nan() != gpu[idx][0].is_nan() {
                nan_mismatch += 1;
                continue;
            }
            if !reference[idx][0].is_nan() {
                max_dist_err = max_dist_err.max((dist(&reference[idx]) - dist(&gpu[idx])).abs());
            }
        }
        assert_eq!(
            nan_mismatch, 0,
            "fused vs unfused reached/unreached status differs"
        );
        assert!(
            max_dist_err < 1e-9,
            "fused GPU vs unfused (GPU seed + CPU flood) flooded distance differs by {max_dist_err}"
        );
    }

    /// Regression: the spatial-hash `nearest_surface_point` must return the SAME
    /// nearest *exposed* surface point as an exhaustive (all-atoms) reference, even
    /// in a dense cluster where deep pockets put the nearest exposed point several
    /// buckets away. A fixed bucket reach (the old `NEAR_REACH = 2`) was unsound and
    /// returned a point up to ~1.2 Å too far on ~1–2% of nodes (found by a GPU
    /// brute-force oracle); the expanding-ring search must match brute force exactly.
    #[test]
    fn nearest_surface_point_matches_brute_force() {
        // Exhaustive reference: nearest exposed radial projection over ALL spheres.
        fn brute(ag: &AtomGrid, p: Vec3) -> Option<Vec3> {
            let mut best: Option<(f64, Vec3)> = None;
            for (i, sp) in ag.spheres.iter().enumerate() {
                let Some(dir) = (p - sp.center).normalized() else {
                    continue;
                };
                let proj = sp.center + dir * sp.radius;
                let exposed = ag.spheres.iter().enumerate().all(|(j, o)| {
                    j == i || proj.square_distance(o.center) >= o.radius * o.radius - 1e-9
                });
                if exposed {
                    let d = p.square_distance(proj);
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, proj));
                    }
                }
            }
            best.map(|(_, q)| q)
        }

        // A dense deterministic cluster (LCG) → lots of burial, deep pockets.
        let mut z = 0x1234_5678u64;
        let mut rng = || {
            z = z.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((z >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let probe = 1.4;
        let atoms: Vec<Sphere> = (0..150)
            .map(|_| s(rng() * 9.0, rng() * 9.0, rng() * 9.0, 1.4 + rng() * 0.5))
            .collect();
        let ag = AtomGrid::build(&atoms, probe);

        // Query a lattice of points spanning (and just outside) the cluster.
        let mut checked = 0usize;
        let mut g = -1.0;
        while g <= 10.0 {
            let mut h = -1.0;
            while h <= 10.0 {
                let mut k = -1.0;
                while k <= 10.0 {
                    let p = Vec3::new(g, h, k);
                    let hash = ag.nearest_surface_point(p);
                    let bf = brute(&ag, p);
                    match (hash, bf) {
                        (Some(a), Some(b)) => {
                            // Compare nearest DISTANCE (the meaningful quantity;
                            // equidistant ties are allowed to differ in point).
                            let da = p.distance(a);
                            let db = p.distance(b);
                            assert!(
                                (da - db).abs() < 1e-9,
                                "hash {da} vs brute {db} at {p:?} (prune missed a closer point)"
                            );
                            checked += 1;
                        }
                        (None, None) => {}
                        _ => panic!("hash/brute disagree on existence at {p:?}"),
                    }
                    k += 0.7;
                }
                h += 0.7;
            }
            g += 0.7;
        }
        assert!(
            checked > 200,
            "expected many exposed queries, got {checked}"
        );
    }

    #[test]
    #[ignore = "diagnostic convergence probe; run manually with --ignored --nocapture"]
    fn sweep_resolution() {
        let single = [s(0.0, 0.0, 0.0, 1.8)];
        let pair = [s(0.0, 0.0, 0.0, 1.8), s(2.5, 0.0, 0.0, 1.8)];
        for h in [0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1] {
            let m1 = ses_mesh_sdf(&single, 1.4, h);
            let m2 = ses_mesh_sdf(&pair, 1.4, h);
            println!(
                "h={h:.2}  single A={:.3} (exact 40.715, {:+.2}%)  pair A={:.3} (ball 67.796, {:+.2}%) V={:.3} (ball 46.62)",
                m1.surface_area(),
                100.0 * (m1.surface_area() - 40.715) / 40.715,
                m2.surface_area(),
                100.0 * (m2.surface_area() - 67.796) / 67.796,
                m2.signed_volume(),
            );
        }
    }

    /// A single atom's SES is its van-der-Waals sphere — the cleanest closed-form
    /// check, plus it confirms the grid/EDT/surface-nets pipeline end-to-end.
    #[test]
    fn single_atom_ses_is_the_vdw_sphere() {
        let mesh = ses_mesh_sdf(&[s(0.0, 0.0, 0.0, 1.8)], 1.4, 0.15);
        assert!(mesh.is_watertight(), "iso-surface must be closed");
        assert!(mesh.is_consistently_oriented());
        assert_eq!(mesh.euler_characteristic(), 2, "topological sphere");
        assert!(mesh.signed_volume() > 0.0, "outward oriented");
        let exact = 4.0 * std::f64::consts::PI * 1.8 * 1.8; // 40.715
        assert!(
            (mesh.surface_area() - exact).abs() / exact < 0.005,
            "area {} vs {exact} (within 0.5% with the vector-DT field)",
            mesh.surface_area()
        );
    }

    /// Finer grid ⇒ closer to the analytic SES area/volume (the convergence the
    /// resolution knob buys), gated against `ball-py 0.1.0a6 ses_area` (analytic
    /// Connolly path, independent of any mesh).
    #[test]
    fn ses_area_and_volume_match_ball_and_converge() {
        // (atoms, probe, ball area, ball volume)
        let pair = vec![s(0.0, 0.0, 0.0, 1.8), s(2.5, 0.0, 0.0, 1.8)];
        let tri = vec![
            s(0.0, 0.0, 0.0, 1.7),
            s(2.5, 0.0, 0.0, 1.7),
            s(1.25, 2.165, 0.0, 1.7),
        ];
        let tri_tight = vec![
            s(0.0, 0.0, 0.0, 1.6),
            s(2.0, 0.0, 0.0, 1.6),
            s(1.0, 1.7, 0.0, 1.6),
        ];
        let cases = [
            ("pair_sym", pair, 1.4, 67.7959, 46.6207),
            ("tri_equilateral", tri, 1.4, 80.0932, 57.9040),
            ("tri_tight", tri_tight, 1.4, 64.5463, 43.6869),
        ];
        for (name, atoms, probe, ball_area, ball_vol) in cases {
            let coarse = ses_mesh_sdf(&atoms, probe, 0.4);
            let fine = ses_mesh_sdf(&atoms, probe, 0.2);

            assert!(fine.is_watertight(), "{name}: closed");
            assert!(fine.is_consistently_oriented(), "{name}: oriented");
            assert_eq!(fine.euler_characteristic(), 2, "{name}: sphere topology");

            let (af, vf) = (fine.surface_area(), fine.signed_volume());
            let ac = coarse.surface_area();
            // Smooth monotone convergence toward the analytic area as h shrinks
            // (the vector-DT field has no occupancy staircase).
            assert!(
                (af - ball_area).abs() < (ac - ball_area).abs() + 1e-9,
                "{name}: area converges ({ac} → {af} vs {ball_area})"
            );
            // Sub-voxel-accurate field: within 1% at h=0.15 (see `sweep_resolution`).
            assert!(
                (af - ball_area).abs() / ball_area < 0.01,
                "{name}: fine area {af} within 1% of {ball_area}"
            );
            assert!(
                (vf - ball_vol).abs() / ball_vol < 0.01,
                "{name}: fine volume {vf} within 1% of {ball_vol}"
            );
        }
    }

    /// Adversarial *connected* surfaces that stress the field where
    /// `nearest_surface_point` is biased (the per-node radial projection is not
    /// the true nearest SAS point — that lies on an intersection rim):
    /// **heterogeneous radii** (also stresses the `cell = maxR` spatial-hash
    /// reach) and a tight **reentrant pocket**, both three-sphere so each carries
    /// rim creases. Staying within 1% is the evidence that dense band seeding +
    /// JFA bound the crease bias to `O(h)` (codex-review). The 2-sphere rim case
    /// is the existing `pair_sym` gate; barely-overlapping (near-disconnection)
    /// is deliberately *not* gated here — that is resolution-limited topology,
    /// not field accuracy. Gated vs `ball-py 0.1.0a6 ses_area`.
    #[test]
    fn crease_and_heterogeneous_cases_stay_within_tolerance() {
        let cases = [
            (
                "hetero",
                vec![
                    s(0.0, 0.0, 0.0, 2.4),
                    s(3.0, 0.0, 0.0, 1.0),
                    s(1.0, 2.2, 0.0, 1.3),
                ],
                87.5941,
                69.2654,
            ),
            (
                "tri_pocket",
                vec![
                    s(0.0, 0.0, 0.0, 1.7),
                    s(2.4, 0.0, 0.0, 1.7),
                    s(1.2, 2.0, 0.0, 1.7),
                ],
                77.4331,
                55.9095,
            ),
        ];
        for (name, atoms, ball_area, ball_vol) in cases {
            let m = ses_mesh_sdf(&atoms, 1.4, 0.15);
            assert!(m.is_watertight(), "{name}: closed");
            assert!(m.is_consistently_oriented(), "{name}: oriented");
            assert_eq!(m.euler_characteristic(), 2, "{name}: sphere topology");
            let (a, v) = (m.surface_area(), m.signed_volume());
            assert!(
                (a - ball_area).abs() / ball_area < 0.01,
                "{name}: area {a} within 1% of {ball_area}"
            );
            assert!(
                (v - ball_vol).abs() / ball_vol < 0.01,
                "{name}: volume {v} within 1% of {ball_vol}"
            );
        }
    }
}
