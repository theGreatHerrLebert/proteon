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
//! iso-surface with **naive surface nets** (one vertex per sign-changing cell,
//! one quad per sign-changing grid edge) — watertight by construction, any
//! topology, no patch stitching.
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
    let grid = Grid::enclosing(atoms, probe, spacing);
    let f = grid.distance_field(atoms, probe);
    let mut mesh = surface_nets(&grid, &f);
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
        let grid = AtomGrid::build(atoms, probe);

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

        // Seed: nodes adjacent (6-neighbour) to a sign change carry their analytic
        // nearest point on the SAS. NaN x marks "no feature yet".
        const NONE: [f64; 3] = [f64::NAN; 3];
        let mut feat = vec![NONE; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = self.idx(i, j, k);
                    let ins = inside[idx];
                    let boundary = [
                        (i + 1 < nx).then(|| self.idx(i + 1, j, k)),
                        (i > 0).then(|| self.idx(i - 1, j, k)),
                        (j + 1 < ny).then(|| self.idx(i, j + 1, k)),
                        (j > 0).then(|| self.idx(i, j - 1, k)),
                        (k + 1 < nz).then(|| self.idx(i, j, k + 1)),
                        (k > 0).then(|| self.idx(i, j, k - 1)),
                    ]
                    .into_iter()
                    .flatten()
                    .any(|nb| inside[nb] != ins);
                    if boundary {
                        if let Some(s) = grid.nearest_surface_point(self.pos(i, j, k)) {
                            feat[idx] = [s.x, s.y, s.z];
                        }
                    }
                }
            }
        }

        // Features only need to reach nodes within ~probe of the surface (that's
        // the band straddling f = 0); beyond it the sign alone is correct.
        let reach = (probe / self.spacing).ceil() as usize + 4;
        jump_flood(&mut feat, self.dims, reach, &|i, j, k| self.pos(i, j, k));

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
                    f[idx] = if inside[idx] { dist } else { -dist } - probe;
                }
            }
        }
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
}

/// Bucket reach for `AtomGrid` queries — valid only while `cell` is the maximum
/// inflated radius (see `for_each_near`).
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
        for (idx, s) in spheres.iter().enumerate() {
            buckets
                .entry(Self::key(s.center, cell))
                .or_default()
                .push(idx);
        }
        AtomGrid {
            spheres,
            cell,
            buckets,
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

    /// The analytic nearest point on the SAS (the boundary of the union): the
    /// closest *exposed* radial projection onto a nearby inflated sphere. Returns
    /// `None` only if no atom is near (shouldn't happen for a boundary node).
    fn nearest_surface_point(&self, p: Vec3) -> Option<Vec3> {
        let mut best: Option<(f64, Vec3)> = None;
        self.for_each_near(p, NEAR_REACH, |i| {
            let s = self.spheres[i];
            let dir = match (p - s.center).normalized() {
                Some(d) => d,
                None => return, // p at the centre — degenerate, skip
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
    pos: &impl Fn(usize, usize, usize) -> Vec3,
) {
    let [nx, ny, nz] = dims;
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
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let here = pos(i, j, k);
                    let cur = src[idx(i, j, k)];
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
                                let cand = src[idx(ii as usize, jj as usize, kk as usize)];
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
                    dst[idx(i, j, k)] = best;
                }
            }
        }
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

/// Naive surface nets: one vertex per sign-changing cell (at the mean of its
/// edge zero-crossings), one quad per interior sign-changing grid edge. The
/// result is a closed two-manifold; orientation is fixed by the caller.
fn surface_nets(grid: &Grid, f: &[f64]) -> Mesh {
    let [nx, ny, nz] = grid.dims;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1); // cells per axis
    let cell_index = |i: usize, j: usize, k: usize| i + cx * (j + cy * k);
    let mut cell_vert = vec![u32::MAX; cx * cy * cz];
    let mut verts: Vec<Vec3> = Vec::new();

    // Place a vertex in every sign-changing cell.
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut corner_f = [0.0f64; 8];
                for (c, off) in CORNER.iter().enumerate() {
                    corner_f[c] = f[grid.idx(i + off[0], j + off[1], k + off[2])];
                }
                let neg = corner_f.iter().filter(|&&x| x < 0.0).count();
                if neg == 0 || neg == 8 {
                    continue; // not crossed
                }
                let mut acc = Vec3::new(0.0, 0.0, 0.0);
                let mut m = 0.0;
                for &(a, b) in &CUBE_EDGES {
                    let (fa, fb) = (corner_f[a], corner_f[b]);
                    if (fa < 0.0) == (fb < 0.0) {
                        continue;
                    }
                    // A crossing endpoint must be a reached (finite, near-band)
                    // node; the UNREACHED sentinel never borders the surface, so
                    // this guards against a NaN/∞ interpolation slipping through.
                    debug_assert!(
                        fa.abs() < 1e17 && fb.abs() < 1e17,
                        "sign-changing edge touches an unreached node"
                    );
                    let t = fa / (fa - fb);
                    let pa = grid.pos(i + CORNER[a][0], j + CORNER[a][1], k + CORNER[a][2]);
                    let pb = grid.pos(i + CORNER[b][0], j + CORNER[b][1], k + CORNER[b][2]);
                    acc = acc + pa + (pb - pa) * t;
                    m += 1.0;
                }
                cell_vert[cell_index(i, j, k)] = verts.len() as u32;
                verts.push(acc * (1.0 / m));
            }
        }
    }

    // One quad per interior grid edge whose endpoints differ in sign; the four
    // cells around that edge own the quad's corners.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let quad = |a: u32, b: u32, c: u32, d: u32, flip: bool, tris: &mut Vec<[u32; 3]>| {
        // The four cells sharing a sign-changing grid edge each straddle that
        // edge, so each is itself sign-changing and owns a vertex — a MAX here
        // would mean a dropped quad (a hole), so trip the build rather than
        // silently leave one.
        debug_assert!(
            a != u32::MAX && b != u32::MAX && c != u32::MAX && d != u32::MAX,
            "sign-changing edge with a non-crossed incident cell → would hole the mesh"
        );
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || d == u32::MAX {
            return;
        }
        if flip {
            tris.push([a, c, b]);
            tris.push([a, d, c]);
        } else {
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let here = f[grid.idx(i, j, k)] < 0.0;
                // +x edge → quad in the y/z cells around it
                if i + 1 < nx && j > 0 && k > 0 {
                    let there = f[grid.idx(i + 1, j, k)] < 0.0;
                    if here != there {
                        quad(
                            cell_vert[cell_index(i, j - 1, k - 1)],
                            cell_vert[cell_index(i, j, k - 1)],
                            cell_vert[cell_index(i, j, k)],
                            cell_vert[cell_index(i, j - 1, k)],
                            here,
                            &mut tris,
                        );
                    }
                }
                // +y edge → quad in the x/z cells
                if j + 1 < ny && i > 0 && k > 0 {
                    let there = f[grid.idx(i, j + 1, k)] < 0.0;
                    if here != there {
                        quad(
                            cell_vert[cell_index(i - 1, j, k - 1)],
                            cell_vert[cell_index(i, j, k - 1)],
                            cell_vert[cell_index(i, j, k)],
                            cell_vert[cell_index(i - 1, j, k)],
                            !here,
                            &mut tris,
                        );
                    }
                }
                // +z edge → quad in the x/y cells
                if k + 1 < nz && i > 0 && j > 0 {
                    let there = f[grid.idx(i, j, k + 1)] < 0.0;
                    if here != there {
                        quad(
                            cell_vert[cell_index(i - 1, j - 1, k)],
                            cell_vert[cell_index(i, j - 1, k)],
                            cell_vert[cell_index(i, j, k)],
                            cell_vert[cell_index(i - 1, j, k)],
                            here,
                            &mut tris,
                        );
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    #[test]
    #[ignore]
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
