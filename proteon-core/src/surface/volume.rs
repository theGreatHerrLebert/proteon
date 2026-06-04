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
//! `dist(x, complement(A_p))` is the distance from an interior node to the
//! nearest node *outside* every inflated atom; we get it from an exact separable
//! squared-Euclidean distance transform (Felzenszwalb–Huttenlocher) seeded on the
//! outside nodes. Grid spacing is the single accuracy knob; gated against
//! `ball-py ses_area` (analytic area + volume).

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
    /// `signed_dist` comes from a **binary** occupancy of `A_p` plus a squared
    /// Euclidean distance transform: inside nodes get their grid-distance to the
    /// nearest outside node (= distance to the complement of `A_p` = distance to
    /// the SAS). This quantizes the SAS to voxel resolution, so the area/volume
    /// converge from slightly high as `h → 0` (see `sweep_resolution`); a
    /// sub-voxel-accurate field would need a vector/Danielsson EDT seeded with
    /// the analytic distance to the nearest inflated atom.
    fn distance_field(&self, atoms: &[Sphere], probe: f64) -> Vec<f64> {
        let [nx, ny, nz] = self.dims;
        let h = self.spacing;
        const BIG: f64 = 1e20;
        let mut inside = vec![false; nx * ny * nz];
        let mut d = vec![0.0f64; nx * ny * nz]; // FH seed: 0 outside A_p, BIG inside
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = self.pos(i, j, k);
                    let ins = atoms.iter().any(|a| {
                        let r = a.radius + probe;
                        p.square_distance(a.center) <= r * r
                    });
                    let idx = self.idx(i, j, k);
                    inside[idx] = ins;
                    d[idx] = if ins { BIG } else { 0.0 };
                }
            }
        }
        // D(p) = squared index-distance from inside nodes to the nearest outside
        // node = squared distance to the complement of A_p (grid-quantized).
        squared_edt_3d(&mut d, self.dims);
        for (v, &ins) in d.iter_mut().zip(inside.iter()) {
            let dist = v.sqrt() * h; // distance to the SAS surface, in Å
            *v = if ins { dist } else { -dist } - probe; // + inside A_p, erode by probe
        }
        d
    }
}

/// In-place separable squared Euclidean distance transform over a 3D grid stored
/// `i + nx·(j + ny·k)`. Input holds the seed cost per node (0 at seeds, large
/// elsewhere); output holds `min_q cost(q) + |p−q|²` in node² units.
///
/// `Grid::enclosing` pads by `probe + 2·spacing` on every side, so the first and
/// last node of every axis line are outside `A_p` (seed cost 0). No line is ever
/// all-`BIG`, which keeps the parabola lower-envelope finite (the `BIG` sentinel
/// never has to compete against itself).
fn squared_edt_3d(d: &mut [f64], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    // along x
    let mut line = vec![0.0f64; nx.max(ny).max(nz)];
    for k in 0..nz {
        for j in 0..ny {
            let base = nx * (j + ny * k);
            line[..nx].copy_from_slice(&d[base..base + nx]);
            let out = edt_1d(&line[..nx]);
            d[base..base + nx].copy_from_slice(&out);
        }
    }
    // along y
    for k in 0..nz {
        for i in 0..nx {
            for j in 0..ny {
                line[j] = d[i + nx * (j + ny * k)];
            }
            let out = edt_1d(&line[..ny]);
            for j in 0..ny {
                d[i + nx * (j + ny * k)] = out[j];
            }
        }
    }
    // along z
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                line[k] = d[i + nx * (j + ny * k)];
            }
            let out = edt_1d(&line[..nz]);
            for k in 0..nz {
                d[i + nx * (j + ny * k)] = out[k];
            }
        }
    }
}

/// 1D distance transform of a sampled function `f`: returns
/// `D[q] = min_p f[p] + (q − p)²` (Felzenszwalb–Huttenlocher lower envelope of
/// parabolas).
fn edt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut d = vec![0.0f64; n];
    let mut v = vec![0usize; n]; // locations of parabolas in the envelope
    let mut z = vec![0.0f64; n + 1]; // breakpoints between them
    let mut k: isize = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let q2 = (q * q) as f64;
        loop {
            let p = v[k as usize];
            let s = ((f[q] + q2) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if s <= z[k as usize] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k as usize] = q;
                z[k as usize] = s;
                z[k as usize + 1] = f64::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for q in 0..n {
        while z[k as usize + 1] < q as f64 {
            k += 1;
        }
        let p = v[k as usize];
        let dq = q as f64 - p as f64;
        d[q] = dq * dq + f[p];
    }
    d
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
            (mesh.surface_area() - exact).abs() / exact < 0.02,
            "area {} vs {exact}",
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
            let coarse = ses_mesh_sdf(&atoms, probe, 0.3);
            let fine = ses_mesh_sdf(&atoms, probe, 0.12);

            assert!(fine.is_watertight(), "{name}: closed");
            assert!(fine.is_consistently_oriented(), "{name}: oriented");
            assert_eq!(fine.euler_characteristic(), 2, "{name}: sphere topology");

            let (af, vf) = (fine.surface_area(), fine.signed_volume());
            let ac = coarse.surface_area();
            // Monotone convergence toward the analytic area as the grid refines.
            assert!(
                (af - ball_area).abs() < (ac - ball_area).abs() + 1e-9,
                "{name}: area converges ({ac} → {af} vs {ball_area})"
            );
            // Resolution-limited (grid SES): ~1.5-2% high at h=0.15, shrinking
            // with h (see the `sweep_resolution` probe).
            assert!(
                (af - ball_area).abs() / ball_area < 0.025,
                "{name}: fine area {af} within 2.5% of {ball_area}"
            );
            assert!(
                (vf - ball_vol).abs() / ball_vol < 0.03,
                "{name}: fine volume {vf} within 3% of {ball_vol}"
            );
        }
    }
}
