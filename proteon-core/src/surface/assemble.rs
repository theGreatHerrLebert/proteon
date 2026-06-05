//! SES assembler — stitch the analytic patches (contact caps, toric and spheric
//! faces) into one watertight mesh, sharing boundaries through the [`registry`].
//!
//! Built bottom-up: this first step meshes a single atom's **contact face** end to
//! end, tying together the validated pieces — [`elements::buried_cap`] (one cap
//! per neighbour), [`arrangement::boundary_loops`] (the exposed-region boundary),
//! and [`chart::fill_spherical_region`] (the multi-hole interior fill). The toric
//! and spheric faces and the full multi-atom assembly follow.

use super::arrangement::{boundary_loops, sample_loop, SphereCircle};
use super::chart::fill_spherical_region;
use super::elements::{arc_on_sphere, buried_cap, ses_vertex};
use super::geom::{intersect_two_spheres, plane_basis, Sphere, Vec3};
use super::graph::build_graph;
use super::mesh::Mesh;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

/// Mesh `atom`'s contact face: its sphere outside the union of the buried caps
/// carved by each of `neighbours`. `grid` is the interior chart-plane spacing
/// (≈ angular spacing); `n_boundary` the samples per boundary arc.
///
/// Returns an open patch whose boundary is exactly the contact-circle arcs
/// (later shared with the toric faces); every vertex lies on `atom`'s sphere.
pub fn contact_cap_mesh(
    atom: Sphere,
    neighbours: &[Sphere],
    probe: f64,
    grid: f64,
    n_boundary: usize,
) -> Result<Mesh> {
    let caps: Vec<SphereCircle> = neighbours
        .iter()
        .filter_map(|&b| buried_cap(atom, b, probe))
        .collect();
    ensure!(
        !caps.is_empty(),
        "atom has no buried caps — a free atom's contact face is the whole sphere"
    );
    let loops = boundary_loops(&caps)?;
    let pole = pick_chart_pole(&caps).context(
        "no single chart pole with enough clearance — contact face needs \
         multi-chart handling (e.g. a band around an atom with opposite neighbours)",
    )?;

    let world_loops: Vec<Vec<Vec3>> = loops
        .iter()
        .map(|lp| {
            sample_loop(lp, &caps, n_boundary)
                .into_iter()
                .map(|d| atom.center + d * atom.radius)
                .collect()
        })
        .collect();

    fill_spherical_region(atom.center, atom.radius, &world_loops, pole, grid)
}

/// Choose an azimuthal-chart pole for a contact cap. The chart is well-posed iff
/// the exposed region avoids the pole's **antipode**; the pole itself need not be
/// exposed. Taking `P = −a_k` (away from neighbour `k`) puts the antipode `a_k`
/// at the centre of buried cap `k`, so the region stays `half_angle_k` away from
/// it — bounded for *any* contact cap, including a band/annulus around an atom
/// with opposite neighbours (the case the old "pole must be exposed" rule wrongly
/// rejected). Pick the deepest cap for the largest margin.
fn pick_chart_pole(caps: &[SphereCircle]) -> Option<Vec3> {
    caps.iter()
        .max_by(|x, y| x.half_angle.total_cmp(&y.half_angle))
        .filter(|c| c.half_angle > 0.01)
        .map(|c| -c.axis)
}

/// The probe centre whose contact point on `a` is `t_a`: along the ray
/// `a.center → t_a`, at distance `r_a + probe`. Normalizing the direction (rather
/// than scaling `t_a − a.center` by `(r_a+probe)/r_a`) avoids amplifying any
/// radial drift in `t_a` by that factor (codex-review).
fn probe_center_from_contact(t_a: Vec3, a: Sphere, probe: f64) -> Option<Vec3> {
    let dir = (t_a - a.center).normalized()?;
    Some(a.center + dir * (a.radius + probe))
}

/// Mesh a toric (reentrant) face from its θ-aligned probe-centre chain and the two
/// φ-rims: `probe_centers[t]` is the probe at column `t`, `rim_a[t]`/`rim_b[t]` its
/// contacts on the two atoms (each on `probe_centers[t]`'s sphere). At each θ the
/// reentrant surface is the probe-sphere arc from `rim_a[t]` to `rim_b[t]` with
/// `n_phi` interior points. `wrap` closes the θ-ring for a *free* toric face; a
/// bounded face leaves its two θ-end columns (the concave arcs) as the open
/// boundary shared with the spheric faces.
///
/// Taking the probe-centre chain explicitly (rather than recovering it from a rim)
/// is what makes the θ-end concave arcs **bit-identical** to the spheric faces'
/// edges — both use the same exact probe centre — so they weld watertight.
pub fn toric_face_mesh(
    rim_a: &[Vec3],
    rim_b: &[Vec3],
    probe_centers: &[Vec3],
    probe: f64,
    n_phi: usize,
    wrap: bool,
) -> Result<Mesh> {
    ensure!(
        rim_a.len() == rim_b.len() && rim_a.len() == probe_centers.len(),
        "toric φ-rims and probe-centre chain must be θ-aligned (equal length)"
    );
    let n_theta = rim_a.len();
    ensure!(
        n_theta >= if wrap { 3 } else { 2 },
        "toric face needs ≥{} θ columns",
        if wrap { 3 } else { 2 }
    );
    let row = n_phi + 2; // rim_a + n_phi interior + rim_b
    let mut verts = Vec::with_capacity(n_theta * row);
    for t in 0..n_theta {
        let p = probe_centers[t];
        // Both rims must lie on this probe sphere — else the torus tears.
        ensure!(
            (rim_a[t].distance(p) - probe).abs() < 1e-6
                && (rim_b[t].distance(p) - probe).abs() < 1e-6,
            "toric rims at θ={t} are not on probe_centers[{t}]'s sphere"
        );
        verts.push(rim_a[t]);
        verts.extend(arc_on_sphere(p, probe, rim_a[t], rim_b[t], n_phi));
        verts.push(rim_b[t]);
    }
    let mut tris = Vec::new();
    let cols = if wrap { n_theta } else { n_theta - 1 };
    for t in 0..cols {
        let t2 = (t + 1) % n_theta;
        for p in 0..row - 1 {
            let (a0, b0) = ((t * row + p) as u32, (t * row + p + 1) as u32);
            let (a1, b1) = ((t2 * row + p) as u32, (t2 * row + p + 1) as u32);
            tris.push([a0, b0, b1]);
            tris.push([a0, b1, a1]);
        }
    }
    Ok(Mesh {
        verts,
        normals: Vec::new(),
        tris,
    })
}

/// The full SES of **two atoms** (one free toric ring + two contact caps, no
/// spheric faces) — the smallest end-to-end analytic assembly, and the proof that
/// the patches stitch watertight.
///
/// Watertightness by **bit-identical shared samples**: the toric face's two
/// φ-rims (`rim_a`/`rim_b`, sampled once per probe position) are passed *verbatim*
/// as the two caps' boundary loops, so the welded vertices coincide exactly rather
/// than within a tolerance. `n_theta` probe positions, `n_phi` toric φ-samples,
/// `grid` cap-chart spacing.
pub fn two_atom_ses(
    a: Sphere,
    b: Sphere,
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    grid: f64,
) -> Result<Mesh> {
    use super::elements::contact_circle;
    use super::geom::plane_basis;
    use std::f64::consts::TAU;

    let circle_a = contact_circle(a, b, probe).context("atoms share no toric face")?;
    let (u, v) = plane_basis(circle_a.normal);
    // One probe-position sweep defines BOTH rims + the probe-centre chain.
    let mut rim_a = Vec::with_capacity(n_theta);
    let mut rim_b = Vec::with_capacity(n_theta);
    let mut centers = Vec::with_capacity(n_theta);
    for t in 0..n_theta {
        let th = TAU * t as f64 / n_theta as f64;
        let ta = circle_a.center + (u * th.cos() + v * th.sin()) * circle_a.radius;
        let p = probe_center_from_contact(ta, a, probe).context("degenerate rim point")?;
        rim_a.push(ta);
        rim_b.push(b.center + (p - b.center).normalized().context("probe at b centre")? * b.radius);
        centers.push(p);
    }

    let toward_b = (b.center - a.center)
        .normalized()
        .context("coincident atoms")?;
    let mut mesh = toric_face_mesh(&rim_a, &rim_b, &centers, probe, n_phi, true)?;
    // Caps: boundary = the *same* rim Vec the toric used (→ exact weld). Pole away
    // from the neighbour (the buried cap's antipode), deep in the exposed region.
    mesh.append(&fill_spherical_region(
        a.center,
        a.radius,
        &[rim_a],
        -toward_b,
        grid,
    )?);
    mesh.append(&fill_spherical_region(
        b.center,
        b.radius,
        &[rim_b],
        toward_b,
        grid,
    )?);

    let mut mesh = mesh.welded(); // fuse the bit-identical shared rims
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip();
    }
    Ok(mesh)
}

/// The full analytic SES of **three mutually-contacting atoms** (the triangle3
/// case): 3 multi-hole contact caps + 3 bounded toric faces + 2 spheric faces,
/// stitched watertight by bit-identical shared samples keyed on probe position.
///
/// Requires exactly two RS faces (a probe resting on the triple from each side).
/// `n_theta` toric θ-columns, `n_phi` reentrant-arc samples (shared between toric
/// θ-ends and spheric edges), `grid` cap/spheric chart spacing.
pub fn triangle3_ses(
    atoms: [Sphere; 3],
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    grid: f64,
) -> Result<Mesh> {
    use super::elements::{arc_on_sphere, ses_vertex};
    use super::geom::{intersect_two_spheres, plane_basis};
    use super::rs;
    use std::collections::HashMap;
    use std::f64::consts::TAU;

    ensure!(
        probe.is_finite()
            && probe > 0.0
            && n_theta >= 3
            && n_phi >= 1
            && grid.is_finite()
            && grid > 0.0,
        "triangle3 needs finite probe>0, grid>0, n_theta≥3, n_phi≥1"
    );
    let r = rs::compute(&atoms, probe);
    ensure!(
        r.faces.len() == 2,
        "triangle3 expects two RS faces (probe on each side), got {}",
        r.faces.len()
    );
    let probes = [r.faces[0].probe_center, r.faces[1].probe_center];
    // The two probes must be distinct (degenerate/tangent triples coincide them).
    ensure!(
        probes[0].distance(probes[1]) > 1e-6,
        "triangle3 probe positions coincide (tangent/degenerate triple)"
    );
    // SES corner: the contact of probe `f` on atom `a` (bit-identical everywhere).
    let corner = |f: usize, a: usize| ses_vertex(probes[f], atoms[a]);

    // For each ordered pair (i,j): the contact chain on atom i as the probe sweeps
    // the free arc P0→P1 of the (i,j) roll circle. rim_of[(i,j)][0]=corner(0,i),
    // last=corner(1,i); same probe positions for (i,j) and (j,i) → θ-aligned.
    let mut rim_of: HashMap<(usize, usize), Vec<Vec3>> = HashMap::new();
    let mut centers_of: HashMap<(usize, usize), Vec<Vec3>> = HashMap::new();
    for &(i, j, k) in &[(0usize, 1usize, 2usize), (0, 2, 1), (1, 2, 0)] {
        let roll = intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
            .context("pair shares no roll circle")?;
        let (u, v) = plane_basis(roll.normal);
        let recon = |th: f64| roll.center + (u * th.cos() + v * th.sin()) * roll.radius;
        let ang = |p: Vec3| {
            let d = p - roll.center;
            let t = d.dot(v).atan2(d.dot(u));
            if t < 0.0 {
                t + TAU
            } else {
                t
            }
        };
        let (th0, th1) = (ang(probes[0]), ang(probes[1]));
        // Free arc = the side where the rolling probe clears the third atom k. The
        // clearance along the roll circle is a constant plus one sinusoid, so it
        // crosses k's inflated sphere at most twice (codex-review): the two probe
        // positions are exactly those crossings, so one arc is wholly clear and
        // the other wholly blocked. Classify BOTH midpoints decisively (not a
        // single `>=`) and require them opposite — else the triple is tangent/
        // degenerate and this is refused rather than silently mis-sided.
        let span_inc = (th1 - th0).rem_euclid(TAU);
        let clearance = |frac: f64| {
            recon(th0 + span_inc * frac).distance(atoms[k].center) - (atoms[k].radius + probe)
        };
        let inc_mid = clearance(0.5); // midpoint of the th0→th1 (increasing) arc
        let dec_mid = clearance(-0.5); // midpoint of the complementary arc
        const TOL: f64 = 1e-7;
        let span = if inc_mid > TOL && dec_mid < -TOL {
            span_inc
        } else if dec_mid > TOL && inc_mid < -TOL {
            span_inc - TAU
        } else {
            bail!("triangle3 free arc is ambiguous (tangent/degenerate triple)");
        };
        let (mut ri, mut rj, mut cen) = (
            Vec::with_capacity(n_theta + 1),
            Vec::with_capacity(n_theta + 1),
            Vec::with_capacity(n_theta + 1),
        );
        for t in 0..=n_theta {
            let p = if t == 0 {
                probes[0]
            } else if t == n_theta {
                probes[1]
            } else {
                recon(th0 + span * t as f64 / n_theta as f64)
            };
            ri.push(ses_vertex(p, atoms[i]));
            rj.push(ses_vertex(p, atoms[j]));
            cen.push(p);
        }
        rim_of.insert((i, j), ri);
        rim_of.insert((j, i), rj);
        centers_of.insert((i, j), cen);
    }

    let mut mesh = Mesh::default();

    // 3 bounded toric faces.
    for &(i, j) in &[(0usize, 1usize), (0, 2), (1, 2)] {
        let rim_a = &rim_of[&(i, j)];
        let rim_b = &rim_of[&(j, i)];
        mesh.append(&toric_face_mesh(
            rim_a,
            rim_b,
            &centers_of[&(i, j)],
            probe,
            n_phi,
            false,
        )?);
    }

    // 3 contact caps: boundary = rim toward one neighbour, then the reversed rim
    // toward the other (sharing the two SES corners) → one loop.
    for &(i, j, k) in &[(0usize, 1usize, 2usize), (1, 0, 2), (2, 0, 1)] {
        let mut loop_pts = rim_of[&(i, j)].clone();
        let back = &rim_of[&(i, k)];
        // append the other rim reversed, dropping its shared end corners.
        for t in (1..back.len() - 1).rev() {
            loop_pts.push(back[t]);
        }
        let cap_ab = buried_cap(atoms[i], atoms[j], probe).context("cap ij")?;
        let cap_ac = buried_cap(atoms[i], atoms[k], probe).context("cap ik")?;
        let pole =
            pick_chart_pole(&[cap_ab, cap_ac]).context("contact cap pole (triangle3 atom)")?;
        mesh.append(&fill_spherical_region(
            atoms[i].center,
            atoms[i].radius,
            &[loop_pts],
            pole,
            grid,
        )?);
    }

    // 2 spheric faces: the concave probe-cap triangle, one per probe position.
    for f in 0..2 {
        let p = probes[f];
        let cs = [corner(f, 0), corner(f, 1), corner(f, 2)];
        let mut loop_pts = Vec::new();
        for e in 0..3 {
            let (i, j) = (e, (e + 1) % 3);
            loop_pts.push(cs[i]);
            // Sample each concave arc in the canonical (low→high atom) order the
            // toric θ-end uses, reversing the Vec (bit-exact) when the loop runs
            // high→low — so the shared arc points are *bit-identical* and weld.
            // arc_on_sphere(A,B) vs arc_on_sphere(B,A) differ by float reordering.
            if i < j {
                loop_pts.extend(arc_on_sphere(p, probe, cs[i], cs[j], n_phi));
            } else {
                let mut arc = arc_on_sphere(p, probe, cs[j], cs[i], n_phi);
                arc.reverse();
                loop_pts.extend(arc);
            }
        }
        let centroid = (cs[0] + cs[1] + cs[2]) * (1.0 / 3.0);
        let inward = (centroid - p)
            .normalized()
            .context("spheric centroid at probe")?;
        mesh.append(&fill_spherical_region(p, probe, &[loop_pts], inward, grid)?);
    }

    let mut mesh = mesh.welded();
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip();
    }
    Ok(mesh)
}

/// One contact-circle arc on an atom: the φ-rim of an incident toric arc, between
/// its two endpoint RS faces (or a full ring when both are `None`). `pts` runs
/// from `ends[0]` to `ends[1]`.
struct ContactArc {
    ends: [Option<usize>; 2],
    neighbour: usize,
    pts: Vec<Vec3>,
}

/// Walk an atom's contact arcs into closed boundary loops, joining them at shared
/// RS-face indices. A full-ring arc (`ends = [None, None]`) is its own loop.
/// Joining by **index** (not coordinate) keeps it exact.
///
/// **Fails loud** on the non-generic graph (codex-review): every bounded RS-face
/// incidence on the atom must have degree exactly 2 (degree ≥3 ⇒ a ≥4-cospherical
/// singular vertex the greedy walk would mis-splice); each join is bit-exact; the
/// loop must close; every bounded arc must be consumed.
fn walk_cap_loops(arcs: &[ContactArc]) -> Result<Vec<Vec<Vec3>>> {
    // Validate arc end-shapes and build the RS-face → incidences map.
    let mut at_face: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (ai, a) in arcs.iter().enumerate() {
        match a.ends {
            [None, None] => ensure!(a.pts.len() >= 3, "degenerate full-ring contact arc"),
            [Some(p), Some(q)] => {
                ensure!(p != q && a.pts.len() >= 2, "degenerate bounded contact arc");
                at_face.entry(p).or_default().push((ai, 0));
                at_face.entry(q).or_default().push((ai, 1));
            }
            _ => bail!("half-ring contact arc (one end None) — invalid"),
        }
    }
    for (f, inc) in &at_face {
        ensure!(
            inc.len() == 2,
            "RS face {f} touches {} contact arcs on this atom (expected 2; \
             ≥3 ⇒ cospherical/singular)",
            inc.len()
        );
    }

    let mut loops = Vec::new();
    let mut used = vec![false; arcs.len()];
    for (ai, a) in arcs.iter().enumerate() {
        if used[ai] {
            continue;
        }
        if a.ends[0].is_none() {
            used[ai] = true;
            loops.push(a.pts.clone()); // full ring
            continue;
        }
        used[ai] = true;
        let mut chain: Vec<Vec3> = a.pts.clone();
        let start_face = a.ends[0].unwrap();
        let mut cur_face = a.ends[1].unwrap();
        let mut steps = 0;
        while cur_face != start_face {
            steps += 1;
            ensure!(steps <= arcs.len(), "cap loop did not close");
            // Exactly one unused continuation (degree-2 already guaranteed above).
            let unused: Vec<_> = at_face[&cur_face]
                .iter()
                .filter(|&&(na, _)| !used[na])
                .collect();
            ensure!(
                unused.len() == 1,
                "ambiguous cap continuation at RS face {cur_face}"
            );
            let (na, nslot) = *unused[0];
            used[na] = true;
            let mut seg = arcs[na].pts.clone();
            if nslot == 1 {
                seg.reverse();
            }
            ensure!(
                chain.last() == seg.first(),
                "cap arcs do not meet bit-exactly at RS face {cur_face}"
            );
            chain.extend(seg.into_iter().skip(1)); // skip the shared vertex
            cur_face = arcs[na].ends[1 - nslot].unwrap();
        }
        // The walk closed onto start_face: the last point must equal the first.
        ensure!(
            chain.last() == chain.first(),
            "cap loop closure is not bit-exact"
        );
        chain.pop();
        loops.push(chain);
    }
    ensure!(used.iter().all(|&u| u), "some contact arcs left unwalked");
    Ok(loops)
}

/// The full analytic SES of `atoms` (general-N, **non-singular**), gated against
/// BALL. Builds the SES element graph, samples every patch by probe position off
/// the *canonical* SES vertices, and stitches contact caps + toric + spheric into
/// one watertight mesh (bit-identical shared samples + exact weld).
///
/// CAVEAT (codex-review): the result is *combinatorially* watertight, which is
/// **not** the same as a valid embedding. Where the probe self-intersects (tight
/// pockets), independently-valid toric patches can geometrically overlap while the
/// mesh stays closed — the "closed but wrong" case. On crambin this shows as a
/// stable ≈1% area excess that does *not* shrink with finer sampling. Trimming
/// those singular events (the geometric singularity resolver) and a self-
/// intersection gate are required before this is correct on arbitrary proteins.
pub fn ses_mesh_analytic(
    atoms: &[Sphere],
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    grid: f64,
) -> Result<Mesh> {
    ensure!(
        n_theta >= 3 && n_phi >= 1 && grid > 0.0,
        "bad sampling params"
    );
    let g = build_graph(atoms, probe)?;
    let mut mesh = Mesh::default();
    let mut contact: HashMap<usize, Vec<ContactArc>> = HashMap::new();

    // --- toric faces; collect each atom's contact arcs ---
    for arc in &g.toric {
        let [i, j] = arc.edge;
        let roll = intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
            .context("toric pair lost its roll circle")?;
        let (u, v) = plane_basis(roll.normal);
        let wrap = arc.end_faces[0].is_none();
        let (s, e) = arc.theta;
        let count = if wrap { n_theta } else { n_theta + 1 };
        let (mut cen, mut rim_i, mut rim_j) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..count {
            let p = if !wrap && t == 0 {
                g.rs_faces[arc.end_faces[0].unwrap()].probe
            } else if !wrap && t == n_theta {
                g.rs_faces[arc.end_faces[1].unwrap()].probe
            } else {
                let th = s + (e - s) * t as f64 / n_theta as f64;
                roll.center + (u * th.cos() + v * th.sin()) * roll.radius
            };
            cen.push(p);
            rim_i.push(ses_vertex(p, atoms[i]));
            rim_j.push(ses_vertex(p, atoms[j]));
        }
        mesh.append(&toric_face_mesh(&rim_i, &rim_j, &cen, probe, n_phi, wrap)?);
        contact.entry(i).or_default().push(ContactArc {
            ends: arc.end_faces,
            neighbour: j,
            pts: rim_i,
        });
        contact.entry(j).or_default().push(ContactArc {
            ends: arc.end_faces,
            neighbour: i,
            pts: rim_j,
        });
    }

    // --- contact caps: walk loops, fill ---
    for (&a, arcs) in &contact {
        let loops = walk_cap_loops(arcs)?;
        let caps: Vec<SphereCircle> = arcs
            .iter()
            .map(|c| c.neighbour)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter_map(|nb| buried_cap(atoms[a], atoms[nb], probe))
            .collect();
        let pole = pick_chart_pole(&caps).context("contact cap pole")?;
        mesh.append(&fill_spherical_region(
            atoms[a].center,
            atoms[a].radius,
            &loops,
            pole,
            grid,
        )?);
    }

    // --- spheric faces (one per RS face) ---
    for f in &g.rs_faces {
        let p = f.probe;
        let cs = [
            ses_vertex(p, atoms[f.atoms[0]]),
            ses_vertex(p, atoms[f.atoms[1]]),
            ses_vertex(p, atoms[f.atoms[2]]),
        ];
        let mut loop_pts = Vec::new();
        for e in 0..3 {
            let (x, y) = (e, (e + 1) % 3);
            loop_pts.push(cs[x]);
            // Canonical low→high atom order (reverse bit-exactly) so the concave
            // arc is bit-identical to the toric θ-end (same as triangle3).
            if f.atoms[x] < f.atoms[y] {
                loop_pts.extend(arc_on_sphere(p, probe, cs[x], cs[y], n_phi));
            } else {
                let mut arc = arc_on_sphere(p, probe, cs[y], cs[x], n_phi);
                arc.reverse();
                loop_pts.extend(arc);
            }
        }
        let centroid = (cs[0] + cs[1] + cs[2]) * (1.0 / 3.0);
        let inward = (centroid - p)
            .normalized()
            .context("spheric centroid at probe")?;
        mesh.append(&fill_spherical_region(p, probe, &[loop_pts], inward, grid)?);
    }

    let mut mesh = mesh.welded();
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip();
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::super::elements::buried_cap;
    use super::*;
    use std::f64::consts::PI;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// One neighbour ⇒ the contact face is the sphere minus one buried cap (a
    /// spherical zone). Its area is `2πr²(1+cos half_angle)` — the analytic check
    /// that the buried_cap → boundary_loops → chart-fill pipeline is correct, end
    /// to end, on a real atom.
    #[test]
    fn single_neighbour_contact_face_matches_the_analytic_zone() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let neighbour = sph(3.0, 0.0, 0.0, 1.6);
        let probe = 1.4;
        let cap = buried_cap(atom, neighbour, probe).unwrap();
        let exact = 2.0 * PI * atom.radius * atom.radius * (1.0 + cap.half_angle.cos());

        let coarse = contact_cap_mesh(atom, &[neighbour], probe, 0.12, 48).unwrap();
        let fine = contact_cap_mesh(atom, &[neighbour], probe, 0.06, 96).unwrap();
        let (ac, af) = (coarse.surface_area(), fine.surface_area());
        assert!(
            (af - exact).abs() < (ac - exact).abs() + 1e-9,
            "contact-face area converges {ac} → {af} vs {exact}"
        );
        assert!(
            (af - exact).abs() / exact < 0.01,
            "fine contact-face area {af} within 1% of {exact}"
        );
        // Open patch with a single boundary loop (the one contact circle).
        assert!(
            fine.num_nonmanifold_edges() > 0,
            "contact face has a boundary"
        );
        for v in &fine.verts {
            assert!(
                (v.distance(atom.center) - atom.radius).abs() < 1e-9,
                "every vertex on the atom sphere"
            );
        }
    }

    /// Two neighbours (the triangle3 atom) ⇒ the contact face is the sphere minus
    /// two buried caps. It must still mesh — a closed boundary, all vertices on
    /// the sphere — exercising the multi-cap arrangement + chart fill.
    #[test]
    fn two_neighbour_contact_face_meshes_on_the_sphere() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let n1 = sph(2.6, 0.0, 0.0, 1.7);
        let n2 = sph(1.3, 2.1, 0.0, 1.7);
        let m = contact_cap_mesh(atom, &[n1, n2], 1.4, 0.07, 48).unwrap();
        assert!(!m.tris.is_empty(), "non-empty mesh");
        assert!(m.num_nonmanifold_edges() > 0, "open patch with a boundary");
        for v in &m.verts {
            assert!((v.distance(atom.center) - atom.radius).abs() < 1e-9);
        }
    }

    /// The first watertight ANALYTIC SES assembled from contact caps + a toric
    /// face, gated against `ball-py 0.1.0a6 ses_area` — the proof the patches
    /// stitch closed and on-surface. Within 1% of BALL's analytic area/volume.
    #[test]
    fn two_atom_ses_is_watertight_and_matches_ball() {
        // (atom_a, atom_b, probe, ball area, ball volume)
        let cases = [
            (
                sph(0.0, 0.0, 0.0, 1.8),
                sph(2.5, 0.0, 0.0, 1.8),
                1.4,
                67.7959,
                46.6207,
            ),
            (
                sph(0.0, 0.0, 0.0, 2.0),
                sph(3.0, 0.0, 0.0, 1.2),
                1.4,
                64.3406,
                42.1575,
            ),
        ];
        for (a, b, probe, ball_area, ball_vol) in cases {
            let m = two_atom_ses(a, b, probe, 96, 10, 0.05).unwrap();
            assert!(m.is_watertight(), "assembled SES must be closed");
            assert!(m.is_consistently_oriented());
            assert_eq!(m.euler_characteristic(), 2, "sphere topology");
            let (area, vol) = (m.surface_area(), m.signed_volume());
            assert!(vol > 0.0, "outward oriented");
            assert!(
                (area - ball_area).abs() / ball_area < 0.01,
                "SES area {area} within 1% of ball {ball_area}"
            );
            assert!(
                (vol - ball_vol).abs() / ball_vol < 0.01,
                "SES volume {vol} within 1% of ball {ball_vol}"
            );
        }
    }

    /// **The general-N gate.** `ses_mesh_analytic` (graph-driven) must produce a
    /// watertight SES matching `ball-py ses_area` for: triangle3 (reproduced via
    /// the general path), tetra (the smallest >2-neighbour case), and a 4-chain
    /// (free-ring toric faces). euler may exceed 2 (internal cavities are fine).
    #[test]
    fn general_n_ses_is_watertight_and_matches_ball() {
        let tri = vec![
            sph(0.0, 0.0, 0.0, 1.7),
            sph(2.5, 0.0, 0.0, 1.7),
            sph(1.25, 2.165, 0.0, 1.7),
        ];
        let tetra = vec![
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        let chain = vec![
            sph(0.0, 0.0, 0.0, 1.5),
            sph(2.6, 0.0, 0.0, 1.5),
            sph(5.2, 0.0, 0.0, 1.5),
            sph(7.8, 0.0, 0.0, 1.5),
        ];
        let cases = [
            ("tri", tri, 80.0932, 57.9040),
            ("tetra", tetra, 74.1161, 54.0987),
            ("chain", chain, 96.7732, 58.9781),
        ];
        for (name, atoms, ball_area, ball_vol) in cases {
            let m = ses_mesh_analytic(&atoms, 1.4, 48, 10, 0.05).unwrap();
            assert!(m.is_watertight(), "{name}: SES must be closed");
            assert!(m.is_consistently_oriented(), "{name}: oriented");
            // Valid embedding, not just combinatorially closed (codex-review).
            assert_eq!(
                super::super::intersect::self_intersections(&m, 0.5, 1),
                0,
                "{name}: SES must not self-intersect"
            );
            let (area, vol) = (m.surface_area(), m.signed_volume());
            assert!(vol > 0.0, "{name}: outward");
            assert!(
                (area - ball_area).abs() / ball_area < 0.02,
                "{name}: area {area} within 2% of ball {ball_area}"
            );
            assert!(
                (vol - ball_vol).abs() / ball_vol < 0.02,
                "{name}: volume {vol} within 2% of ball {ball_vol}"
            );
        }
    }

    /// **The triangle3 gate.** The full analytic SES of three mutually-contacting
    /// atoms — all three patch types (contact caps, bounded toric, spheric) — must
    /// stitch watertight and match `ball-py ses_area`. This is the case BALL's
    /// gnarly clip-and-gift-wrap was claimed un-portable for.
    #[test]
    fn triangle3_ses_is_watertight_and_matches_ball() {
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.7),
            sph(2.5, 0.0, 0.0, 1.7),
            sph(1.25, 2.165, 0.0, 1.7),
        ];
        let (ball_area, ball_vol) = (80.0932, 57.9040);
        let m = triangle3_ses(atoms, 1.4, 48, 10, 0.05).unwrap();
        assert!(m.is_watertight(), "triangle3 SES must be closed");
        assert!(m.is_consistently_oriented());
        assert_eq!(m.euler_characteristic(), 2, "sphere topology");
        let (area, vol) = (m.surface_area(), m.signed_volume());
        assert!(vol > 0.0, "outward");
        assert!(
            (area - ball_area).abs() / ball_area < 0.02,
            "SES area {area} within 2% of ball {ball_area}"
        );
        assert!(
            (vol - ball_vol).abs() / ball_vol < 0.02,
            "SES volume {vol} within 2% of ball {ball_vol}"
        );
    }

    /// Two opposite neighbours leave a band (annulus) around the atom. With the
    /// pole at a buried cap's antipode the single azimuthal chart still handles it
    /// — the band meshes, every vertex on the atom sphere, two boundary loops.
    #[test]
    fn opposite_neighbours_band_cap_meshes() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let n1 = sph(2.6, 0.0, 0.0, 1.7);
        let n2 = sph(-2.6, 0.0, 0.0, 1.7);
        let m = contact_cap_mesh(atom, &[n1, n2], 1.4, 0.1, 48).unwrap();
        assert!(!m.tris.is_empty());
        for v in &m.verts {
            assert!((v.distance(atom.center) - atom.radius).abs() < 1e-9);
        }
    }

    /// A *free* toric face (two atoms, probe rolls all the way around) from
    /// θ-aligned φ-rims: every interior vertex is exactly `probe` from its rolling
    /// probe centre, the rims lie on the atoms, and the ring is a clean grid whose
    /// only boundary is the two φ-rim circles.
    #[test]
    fn free_toric_face_lies_on_the_probe_surface() {
        use super::super::elements::contact_circle;
        use super::super::geom::plane_basis;
        use std::f64::consts::TAU;
        let a = sph(0.0, 0.0, 0.0, 1.8);
        let b = sph(2.5, 0.0, 0.0, 1.8);
        let probe = 1.4;
        let circle_a = contact_circle(a, b, probe).unwrap();
        let (u, v) = plane_basis(circle_a.normal);
        let n_theta = 48;
        let n_phi = 6;
        // rim_a sampled around contact circle A; rim_b is the same probe's contact
        // point on B (θ-aligned by construction).
        let mut rim_a = Vec::new();
        let mut rim_b = Vec::new();
        let mut centers = Vec::new();
        for t in 0..n_theta {
            let th = TAU * t as f64 / n_theta as f64;
            let ta = circle_a.center + (u * th.cos() + v * th.sin()) * circle_a.radius;
            let p = probe_center_from_contact(ta, a, probe).unwrap();
            rim_a.push(ta);
            rim_b.push(b.center + (p - b.center).normalized().unwrap() * b.radius);
            centers.push(p);
        }
        let m = toric_face_mesh(&rim_a, &rim_b, &centers, probe, n_phi, true).unwrap();
        for t in 0..n_theta {
            let p = centers[t];
            let row = n_phi + 2;
            for q in 1..=n_phi {
                let pt = m.verts[t * row + q];
                assert!(
                    (pt.distance(p) - probe).abs() < 1e-9,
                    "interior on probe sphere"
                );
            }
            assert!((rim_a[t].distance(a.center) - a.radius).abs() < 1e-9);
            assert!((rim_b[t].distance(b.center) - b.radius).abs() < 1e-9);
        }
        // A wrapped ring: boundary = the two φ-rim circles (2·n_theta edges).
        assert_eq!(m.num_nonmanifold_edges(), 2 * n_theta);
        // Consistent winding (open patch): no directed edge is traversed twice.
        let mut seen = std::collections::HashSet::new();
        for t in &m.tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                assert!(seen.insert(e), "a directed edge repeats — winding flipped");
            }
        }
    }
}
