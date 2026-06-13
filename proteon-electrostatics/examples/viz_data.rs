//! Dump data for the electrostatics visualizations into a JSON file:
//!   cargo run --release --example viz_data -- /tmp/electro_viz.json
//!
//! Produces (a) the BEM surface potential on a sphere with off-centre charges,
//! (b) the local-vs-nonlocal reaction potential radial profile, (c) the BEM→Born
//! energy convergence, and (d) the nine-ion closed-form Born energies.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, espotential, rfenergy, solve_local_elements,
    solve_nonlocal_elements, Charge, Domain, Locality, Params, SolveConfig, Tri,
};
use serde_json::{json, Value};

fn params() -> Params {
    Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

fn sphere(radius: f64, subdiv: u32) -> (Vec<Tri>, Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mesh = analytic_sphere_mesh(radius, subdiv);
    let tris: Vec<Tri> = mesh
        .tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect();
    let verts = mesh.verts.iter().map(|v| [v.x, v.y, v.z]).collect();
    (tris, verts, mesh.tris.clone())
}

/// (a) Surface potential on a radius-2 sphere with a few off-centre charges.
fn surface_potential() -> Value {
    let radius = 2.0;
    let (tris, verts, idx) = sphere(radius, 3);
    let charges = vec![
        Charge {
            pos: Vec3::new(1.0, 0.0, 0.0),
            val: 1.0,
        },
        Charge {
            pos: Vec3::new(-0.7, 0.8, 0.3),
            val: -1.0,
        },
        Charge {
            pos: Vec3::new(0.2, -1.1, -0.6),
            val: 0.6,
        },
    ];
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&tris, &charges, &params(), &cfg).unwrap();

    // Surface potential (V) at each vertex (the vertices lie on the sphere).
    let phi: Vec<f64> = verts
        .iter()
        .map(|v| {
            let xi = Vec3::new(v[0], v[1], v[2]);
            espotential(Domain::Gamma, xi, &tris, &charges, &params(), &res)
        })
        .collect();
    json!({
        "radius": radius,
        "verts": verts,
        "tris": idx,
        "vert_potential": phi,
        "charges": charges.iter().map(|c| json!({"pos": [c.pos.x, c.pos.y, c.pos.z], "val": c.val})).collect::<Vec<_>>(),
    })
}

/// (b) Reaction potential along the +x ray, local vs nonlocal, for a central charge.
fn radial_profile() -> Value {
    let radius = 2.0;
    let (tris, _, _) = sphere(radius, 2);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let cfg = SolveConfig {
        tol: 1e-10,
        ..Default::default()
    };
    let (loc, _) = solve_local_elements(&tris, &charges, &params(), &cfg).unwrap();
    let (nl, _) = solve_nonlocal_elements(&tris, &charges, &params(), &cfg).unwrap();

    let mut rs = vec![];
    let (mut pl, mut pn) = (vec![], vec![]);
    let mut r = 0.25_f64;
    while r <= 4.8 {
        let xi = Vec3::new(r, 0.0, 0.0);
        // Inside the sphere → Ω, outside → Σ (skip the thin on-surface band).
        if (r - radius).abs() > 0.12 {
            let dom = if r < radius {
                Domain::Omega
            } else {
                Domain::Sigma
            };
            rs.push(r);
            pl.push(espotential(dom, xi, &tris, &charges, &params(), &loc));
            pn.push(espotential(dom, xi, &tris, &charges, &params(), &nl));
        }
        r += 0.1;
    }
    json!({"radius": radius, "r": rs, "phi_local": pl, "phi_nonlocal": pn})
}

/// (c) BEM reaction-field energy vs the analytic Born energy as the mesh refines.
fn convergence() -> Value {
    let radius = 2.0;
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let born_l = born_rfenergy(1.0, radius, &params(), Locality::Local);
    let born_n = born_rfenergy(1.0, radius, &params(), Locality::Nonlocal);

    let (mut ntri, mut el, mut en) = (vec![], vec![], vec![]);
    for s in [0u32, 1, 2, 3] {
        let (tris, _, _) = sphere(radius, s);
        let (l, _) = solve_local_elements(&tris, &charges, &params(), &cfg).unwrap();
        let (n, _) = solve_nonlocal_elements(&tris, &charges, &params(), &cfg).unwrap();
        ntri.push(tris.len());
        el.push(rfenergy(&tris, &charges, &l));
        en.push(rfenergy(&tris, &charges, &n));
    }
    json!({"n_tri": ntri, "bem_local": el, "bem_nonlocal": en, "born_local": born_l, "born_nonlocal": born_n})
}

/// (d) Closed-form Born energies for the nine built-in ions.
fn ion_energies() -> Value {
    // name, charge, radius (NESSie's built-in Born ions).
    let ions = [
        ("Li", 1.0, 0.645),
        ("Na", 1.0, 1.005),
        ("K", 1.0, 1.365),
        ("Rb", 1.0, 1.505),
        ("Cs", 1.0, 1.715),
        ("Mg", 2.0, 0.615),
        ("Ca", 2.0, 1.015),
        ("Sr", 2.0, 1.195),
        ("Ba", 2.0, 1.385),
    ];
    let p = params();
    let names: Vec<&str> = ions.iter().map(|i| i.0).collect();
    let local: Vec<f64> = ions
        .iter()
        .map(|i| born_rfenergy(i.1, i.2, &p, Locality::Local))
        .collect();
    let nonlocal: Vec<f64> = ions
        .iter()
        .map(|i| born_rfenergy(i.1, i.2, &p, Locality::Nonlocal))
        .collect();
    json!({"names": names, "local": local, "nonlocal": nonlocal})
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/electro_viz.json".into());
    eprintln!("computing surface potential…");
    let surf = surface_potential();
    eprintln!("computing radial profile…");
    let radial = radial_profile();
    eprintln!("computing convergence…");
    let conv = convergence();
    let ions = ion_energies();

    let doc = json!({
        "surface": surf,
        "radial": radial,
        "convergence": conv,
        "ions": ions,
    });
    std::fs::write(&out, serde_json::to_string(&doc).unwrap()).unwrap();
    eprintln!("wrote {out}");
}
