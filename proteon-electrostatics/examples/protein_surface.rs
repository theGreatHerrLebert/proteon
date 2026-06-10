//! BEM electrostatic potential on a real protein surface.
//!
//!   cargo run --release --example protein_surface -- test-pdbs/1crn.pdb /tmp/electro_protein.json [spacing]
//!
//! Meshes a protein's solvent-excluded surface (proteon-core), places illustrative
//! atom charges (electronegative O⁻ / amine-amide N⁺ — NOT a force field), solves the
//! local BEM (proteon-electrostatics), and dumps the SES coloured by the surface
//! electrostatic potential. A coarse SES keeps the dense O(N²) solve tractable.

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::geom::{Sphere, Vec3};
use proteon_core::surface::volume::ses_mesh_sdf;
use proteon_electrostatics::{
    espotential, solve_local_elements, Charge, Domain, Params, SolveConfig, Tri,
};
use serde_json::json;

/// Illustrative per-atom partial charge by atom name (chemistry, not a force field):
/// carbonyl/carboxyl O is electronegative, amide/amine N slightly positive, S mild.
fn illustrative_charge(name: &str) -> f64 {
    match name.trim().chars().next() {
        Some('O') => -0.55,
        Some('N') => 0.35,
        Some('S') => -0.20,
        _ => 0.0,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pdb_path = args.next().unwrap_or_else(|| "test-pdbs/1crn.pdb".into());
    let out = args
        .next()
        .unwrap_or_else(|| "/tmp/electro_protein.json".into());
    let spacing: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.9);

    let pdb = proteon_io::pdb_io::load(&pdb_path).expect("load PDB");
    let model = pdb.models().next().expect("a model");

    // Heavy atoms → spheres (vdW radii) + illustrative charges at the atom centres.
    let mut atoms = Vec::new();
    let mut charges = Vec::new();
    for chain in model.chains() {
        for res in chain.residues() {
            if matches!(res.name().unwrap_or(""), "HOH" | "WAT" | "DOD") {
                continue;
            }
            for a in res.atoms() {
                let (x, y, z) = a.pos();
                let p = Vec3::new(x, y, z);
                let elem = a.element().map(|e| e.symbol()).unwrap_or("");
                atoms.push(Sphere::new(p, vdw_radius(elem).unwrap_or(DEFAULT_RADIUS)));
                let q = illustrative_charge(a.name());
                if q != 0.0 {
                    charges.push(Charge { pos: p, val: q });
                }
            }
        }
    }
    eprintln!("{} atoms, {} charges", atoms.len(), charges.len());

    // Coarse SES (SDF marching cubes) so the dense BEM stays tractable.
    let mesh = ses_mesh_sdf(&atoms, 1.4, spacing);
    let verts: Vec<[f64; 3]> = mesh.verts.iter().map(|v| [v.x, v.y, v.z]).collect();
    eprintln!(
        "SES: {} vertices, {} triangles (spacing {spacing} Å)",
        mesh.verts.len(),
        mesh.tris.len()
    );

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

    // εΩ=1 (solute), εΣ=78 (water). Local Poisson solve.
    let params = Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    };
    let cfg = SolveConfig {
        tol: 1e-7,
        ..Default::default()
    };
    eprintln!("solving local BEM ({} elements)…", tris.len());
    let (res, stats) = solve_local_elements(&tris, &charges, &params, &cfg).expect("BEM solve");
    eprintln!(
        "converged: {} iters, residual {:.2e}",
        stats.iterations, stats.residual
    );

    // Surface electrostatic potential at each vertex.
    let phi: Vec<f64> = verts
        .iter()
        .map(|v| {
            let xi = Vec3::new(v[0], v[1], v[2]);
            espotential(Domain::Gamma, xi, &tris, &charges, &params, &res)
        })
        .collect();

    let doc = json!({
        "pdb": pdb_path,
        "verts": verts,
        "tris": mesh.tris,
        "vert_potential": phi,
        "n_atoms": atoms.len(),
        "n_charges": charges.len(),
    });
    std::fs::write(&out, serde_json::to_string(&doc).unwrap()).unwrap();
    eprintln!("wrote {out}");
}
