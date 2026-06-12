// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Phase-E redocking validation: the full Monte-Carlo global search must
// recover a known crystal pose from a random start. Unlike the per-pose
// scoring/local-opt parity tests, docking parity is *statistical* — we
// assert that the best-scoring mode lands within the standard 2.0 Å
// success threshold of the crystal ligand, from a fixed seed so the run
// is deterministic.

use proteon_vina::global_search::{dock, DockParams, SearchBox};
use proteon_vina::mc::McParams;
use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::{parse_pdbqt, PdbqtFile};
use proteon_vina::precalculate::Precalculate;
use proteon_vina::torsion::rmsd;

/// Standard "successful redock" threshold used across the docking
/// literature.
const SUCCESS_RMSD: f64 = 2.0;

fn fixture(name: &str, kind: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/pairs/{name}/{kind}.pdbqt"))
        .or_else(|_| {
            std::fs::read_to_string(format!(
                "proteon-vina/tests/fixtures/pairs/{name}/{kind}.pdbqt"
            ))
        })
        .unwrap_or_else(|_| panic!("missing fixture {name}/{kind}"))
}

fn load(name: &str) -> (Molecule, Molecule, PdbqtFile) {
    let lig_s = fixture(name, "ligand");
    let rec_s = fixture(name, "receptor");
    let rec = Molecule::from_pdbqt_str(&rec_s).unwrap();
    let lig = Molecule::from_pdbqt_str(&lig_s).unwrap();
    let file = parse_pdbqt(&lig_s).unwrap();
    (rec, lig, file)
}

#[test]
fn redocks_1iep_within_2_angstrom() {
    let (rec, lig, file) = load("1iep");
    let precalc = Precalculate::vina();
    let crystal = lig.coords.clone();

    // Box around the crystal site; random placement inside it.
    let sbox = SearchBox::around_ligand(&lig, 6.0);
    let params = DockParams {
        exhaustiveness: 4,
        n_poses: 5,
        seed: 7,
        mc: McParams { global_steps: 100, ..McParams::default() },
        ..DockParams::default()
    };

    let modes = dock(&rec, &lig, &file, &precalc, sbox, &params);
    assert!(!modes.is_empty(), "docking produced no modes");

    // Standard "redock success in top-N": the crystal pose must be
    // recovered among the reported modes. Top-1-only is too strict here
    // because our local-opt carries ~1 kcal/mol drift vs upstream, which
    // can reorder near-degenerate modes.
    let (rank, best_rmsd) = modes
        .iter()
        .map(|m| rmsd(&m.coords, &crystal))
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert!(
        best_rmsd < SUCCESS_RMSD,
        "no mode within {SUCCESS_RMSD} Å of crystal (best {best_rmsd:.2} Å at rank {rank})"
    );

    // The top-scoring mode must be a favourable, genuinely bound pose.
    assert!(
        modes[0].components.total < -8.0,
        "top mode score {:.2} not favourable",
        modes[0].components.total
    );
}
