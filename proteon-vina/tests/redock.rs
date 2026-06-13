// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Phase-E redocking validation: the full Monte-Carlo global search must
// recover a known crystal pose from a random start.
//
// This is a *deterministic sanity check on a pinned seed*, not a
// success-rate claim. Docking is stochastic and this test runs at a tiny
// fraction of Vina's default budget (ex=4, 120 MC steps vs 8 × 2500), so
// recovery is seed-dependent: across seeds 0–9 here, 5/10 land < 2 Å and
// the rest report shallower, worse-scoring minima — the search funnels to
// the pocket correctly (recovered modes also score best), it just doesn't
// always *sample* it at this budget. Seed 0 reliably recovers the pose to
// ~0.4 Å. Proper success-rate-over-seeds validation on a PDBbind subset is
// the multi-seed benchmark tracked in devdocs/VINA_ROADMAP.md.

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
    let sbox = SearchBox::around_ligand(&lig, 5.0);
    let params = DockParams {
        exhaustiveness: 4,
        n_poses: 5,
        seed: 0, // pinned: recovers the crystal pose to ~0.4 Å (see header)
        mc: McParams { global_steps: 120, ..McParams::default() },
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

#[test]
fn redocks_1iep_on_the_grid() {
    // Same redock through the precomputed affinity-grid path (use_grid):
    // ~3× faster on this 2229-atom receptor, and the grid's smoothed
    // landscape still recovers the crystal pose (slightly looser than the
    // exact path, ~1.1 Å for this seed).
    let (rec, lig, file) = load("1iep");
    let precalc = Precalculate::vina();
    let crystal = lig.coords.clone();
    let sbox = SearchBox::around_ligand(&lig, 5.0);
    let params = DockParams {
        exhaustiveness: 4,
        n_poses: 5,
        seed: 0,
        mc: McParams { global_steps: 120, use_grid: true, ..McParams::default() },
        ..DockParams::default()
    };
    let modes = dock(&rec, &lig, &file, &precalc, sbox, &params);
    assert!(!modes.is_empty(), "grid docking produced no modes");
    let best_rmsd = modes
        .iter()
        .map(|m| rmsd(&m.coords, &crystal))
        .fold(f64::INFINITY, f64::min);
    assert!(
        best_rmsd < SUCCESS_RMSD,
        "grid path: no mode within {SUCCESS_RMSD} Å of crystal (best {best_rmsd:.2} Å)"
    );
    assert!(modes[0].components.total < 0.0, "grid top mode not favourable");
}
