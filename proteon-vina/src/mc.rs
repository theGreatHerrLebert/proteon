// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Single-replicate Metropolis Monte-Carlo search, ported from
// AutoDock-Vina src/lib/monte_carlo.cpp (Apache-2.0).

//! One Monte-Carlo replicate: the outer search that turns the scorer +
//! local optimiser into docking.
//!
//! Each replicate is upstream's `monte_carlo::operator()`:
//!
//! 1. Random placement of the ligand inside the search box.
//! 2. `global_steps` iterations of *mutate one DoF → BFGS-minimise →
//!    Metropolis accept/reject* at temperature `T`.
//! 3. Every accepted, improving pose is rescored with the authentic
//!    `v` cap and folded into an RMSD-clustered pool of the best minima.
//!
//! [`crate::global_search`] runs many of these in parallel and merges
//! the pools. The Metropolis energy (and the pose ranking) is the
//! inter+intra *search* energy — the BFGS objective — exactly as
//! upstream uses `output_type::e`.

use crate::conf::{Conf, Vec3};
use crate::local_only::{minimise_conf_confined, BoxPenalty};
use crate::molecule::Molecule;
use crate::mutate::{gyration_radius, mutate_conf, randomize_conf};
use crate::pdbqt::PdbqtFile;
use crate::precalculate::Precalculate;
use crate::score::{intra_pair_list, score_only, ScoreComponents};
use crate::torsion::{rmsd, TorsionTree};
use rand::Rng;

/// Tunable parameters for one Monte-Carlo replicate. [`Default`] mirrors
/// upstream `monte_carlo`'s constructor defaults.
#[derive(Clone, Copy, Debug)]
pub struct McParams {
    /// Number of MC steps per replicate (upstream default 2500).
    pub global_steps: usize,
    /// Metropolis temperature `RT` (upstream default 1.2 ≈ R·600 K).
    pub temperature: f64,
    /// Mutation step size in Å (upstream default 2.0). Also scales the
    /// orientation step via the pose's radius of gyration.
    pub mutation_amplitude: f64,
    /// BFGS iteration budget per local minimisation. `None` ⇒ upstream's
    /// `(25 + num_movable_atoms) / 3`.
    pub local_steps: Option<usize>,
    /// Capacity of the per-replicate clustered pool of best minima
    /// (upstream default 50).
    pub num_saved_mins: usize,
    /// Two poses closer than this RMSD (Å) are treated as the same
    /// minimum; the lower-energy one is kept (upstream default 0.5).
    pub min_rmsd: f64,
    /// Soft energy cap `v` used during search minimisation *and* final
    /// rescoring. Upstream uses a tighter `hunt_cap` during search and
    /// the authentic `(1000,1000,1000)` for scoring; we use the
    /// authentic cap throughout (the value the parity-validated
    /// `local_only` path uses) to keep the BFGS gradient exactly
    /// consistent with the capped energy.
    pub v_curl: f64,
    /// Out-of-box confinement slope (kcal/mol per Å per axis). Without a
    /// receptor grid, nothing stops BFGS translating the ligand into
    /// empty space; this linear penalty (see
    /// [`crate::local_only::BoxPenalty`]) keeps every atom inside the
    /// search box during minimisation. Poses fully inside pay nothing.
    pub box_slope: f64,
}

impl Default for McParams {
    fn default() -> Self {
        Self {
            global_steps: 2500,
            temperature: 1.2,
            mutation_amplitude: 2.0,
            local_steps: None,
            num_saved_mins: 50,
            min_rmsd: 0.5,
            v_curl: 1000.0,
            box_slope: 100.0,
        }
    }
}

/// A scored docked pose.
#[derive(Clone, Debug)]
pub struct DockPose {
    /// The minimised conformation.
    pub conf: Conf,
    /// World-space ligand atom coordinates of this pose (parallel to the
    /// ligand `Molecule`'s atoms).
    pub coords: Vec<Vec3>,
    /// Inter+intra *search* energy (the BFGS objective). This is the
    /// ranking key and the Metropolis energy — equivalent to upstream
    /// `output_type::e`.
    pub search_energy: f64,
    /// Authentic 8-component score of the pose, as `--score_only` would
    /// report it.
    pub components: ScoreComponents,
}

/// Metropolis acceptance test (upstream `metropolis_accept`): always
/// accept a downhill move; accept an uphill move with probability
/// `exp((old − new) / T)`.
fn metropolis_accept<R: Rng + ?Sized>(old_f: f64, new_f: f64, temperature: f64, rng: &mut R) -> bool {
    if new_f < old_f {
        return true;
    }
    let p = ((old_f - new_f) / temperature).exp();
    rng.gen_range(0.0..1.0) < p
}

/// Run one Monte-Carlo replicate and return its RMSD-clustered pool of
/// best minima, sorted by ascending search energy (best first).
///
/// `corner1`/`corner2` are the lower/upper corners of the search box
/// (expects `corner1[i] < corner2[i]`); the ligand is randomly placed
/// inside it to seed the walk.
#[must_use]
pub fn monte_carlo_replicate<R: Rng + ?Sized>(
    receptor: &Molecule,
    ligand: &Molecule,
    ligand_file: &PdbqtFile,
    precalc: &Precalculate,
    corner1: Vec3,
    corner2: Vec3,
    params: &McParams,
    rng: &mut R,
) -> Vec<DockPose> {
    let tree = TorsionTree::from_molecule(ligand, ligand_file);
    let pairs = intra_pair_list(ligand);
    let mut scratch = ligand.clone();
    let n_tors = tree.num_torsions();
    let max_steps = params.local_steps.unwrap_or((25 + ligand.len()) / 3);
    let confine = Some(BoxPenalty { corner1, corner2, slope: params.box_slope });

    // Seed: random placement, then minimise.
    let start = randomize_conf(corner1, corner2, n_tors, rng);
    let (mut cur_conf, mut cur_out) = minimise_conf_confined(
        receptor, &tree, &pairs, &mut scratch, precalc, start, max_steps, params.v_curl, confine,
    );
    let mut cur_energy = cur_out.final_energy;

    let mut pool: Vec<DockPose> = Vec::new();
    let mut best_e = f64::INFINITY;

    for step in 0..params.global_steps {
        // Mutate a copy of the current pose, then locally minimise it.
        let gr = gyration_radius(&tree.apply(&cur_conf), cur_conf.center);
        let mut candidate = cur_conf.clone();
        mutate_conf(&mut candidate, gr, params.mutation_amplitude, rng);
        let (cand_conf, cand_out) = minimise_conf_confined(
            receptor, &tree, &pairs, &mut scratch, precalc, candidate, max_steps, params.v_curl,
            confine,
        );
        let cand_energy = cand_out.final_energy;

        if step == 0 || metropolis_accept(cur_energy, cand_energy, params.temperature, rng) {
            cur_conf = cand_conf;
            cur_out = cand_out;
            cur_energy = cand_energy;

            if cur_energy < best_e || pool.len() < params.num_saved_mins {
                let pose = build_pose(
                    receptor, &tree, &mut scratch, precalc, &cur_conf, cur_energy,
                    ligand_file, params.v_curl,
                );
                insert_clustered(&mut pool, pose, params.min_rmsd, params.num_saved_mins);
                best_e = best_e.min(cur_energy);
            }
        }
    }

    let _ = cur_out; // last accepted outcome; retained for clarity, not exported
    pool
}

/// Materialise a [`DockPose`]: world coords + authentic 8-component score.
fn build_pose(
    receptor: &Molecule,
    tree: &TorsionTree,
    scratch: &mut Molecule,
    precalc: &Precalculate,
    conf: &Conf,
    search_energy: f64,
    ligand_file: &PdbqtFile,
    v_curl: f64,
) -> DockPose {
    let coords = tree.apply(conf);
    scratch.coords.clone_from(&coords);
    let components = score_only(receptor, scratch, &ligand_file.rotatable_bonds, precalc, v_curl);
    DockPose { conf: conf.clone(), coords, search_energy, components }
}

/// Insert `pose` into the energy-sorted `pool`, collapsing near-duplicate
/// minima (RMSD < `min_rmsd`) to their lowest-energy representative and
/// capping the pool at `capacity`. Mirrors upstream
/// `add_to_output_container`.
fn insert_clustered(pool: &mut Vec<DockPose>, pose: DockPose, min_rmsd: f64, capacity: usize) {
    // If a near-duplicate already sits at lower-or-equal energy, drop the
    // newcomer; if the duplicate is worse, the newcomer replaces it.
    for existing in pool.iter_mut() {
        if rmsd(&existing.coords, &pose.coords) < min_rmsd {
            if pose.search_energy < existing.search_energy {
                *existing = pose;
                sort_and_cap(pool, capacity);
            }
            return;
        }
    }
    pool.push(pose);
    sort_and_cap(pool, capacity);
}

fn sort_and_cap(pool: &mut Vec<DockPose>, capacity: usize) {
    pool.sort_by(|a, b| a.search_energy.total_cmp(&b.search_energy));
    pool.truncate(capacity);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::parse_pdbqt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const REC_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");

    fn load() -> (Molecule, Molecule, PdbqtFile) {
        (
            Molecule::from_pdbqt_str(REC_1IEP).unwrap(),
            Molecule::from_pdbqt_str(LIG_1IEP).unwrap(),
            parse_pdbqt(LIG_1IEP).unwrap(),
        )
    }

    /// A box centred on the crystal ligand, big enough to dock in.
    fn box_around(lig: &Molecule, pad: f64) -> (Vec3, Vec3) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in &lig.coords {
            for i in 0..3 {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i]);
            }
        }
        (
            [lo[0] - pad, lo[1] - pad, lo[2] - pad],
            [hi[0] + pad, hi[1] + pad, hi[2] + pad],
        )
    }

    #[test]
    fn metropolis_always_accepts_downhill() {
        let mut r = ChaCha8Rng::seed_from_u64(1);
        assert!(metropolis_accept(0.0, -5.0, 1.2, &mut r));
    }

    #[test]
    fn metropolis_sometimes_accepts_uphill() {
        let mut r = ChaCha8Rng::seed_from_u64(2);
        let accepts = (0..1000)
            .filter(|_| metropolis_accept(0.0, 0.5, 1.2, &mut r))
            .count();
        // exp(-0.5/1.2) ≈ 0.66 — expect a clear majority but not all.
        assert!((400..900).contains(&accepts), "uphill acceptance {accepts}/1000 off");
    }

    #[test]
    fn replicate_returns_sorted_nonempty_pool() {
        let (rec, lig, file) = load();
        let precalc = Precalculate::vina();
        let (c1, c2) = box_around(&lig, 4.0);
        let params = McParams { global_steps: 30, ..McParams::default() };
        let mut r = ChaCha8Rng::seed_from_u64(42);
        let pool = monte_carlo_replicate(&rec, &lig, &file, &precalc, c1, c2, &params, &mut r);
        assert!(!pool.is_empty(), "replicate produced no poses");
        for w in pool.windows(2) {
            assert!(w[0].search_energy <= w[1].search_energy, "pool not sorted");
        }
        for p in &pool {
            assert!(p.search_energy.is_finite());
            assert!(p.components.total.is_finite());
            assert_eq!(p.coords.len(), lig.len());
        }
    }

    #[test]
    fn pool_respects_min_rmsd_clustering() {
        // Two near-identical poses collapse; distinct poses don't.
        let (_rec, lig, _file) = load();
        let coords = lig.coords.clone();
        let mut shifted = coords.clone();
        for c in &mut shifted {
            c[0] += 0.05; // < 0.5 Å everywhere → same cluster
        }
        let mk = |co: Vec<Vec3>, e: f64| DockPose {
            conf: Conf::identity_at([0.0; 3], 0),
            coords: co,
            search_energy: e,
            components: ScoreComponents::default(),
        };
        let mut pool = Vec::new();
        insert_clustered(&mut pool, mk(coords.clone(), -5.0), 0.5, 50);
        insert_clustered(&mut pool, mk(shifted, -7.0), 0.5, 50);
        assert_eq!(pool.len(), 1, "near-duplicate poses should cluster");
        assert!((pool[0].search_energy - (-7.0)).abs() < 1e-9, "kept the worse pose");
    }
}
