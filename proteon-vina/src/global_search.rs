// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Parallel multi-replicate global search, ported from AutoDock-Vina
// src/lib/{parallel_mc.cpp,vina.cpp} (Apache-2.0).

//! Full docking: many [`crate::mc`] replicates in parallel, merged and
//! RMSD-clustered into a ranked list of distinct binding modes.
//!
//! This is the `vina --dock` equivalent. Each replicate is an
//! independent Metropolis walk from its own random placement, seeded
//! deterministically from `seed ⊕ replicate_index` so a whole run is
//! reproducible. The replicates' pose pools are concatenated, sorted by
//! search energy, and greedily de-duplicated at `min_rmsd` to yield the
//! final modes (upstream's `merge_output_containers` +
//! `remove_redundant`).

use crate::conf::Vec3;
use crate::mc::{monte_carlo_replicate, DockPose, McParams};
use crate::molecule::Molecule;
use crate::pdbqt::PdbqtFile;
use crate::precalculate::Precalculate;
use crate::torsion::rmsd;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// An axis-aligned cubic search box, defined by its center and side
/// lengths in Å — the same `--center_* / --size_*` parameterisation Vina
/// exposes on the command line.
#[derive(Clone, Copy, Debug)]
pub struct SearchBox {
    /// Box center in world coordinates.
    pub center: Vec3,
    /// Box side lengths along x, y, z.
    pub size: Vec3,
}

impl SearchBox {
    /// Construct from explicit center and size.
    #[must_use]
    pub fn new(center: Vec3, size: Vec3) -> Self {
        Self { center, size }
    }

    /// The smallest box enclosing a ligand's current coordinates, grown
    /// by `padding` Å on every side. This is the standard "redock"
    /// box for benchmarking against a crystal pose.
    ///
    /// # Panics
    /// Panics if `ligand` has no atoms.
    #[must_use]
    pub fn around_ligand(ligand: &Molecule, padding: f64) -> Self {
        assert!(!ligand.coords.is_empty(), "cannot box an empty ligand");
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in &ligand.coords {
            for i in 0..3 {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i]);
            }
        }
        let center = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let size = [
            hi[0] - lo[0] + 2.0 * padding,
            hi[1] - lo[1] + 2.0 * padding,
            hi[2] - lo[2] + 2.0 * padding,
        ];
        Self { center, size }
    }

    /// Lower and upper corners `(corner1, corner2)` with
    /// `corner1[i] < corner2[i]`.
    #[must_use]
    pub fn corners(&self) -> (Vec3, Vec3) {
        let h = [self.size[0] * 0.5, self.size[1] * 0.5, self.size[2] * 0.5];
        (
            [
                self.center[0] - h[0],
                self.center[1] - h[1],
                self.center[2] - h[2],
            ],
            [
                self.center[0] + h[0],
                self.center[1] + h[1],
                self.center[2] + h[2],
            ],
        )
    }
}

/// Parameters for a full docking run.
#[derive(Clone, Copy, Debug)]
pub struct DockParams {
    /// Number of independent MC replicates run in parallel. Higher =
    /// more thorough but slower (upstream's `--exhaustiveness`, default 8).
    pub exhaustiveness: usize,
    /// Maximum number of distinct binding modes to return (upstream
    /// `--num_modes`, default 9).
    pub n_poses: usize,
    /// Only report modes within this many kcal/mol of the best mode
    /// (upstream `--energy_range`, default 3.0).
    pub energy_range: f64,
    /// Master RNG seed; each replicate derives a distinct stream from it,
    /// so a given `(seed, params)` reproduces the run exactly.
    pub seed: u64,
    /// Per-replicate Monte-Carlo parameters.
    pub mc: McParams,
}

impl Default for DockParams {
    fn default() -> Self {
        Self {
            exhaustiveness: 8,
            n_poses: 9,
            energy_range: 3.0,
            seed: 0,
            mc: McParams::default(),
        }
    }
}

/// Dock `ligand` into `receptor` within `search_box`, returning distinct
/// binding modes sorted by ascending search energy (best first).
///
/// Runs `params.exhaustiveness` Monte-Carlo replicates over a rayon
/// thread pool, merges their pose pools, de-duplicates at
/// `params.mc.min_rmsd`, and trims to `n_poses` / `energy_range`.
#[must_use]
pub fn dock(
    receptor: &Molecule,
    ligand: &Molecule,
    ligand_file: &PdbqtFile,
    precalc: &Precalculate,
    search_box: SearchBox,
    params: &DockParams,
) -> Vec<DockPose> {
    let (corner1, corner2) = search_box.corners();

    // Independent replicates, each with a deterministic distinct stream.
    let pools: Vec<Vec<DockPose>> = (0..params.exhaustiveness)
        .into_par_iter()
        .map(|i| {
            let stream = params
                .seed
                .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = ChaCha8Rng::seed_from_u64(stream);
            monte_carlo_replicate(
                receptor,
                ligand,
                ligand_file,
                precalc,
                corner1,
                corner2,
                &params.mc,
                &mut rng,
            )
        })
        .collect();

    merge_and_cluster(pools, params)
}

/// Concatenate replicate pools, sort by energy, greedily keep modes that
/// are ≥ `min_rmsd` from every better mode already kept, then apply the
/// `n_poses` and `energy_range` cutoffs.
fn merge_and_cluster(pools: Vec<Vec<DockPose>>, params: &DockParams) -> Vec<DockPose> {
    let mut merged: Vec<DockPose> = pools.into_iter().flatten().collect();
    merged.sort_by(|a, b| a.search_energy.total_cmp(&b.search_energy));

    let mut modes: Vec<DockPose> = Vec::new();
    let best = merged.first().map_or(f64::INFINITY, |p| p.search_energy);
    for pose in merged {
        if pose.search_energy > best + params.energy_range {
            break; // sorted: everything after is also out of range
        }
        if modes
            .iter()
            .all(|m| rmsd(&m.coords, &pose.coords) >= params.mc.min_rmsd)
        {
            modes.push(pose);
            if modes.len() >= params.n_poses {
                break;
            }
        }
    }
    modes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::parse_pdbqt;

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const REC_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");

    fn load() -> (Molecule, Molecule, PdbqtFile) {
        (
            Molecule::from_pdbqt_str(REC_1IEP).unwrap(),
            Molecule::from_pdbqt_str(LIG_1IEP).unwrap(),
            parse_pdbqt(LIG_1IEP).unwrap(),
        )
    }

    #[test]
    fn search_box_corners_and_autobox() {
        let b = SearchBox::new([0.0, 0.0, 0.0], [10.0, 20.0, 4.0]);
        let (c1, c2) = b.corners();
        assert_eq!(c1, [-5.0, -10.0, -2.0]);
        assert_eq!(c2, [5.0, 10.0, 2.0]);

        let (_rec, lig, _file) = load();
        let ab = SearchBox::around_ligand(&lig, 4.0);
        let (lo, hi) = ab.corners();
        for c in &lig.coords {
            for i in 0..3 {
                assert!(
                    c[i] >= lo[i] - 1e-9 && c[i] <= hi[i] + 1e-9,
                    "atom outside autobox"
                );
            }
        }
    }

    #[test]
    fn dock_returns_ranked_distinct_modes() {
        let (rec, lig, file) = load();
        let precalc = Precalculate::vina();
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let params = DockParams {
            exhaustiveness: 4,
            n_poses: 5,
            seed: 7,
            mc: McParams {
                global_steps: 50,
                ..McParams::default()
            },
            ..DockParams::default()
        };
        let modes = dock(&rec, &lig, &file, &precalc, sbox, &params);
        assert!(!modes.is_empty(), "docking produced no modes");
        // Sorted ascending.
        for w in modes.windows(2) {
            assert!(w[0].search_energy <= w[1].search_energy);
        }
        // Distinct (pairwise RMSD >= min_rmsd).
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert!(
                    rmsd(&modes[i].coords, &modes[j].coords) >= params.mc.min_rmsd - 1e-9,
                    "modes {i} and {j} are not distinct"
                );
            }
        }
        // The best mode is a genuinely bound pose (favourable score).
        assert!(
            modes[0].components.total < 0.0,
            "best mode not favourable: {}",
            modes[0].components.total
        );
    }

    #[test]
    fn dock_is_reproducible_for_fixed_seed() {
        let (rec, lig, file) = load();
        let precalc = Precalculate::vina();
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let params = DockParams {
            exhaustiveness: 3,
            n_poses: 3,
            seed: 123,
            mc: McParams {
                global_steps: 40,
                ..McParams::default()
            },
            ..DockParams::default()
        };
        let a = dock(&rec, &lig, &file, &precalc, sbox, &params);
        let b = dock(&rec, &lig, &file, &precalc, sbox, &params);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.search_energy - y.search_energy).abs() < 1e-9,
                "run not reproducible"
            );
        }
    }
}
