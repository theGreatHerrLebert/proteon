// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Receptor affinity-grid cache, ported from AutoDock-Vina
// src/lib/cache.cpp (Apache-2.0).

//! Precomputed receptor affinity grids — the docking speedup.
//!
//! [`GridCache::build`] precomputes, for each ligand XS interaction type
//! present, a [`Grid`] over the search box holding the energy a ligand
//! atom of that type would feel from the *whole* receptor at each sample
//! point (the inner `O(N_receptor)` sum of [`crate::score::inter_pair_energy`],
//! done once). [`GridCache::eval`] then scores a ligand pose with one
//! trilinear lookup per atom — `O(N_ligand)` instead of
//! `O(N_ligand · N_receptor)` — and its out-of-box slope subsumes the
//! search-time `BoxPenalty`.
//!
//! Energies match the direct pair path to trilinear-interpolation error
//! (smaller with finer granularity); see the parity test.

use crate::atom_types::XsType;
use crate::conf::Vec3;
use crate::grid::{Grid, GridDim};
use crate::molecule::Molecule;
use crate::precalculate::Precalculate;
use crate::score::xs_for_receptor_interaction;
use rayon::prelude::*;
use std::collections::HashMap;

/// Per-XS-type receptor affinity grids over a search box.
#[derive(Clone, Debug)]
pub struct GridCache {
    grids: HashMap<XsType, Grid>,
    slope: f64,
}

impl GridCache {
    /// Default grid spacing in Å (upstream Vina default).
    pub const DEFAULT_GRANULARITY: f64 = 0.375;
    /// Default out-of-box penalty slope (upstream `cache` default).
    pub const DEFAULT_SLOPE: f64 = 1e6;

    /// Build affinity grids over the box `[corner1, corner2]` for every
    /// receptor-interaction XS type present in `ligand`. `granularity` is
    /// the grid spacing in Å (use [`Self::DEFAULT_GRANULARITY`]); `slope`
    /// is the out-of-box penalty (use [`Self::DEFAULT_SLOPE`]).
    #[must_use]
    pub fn build(
        receptor: &Molecule,
        ligand: &Molecule,
        precalc: &Precalculate,
        corner1: Vec3,
        corner2: Vec3,
        granularity: f64,
        slope: f64,
    ) -> Self {
        let (c1, c2) = (corner1, corner2);
        let gd = [
            axis_dim(c1[0], c2[0], granularity),
            axis_dim(c1[1], c2[1], granularity),
            axis_dim(c1[2], c2[2], granularity),
        ];

        // Distinct ligand interaction types that need a grid.
        let mut needed: Vec<XsType> = Vec::new();
        for &xs in &ligand.xs_types {
            if let Some(t) = xs_for_receptor_interaction(xs) {
                if !needed.contains(&t) {
                    needed.push(t);
                }
            }
        }

        // Receptor atoms that interact, with their interaction type.
        let rec: Vec<(Vec3, XsType)> = receptor
            .coords
            .iter()
            .zip(receptor.xs_types.iter())
            .filter_map(|(&c, &xs)| xs_for_receptor_interaction(xs).map(|t| (c, t)))
            .collect();

        let cutoff_sqr = precalc.cutoff_sqr();
        let template = Grid::new(gd);
        let [nx, ny, nz] = template.dims();

        // Populate is O(voxels × receptor atoms) — the one-time cost the
        // grid amortises. Parallelise over x-slabs; each slab independently
        // accumulates `needed.len()` affinity planes of size ny·nz.
        let slabs: Vec<Vec<f64>> = (0..nx)
            .into_par_iter()
            .map(|ix| {
                let mut slab = vec![0.0_f64; needed.len() * ny * nz];
                for iy in 0..ny {
                    for iz in 0..nz {
                        let probe = template.index_to_argument(ix, iy, iz);
                        let cell = iy * nz + iz;
                        for &(rc, rt) in &rec {
                            let dx = rc[0] - probe[0];
                            let dy = rc[1] - probe[1];
                            let dz = rc[2] - probe[2];
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 <= cutoff_sqr {
                                for (k, &t) in needed.iter().enumerate() {
                                    slab[k * ny * nz + cell] += precalc.eval_fast(t, rt, r2);
                                }
                            }
                        }
                    }
                }
                slab
            })
            .collect();

        let mut grids: HashMap<XsType, Grid> =
            needed.iter().map(|&t| (t, Grid::new(gd))).collect();
        for (ix, slab) in slabs.iter().enumerate() {
            for (k, &t) in needed.iter().enumerate() {
                let g = grids.get_mut(&t).unwrap();
                for iy in 0..ny {
                    for iz in 0..nz {
                        g.set(ix, iy, iz, slab[k * ny * nz + iy * nz + iz]);
                    }
                }
            }
        }

        Self { grids, slope }
    }

    /// Number of distinct atom-type grids held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grids.len()
    }

    /// True if no grids are held (no interacting ligand atom types).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }

    /// Grid-based inter-molecular energy and per-atom physics forces
    /// (`−∂E/∂x`) for a ligand pose, drop-in for
    /// [`crate::score::inter_pair_energy_with_forces`]. `v_curl` is the
    /// soft cap (1000.0 authentic). Atoms whose type has no grid
    /// (non-interacting / glue) contribute zero.
    #[must_use]
    pub fn eval(&self, coords: &[Vec3], xs_types: &[XsType], v_curl: f64) -> (f64, Vec<Vec3>) {
        let mut energy = 0.0;
        let mut forces = vec![[0.0_f64; 3]; coords.len()];
        for (i, (&c, &xs)) in coords.iter().zip(xs_types.iter()).enumerate() {
            let Some(t) = xs_for_receptor_interaction(xs) else { continue };
            if let Some(g) = self.grids.get(&t) {
                let (e_i, grad) = g.evaluate(c, self.slope, v_curl);
                energy += e_i;
                forces[i] = [-grad[0], -grad[1], -grad[2]]; // physics force = −∂E/∂x
            }
        }
        (energy, forces)
    }
}

/// Grid dimensions for one axis: enough whole voxels of ~`granularity` to
/// span `[lo, hi]` (at least one voxel).
fn axis_dim(lo: f64, hi: f64, granularity: f64) -> GridDim {
    let span = hi - lo;
    let n_voxels = (span / granularity).ceil().max(1.0) as usize;
    GridDim { begin: lo, end: hi, n_voxels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_search::SearchBox;
    use crate::score::{inter_pair_energy, inter_pair_energy_with_forces};

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const REC_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");

    fn load() -> (Molecule, Molecule) {
        (
            Molecule::from_pdbqt_str(REC_1IEP).unwrap(),
            Molecule::from_pdbqt_str(LIG_1IEP).unwrap(),
        )
    }

    #[test]
    fn grid_energy_approximates_pair_energy_for_in_box_pose() {
        let (rec, lig) = load();
        let precalc = Precalculate::vina();
        // Box enclosing the crystal pose so no atom is out of bounds (the
        // slope penalty would otherwise diverge from the pair sum).
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let cache = { let (c1, c2) = sbox.corners(); GridCache::build(
            &rec, &lig, &precalc, c1, c2, GridCache::DEFAULT_GRANULARITY, GridCache::DEFAULT_SLOPE,
        ) };
        assert!(!cache.is_empty());

        let v = 1000.0;
        let pair = inter_pair_energy(&rec, &lig, &precalc, v);
        let (grid_e, _f) = cache.eval(&lig.coords, &lig.xs_types, v);

        // The grid trilinearly interpolates a steep potential at 0.375 Å, so
        // it smooths the wells slightly (a few % shallower) — the standard
        // grid-docking approximation. Correctness (it converges to the pair
        // energy) is asserted separately. Here we just bound the gap.
        let rel = (grid_e - pair).abs() / pair.abs().max(1.0);
        assert!(rel < 0.10, "grid {grid_e} vs pair {pair} (rel {rel:.4})");
        // Same sign and same ballpark — a favourable pose stays favourable.
        assert!(grid_e < 0.0 && pair < 0.0);
    }

    #[test]
    fn grid_forces_track_pair_forces() {
        let (rec, lig) = load();
        let precalc = Precalculate::vina();
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let cache = { let (c1, c2) = sbox.corners(); GridCache::build(
            &rec, &lig, &precalc, c1, c2, GridCache::DEFAULT_GRANULARITY, GridCache::DEFAULT_SLOPE,
        ) };
        let v = 1000.0;
        let (_e, pair_f) = inter_pair_energy_with_forces(&rec, &lig, &precalc, v);
        let (_ge, grid_f) = cache.eval(&lig.coords, &lig.xs_types, v);
        assert_eq!(pair_f.len(), grid_f.len());
        // Aggregate force magnitude tracks the pair path (the grid smooths
        // the field, so individual atoms differ more than the total).
        let mag = |fs: &[Vec3]| -> f64 {
            fs.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).sum()
        };
        let (mp, mg) = (mag(&pair_f), mag(&grid_f));
        let rel = (mp - mg).abs() / mp.max(1.0);
        assert!(rel < 0.25, "force magnitude pair {mp} vs grid {mg} (rel {rel:.3})");
    }

    #[test]
    fn grid_force_sign_matches_finite_difference() {
        // Guards the end-to-end force *direction* (the magnitude test does
        // not): the returned per-atom force must be −∂E/∂x of cache.eval's
        // own energy. Central differences on a few real ligand atoms.
        let (rec, lig) = load();
        let precalc = Precalculate::vina();
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let (c1, c2) = sbox.corners();
        let cache = GridCache::build(
            &rec, &lig, &precalc, c1, c2, GridCache::DEFAULT_GRANULARITY, GridCache::DEFAULT_SLOPE,
        );
        let v = 1000.0;
        let (_e, forces) = cache.eval(&lig.coords, &lig.xs_types, v);
        let h = 1e-4;
        // A handful of interacting atoms (skip any with ~zero force).
        let mut checked = 0;
        for i in 0..lig.coords.len() {
            let fmag = forces[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if fmag < 1.0 {
                continue;
            }
            for axis in 0..3 {
                let mut cp = lig.coords.clone();
                let mut cm = lig.coords.clone();
                cp[i][axis] += h;
                cm[i][axis] -= h;
                let ep = cache.eval(&cp, &lig.xs_types, v).0;
                let em = cache.eval(&cm, &lig.xs_types, v).0;
                let fd_force = -(ep - em) / (2.0 * h); // −∂E/∂x
                let rel = (forces[i][axis] - fd_force).abs() / forces[i][axis].abs().max(1.0);
                assert!(
                    rel < 0.05,
                    "atom {i} axis {axis}: force {} vs fd {fd_force} (rel {rel:.3})",
                    forces[i][axis]
                );
            }
            checked += 1;
            if checked >= 4 {
                break;
            }
        }
        assert!(checked > 0, "no interacting atom found to check");
    }

    #[test]
    fn finer_granularity_converges_to_pair_energy() {
        // The correctness proof: as spacing → 0 the grid energy converges to
        // the direct pair energy, so the grid computes the right quantity
        // (the in-box-pose gap is pure discretisation error).
        let (rec, lig) = load();
        let precalc = Precalculate::vina();
        let sbox = SearchBox::around_ligand(&lig, 4.0);
        let v = 1000.0;
        let pair = inter_pair_energy(&rec, &lig, &precalc, v);
        let err = |gran: f64| {
            let c = { let (c1, c2) = sbox.corners(); GridCache::build(&rec, &lig, &precalc, c1, c2, gran, GridCache::DEFAULT_SLOPE) };
            (c.eval(&lig.coords, &lig.xs_types, v).0 - pair).abs()
        };
        let coarse = err(0.5);
        let fine = err(0.25);
        assert!(fine < coarse, "finer grid not closer: fine {fine} coarse {coarse}");
    }
}
