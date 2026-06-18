//! `NbCache` — a reusable nonbonded-evaluation cache shared by the minimizer and MD.
//!
//! It dispatches energy/force calls to the cutoff neighbor-list path (and, with the
//! `cuda` feature, the GPU) when the structure is large enough to benefit, and falls
//! back to the exact all-pair path otherwise. The neighbor list is built once and
//! rebuilt on drift via [`NbCache::refresh`] (the standard Verlet rebuild criterion),
//! so a long-running caller (an iterative minimizer, or an MD trajectory) pays the
//! O(N²) build only occasionally instead of every evaluation.

use super::energy::{
    compute_energy, compute_energy_and_forces, compute_energy_and_forces_nbl_inner, EnergyResult,
};
use super::neighbor_list::NeighborList;
use super::params::ForceField;
use super::topology::Topology;

/// Atom count at/above which `NbExec::Auto` switches to the neighbor-list path. Matches
/// `energy::NBL_AUTO_THRESHOLD`. Below it the O(N²) loop is faster than building/querying
/// a list.
pub(crate) const MIN_NBL_THRESHOLD: usize = 2000;

/// Nonbonded execution policy for [`NbCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NbExec {
    /// Always the exact all-pair path (no neighbor list, no GPU). (Test/A-B hook;
    /// production paths use `Auto`.)
    #[cfg_attr(not(test), allow(dead_code))]
    AllPair,
    /// Always the CPU neighbor-list path (no GPU) — lets a test exercise the NBL code on
    /// a small fixture with deterministic CPU accumulation.
    #[cfg_attr(not(test), allow(dead_code))]
    CpuNbl,
    /// Size-gated: neighbor list (+ GPU when available) iff `n >= MIN_NBL_THRESHOLD`.
    Auto,
}

/// Caches a neighbor list (and optional GPU state) and dispatches energy/force calls.
pub(crate) struct NbCache {
    nbl: Option<NeighborList>,
    cutoff: f64,
    /// Exclusion-free GB neighbor list for the CutoffNonPeriodic OBC GB method.
    /// `Some` only when the force field selects cutoff GB AND the NBL path is
    /// active; refreshed atomically with `nbl` (same coords/cutoff/buffer). GB
    /// has no bonded exclusions, so this is a SEPARATE list from `nbl`.
    gb_nbl: Option<NeighborList>,
    /// GB cutoff captured at construction. `debug_assert`-ed against the
    /// evaluation-time force field so a caller can't swap in a force field whose
    /// GB method differs from the one `gb_nbl` was built for.
    gb_cutoff: Option<f64>,
    /// Bumped every time `refresh` actually rebuilds the lists. Lets tests prove
    /// a drift triggered (or did not trigger) a rebuild rather than inferring it.
    pub(crate) rebuild_generation: u64,
    /// GPU state for CUDA-accelerated energy+forces evaluation. Present only with the
    /// `cuda` feature AND a detected GPU AND a policy that enables it. Dropped to `None`
    /// (silent CPU degrade) if a neighbor-list re-upload ever fails, so a stale GPU pair
    /// list can never silently produce wrong forces (critical for MD trajectories).
    #[cfg(feature = "cuda")]
    gpu: Option<super::gpu::GpuStructState>,
}

impl NbCache {
    /// Build with the default [`NbExec::Auto`] policy.
    pub(crate) fn new<F: ForceField>(coords: &[[f64; 3]], topo: &Topology, params: &F) -> Self {
        Self::new_with_exec(coords, topo, params, NbExec::Auto)
    }

    /// Build with an explicit execution policy.
    pub(crate) fn new_with_exec<F: ForceField>(
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
        exec: NbExec,
    ) -> Self {
        let cutoff = params.nonbonded_cutoff();
        let use_nbl = match exec {
            NbExec::AllPair => false,
            NbExec::CpuNbl => true,
            NbExec::Auto => coords.len() >= MIN_NBL_THRESHOLD,
        };
        let nbl = if use_nbl {
            Some(NeighborList::build(
                coords,
                cutoff,
                &topo.excluded_pairs,
                &topo.pairs_14,
            ))
        } else {
            None
        };

        // Cutoff GB needs its OWN exclusion-free list (GB has no bonded
        // exclusions). Built only on the NBL path under the cutoff GB method.
        let gb_cutoff = params.gb_cutoff();
        let gb_nbl = if let (true, Some(rc)) = (use_nbl, gb_cutoff) {
            Some(NeighborList::build(
                coords,
                rc,
                &std::collections::HashSet::new(),
                &std::collections::HashSet::new(),
            ))
        } else {
            None
        };

        // GPU only under Auto (CpuNbl/AllPair are explicitly CPU). Falls back silently.
        #[cfg(feature = "cuda")]
        let gpu = if matches!(exec, NbExec::Auto) && coords.len() >= MIN_NBL_THRESHOLD {
            super::gpu::GpuContext::try_global().and_then(|gpu_ctx| {
                let nbl_ref = nbl.as_ref()?;
                match super::gpu::GpuStructState::new(gpu_ctx, topo, nbl_ref, params) {
                    Ok(state) => Some(state),
                    Err(e) => {
                        eprintln!("[proteon-gpu] Failed to upload topology: {}", e);
                        None
                    }
                }
            })
        } else {
            None
        };

        Self {
            nbl,
            cutoff,
            gb_nbl,
            gb_cutoff,
            rebuild_generation: 0,
            #[cfg(feature = "cuda")]
            gpu,
        }
    }

    /// Rebuild the neighbor list if any atom has drifted further than the buffer allows.
    /// Cheap no-op if not using NBL or if atoms haven't moved much.
    pub(crate) fn refresh(&mut self, coords: &[[f64; 3]], topo: &Topology) {
        if let Some(ref nbl) = self.nbl {
            if nbl.needs_rebuild(coords) {
                let new_nbl =
                    NeighborList::build(coords, self.cutoff, &topo.excluded_pairs, &topo.pairs_14);
                // Re-upload NBL pairs to GPU. On failure DROP the GPU state and degrade to
                // the CPU NBL path — keeping the GPU enabled would dispatch against a stale
                // pair list and silently produce wrong forces (corrupting an MD trajectory).
                #[cfg(feature = "cuda")]
                if let Some(mut gpu) = self.gpu.take() {
                    match gpu.refresh_nbl(&new_nbl) {
                        Ok(()) => self.gpu = Some(gpu),
                        Err(e) => eprintln!(
                            "[proteon-gpu] NBL re-upload failed, dropping GPU (CPU fallback): {}",
                            e
                        ),
                    }
                }
                self.nbl = Some(new_nbl);
                // Rebuild the GB list (at its OWN cutoff) atomically on the SAME
                // drift trigger. Validity is bounded by displacement vs the
                // BUFFER, not the cutoff: both lists are built from identical
                // coords with the same 2 Å Verlet buffer, so the LJ list's
                // `needs_rebuild` (displacement > buffer/2) is exactly the GB
                // list's trigger too — even for a custom force field whose GB
                // cutoff differs from the nonbonded cutoff. No cutoff-equality
                // requirement (codex review).
                if let Some(rc) = self.gb_cutoff {
                    self.gb_nbl = Some(NeighborList::build(
                        coords,
                        rc,
                        &std::collections::HashSet::new(),
                        &std::collections::HashSet::new(),
                    ));
                }
                self.rebuild_generation += 1;
            }
        }
    }

    pub(crate) fn energy<F: ForceField>(
        &mut self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> EnergyResult {
        // The GB method must match the one the cache (CPU list AND GPU constants,
        // both captured at construction) was built for. Checked BEFORE the GPU
        // dispatch — the GPU's cutoff is baked in at `GpuStructState::new`, so a
        // reused cache with a swapped force field would otherwise silently use
        // the stale GPU cutoff (codex review).
        debug_assert!(
            params.gb_cutoff() == self.gb_cutoff,
            "NbCache GB method changed since construction"
        );
        // GPU path: launch kernels, sync, read energy (no forces download)
        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu {
            let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
            if let Ok(gpu_ctx) = super::gpu::GpuContext::try_global().ok_or(()) {
                match gpu.energy(gpu_ctx, &coords_flat) {
                    Ok(result) => return result,
                    Err(e) => eprintln!(
                        "[proteon-gpu] GPU energy failed, falling back to CPU: {}",
                        e
                    ),
                }
            }
        }

        // CPU fallback. The cached GB list (when present) feeds the GB term so
        // iterative callers don't rebuild it each eval.
        match &self.nbl {
            Some(nbl) => {
                compute_energy_and_forces_nbl_inner(coords, topo, params, nbl, self.gb_nbl.as_ref())
                    .0
            }
            None => compute_energy(coords, topo, params),
        }
    }

    pub(crate) fn energy_and_forces<F: ForceField>(
        &mut self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> (EnergyResult, Vec<[f64; 3]>) {
        // Method-consistency check BEFORE GPU dispatch (see `energy`).
        debug_assert!(
            params.gb_cutoff() == self.gb_cutoff,
            "NbCache GB method changed since construction"
        );
        // GPU path: launch kernels, sync, read energy + forces
        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu {
            let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
            if let Ok(gpu_ctx) = super::gpu::GpuContext::try_global().ok_or(()) {
                match gpu.energy_and_forces(gpu_ctx, &coords_flat) {
                    Ok(result) => return result,
                    Err(e) => eprintln!(
                        "[proteon-gpu] GPU energy+forces failed, falling back to CPU: {}",
                        e
                    ),
                }
            }
        }

        // CPU fallback. Pass the cached GB list (when present) into the GB term.
        match &self.nbl {
            Some(nbl) => {
                compute_energy_and_forces_nbl_inner(coords, topo, params, nbl, self.gb_nbl.as_ref())
            }
            None => compute_energy_and_forces(coords, topo, params),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::params::{self, AmberParams};
    use crate::forcefield::topology::{FFAtom, Topology};
    use std::collections::HashSet;

    fn ff_atom_q(amber_type: &str, element: &str, pos: [f64; 3], charge: f64) -> FFAtom {
        FFAtom {
            pos,
            amber_type: amber_type.to_string(),
            charge,
            residue_name: "XXX".into(),
            atom_name: "X".into(),
            element: element.into(),
            residue_idx: 0,
            is_hydrogen: element == "H",
        }
    }

    fn topo_of(atoms: Vec<FFAtom>, excluded: HashSet<(usize, usize)>) -> Topology {
        Topology {
            atoms,
            bonds: vec![],
            angles: vec![],
            torsions: vec![],
            improper_torsions: vec![],
            excluded_pairs: excluded,
            pairs_14: HashSet::new(),
            lj_excluded_pairs: HashSet::new(),
            unassigned_atoms: vec![],
            inferred_bonds: false,
        }
    }

    /// AMBER96+OBC with the cutoff GB method at an explicit small cutoff so test
    /// geometries stay compact.
    fn cutoff_params(rc: f64) -> AmberParams {
        let mut p = params::amber96_obc_cutoff();
        p.cutoff_override = Some(rc);
        p
    }

    fn has_pair(nbl: &NeighborList, i: usize, j: usize) -> bool {
        let (lo, hi) = (i.min(j), i.max(j));
        nbl.pairs.iter().any(|p| p.i == lo && p.j == hi)
    }

    fn forces_close(a: &[[f64; 3]], b: &[[f64; 3]], tol: f64) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (0..3).all(|k| (x[k] - y[k]).abs() <= tol))
    }

    #[test]
    fn cached_gb_equals_internal_build_and_all_pairs() {
        // Same coords ⇒ the cached GB list is byte-identical to a fresh internal
        // build (deterministic cell list), so NbCache must equal the public
        // _nbl path (builds internally) exactly, and the all-pairs cutoff path
        // to FP tolerance.
        let params = cutoff_params(6.0);
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.6, 0.3, 0.1],
            [3.1, 1.2, -0.4],
            [2.0, -1.5, 0.8],
        ];
        let topo = topo_of(
            vec![
                ff_atom_q("CT", "C", coords[0], 0.3),
                ff_atom_q("N", "N", coords[1], -0.4),
                ff_atom_q("O", "O", coords[2], -0.5),
                ff_atom_q("CT", "C", coords[3], 0.2),
            ],
            HashSet::new(),
        );

        let mut nbc = NbCache::new_with_exec(&coords, &topo, &params, NbExec::CpuNbl);
        let (e_cached, f_cached) = nbc.energy_and_forces(&coords, &topo, &params);

        // Public _nbl path builds the GB list internally each call.
        let lj_nbl = NeighborList::build(
            &coords,
            params.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        let (e_internal, f_internal) =
            super::compute_energy_and_forces_nbl_inner(&coords, &topo, &params, &lj_nbl, None);
        assert_eq!(
            e_cached.solvation, e_internal.solvation,
            "cached != internal"
        );
        assert!(
            forces_close(&f_cached, &f_internal, 0.0),
            "forces cached != internal"
        );

        // All-pairs cutoff path: same method, different enumeration ⇒ tolerance.
        let (e_all, f_all) = super::compute_energy_and_forces(&coords, &topo, &params);
        assert!(
            (e_cached.total - e_all.total).abs() < 1e-9,
            "cached {} vs all-pairs {}",
            e_cached.total,
            e_all.total
        );
        assert!(
            forces_close(&f_cached, &f_all, 1e-9),
            "forces cached != all-pairs"
        );
    }

    #[test]
    fn drift_past_buffer_rebuilds_gb_list() {
        // A GB pair initially beyond cutoff+buffer is absent; after an atom moves
        // it inside the physical cutoff, refresh() must REBUILD the list so the
        // pair appears (proving the cache isn't serving a stale, pair-missing list).
        let rc = 5.0; // buffer 2 Å ⇒ list reaches 7 Å
        let params = cutoff_params(rc);
        // atom2 starts at 10 Å (> rc+buffer from both anchors) → absent.
        let coords0 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let topo = topo_of(
            vec![
                ff_atom_q("CT", "C", coords0[0], 0.3),
                ff_atom_q("CT", "C", coords0[1], -0.3),
                ff_atom_q("CT", "C", coords0[2], 0.4),
            ],
            HashSet::new(),
        );
        let mut nbc = NbCache::new_with_exec(&coords0, &topo, &params, NbExec::CpuNbl);

        let gb0 = nbc.gb_nbl.as_ref().expect("gb list built");
        assert!(!has_pair(gb0, 0, 2), "pair (0,2) must be absent initially");
        assert!(!has_pair(gb0, 1, 2), "pair (1,2) must be absent initially");
        assert_eq!(nbc.rebuild_generation, 0);

        // Move atom2 to 4 Å (within rc of atom0); displacement 6 Å >> buffer/2.
        let coords1 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        nbc.refresh(&coords1, &topo);
        assert_eq!(nbc.rebuild_generation, 1, "refresh must have rebuilt");
        let gb1 = nbc.gb_nbl.as_ref().unwrap();
        assert!(has_pair(gb1, 0, 2), "pair (0,2) must appear after refresh");
        assert!(has_pair(gb1, 1, 2), "pair (1,2) must appear after refresh");

        // The freshly-refreshed cache must match a from-scratch all-pairs eval.
        let (e_cached, f_cached) = nbc.energy_and_forces(&coords1, &topo, &params);
        let (e_all, f_all) = super::compute_energy_and_forces(&coords1, &topo, &params);
        assert!(
            (e_cached.total - e_all.total).abs() < 1e-9,
            "post-refresh energy mismatch"
        );
        assert!(
            forces_close(&f_cached, &f_all, 1e-9),
            "post-refresh force mismatch"
        );
    }

    #[test]
    fn sub_threshold_move_does_not_rebuild() {
        let params = cutoff_params(5.0);
        let coords0 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.5, 0.0]];
        let topo = topo_of(
            vec![
                ff_atom_q("CT", "C", coords0[0], 0.3),
                ff_atom_q("CT", "C", coords0[1], -0.3),
                ff_atom_q("CT", "C", coords0[2], 0.0),
            ],
            HashSet::new(),
        );
        let mut nbc = NbCache::new_with_exec(&coords0, &topo, &params, NbExec::CpuNbl);
        // Move every atom < buffer/2 (= 1 Å).
        let coords1 = vec![[0.1, 0.0, 0.0], [2.05, 0.0, 0.0], [3.45, 0.5, 0.0]];
        nbc.refresh(&coords1, &topo);
        assert_eq!(
            nbc.rebuild_generation, 0,
            "sub-threshold move must NOT rebuild"
        );
    }

    #[test]
    fn gb_list_includes_bonded_pairs_that_lj_excludes() {
        // GB has no exclusions: a 1-2/1-3 bonded pair is absent from the LJ list
        // but present in the exclusion-free GB list.
        let params = cutoff_params(6.0);
        let coords = vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        let topo = topo_of(
            vec![
                ff_atom_q("CT", "C", coords[0], 0.3),
                ff_atom_q("CT", "C", coords[1], -0.3),
            ],
            excluded,
        );
        let nbc = NbCache::new_with_exec(&coords, &topo, &params, NbExec::CpuNbl);
        let lj = nbc.nbl.as_ref().unwrap();
        let gb = nbc.gb_nbl.as_ref().unwrap();
        assert!(!has_pair(lj, 0, 1), "LJ list must exclude the bonded pair");
        assert!(has_pair(gb, 0, 1), "GB list must include the bonded pair");
    }

    #[test]
    fn nocutoff_gb_builds_no_gb_list() {
        // Default NoCutoff GB (amber96_obc) must not build a GB list at all.
        let params = params::amber96_obc();
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let topo = topo_of(
            vec![
                ff_atom_q("CT", "C", coords[0], 0.3),
                ff_atom_q("CT", "C", coords[1], -0.3),
            ],
            HashSet::new(),
        );
        let nbc = NbCache::new_with_exec(&coords, &topo, &params, NbExec::CpuNbl);
        assert!(
            nbc.gb_nbl.is_none(),
            "NoCutoff GB must not allocate a GB list"
        );
        assert!(nbc.gb_cutoff.is_none());
    }
}
