//! `NbCache` — a reusable nonbonded-evaluation cache shared by the minimizer and MD.
//!
//! It dispatches energy/force calls to the cutoff neighbor-list path (and, with the
//! `cuda` feature, the GPU) when the structure is large enough to benefit, and falls
//! back to the exact all-pair path otherwise. The neighbor list is built once and
//! rebuilt on drift via [`NbCache::refresh`] (the standard Verlet rebuild criterion),
//! so a long-running caller (an iterative minimizer, or an MD trajectory) pays the
//! O(N²) build only occasionally instead of every evaluation.

use super::energy::{
    compute_energy, compute_energy_and_forces, compute_energy_and_forces_nbl, compute_energy_nbl,
    EnergyResult,
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
            }
        }
    }

    pub(crate) fn energy<F: ForceField>(
        &mut self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> EnergyResult {
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

        // CPU fallback
        match &self.nbl {
            Some(nbl) => compute_energy_nbl(coords, topo, params, nbl),
            None => compute_energy(coords, topo, params),
        }
    }

    pub(crate) fn energy_and_forces<F: ForceField>(
        &mut self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> (EnergyResult, Vec<[f64; 3]>) {
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

        // CPU fallback
        match &self.nbl {
            Some(nbl) => compute_energy_and_forces_nbl(coords, topo, params, nbl),
            None => compute_energy_and_forces(coords, topo, params),
        }
    }
}
