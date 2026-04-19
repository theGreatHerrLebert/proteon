//! CUDA-accelerated search paths.
//!
//! **Phase 4.1 scaffold only — no kernels yet.** This module establishes
//! the dispatch framework that the kernel-bearing phases (4.2 batched
//! diagonal scoring, 4.3 GPU SW, 4.4 libmarv-equivalent) will plug into.
//!
//! Reference for the GPU Smith-Waterman / PSSM-SW kernel family:
//! Kallenborn, Chacon, Hundt, Sirelkhatim, Didi, Cha, Dallago, Mirdita,
//! Schmidt, Steinegger, "GPU-accelerated homology search with MMseqs2",
//! *Nat. Methods* 22, 2024-2027 (2025). The "libmarv" naming throughout
//! the `pssm_sw*.rs` modules refers to the canonical kernel design from
//! that work.
//!
//! Architecture mirrors `proteon-connector::forcefield::gpu`:
//!
//! - [`GpuContext`] is a process-global singleton via `OnceLock`. Probe
//!   on first access; cache the result; return `Option<&'static
//!   GpuContext>` so callers can fall back to CPU silently when no GPU
//!   is available.
//! - Kernel sources live as `.cu` files alongside this module and are
//!   compiled at runtime via NVRTC, targeted at the detected compute
//!   capability (`sm_75` on the dev 2070, `sm_120` on the production
//!   5090). Per-kernel module structs hold the resulting
//!   `CudaFunction` handles after compile.
//! - All cudarc usage is encapsulated here. The rest of the crate sees
//!   only `GpuContext::try_global()` and the dispatch helpers each
//!   future kernel module exports (e.g. `gpu::diagonal::score_batch`).
//!
//! Validation strategy (per the user's instruction): every GPU kernel
//! ports a CPU primitive that already has unit tests, and ships
//! alongside a parity test that asserts GPU output equals CPU output
//! on hand-crafted inputs. The CPU implementation is the oracle —
//! identical to how upstream MMseqs2's CPU paths are oracles for its
//! own GPU kernels.

use std::sync::{Arc, OnceLock};

use cudarc::driver::*;

pub mod diagonal;
pub mod pssm_diagonal;
pub mod pssm_sw;
pub mod pssm_sw_warp;
pub mod pssm_sw_warp_multitile;
pub mod sw;

/// Process-global GPU context. Holds the CUDA device + a stream pool
/// + (in future phases) compiled kernel handles. Created on first
/// `try_global()` call; `None` if no GPU is reachable.
pub struct GpuContext {
    ctx: Arc<CudaContext>,
    /// Detected compute capability, used to target NVRTC compiles.
    /// (major, minor), e.g. (7, 5) for Turing, (12, 0) for Blackwell.
    cc: (i32, i32),
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get the global GPU context, initializing on first call.
    /// Returns `None` if no GPU is reachable. Callers should treat
    /// that as "use the CPU path."
    pub fn try_global() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(|| match Self::init() {
                Ok(ctx) => {
                    eprintln!(
                        "[proteon-search-gpu] CUDA initialized: {} (CC {}.{})",
                        ctx.ctx.name().unwrap_or_default(),
                        ctx.cc.0,
                        ctx.cc.1,
                    );
                    Some(ctx)
                }
                Err(e) => {
                    eprintln!(
                        "[proteon-search-gpu] No GPU available, using CPU path: {}",
                        e
                    );
                    None
                }
            })
            .as_ref()
    }

    fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let cc = ctx.compute_capability()?;
        Ok(Self { ctx, cc })
    }

    /// Detected device name (e.g. "NVIDIA GeForce RTX 5090").
    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_default()
    }

    /// `(major, minor)` compute capability. Used internally to pick the
    /// NVRTC `--gpu-architecture=sm_<MM>` flag for kernel compilation.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    /// `--gpu-architecture` string for NVRTC (e.g. `"sm_75"`, `"sm_120"`).
    /// Future kernel modules call this to build their compile options.
    pub fn arch_flag(&self) -> String {
        format!("sm_{}{}", self.cc.0, self.cc.1)
    }

    /// Total device memory in MB (for diagnostics only).
    pub fn total_memory_mb(&self) -> usize {
        self.ctx.total_mem().unwrap_or(0) / (1024 * 1024)
    }

    /// Underlying cudarc context. Pub(crate) so future kernel modules
    /// inside `gpu/` can build streams + load PTX without re-exposing
    /// cudarc to the broader crate.
    #[allow(dead_code)]
    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

/// Capability probe for callers that want a quick "is GPU usable?"
/// answer without holding onto the context. Cheap after the first call.
pub fn is_available() -> bool {
    GpuContext::try_global().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_global()` must never panic regardless of host GPU state.
    /// On a CPU-only CI runner it returns `None`; on a GPU host it
    /// returns `Some(_)` with a populated compute capability. Both
    /// outcomes are valid for this scaffold.
    #[test]
    fn try_global_is_safe_to_call_anywhere() {
        let ctx = GpuContext::try_global();
        match ctx {
            Some(g) => {
                let (major, minor) = g.compute_capability();
                assert!(major >= 5, "device CC {major}.{minor} below cudarc minimum",);
                assert_eq!(g.arch_flag(), format!("sm_{major}{minor}"));
                assert!(!g.device_name().is_empty());
            }
            None => {
                // CPU-only — also fine. Just exercise the second call to
                // confirm the OnceLock cache returns the same `None`.
                assert!(GpuContext::try_global().is_none());
            }
        }
    }

    #[test]
    fn is_available_matches_try_global() {
        assert_eq!(is_available(), GpuContext::try_global().is_some());
    }
}
