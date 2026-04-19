//! Secondary structure-based initialization.
//!
//! Ported from C++ TMalign `get_initial_ss` (lines 2499-2504).

use crate::core::nwdp::nwdp_secondary_structure;
use crate::core::types::DPWorkspace;

/// Get initial alignment from secondary structure matching.
///
/// Returns alignment map (j2i).
pub fn get_initial_ss(ws: &mut DPWorkspace, secx: &[char], secy: &[char]) -> Vec<i32> {
    nwdp_secondary_structure(ws, secx, secy, -1.0)
}
