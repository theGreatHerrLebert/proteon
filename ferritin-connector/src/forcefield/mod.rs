//! Force-field layer: AMBER96 + CHARMM19+EEF1, plus OBC Generalized Born
//! implicit solvation, energy minimization, and NVE/NVT molecular dynamics.
//!
//! Energy terms:
//! - Bond stretching: E = k(r - r0)²
//! - Angle bending: E = k(θ - θ0)²
//! - Proper/improper torsion: E = (V/div)(1 + cos(f*φ - φ0))
//! - Nonbonded: Lennard-Jones 12-6 + Coulomb with cubic switching
//! - EEF1 implicit solvation (CHARMM19 only): Gaussian-integral Gsolv term
//! - OBC Generalized Born implicit solvation (AMBER96 path): Eq. 7 of
//!   Onufriev, Bashford, Case (2004)
//!
//! Parameter sources: AMBER96 (`parm96.dat`) and CHARMM19+EEF1
//! (`param19.prm` / `solvpar.inp`). Per-module references live in each
//! submodule's doc comment; per-component parameter citations are in
//! `params.rs`.

pub mod energy;
pub mod gb_obc;
#[cfg(feature = "cuda")]
pub(crate) mod gpu;
pub mod md;
pub mod minimize;
pub mod neighbor_list;
pub mod params;
pub mod topology;
