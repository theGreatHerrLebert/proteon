//! AMBER force field for energy minimization and molecular dynamics.
//!
//! Energy terms:
//! - Bond stretching: E = k(r - r0)²
//! - Angle bending: E = k(θ - θ0)²
//! - Proper/improper torsion: E = (V/div)(1 + cos(f*φ - φ0))
//! - Nonbonded: Lennard-Jones 12-6 + Coulomb with cubic switching
//!
//! Parameters from AMBER96 (parm96.dat).

pub mod params;
pub mod topology;
pub mod energy;
pub mod gb_obc;
pub mod neighbor_list;
pub mod minimize;
pub mod md;
#[cfg(feature = "cuda")]
pub(crate) mod gpu;
