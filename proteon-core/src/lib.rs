//! proteon-core — pure-Rust structural-bioinformatics compute.
//!
//! The molecular-mechanics / analysis half of proteon, with **no Python or
//! PyO3 dependency**: SASA, DSSP, H-bonds, hydrogen placement, missing-atom
//! reconstruction, force fields (CHARMM19+EEF1, AMBER96, OBC GB), energy
//! minimization / MD, and the `prepare` pipeline. Every entry point operates on
//! `pdbtbx::PDB` and plain Rust types.
//!
//! Two consumers share this crate so they cannot drift:
//! - `proteon-connector` — the PyO3 bindings (thin `py_*` shims over these).
//! - `proteon-bin` — the `proteon` CLI, which links this directly and never
//!   pulls PyO3 into a plain binary.

pub mod add_hydrogens;
pub mod altloc;
pub mod bond_order;
pub mod dssp;
pub mod forcefield;
pub mod fragment_templates;
pub mod hbond;
pub mod prepare;
pub mod reconstruct;
pub mod sasa;
pub mod surface;
