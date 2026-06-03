//! Molecular surface (solvent-excluded surface) construction.
//!
//! Port of BALL's Connolly/Sanner pipeline (reduced surface → SES →
//! triangulation), built bottom-up and gated layer-by-layer against BALL as an
//! oracle (`ball-py`; see proteon `TO_SES_TRIANGULATION.md`).
//!
//! - `geom` (L0) — the geometry kernel: vectors, spheres, circles, planes, and
//!   the probe/contact predicates the reduced-surface algorithm rolls on. No
//!   oracle dependency — validated by closed-form unit tests.

pub mod geom;
