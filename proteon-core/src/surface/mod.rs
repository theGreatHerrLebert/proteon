//! Molecular surface (solvent-excluded surface) construction.
//!
//! Port of BALL's Connolly/Sanner pipeline (reduced surface → SES →
//! triangulation), built bottom-up and gated layer-by-layer against BALL as an
//! oracle (`ball-py`; see proteon `TO_SES_TRIANGULATION.md`).
//!
//! - `geom` (L0) — the geometry kernel: vectors, spheres, circles, planes, and
//!   the probe/contact predicates the reduced-surface algorithm rolls on. No
//!   oracle dependency — validated by closed-form unit tests.
//! - `rs` (L1) — the reduced surface: atom triples carrying a non-intersecting
//!   probe (faces), the pairs they share (edges), and surface atoms (vertices).
//!   Gated against `ball-py reduced_surface_stats`.

pub mod geom;
pub mod rs;
