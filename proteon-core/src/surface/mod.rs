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
//! - `ses` (L2) — the solvent-excluded-surface element graph: RS vertices/edges/
//!   faces → contact/toric/spheric SES faces + ownership. Gated against
//!   `ball-py ses_graph` (non-singular corpus; the L3 singular cleaner follows).
//! - `mesh` (L4 foundation) — index triangle mesh + invariants (area,
//!   watertightness, signed volume, Euler χ) and the `icosphere` contact-face
//!   primitive. An isolated atom's SES is its sphere, gated against `ses_area`.
//! - `patches` (L4) — reentrant-face meshing: the spheric (concave probe-cap)
//!   face, gated on its closed-form area (`radius² × spherical excess`). Toric
//!   faces + watertight stitching follow.

pub mod geom;
pub mod mesh;
pub mod patches;
pub mod rs;
pub mod ses;
