//! Shared-boundary registry — the backbone of watertight SES assembly.
//!
//! Per `TO_SES_STITCHING.md`: the only robust way to stitch the analytic patches
//! (contact caps, toric faces, spheric faces) into a closed mesh is to discretize
//! every shared boundary **once** and have both adjacent patches index the *same*
//! vertices. The registry is that single source of truth. Keys are **topological
//! identities** (which RS element creates the feature), never coordinates — so
//! two patches that meet on a curve resolve to bit-identical indices even when
//! their coordinates are only numerically close (codex-review: coordinate welding
//! would merge distinct near-degenerate features and crack at triple points).
//!
//! Two kinds of shared feature:
//! - **SES vertices** — the probe-contact corner points where a probe fixed on an
//!   RS face touches one of its atoms; shared by the contact, toric and spheric
//!   patches meeting there. Interned by [`Registry::vertex`].
//! - **boundary curves** — a contact-circle arc (atom ∩ toric) or a concave arc
//!   (spheric edge ∩ toric); sampled once into an ordered vertex list by
//!   [`Registry::curve_between`]. A curve is identified by its endpoint SES
//!   vertices (so a circle split into several arcs never welds) and each patch
//!   passes its own endpoint order to get the chain wound its way — orientation
//!   is data-derived, not a trusted flag.

use super::geom::Vec3;
use std::collections::HashMap;

/// A topological identity for a shared vertex or curve. Built via the semantic
/// constructors so callers never collide two different features.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct Key {
    kind: u8,
    a: u32,
    b: u32,
    c: u32,
}

impl Key {
    /// The probe-contact corner where the probe of `rs_face` touches `atom`.
    pub fn ses_vertex(rs_face: u32, atom: u32) -> Self {
        Key {
            kind: 0,
            a: rs_face,
            b: atom,
            c: 0,
        }
    }
    /// An arc of `atom`'s contact circle toward neighbour across RS `edge`.
    pub fn contact_arc(atom: u32, edge: u32) -> Self {
        Key {
            kind: 1,
            a: atom,
            b: edge,
            c: 0,
        }
    }
    /// The concave (spheric-triangle) arc of `rs_face` between its two atoms
    /// `p`/`q` — order-independent so both atoms' bookkeeping agree.
    pub fn concave_arc(rs_face: u32, p: u32, q: u32) -> Self {
        Key {
            kind: 2,
            a: rs_face,
            b: p.min(q),
            c: p.max(q),
        }
    }
}

/// A curve's storage identity: a semantic discriminator **plus its canonical
/// endpoint pair**. Folding the endpoints in (codex-review) means a contact
/// circle broken into several arcs by intervening SES vertices gets a distinct
/// identity per arc — it can never weld two arcs — and direction is derivable
/// from the endpoints rather than trusted from a bool.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct CurveId {
    disc: Key,
    lo: u32,
    hi: u32,
}

struct Curve {
    /// vertex chain in canonical `lo → … → hi` order.
    chain: Vec<u32>,
}

/// The single source of truth for shared vertices and sampled boundary curves.
#[derive(Default)]
pub struct Registry {
    /// All shared 3D vertices, in allocation order. Patch meshes index into this.
    pub verts: Vec<Vec3>,
    vertex_of: HashMap<Key, u32>,
    curve_of: HashMap<CurveId, Curve>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a single shared SES-corner vertex by topological `key`. The first
    /// call fixes its position and index; later calls with the same key return
    /// that index (the key, not the coordinate, is identity). In debug builds the
    /// repeat position must agree with the first — a large disagreement means two
    /// patches computed the corner differently, which is a wiring bug, not a
    /// rounding one (codex-review).
    pub fn vertex(&mut self, key: Key, pos: Vec3) -> u32 {
        debug_assert_eq!(key.kind, 0, "vertex() needs an SES-vertex key");
        if let Some(&v) = self.vertex_of.get(&key) {
            debug_assert!(
                self.verts[v as usize].distance(pos) < 1e-6,
                "SES vertex {key:?} re-interned at a divergent position"
            );
            return v;
        }
        let v = self.verts.len() as u32;
        self.verts.push(pos);
        self.vertex_of.insert(key, v);
        v
    }

    /// Sample/lookup the boundary curve between already-interned endpoints `a` and
    /// `b`, under semantic discriminator `disc`. Returns the shared vertex chain
    /// ordered `a → … → b`; `interior` are the fresh samples between them in that
    /// same direction (used only on the *first* call — later patches reuse the
    /// chain). **Orientation is derived from `a`/`b`**, not a caller-supplied bool,
    /// so the patch on each side passes its own endpoint order and gets the chain
    /// wound its way over the exact same vertices.
    pub fn curve_between(&mut self, disc: Key, a: u32, b: u32, interior: &[Vec3]) -> Vec<u32> {
        debug_assert_ne!(disc.kind, 0, "curve discriminator must not be a vertex key");
        assert_ne!(a, b, "degenerate curve (a == b)");
        let n = self.verts.len() as u32;
        assert!(a < n && b < n, "curve endpoints must be interned vertices");
        let (lo, hi) = (a.min(b), a.max(b));
        let id = CurveId { disc, lo, hi };
        if let Some(c) = self.curve_of.get(&id) {
            debug_assert_eq!(
                interior.len(),
                c.chain.len().saturating_sub(2),
                "curve {disc:?} reused with a different sample count"
            );
            // Hand back in the requested a→b direction.
            return if a == lo {
                c.chain.clone()
            } else {
                c.chain.iter().rev().copied().collect()
            };
        }
        // First definition: build a→b, then store canonically as lo→hi.
        let mut ab = Vec::with_capacity(interior.len() + 2);
        ab.push(a);
        for &p in interior {
            ab.push(self.verts.len() as u32);
            self.verts.push(p);
        }
        ab.push(b);
        let chain = if a == lo {
            ab.clone()
        } else {
            ab.iter().rev().copied().collect()
        };
        self.curve_of.insert(id, Curve { chain });
        ab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64) -> Vec3 {
        Vec3::new(x, 0.0, 0.0)
    }

    #[test]
    fn vertex_interning_is_keyed_by_identity_not_coordinate() {
        let mut r = Registry::new();
        let a = r.vertex(Key::ses_vertex(3, 7), v(1.0));
        // same key, a second patch's numerically-divergent coordinate → same
        // index, the first position kept (the key is identity, not the coord).
        let b = r.vertex(Key::ses_vertex(3, 7), v(1.0 + 1e-9));
        assert_eq!(a, b);
        assert_eq!(r.verts[a as usize], v(1.0));
        // different key → different vertex.
        let c = r.vertex(Key::ses_vertex(3, 8), v(1.0));
        assert_ne!(a, c);
        assert_eq!(r.verts.len(), 2);
    }

    #[test]
    fn concave_arc_key_is_atom_order_independent() {
        assert_eq!(Key::concave_arc(5, 2, 9), Key::concave_arc(5, 9, 2));
        assert_ne!(Key::concave_arc(5, 2, 9), Key::concave_arc(6, 2, 9));
    }

    #[test]
    fn a_curve_is_sampled_once_and_shared_by_both_patches() {
        let mut r = Registry::new();
        let s = r.vertex(Key::ses_vertex(0, 0), v(0.0));
        let e = r.vertex(Key::ses_vertex(0, 1), v(3.0));
        let disc = Key::concave_arc(0, 0, 1);
        let first = r.curve_between(disc, s, e, &[v(1.0), v(2.0)]);
        assert_eq!(first, vec![s, 2, 3, e]); // endpoints + 2 fresh interior
        let nverts = r.verts.len();

        // The patch on the other side walks the arc end→start. It gets the SAME
        // vertices in reversed order — orientation derived from the endpoints, no
        // bool — and allocates nothing. That is the anti-crack guarantee.
        let other = r.curve_between(disc, e, s, &[v(99.0), v(99.0)]);
        assert_eq!(other, vec![e, 3, 2, s]);
        assert_eq!(r.verts.len(), nverts);

        // Same direction again → identical to the first.
        assert_eq!(r.curve_between(disc, s, e, &[v(1.0), v(2.0)]), first);
    }

    #[test]
    fn a_split_circle_keeps_its_arcs_distinct() {
        // Three SES vertices on one atom's contact circle (same atom+edge disc):
        // the arcs A→B and B→C must be different curves, never welded.
        let mut r = Registry::new();
        let a = r.vertex(Key::ses_vertex(0, 5), v(0.0));
        let b = r.vertex(Key::ses_vertex(1, 5), v(1.0));
        let c = r.vertex(Key::ses_vertex(2, 5), v(2.0));
        let disc = Key::contact_arc(5, 0);
        let ab = r.curve_between(disc, a, b, &[v(0.5)]);
        let bc = r.curve_between(disc, b, c, &[v(1.5)]);
        assert_ne!(ab, bc);
        assert_eq!(ab.first(), Some(&a));
        assert_eq!(bc.first(), Some(&b));
    }
}
