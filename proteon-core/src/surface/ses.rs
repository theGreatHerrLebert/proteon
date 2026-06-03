//! L2 solvent-excluded surface (SES) element graph.
//!
//! The SES is built from the reduced surface by turning each RS element into a
//! reentrant/contact patch (Connolly):
//!
//! - RS **vertex** (surface atom)  → SES **contact** face (the convex exposed
//!   cap of the atom)
//! - RS **edge** (atom pair)       → SES **toric** face (the reentrant torus
//!   swept by the probe rolling between the two atoms)
//! - RS **face** (atom triple)     → SES **spheric** face (the concave probe cap
//!   where the probe rests on three atoms)
//!
//! This layer builds that element graph and its atom ownership — exactly what
//! `ball-py ses_graph` exposes (face-type counts + ownership + singular-edge
//! count). For non-singular inputs the mapping is 1:1 from the RS, so the gate
//! is direct. **Singular** configurations (probe tori that self-intersect) split
//! toric/spheric faces and introduce singular edges — that is the L3 cleaner,
//! tracked separately; this layer reports `n_singular_edges = 0` and is gated on
//! the non-singular corpus. The SES element *geometry* (vertex positions, edge
//! arcs) needed by the L4 triangulator is also a follow-up; here we establish
//! the combinatorial graph + ownership.

use super::rs::ReducedSurface;

/// Which kind of reentrant/contact patch an SES face is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceKind {
    Contact,
    Toric,
    Spheric,
}

/// One SES face, tagged by kind and owning the RS element (and thus the atoms)
/// it came from: a contact face owns one atom, a toric face an atom pair, a
/// spheric face an atom triple.
#[derive(Clone, Debug)]
pub struct SesFace {
    pub kind: FaceKind,
    /// Owning atom indices (1 for contact, 2 for toric, 3 for spheric), sorted.
    pub atoms: Vec<usize>,
}

/// The SES element graph of a reduced surface.
#[derive(Clone, Debug)]
pub struct SolventExcludedSurface {
    pub faces: Vec<SesFace>,
    /// Singular edges (probe-probe self-intersections). Zero for non-singular
    /// inputs; populated by the L3 cleaner.
    pub n_singular_edges: usize,
}

impl SolventExcludedSurface {
    fn faces_of(&self, kind: FaceKind) -> impl Iterator<Item = &SesFace> {
        self.faces.iter().filter(move |f| f.kind == kind)
    }

    pub fn num_contact_faces(&self) -> usize {
        self.faces_of(FaceKind::Contact).count()
    }
    pub fn num_toric_faces(&self) -> usize {
        self.faces_of(FaceKind::Toric).count()
    }
    pub fn num_spheric_faces(&self) -> usize {
        self.faces_of(FaceKind::Spheric).count()
    }

    /// Owning atom of each contact face (matches `ses_graph` `contact_atoms`).
    pub fn contact_atoms(&self) -> Vec<usize> {
        self.faces_of(FaceKind::Contact)
            .map(|f| f.atoms[0])
            .collect()
    }
    /// Owning atom pair of each toric face (`ses_graph` `toric_atoms`).
    pub fn toric_atoms(&self) -> Vec<[usize; 2]> {
        self.faces_of(FaceKind::Toric)
            .map(|f| [f.atoms[0], f.atoms[1]])
            .collect()
    }
    /// Owning atom triple of each spheric face (`ses_graph` `spheric_atoms`).
    pub fn spheric_atoms(&self) -> Vec<[usize; 3]> {
        self.faces_of(FaceKind::Spheric)
            .map(|f| [f.atoms[0], f.atoms[1], f.atoms[2]])
            .collect()
    }
}

/// Build the SES element graph from a reduced surface (non-singular mapping).
pub fn compute(rs: &ReducedSurface) -> SolventExcludedSurface {
    let mut faces = Vec::with_capacity(rs.vertices.len() + rs.edges.len() + rs.faces.len());

    for &atom in &rs.vertices {
        faces.push(SesFace {
            kind: FaceKind::Contact,
            atoms: vec![atom],
        });
    }
    for &[i, j] in &rs.edges {
        faces.push(SesFace {
            kind: FaceKind::Toric,
            atoms: vec![i, j],
        });
    }
    for f in &rs.faces {
        faces.push(SesFace {
            kind: FaceKind::Spheric,
            atoms: f.atoms.to_vec(),
        });
    }

    SolventExcludedSurface {
        faces,
        n_singular_edges: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::geom::{Sphere, Vec3};
    use super::super::rs;
    use super::*;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// L2 oracle gate. SES face-type counts + atom ownership match `ball-py
    /// 0.1.0a6 ses_graph(spheres, probe_radius=1.4)` on the non-singular corpus.
    #[test]
    fn ses_element_graph_matches_ball_oracle() {
        // (config, contact, toric_pairs, spheric_triples) straight from ses_graph.
        let single = vec![sph(0.0, 0.0, 0.0, 1.5)];
        let pair = vec![sph(0.0, 0.0, 0.0, 1.8), sph(2.5, 0.0, 0.0, 1.8)];
        let chain = vec![
            sph(0.0, 0.0, 0.0, 1.5),
            sph(2.6, 0.0, 0.0, 1.5),
            sph(5.2, 0.0, 0.0, 1.5),
            sph(7.8, 0.0, 0.0, 1.5),
        ];
        let tri = vec![
            sph(0.0, 0.0, 0.0, 2.0),
            sph(2.5, 0.0, 0.0, 2.0),
            sph(1.25, 2.0, 0.0, 2.0),
        ];
        let tetra = vec![
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];

        type Case = (Vec<Sphere>, Vec<usize>, Vec<[usize; 2]>, Vec<[usize; 3]>);
        let cases: Vec<Case> = vec![
            (single, vec![0], vec![], vec![]),
            (pair, vec![0, 1], vec![[0, 1]], vec![]),
            (
                chain,
                vec![0, 1, 2, 3],
                vec![[0, 1], [1, 2], [2, 3]],
                vec![],
            ),
            (
                tri,
                vec![0, 1, 2],
                vec![[0, 1], [0, 2], [1, 2]],
                vec![[0, 1, 2], [0, 1, 2]],
            ),
            (
                tetra,
                vec![0, 1, 2, 3],
                vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            ),
        ];

        for (atoms, contact, toric, spheric) in cases {
            let ses = compute(&rs::compute(&atoms, 1.4));
            assert_eq!(ses.n_singular_edges, 0);
            // Counts.
            assert_eq!(ses.num_contact_faces(), contact.len());
            assert_eq!(ses.num_toric_faces(), toric.len());
            assert_eq!(ses.num_spheric_faces(), spheric.len());
            // Ownership (sorted multisets).
            let mut c = ses.contact_atoms();
            c.sort_unstable();
            assert_eq!(c, contact);
            let mut t = ses.toric_atoms();
            t.sort_unstable();
            assert_eq!(t, toric);
            let mut s = ses.spheric_atoms();
            s.sort_unstable();
            assert_eq!(s, spheric);
        }
    }
}
