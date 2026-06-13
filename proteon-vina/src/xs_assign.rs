// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/model.cpp::assign_types
// (Apache-2.0). Upstream author: Oleg Trott, Scripps Research Institute.

//! XS-type assignment from AD type + bond topology.
//!
//! Upstream `model::assign_types` combines element, AD type, and
//! neighbor information to pick the X-Score type used by the Vina
//! scoring function. Two pieces of topology matter:
//!
//! * for carbons: whether the atom is bonded to any heteroatom
//!   (distinguishes hydrophobic `C_H` from polar `C_P`);
//! * for nitrogens and oxygens: whether the atom is bonded to any
//!   polar hydrogen (`HD`) — together with the AD-level acceptor
//!   flag (`NA`/`OA`), this selects from `{*_P, *_D, *_A, *_DA}`.
//!
//! Hydrogens and hydrated-docking waters carry no XS type (upstream
//! leaves `xs` at `XS_TYPE_SIZE`); we represent that as `None`.

use crate::ad_types::{element, AdType, Element};
use crate::atom_types::XsType;
use crate::bonds::{bonded_to_hd, bonded_to_heteroatom, BondGraph, BondInput};

/// Assign an XS type to each atom.
///
/// Returns `None` for atoms that have no XS type in the Vina scoring
/// function (hydrogens — both `H` and `HD` — and hydrated-docking
/// waters `W`). Callers are expected to filter these out before
/// handing coordinates to the scorer.
#[must_use]
pub fn assign_xs_types<A: BondInput>(atoms: &[A], graph: &BondGraph) -> Vec<Option<XsType>> {
    (0..atoms.len())
        .map(|i| classify_single(atoms, graph, i))
        .collect()
}

fn classify_single<A: BondInput>(atoms: &[A], graph: &BondGraph, i: usize) -> Option<XsType> {
    let ad = atoms[i].ad_type();

    // Upstream sets `acceptor = (ad == OA || ad == NA)` and
    // `donor_NorO = (el == Met || bonded_to_HD)`. Note SA is
    // intentionally excluded — upstream comment: "X-Score
    // formulation apparently ignores SA".
    let acceptor = matches!(ad, AdType::Oa | AdType::Na);

    match element(ad) {
        Element::H => None, // H and HD: no XS type
        Element::C => {
            let polar = bonded_to_heteroatom(graph, atoms, i);
            Some(match ad {
                AdType::Cg0 => {
                    if polar {
                        XsType::CPCG0
                    } else {
                        XsType::CHCG0
                    }
                }
                AdType::Cg1 => {
                    if polar {
                        XsType::CPCG1
                    } else {
                        XsType::CHCG1
                    }
                }
                AdType::Cg2 => {
                    if polar {
                        XsType::CPCG2
                    } else {
                        XsType::CHCG2
                    }
                }
                AdType::Cg3 => {
                    if polar {
                        XsType::CPCG3
                    } else {
                        XsType::CHCG3
                    }
                }
                _ => {
                    if polar {
                        XsType::CP
                    } else {
                        XsType::CH
                    }
                }
            })
        }
        Element::N => {
            let donor = element(ad) == Element::Met || bonded_to_hd(graph, atoms, i);
            Some(match (acceptor, donor) {
                (true, true) => XsType::NDA,
                (true, false) => XsType::NA,
                (false, true) => XsType::ND,
                (false, false) => XsType::NP,
            })
        }
        Element::O => {
            let donor = bonded_to_hd(graph, atoms, i);
            // Element::Met can't appear here — Met atoms land in the
            // Met branch below. Mirror upstream's short-circuit.
            Some(match (acceptor, donor) {
                (true, true) => XsType::ODA,
                (true, false) => XsType::OA,
                (false, true) => XsType::OD,
                (false, false) => XsType::OP,
            })
        }
        Element::S => Some(XsType::SP),
        Element::P => Some(XsType::PP),
        Element::F => Some(XsType::FH),
        Element::Cl => Some(XsType::ClH),
        Element::Br => Some(XsType::BrH),
        Element::I => Some(XsType::IH),
        Element::Si => Some(XsType::Si),
        Element::At => Some(XsType::At),
        Element::Met => Some(XsType::MetD),
        Element::Dummy => match ad {
            AdType::G0 => Some(XsType::G0),
            AdType::G1 => Some(XsType::G1),
            AdType::G2 => Some(XsType::G2),
            AdType::G3 => Some(XsType::G3),
            AdType::W => None, // hydrated ligand waters carry no XS type
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::infer_bonds;

    struct T {
        ad: AdType,
        xyz: [f64; 3],
    }
    impl BondInput for T {
        fn ad_type(&self) -> AdType {
            self.ad
        }
        fn coords(&self) -> [f64; 3] {
            self.xyz
        }
    }
    fn a(ad: AdType, x: f64, y: f64, z: f64) -> T {
        T { ad, xyz: [x, y, z] }
    }

    /// Classify an atom directly, given its neighbours as a slice.
    /// Builds a bond graph via `infer_bonds` and returns the XS type
    /// for atom index 0 (the first entry in `atoms`).
    fn classify(atoms: &[T]) -> Option<XsType> {
        let g = infer_bonds(atoms);
        assign_xs_types(atoms, &g)[0]
    }

    // --- hydrogens and waters have no XS type ----------------------------

    #[test]
    fn hydrogens_and_waters_have_no_xs_type() {
        let atoms = vec![
            a(AdType::H, 0.0, 0.0, 0.0),
            a(AdType::Hd, 2.0, 0.0, 0.0),
            a(AdType::W, 4.0, 0.0, 0.0),
        ];
        let g = infer_bonds(&atoms);
        let xs = assign_xs_types(&atoms, &g);
        assert!(xs.iter().all(|x| x.is_none()));
    }

    // --- carbon ----------------------------------------------------------

    #[test]
    fn carbon_without_heteroatom_neighbors_is_hydrophobic() {
        // C-C-H-H → central C sees only C + H: no heteroatom.
        let atoms = vec![
            a(AdType::C, 0.0, 0.0, 0.0),
            a(AdType::C, 1.54, 0.0, 0.0),
            a(AdType::H, -0.5, 0.9, 0.0),
            a(AdType::H, -0.5, -0.9, 0.0),
        ];
        assert_eq!(classify(&atoms), Some(XsType::CH));
    }

    #[test]
    fn carbon_bonded_to_oxygen_is_polar() {
        let atoms = vec![
            a(AdType::C, 0.0, 0.0, 0.0),
            a(AdType::Oa, 1.4, 0.0, 0.0),
            a(AdType::H, -0.5, 0.9, 0.0),
        ];
        assert_eq!(classify(&atoms), Some(XsType::CP));
    }

    #[test]
    fn aromatic_carbon_is_not_intrinsically_polar() {
        // AD "A" (aromatic C) surrounded by other carbons → C_H.
        let atoms = vec![
            a(AdType::A, 0.0, 0.0, 0.0),
            a(AdType::A, 1.4, 0.0, 0.0),
            a(AdType::A, -1.4, 0.0, 0.0),
        ];
        assert_eq!(classify(&atoms), Some(XsType::CH));
    }

    #[test]
    fn cg0_carbon_routes_to_matching_macrocycle_variant() {
        // CG0 with only C neighbors → C_H_CG0.
        let atoms = vec![a(AdType::Cg0, 0.0, 0.0, 0.0), a(AdType::C, 1.54, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::CHCG0));

        // CG0 with an O neighbor → C_P_CG0.
        let atoms2 = vec![a(AdType::Cg0, 0.0, 0.0, 0.0), a(AdType::Oa, 1.4, 0.0, 0.0)];
        assert_eq!(classify(&atoms2), Some(XsType::CPCG0));
    }

    // --- nitrogen --------------------------------------------------------

    #[test]
    fn nitrogen_na_with_hd_neighbor_is_n_da() {
        let atoms = vec![a(AdType::Na, 0.0, 0.0, 0.0), a(AdType::Hd, 1.0, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::NDA));
    }

    #[test]
    fn nitrogen_na_without_hd_is_n_a() {
        let atoms = vec![a(AdType::Na, 0.0, 0.0, 0.0), a(AdType::C, 1.47, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::NA));
    }

    #[test]
    fn nitrogen_n_with_hd_is_n_d() {
        let atoms = vec![a(AdType::N, 0.0, 0.0, 0.0), a(AdType::Hd, 1.0, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::ND));
    }

    #[test]
    fn nitrogen_n_without_hd_is_n_p() {
        let atoms = vec![a(AdType::N, 0.0, 0.0, 0.0), a(AdType::C, 1.47, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::NP));
    }

    // --- oxygen ----------------------------------------------------------

    #[test]
    fn oxygen_oa_with_hd_is_o_da() {
        // Carboxylic OH oxygen: OA bonded to HD.
        let atoms = vec![a(AdType::Oa, 0.0, 0.0, 0.0), a(AdType::Hd, 0.96, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::ODA));
    }

    #[test]
    fn oxygen_oa_without_hd_is_o_a() {
        // Carbonyl oxygen: OA bonded to C.
        let atoms = vec![a(AdType::Oa, 0.0, 0.0, 0.0), a(AdType::C, 1.23, 0.0, 0.0)];
        assert_eq!(classify(&atoms), Some(XsType::OA));
    }

    // --- other elements --------------------------------------------------

    #[test]
    fn sulfur_is_s_p_regardless_of_ad_subtype() {
        let s = vec![a(AdType::S, 0.0, 0.0, 0.0), a(AdType::C, 1.8, 0.0, 0.0)];
        assert_eq!(classify(&s), Some(XsType::SP));
        let sa = vec![a(AdType::Sa, 0.0, 0.0, 0.0), a(AdType::C, 1.8, 0.0, 0.0)];
        // "X-Score formulation apparently ignores SA" — still S_P.
        assert_eq!(classify(&sa), Some(XsType::SP));
    }

    #[test]
    fn halogens_map_to_halogen_xs_types() {
        assert_eq!(
            classify(&[a(AdType::F, 0.0, 0.0, 0.0), a(AdType::C, 1.35, 0.0, 0.0)]),
            Some(XsType::FH)
        );
        assert_eq!(
            classify(&[a(AdType::Cl, 0.0, 0.0, 0.0), a(AdType::C, 1.77, 0.0, 0.0)]),
            Some(XsType::ClH)
        );
        assert_eq!(
            classify(&[a(AdType::Br, 0.0, 0.0, 0.0), a(AdType::C, 1.94, 0.0, 0.0)]),
            Some(XsType::BrH)
        );
        assert_eq!(
            classify(&[a(AdType::I, 0.0, 0.0, 0.0), a(AdType::C, 2.14, 0.0, 0.0)]),
            Some(XsType::IH)
        );
    }

    #[test]
    fn metals_map_to_met_d() {
        for m in [AdType::Mg, AdType::Mn, AdType::Zn, AdType::Ca, AdType::Fe] {
            let atoms = vec![a(m, 0.0, 0.0, 0.0), a(AdType::Oa, 2.0, 0.0, 0.0)];
            assert_eq!(classify(&atoms), Some(XsType::MetD));
        }
    }

    #[test]
    fn glue_atoms_map_to_matching_g_types() {
        for (ad, expected) in [
            (AdType::G0, XsType::G0),
            (AdType::G1, XsType::G1),
            (AdType::G2, XsType::G2),
            (AdType::G3, XsType::G3),
        ] {
            let atoms = vec![a(ad, 0.0, 0.0, 0.0)];
            assert_eq!(classify(&atoms), Some(expected));
        }
    }

    // --- integration on a real ligand ------------------------------------

    #[test]
    fn real_ligand_produces_sensible_xs_histogram() {
        use crate::pdbqt::parse_pdbqt_atoms;
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        let atoms = parse_pdbqt_atoms(LIGAND_FIXTURE).unwrap();
        let g = infer_bonds(&atoms);
        let xs = assign_xs_types(&atoms, &g);

        // All heavy atoms (non-H) should have an XS type assigned.
        for (i, (atom, tag)) in atoms.iter().zip(xs.iter()).enumerate() {
            if !matches!(atom.ad_type, AdType::H | AdType::Hd | AdType::W) {
                assert!(
                    tag.is_some(),
                    "heavy atom {i} ({:?}) missing XS type",
                    atom.ad_type
                );
            }
        }
        // Hydrogens should be None.
        for (atom, tag) in atoms.iter().zip(xs.iter()) {
            if matches!(atom.ad_type, AdType::H | AdType::Hd | AdType::W) {
                assert!(tag.is_none());
            }
        }
        // Imatinib has aromatic rings and an amide — expect both C_H
        // (aromatic-backbone carbons) and C_P (amide C / pyridine-
        // adjacent C) present.
        let has_ch = xs.iter().flatten().any(|&x| x == XsType::CH);
        let has_cp = xs.iter().flatten().any(|&x| x == XsType::CP);
        assert!(has_ch && has_cp, "expected both C_H and C_P in imatinib");
        // And at least one acceptor nitrogen.
        assert!(
            xs.iter()
                .flatten()
                .any(|&x| matches!(x, XsType::NA | XsType::NDA)),
            "expected at least one N_A/N_DA in imatinib"
        );
    }
}
