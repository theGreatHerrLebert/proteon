// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/atom_constants.h (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! X-Score (XS) atom typing used by the Vina scoring function.
//!
//! There are 32 XS types. Values match the integer constants `XS_TYPE_*`
//! from the upstream C++ source so lookup tables port 1:1.

/// Number of XS atom types (matches C++ `XS_TYPE_SIZE`).
pub const NUM_XS_TYPES: usize = 32;

/// X-Score atom type.
///
/// Discriminants are the canonical XS indices; `as usize` indexes the
/// per-type lookup tables below.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum XsType {
    /// Hydrophobic carbon (bonded only to C/H).
    CH = 0,
    /// Polar carbon (bonded to a heteroatom).
    CP = 1,
    /// Polar nitrogen (neither donor nor acceptor).
    NP = 2,
    /// Nitrogen donor.
    ND = 3,
    /// Nitrogen acceptor.
    NA = 4,
    /// Nitrogen donor + acceptor.
    NDA = 5,
    /// Polar oxygen.
    OP = 6,
    /// Oxygen donor.
    OD = 7,
    /// Oxygen acceptor.
    OA = 8,
    /// Oxygen donor + acceptor.
    ODA = 9,
    /// Polar sulfur.
    SP = 10,
    /// Polar phosphorus.
    PP = 11,
    /// Hydrophobic fluorine.
    FH = 12,
    /// Hydrophobic chlorine.
    ClH = 13,
    /// Hydrophobic bromine.
    BrH = 14,
    /// Hydrophobic iodine.
    IH = 15,
    /// Silicon.
    Si = 16,
    /// Astatine.
    At = 17,
    /// Metal donor.
    MetD = 18,
    /// Macrocycle-closure carbon, CG0 partner, hydrophobic.
    CHCG0 = 19,
    /// Macrocycle-closure carbon, CG0 partner, polar.
    CPCG0 = 20,
    /// Macrocycle-closure glue atom, CG0.
    G0 = 21,
    CHCG1 = 22,
    CPCG1 = 23,
    G1 = 24,
    CHCG2 = 25,
    CPCG2 = 26,
    G2 = 27,
    CHCG3 = 28,
    CPCG3 = 29,
    G3 = 30,
    /// Water pseudo-atom (hydrated docking).
    W = 31,
}

impl XsType {
    /// Canonical XS index, matching the C++ `XS_TYPE_*` integer constants.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Vina vdW radii (Å), indexed by `XsType as usize`. From
/// `xs_vdw_radii[]` in upstream `atom_constants.h`.
const XS_VDW_RADII: [f64; NUM_XS_TYPES] = [
    1.9, // CH
    1.9, // CP
    1.8, // NP
    1.8, // ND
    1.8, // NA
    1.8, // NDA
    1.7, // OP
    1.7, // OD
    1.7, // OA
    1.7, // ODA
    2.0, // SP
    2.1, // PP
    1.5, // FH
    1.8, // ClH
    2.0, // BrH
    2.2, // IH
    2.2, // Si
    2.3, // At
    1.2, // MetD
    1.9, // CHCG0
    1.9, // CPCG0
    0.0, // G0
    1.9, // CHCG1
    1.9, // CPCG1
    0.0, // G1
    1.9, // CHCG2
    1.9, // CPCG2
    0.0, // G2
    1.9, // CHCG3
    1.9, // CPCG3
    0.0, // G3
    0.0, // W
];

/// Vina vdW radius of an XS atom type (Å).
#[inline]
#[must_use]
pub fn radius(t: XsType) -> f64 {
    XS_VDW_RADII[t.index()]
}

/// True for hydrophobic XS types (carbons bonded only to C/H, and halogens).
/// Mirrors `xs_is_hydrophobic`.
#[inline]
#[must_use]
pub const fn is_hydrophobic(t: XsType) -> bool {
    matches!(
        t,
        XsType::CH | XsType::FH | XsType::ClH | XsType::BrH | XsType::IH
    )
}

/// True for H-bond acceptor XS types. Mirrors `xs_is_acceptor`.
#[inline]
#[must_use]
pub const fn is_acceptor(t: XsType) -> bool {
    matches!(t, XsType::NA | XsType::NDA | XsType::OA | XsType::ODA)
}

/// True for H-bond donor XS types. Mirrors `xs_is_donor`.
#[inline]
#[must_use]
pub const fn is_donor(t: XsType) -> bool {
    matches!(
        t,
        XsType::ND | XsType::NDA | XsType::OD | XsType::ODA | XsType::MetD
    )
}

/// True when `(t1, t2)` is a donor / acceptor pair in either direction.
/// Mirrors `xs_h_bond_possible`.
#[inline]
#[must_use]
pub const fn h_bond_possible(t1: XsType, t2: XsType) -> bool {
    (is_donor(t1) && is_acceptor(t2)) || (is_donor(t2) && is_acceptor(t1))
}

/// Glue atom types used for macrocycle closure. Their pairwise potentials
/// are handled as a special case (see `potentials.rs::optimal_distance`).
#[inline]
#[must_use]
pub const fn is_glue(t: XsType) -> bool {
    matches!(t, XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_cpp() {
        // Spot-check a handful of canonical indices from atom_constants.h.
        assert_eq!(XsType::CH.index(), 0);
        assert_eq!(XsType::NDA.index(), 5);
        assert_eq!(XsType::OA.index(), 8);
        assert_eq!(XsType::SP.index(), 10);
        assert_eq!(XsType::MetD.index(), 18);
        assert_eq!(XsType::G0.index(), 21);
        assert_eq!(XsType::G3.index(), 30);
        assert_eq!(XsType::W.index(), 31);
    }

    #[test]
    fn radii_match_cpp_table() {
        assert!((radius(XsType::CH) - 1.9).abs() < 1e-12);
        assert!((radius(XsType::FH) - 1.5).abs() < 1e-12);
        assert!((radius(XsType::IH) - 2.2).abs() < 1e-12);
        assert!((radius(XsType::At) - 2.3).abs() < 1e-12);
        assert!((radius(XsType::MetD) - 1.2).abs() < 1e-12);
        // Glue atoms have zero radius.
        assert_eq!(radius(XsType::G0), 0.0);
        assert_eq!(radius(XsType::G3), 0.0);
        assert_eq!(radius(XsType::W), 0.0);
    }

    #[test]
    fn hydrophobic_set() {
        for &t in &[XsType::CH, XsType::FH, XsType::ClH, XsType::BrH, XsType::IH] {
            assert!(is_hydrophobic(t), "{t:?} should be hydrophobic");
        }
        for &t in &[
            XsType::CP,
            XsType::NP,
            XsType::NDA,
            XsType::OA,
            XsType::SP,
            XsType::MetD,
        ] {
            assert!(!is_hydrophobic(t), "{t:?} should not be hydrophobic");
        }
    }

    #[test]
    fn donor_acceptor_sets() {
        // Acceptors
        for &t in &[XsType::NA, XsType::NDA, XsType::OA, XsType::ODA] {
            assert!(is_acceptor(t));
        }
        // Donors
        for &t in &[
            XsType::ND,
            XsType::NDA,
            XsType::OD,
            XsType::ODA,
            XsType::MetD,
        ] {
            assert!(is_donor(t));
        }
        // NP / OP are neither.
        assert!(!is_donor(XsType::NP) && !is_acceptor(XsType::NP));
        assert!(!is_donor(XsType::OP) && !is_acceptor(XsType::OP));
    }

    #[test]
    fn h_bond_possible_is_symmetric_and_requires_donor_plus_acceptor() {
        // Canonical yes: donor-N with acceptor-O.
        assert!(h_bond_possible(XsType::ND, XsType::OA));
        assert!(h_bond_possible(XsType::OA, XsType::ND));
        // NDA + anything acceptor-or-donor-compatible.
        assert!(h_bond_possible(XsType::NDA, XsType::OA));
        assert!(h_bond_possible(XsType::NDA, XsType::ND));
        // Two acceptors or two donors: no.
        assert!(!h_bond_possible(XsType::OA, XsType::OA));
        assert!(!h_bond_possible(XsType::ND, XsType::ND));
        // Hydrophobic pairs: no.
        assert!(!h_bond_possible(XsType::CH, XsType::CH));
    }

    #[test]
    fn glue_atoms_are_disjoint_from_real_types() {
        for &t in &[XsType::G0, XsType::G1, XsType::G2, XsType::G3] {
            assert!(is_glue(t));
            assert!(!is_hydrophobic(t));
            assert!(!is_donor(t));
            assert!(!is_acceptor(t));
            assert_eq!(radius(t), 0.0);
        }
    }
}
