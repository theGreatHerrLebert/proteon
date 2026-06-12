// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/atom_constants.h (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! AutoDock (AD) atom typing used in PDBQT input.
//!
//! PDBQT files carry an AD type per atom in their trailing columns
//! (e.g. `C`, `A`, `N`, `NA`, `OA`, `HD`, `Cl`, `SA`). The AD type
//! determines covalent radius for bond inference, element, and is an
//! input to XS-type assignment (together with bond topology).
//!
//! This module matches the upstream `AD_TYPE_*` constants 1:1.

/// Number of AD atom types (matches C++ `AD_TYPE_SIZE`).
pub const NUM_AD_TYPES: usize = 31;

/// AutoDock atom type.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AdType {
    /// Aliphatic / non-aromatic carbon.
    C = 0,
    /// Aromatic carbon.
    A = 1,
    /// Nitrogen (neither donor nor acceptor).
    N = 2,
    /// Oxygen (neither donor nor acceptor — rare).
    O = 3,
    /// Phosphorus.
    P = 4,
    /// Sulfur (non-acceptor).
    S = 5,
    /// Non-polar hydrogen.
    H = 6,
    /// Fluorine.
    F = 7,
    /// Iodine.
    I = 8,
    /// Nitrogen acceptor.
    Na = 9,
    /// Oxygen acceptor.
    Oa = 10,
    /// Sulfur acceptor.
    Sa = 11,
    /// Polar (donor-capable) hydrogen.
    Hd = 12,
    Mg = 13,
    Mn = 14,
    Zn = 15,
    Ca = 16,
    Fe = 17,
    Cl = 18,
    Br = 19,
    Si = 20,
    At = 21,
    /// Glue atoms (macrocycle closure). G0 ↔ CG0 pairs are closure bonds.
    G0 = 22,
    G1 = 23,
    G2 = 24,
    G3 = 25,
    /// Closure carbons partnered with G0-G3.
    Cg0 = 26,
    Cg1 = 27,
    Cg2 = 28,
    Cg3 = 29,
    /// Hydrated-docking pseudo water.
    W = 30,
}

impl AdType {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Element category (mirrors upstream `EL_TYPE_*`). Used to drive the
/// switch in `assign_xs_types`; not otherwise exposed.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Element {
    H = 0,
    C = 1,
    N = 2,
    O = 3,
    S = 4,
    P = 5,
    F = 6,
    Cl = 7,
    Br = 8,
    I = 9,
    Si = 10,
    At = 11,
    Met = 12,
    Dummy = 13,
}

/// Maps an AD type to its element category. Matches
/// `ad_type_to_el_type` in upstream.
#[inline]
#[must_use]
pub const fn element(ad: AdType) -> Element {
    match ad {
        AdType::C | AdType::A | AdType::Cg0 | AdType::Cg1 | AdType::Cg2 | AdType::Cg3 => {
            Element::C
        }
        AdType::N | AdType::Na => Element::N,
        AdType::O | AdType::Oa => Element::O,
        AdType::P => Element::P,
        AdType::S | AdType::Sa => Element::S,
        AdType::H | AdType::Hd => Element::H,
        AdType::F => Element::F,
        AdType::I => Element::I,
        AdType::Mg | AdType::Mn | AdType::Zn | AdType::Ca | AdType::Fe => Element::Met,
        AdType::Cl => Element::Cl,
        AdType::Br => Element::Br,
        AdType::Si => Element::Si,
        AdType::At => Element::At,
        AdType::G0 | AdType::G1 | AdType::G2 | AdType::G3 | AdType::W => Element::Dummy,
    }
}

/// Covalent radius (Å) per AD type, from `atom_kind_data[].covalent_radius`
/// in upstream. Used for distance-based bond inference.
const AD_COVALENT_RADIUS: [f64; NUM_AD_TYPES] = [
    0.77, // C
    0.77, // A
    0.75, // N
    0.73, // O
    1.06, // P
    1.02, // S
    0.37, // H
    0.71, // F
    1.33, // I
    0.75, // Na (N-acceptor)
    0.73, // Oa
    1.02, // Sa
    0.37, // Hd
    1.30, // Mg
    1.39, // Mn
    1.31, // Zn
    1.74, // Ca
    1.25, // Fe
    0.99, // Cl
    1.14, // Br
    1.11, // Si
    1.44, // At
    0.77, // G0
    0.77, // G1
    0.77, // G2
    0.77, // G3
    0.77, // Cg0
    0.77, // Cg1
    0.77, // Cg2
    0.77, // Cg3
    0.00, // W
];

#[inline]
#[must_use]
pub fn covalent_radius(ad: AdType) -> f64 {
    AD_COVALENT_RADIUS[ad.index()]
}

/// Parse the AD-type token from the trailing column of a PDBQT atom
/// record (e.g. `"C"`, `"A"`, `"NA"`, `"HD"`, `"Cl"`, `"Mg"`). Case is
/// preserved — tokens are taken verbatim from the file.
///
/// Returns `None` for unknown tokens. Applies the upstream
/// `atom_equivalence_data` aliasing (currently only `"Se" -> S`).
#[must_use]
pub fn parse_ad_type(s: &str) -> Option<AdType> {
    let s = s.trim();
    // Equivalences first so aliases route to the canonical type.
    if s == "Se" {
        return Some(AdType::S);
    }
    Some(match s {
        "C" => AdType::C,
        "A" => AdType::A,
        "N" => AdType::N,
        "O" => AdType::O,
        "P" => AdType::P,
        "S" => AdType::S,
        "H" => AdType::H,
        "F" => AdType::F,
        "I" => AdType::I,
        "NA" => AdType::Na,
        "OA" => AdType::Oa,
        "SA" => AdType::Sa,
        "HD" => AdType::Hd,
        "Mg" => AdType::Mg,
        "Mn" => AdType::Mn,
        "Zn" => AdType::Zn,
        "Ca" => AdType::Ca,
        "Fe" => AdType::Fe,
        "Cl" => AdType::Cl,
        "Br" => AdType::Br,
        "Si" => AdType::Si,
        "At" => AdType::At,
        "G0" => AdType::G0,
        "G1" => AdType::G1,
        "G2" => AdType::G2,
        "G3" => AdType::G3,
        "CG0" => AdType::Cg0,
        "CG1" => AdType::Cg1,
        "CG2" => AdType::Cg2,
        "CG3" => AdType::Cg3,
        "W" => AdType::W,
        _ => return None,
    })
}

/// Polar (donor-capable) hydrogen.
#[inline]
#[must_use]
pub const fn is_polar_hydrogen(ad: AdType) -> bool {
    matches!(ad, AdType::Hd)
}

/// AD type is intrinsically an acceptor (N-acceptor or O-acceptor).
/// Note: SA is *not* in upstream's acceptor set for XS assignment —
/// the comment in `model::assign_types` notes "X-Score formulation
/// apparently ignores SA". Kept here for symmetry with upstream.
#[inline]
#[must_use]
pub const fn is_acceptor_ad(ad: AdType) -> bool {
    matches!(ad, AdType::Oa | AdType::Na)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_cpp() {
        assert_eq!(AdType::C.index(), 0);
        assert_eq!(AdType::A.index(), 1);
        assert_eq!(AdType::N.index(), 2);
        assert_eq!(AdType::Hd.index(), 12);
        assert_eq!(AdType::Cl.index(), 18);
        assert_eq!(AdType::G0.index(), 22);
        assert_eq!(AdType::Cg0.index(), 26);
        assert_eq!(AdType::W.index(), 30);
    }

    #[test]
    fn element_mapping_matches_cpp() {
        assert_eq!(element(AdType::C), Element::C);
        assert_eq!(element(AdType::A), Element::C);
        assert_eq!(element(AdType::Cg0), Element::C);
        assert_eq!(element(AdType::N), Element::N);
        assert_eq!(element(AdType::Na), Element::N);
        assert_eq!(element(AdType::Oa), Element::O);
        assert_eq!(element(AdType::Sa), Element::S);
        assert_eq!(element(AdType::Hd), Element::H);
        assert_eq!(element(AdType::Mg), Element::Met);
        assert_eq!(element(AdType::Fe), Element::Met);
        assert_eq!(element(AdType::G0), Element::Dummy);
        assert_eq!(element(AdType::W), Element::Dummy);
    }

    #[test]
    fn covalent_radii_spot_check() {
        assert_eq!(covalent_radius(AdType::C), 0.77);
        assert_eq!(covalent_radius(AdType::H), 0.37);
        assert_eq!(covalent_radius(AdType::Hd), 0.37);
        assert_eq!(covalent_radius(AdType::Oa), 0.73);
        assert_eq!(covalent_radius(AdType::Ca), 1.74);
        assert_eq!(covalent_radius(AdType::W), 0.0);
    }

    #[test]
    fn parse_ad_type_known_tokens() {
        assert_eq!(parse_ad_type("C"), Some(AdType::C));
        assert_eq!(parse_ad_type("A"), Some(AdType::A));
        assert_eq!(parse_ad_type("NA"), Some(AdType::Na));
        assert_eq!(parse_ad_type("HD"), Some(AdType::Hd));
        assert_eq!(parse_ad_type("Cl"), Some(AdType::Cl));
        assert_eq!(parse_ad_type("CG0"), Some(AdType::Cg0));
        // Whitespace-tolerant.
        assert_eq!(parse_ad_type(" OA "), Some(AdType::Oa));
    }

    #[test]
    fn parse_ad_type_selenium_aliases_to_sulfur() {
        assert_eq!(parse_ad_type("Se"), Some(AdType::S));
    }

    #[test]
    fn parse_ad_type_unknown_returns_none() {
        assert_eq!(parse_ad_type(""), None);
        assert_eq!(parse_ad_type("XX"), None);
        assert_eq!(parse_ad_type("c"), None); // case-sensitive
    }

    #[test]
    fn polar_hydrogen_and_acceptor_predicates() {
        assert!(is_polar_hydrogen(AdType::Hd));
        assert!(!is_polar_hydrogen(AdType::H));
        assert!(is_acceptor_ad(AdType::Oa));
        assert!(is_acceptor_ad(AdType::Na));
        assert!(!is_acceptor_ad(AdType::Sa));
        assert!(!is_acceptor_ad(AdType::O));
    }
}
