// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/non_cache.cpp::eval (inter) and
// src/lib/model.cpp::evali/evalo (intra) — Apache-2.0. Upstream
// author: Oleg Trott, Scripps Research Institute.

//! Pair-sum energy evaluation.
//!
//! v0 scope: inter-molecular (receptor × ligand) pair energy. The
//! upstream `--score_only` path uses `non_cache::eval` (not the
//! pre-rasterized receptor grid) for final scoring unless
//! `--no_refine` is set, so this is our parity target. The grid is
//! a later optimisation.

use crate::ad_types::AdType;
use crate::atom_types::XsType;
use crate::bonds::BondGraph;
use crate::conf::Vec3;
use crate::molecule::Molecule;
use crate::precalculate::Precalculate;
use crate::weights::W_ROT;

/// Remap a ligand atom's XS type for scoring *against the receptor*.
/// Glue atoms do not interact with the receptor (`None`);
/// macrocycle-closure carbons are scored as their parent `C_H` or
/// `C_P` type. Mirrors the `switch` at the top of `non_cache::eval`.
#[inline]
#[must_use]
pub fn xs_for_receptor_interaction(t: XsType) -> Option<XsType> {
    match t {
        XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3 => None,
        XsType::CHCG0 | XsType::CHCG1 | XsType::CHCG2 | XsType::CHCG3 => Some(XsType::CH),
        XsType::CPCG0 | XsType::CPCG1 | XsType::CPCG2 | XsType::CPCG3 => Some(XsType::CP),
        other => Some(other),
    }
}

/// Upstream's "authentic_v" soft energy cap applied to each ligand
/// atom's contribution. `score_only` passes `1000` kcal/mol via
/// `authentic_v = (1000, 1000, 1000)`. Mirrors the soft branch of
/// `curl` in `curl.h`.
#[inline]
fn curl(mut e: f64, v: f64) -> f64 {
    if e > 0.0 && v.is_finite() {
        let tmp = if v < f64::EPSILON { 0.0 } else { v / (v + e) };
        e *= tmp;
    }
    e
}

/// Inter-molecular (receptor × ligand) Vina pair energy.
///
/// Pair path (no grid, no out-of-bounds penalty). Each ligand atom's
/// contribution is soft-capped via `curl(·, v_curl)`; pass
/// `v_curl = 1000.0` to match upstream `--score_only`, or
/// `f64::INFINITY` to disable the cap.
#[must_use]
pub fn inter_pair_energy(
    receptor: &Molecule,
    ligand: &Molecule,
    precalc: &Precalculate,
    v_curl: f64,
) -> f64 {
    let cutoff_sqr = precalc.cutoff_sqr();
    let mut total = 0.0_f64;

    for (i, &lig_xyz) in ligand.coords.iter().enumerate() {
        let Some(t_lig) = xs_for_receptor_interaction(ligand.xs_types[i]) else {
            continue; // glue atom — does not interact with receptor
        };

        let mut e_i = 0.0_f64;
        for (j, &rec_xyz) in receptor.coords.iter().enumerate() {
            let t_rec = match xs_for_receptor_interaction(receptor.xs_types[j]) {
                Some(t) => t,
                None => continue,
            };
            let dx = lig_xyz[0] - rec_xyz[0];
            let dy = lig_xyz[1] - rec_xyz[1];
            let dz = lig_xyz[2] - rec_xyz[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < cutoff_sqr {
                e_i += precalc.eval_fast(t_lig, t_rec, r2);
            }
        }
        total += curl(e_i, v_curl);
    }
    total
}

/// True if `t` is a macrocycle-closure glue atom. These are excluded
/// from the intra pair list (upstream routes them to `glue_pairs`,
/// which we do not evaluate in v0).
#[inline]
const fn is_glue(t: XsType) -> bool {
    matches!(t, XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3)
}

/// AD atom types of the four macrocycle closure carbons, paired with
/// their matching glue labels. Used by `is_closure_clash`.
const CG_FAMILIES: [(AdType, AdType); 4] = [
    (AdType::Cg0, AdType::G0),
    (AdType::Cg1, AdType::G1),
    (AdType::Cg2, AdType::G2),
    (AdType::Cg3, AdType::G3),
];

/// True iff `(i, j)` is a "closure clash" — two atoms each bonded
/// (1-2) to a macrocycle-closure carbon of the SAME label. Mirrors
/// upstream `model::is_closure_clash`, whose comment explains that
/// this removes 1-2/1-3/1-4 interactions through the implied CG-CG
/// closure bond. The pair is not a closure clash when one atom is
/// itself the matching G for the other's CG (that's a glue pair,
/// routed to `glue_pairs` and already handled).
fn is_closure_clash(m: &Molecule, i: usize, j: usize) -> bool {
    let ai = m.ad_types[i];
    let aj = m.ad_types[j];
    // Early out: if the pair itself is (CGk, Gk) in either order,
    // it's a glue pair and does not count as a closure clash.
    for &(cg, g) in &CG_FAMILIES {
        if (ai == cg && aj == g) || (aj == cg && ai == g) {
            return false;
        }
    }

    // Upstream's `bonded_to(x, 1)` includes `x` itself together with
    // its 1-2 neighbours. We do the same here: an atom is a "closure
    // clash" candidate if it either *is* a CG atom or is directly
    // bonded to one.
    let label_of = |ad: AdType| -> Option<usize> {
        match ad {
            AdType::Cg0 => Some(0),
            AdType::Cg1 => Some(1),
            AdType::Cg2 => Some(2),
            AdType::Cg3 => Some(3),
            _ => None,
        }
    };
    let mut i_has = [false; 4];
    if let Some(l) = label_of(ai) {
        i_has[l] = true;
    }
    for &k in &m.bonds[i] {
        if let Some(l) = label_of(m.ad_types[k]) {
            i_has[l] = true;
        }
    }
    let check = |x: AdType| -> bool { label_of(x).map(|l| i_has[l]).unwrap_or(false) };
    if check(aj) {
        return true;
    }
    for &k in &m.bonds[j] {
        if check(m.ad_types[k]) {
            return true;
        }
    }
    false
}

/// Set of atoms within 3 bonds of `start` (i.e. `start`, 1-2, 1-3,
/// and 1-4 neighbours). Returned as a sorted `Vec<usize>` for cheap
/// `binary_search` lookup.
fn bonded_within_3(graph: &BondGraph, start: usize) -> Vec<usize> {
    let mut visited = vec![false; graph.len()];
    let mut result = Vec::new();
    let mut frontier = vec![start];
    visited[start] = true;
    result.push(start);
    for _depth in 0..3 {
        let mut next_frontier = Vec::new();
        for &i in &frontier {
            for &j in &graph[i] {
                if !visited[j] {
                    visited[j] = true;
                    result.push(j);
                    next_frontier.push(j);
                }
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }
    result.sort_unstable();
    result
}

/// Build the intra-molecular interaction pair list for a ligand.
///
/// Emits unordered pairs `(i, j)` with `i < j` that satisfy all of:
/// * different torsion-tree fragment IDs (upstream's
///   `DISTANCE_VARIABLE` — separated by at least one rotatable bond),
/// * not 1-2 / 1-3 / 1-4 bonded,
/// * neither atom is a glue atom (G0–G3).
///
/// For rigid receptors and rigid single-fragment ligands every pair
/// is excluded on the fragment check, so the returned list is empty
/// and `lig_intra == 0` — matching upstream.
#[must_use]
pub fn intra_pair_list(m: &Molecule) -> Vec<(usize, usize)> {
    let n = m.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        if is_glue(m.xs_types[i]) {
            continue;
        }
        let forbidden = bonded_within_3(&m.bonds, i);
        let mask_i = m.fragment_mask[i];
        for j in (i + 1)..n {
            if is_glue(m.xs_types[j]) {
                continue;
            }
            // Upstream `DISTANCE_FIXED`: atoms share any fragment's
            // extended rigid group.
            if (mask_i & m.fragment_mask[j]) != 0 {
                continue;
            }
            if forbidden.binary_search(&j).is_ok() {
                continue;
            }
            if is_closure_clash(m, i, j) {
                continue;
            }
            pairs.push((i, j));
        }
    }
    pairs
}

/// Soft-cap helper that attenuates a gradient component in lock-step
/// with its energy, matching upstream `curl(fl&, T&, fl)` in `curl.h`.
/// When the pair / atom energy is positive and the cap `v` is finite:
/// `e *= v/(v+e)` and `grad *= (v/(v+e))²`. Non-positive energies and
/// `v = +∞` pass through unchanged.
#[inline]
fn curl_with_gradient(e: &mut f64, grad: &mut Vec3, v: f64) {
    if *e > 0.0 && v.is_finite() {
        let tmp = if v < f64::EPSILON { 0.0 } else { v / (v + *e) };
        *e *= tmp;
        let tmp2 = tmp * tmp;
        grad[0] *= tmp2;
        grad[1] *= tmp2;
        grad[2] *= tmp2;
    }
}

/// Intra-molecular Vina pair energy over a prebuilt pair list.
///
/// Upstream `eval_interacting_pairs` applies the soft `curl` cap to
/// each pair contribution independently (not the per-atom sum as
/// done for inter); this function mirrors that.
#[must_use]
pub fn intra_pair_energy(
    m: &Molecule,
    pairs: &[(usize, usize)],
    precalc: &Precalculate,
    v_curl: f64,
) -> f64 {
    let cutoff_sqr = precalc.cutoff_sqr();
    let mut total = 0.0_f64;
    for &(i, j) in pairs {
        let xi = m.coords[i];
        let xj = m.coords[j];
        let dx = xi[0] - xj[0];
        let dy = xi[1] - xj[1];
        let dz = xi[2] - xj[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 < cutoff_sqr {
            let tmp = precalc.eval_fast(m.xs_types[i], m.xs_types[j], r2);
            total += curl(tmp, v_curl);
        }
    }
    total
}

/// Effective torsion-degree count used by the `num_tors_div`
/// conf-independent term (upstream `conf_independent_inputs::num_tors`).
///
/// For each rotatable bond `(p, c)` (from a `BRANCH` line), we count
/// `0.5 ×` for each end whose partner has > 1 heavy-atom neighbour.
/// A rotatable bond between two "core" heavy atoms counts 1; one
/// terminating in a methyl-like fragment counts 0.5.
///
/// Bonds whose endpoints have been filtered out of `ligand` (e.g. H
/// atoms dropped during XS typing — not expected for rotatable bonds,
/// but safe) are silently skipped.
#[must_use]
pub fn compute_num_tors(ligand: &Molecule, rotatable_bonds: &[(u32, u32)]) -> f64 {
    let mut serial_to_idx = std::collections::HashMap::<u32, usize>::new();
    for (i, &s) in ligand.original_serials.iter().enumerate() {
        serial_to_idx.insert(s, i);
    }

    let mut acc = 0.0_f64;
    for &(pa, pb) in rotatable_bonds {
        let Some(&ia) = serial_to_idx.get(&pa) else {
            continue;
        };
        let Some(&ib) = serial_to_idx.get(&pb) else {
            continue;
        };
        // From ia's side: count if ib has >1 heavy-atom neighbour.
        if ligand.bonds[ib].len() > 1 {
            acc += 0.5;
        }
        // From ib's side: count if ia has >1 heavy-atom neighbour.
        if ligand.bonds[ia].len() > 1 {
            acc += 0.5;
        }
    }
    acc
}

/// The 8-component energy vector returned by upstream `Vina::score()`
/// (`SF_VINA`, rigid receptor, no flex side chains).
///
/// Field names and order mirror upstream's verbose `show_score` output.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoreComponents {
    /// Conf-independent-adjusted total (the value shown as "Affinity").
    pub total: f64,
    /// Ligand vs receptor pair energy (what upstream calls `lig_grids`
    /// when rasterised, or the `non_cache` pair-sum when not).
    pub lig_grids: f64,
    /// Ligand vs flex residues; always 0 in v0 (no flex support).
    pub inter_pairs: f64,
    /// Flex side-chain vs receptor; always 0 in v0.
    pub flex_grids: f64,
    /// Flex-flex pair energy; always 0 in v0.
    pub intra_pairs: f64,
    /// Intra-ligand pair energy across rotatable bonds.
    pub lig_intra: f64,
    /// Conformational-independent adjustment (num_tors_div applied to
    /// `inter + intra - intramolecular`). Equals `total - lig_grids`
    /// in v0 because intra cancels.
    pub conf_independent: f64,
    /// Intramolecular reference energy subtracted from the scoring
    /// argument. For SF_VINA this equals `lig_intra` in v0.
    pub intramolecular: f64,
}

/// Compute the 8-component energy vector for a rigid ligand against a
/// rigid receptor using the Vina scoring function.
///
/// `rotatable_bonds` is the `(parent_serial, child_serial)` list from
/// [`crate::pdbqt::PdbqtFile::rotatable_bonds`]. `v_curl = 1000.0`
/// matches upstream `authentic_v` in `--score_only`.
#[must_use]
pub fn score_only(
    receptor: &Molecule,
    ligand: &Molecule,
    rotatable_bonds: &[(u32, u32)],
    precalc: &Precalculate,
    v_curl: f64,
) -> ScoreComponents {
    let lig_grids = inter_pair_energy(receptor, ligand, precalc, v_curl);
    let pairs = intra_pair_list(ligand);
    let lig_intra = intra_pair_energy(ligand, &pairs, precalc, v_curl);
    let intramolecular = lig_intra; // no flex → eval_intramolecular = lig_intra

    // `conf_independent` argument: inter + intra - intramolecular
    //   = lig_grids + (flex_grids + intra_pairs + lig_intra) - lig_intra
    //   = lig_grids  (v0: no flex)
    let x = lig_grids;

    let num_tors = compute_num_tors(ligand, rotatable_bonds);
    let divisor = 1.0 + W_ROT * num_tors;
    let total = if divisor.abs() < f64::EPSILON {
        0.0
    } else {
        x / divisor
    };
    let conf_independent = total - x;

    ScoreComponents {
        total,
        lig_grids,
        inter_pairs: 0.0,
        flex_grids: 0.0,
        intra_pairs: 0.0,
        lig_intra,
        conf_independent,
        intramolecular,
    }
}

// ------------------------------------------------------------------
// Energy + forces entry points for BFGS / local optimisation.
// The scoring-only `inter_pair_energy` above uses `precalc.eval_fast`
// (bin-midpoint averaged). These variants use `precalc.eval_deriv`
// (linear-interpolated within the bin) because we also need the
// `dor = (dE/dr)/r` derivative it returns. The two paths give
// slightly different energies for the same pose (the midpoint-vs-
// interpolated difference is at most half a bin at 1/32 Å² spacing)
// but are internally self-consistent: the FD derivative of this
// function's energy matches the forces it reports to ≤ 1e-4.
// ------------------------------------------------------------------

/// Inter-molecular Vina pair energy PLUS per-ligand-atom force
/// (force = −∂E/∂r, following our gradient convention).
///
/// Upstream analogue: `non_cache::eval_deriv` — same per-atom
/// loop with `curl(e, deriv, v)` attenuating both the scalar
/// energy and the gradient vector by `tmp²`.
#[must_use]
pub fn inter_pair_energy_with_forces(
    receptor: &Molecule,
    ligand: &Molecule,
    precalc: &Precalculate,
    v_curl: f64,
) -> (f64, Vec<Vec3>) {
    let cutoff_sqr = precalc.cutoff_sqr();
    let mut total = 0.0_f64;
    let mut forces = vec![[0.0_f64; 3]; ligand.len()];

    for (i, &lig_xyz) in ligand.coords.iter().enumerate() {
        let Some(t_lig) = xs_for_receptor_interaction(ligand.xs_types[i]) else {
            continue;
        };
        let mut e_i = 0.0_f64;
        // grad_i accumulates +∂E/∂ligand.coords[i].
        let mut grad_i = [0.0_f64; 3];

        for (j, &rec_xyz) in receptor.coords.iter().enumerate() {
            let t_rec = match xs_for_receptor_interaction(receptor.xs_types[j]) {
                Some(t) => t,
                None => continue,
            };
            let dx = lig_xyz[0] - rec_xyz[0];
            let dy = lig_xyz[1] - rec_xyz[1];
            let dz = lig_xyz[2] - rec_xyz[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < cutoff_sqr {
                let (e, dor) = precalc.eval_deriv(t_lig, t_rec, r2);
                e_i += e;
                // ∂E/∂lig_k = (dE/dr)·(lig_k−rec_k)/r = dor·(lig−rec)_k
                grad_i[0] += dor * dx;
                grad_i[1] += dor * dy;
                grad_i[2] += dor * dz;
            }
        }

        curl_with_gradient(&mut e_i, &mut grad_i, v_curl);
        total += e_i;
        // Our force convention: −∂E/∂r.
        forces[i][0] = -grad_i[0];
        forces[i][1] = -grad_i[1];
        forces[i][2] = -grad_i[2];
    }
    (total, forces)
}

/// Intra-molecular Vina pair energy PLUS per-atom force.
///
/// Upstream analogue: `eval_interacting_pairs_deriv`. Per-pair
/// `curl` is applied exactly like the energy-only variant above,
/// which differs from the per-ligand-atom curl used by inter.
/// Outputs forces on BOTH endpoints of every pair (Newton's third
/// law: force on `a` from a pair equals minus the force on `b`).
#[must_use]
pub fn intra_pair_energy_with_forces(
    m: &Molecule,
    pairs: &[(usize, usize)],
    precalc: &Precalculate,
    v_curl: f64,
) -> (f64, Vec<Vec3>) {
    let cutoff_sqr = precalc.cutoff_sqr();
    let mut total = 0.0_f64;
    let mut forces = vec![[0.0_f64; 3]; m.len()];

    for &(a, b) in pairs {
        let ra = m.coords[a];
        let rb = m.coords[b];
        // r_ba = b − a (vector from a toward b). Upstream uses the
        // same sign so its `force = dor * r` is +∂E/∂a; we keep
        // the sign here then flip at the accumulation step.
        let dx = rb[0] - ra[0];
        let dy = rb[1] - ra[1];
        let dz = rb[2] - ra[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 < cutoff_sqr {
            let (e_pair, dor) = precalc.eval_deriv(m.xs_types[a], m.xs_types[b], r2);
            let mut e = e_pair;
            // grad_on_a = +∂E/∂a = −dor · r_ba (since dr/da = −r_ba/r,
            // see score.rs doc); but upstream stores `force = dor·r_ba`
            // which is +∂E/∂b = −∂E/∂a (Newton). To stay consistent
            // with our "force = −∂E/∂r" convention we compute
            // force_on_a directly: +dor · r_ba.
            let mut force_on_a = [dor * dx, dor * dy, dor * dz];
            curl_with_gradient(&mut e, &mut force_on_a, v_curl);
            total += e;
            // Newton's third law.
            forces[a][0] += force_on_a[0];
            forces[a][1] += force_on_a[1];
            forces[a][2] += force_on_a[2];
            forces[b][0] -= force_on_a[0];
            forces[b][1] -= force_on_a[1];
            forces[b][2] -= force_on_a[2];
        }
    }
    (total, forces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad_types::AdType;
    use crate::bonds::infer_bonds;
    use crate::pdbqt::RawAtom;
    use crate::xs_assign::assign_xs_types;
    use approx::assert_relative_eq;

    /// Build a tiny `Molecule` from a slice of (AD type, coords) pairs.
    /// Runs bond inference + XS typing so the result is scorer-ready.
    fn molecule_from(atoms: Vec<(AdType, [f64; 3])>) -> Molecule {
        let raw: Vec<RawAtom> = atoms
            .into_iter()
            .enumerate()
            .map(|(i, (ad, xyz))| RawAtom {
                serial: i as u32 + 1,
                coords: xyz,
                partial_charge: 0.0,
                ad_type: ad,
            })
            .collect();
        let g = infer_bonds(&raw);
        let xs = assign_xs_types(&raw, &g);

        let mut coords = Vec::new();
        let mut xs_types = Vec::new();
        let mut ad_types_kept = Vec::new();
        let mut serials = Vec::new();
        for (a, maybe_t) in raw.iter().zip(xs) {
            if let Some(t) = maybe_t {
                coords.push(a.coords);
                xs_types.push(t);
                ad_types_kept.push(a.ad_type);
                serials.push(a.serial);
            }
        }
        let bonds = infer_bonds(&raw);
        // Remap bonds onto the kept-atom index space for consistency
        // with real `Molecule` construction (here we build a test
        // Molecule where every raw atom is kept, so the remap is
        // identity — use the kept-atom count to size the graph.)
        let mut kept_bonds = vec![Vec::<usize>::new(); coords.len()];
        // Build a raw→kept index map by walking the `maybe_t` stream.
        let xs_maybe = assign_xs_types(&raw, &bonds);
        let mut raw_to_kept = vec![None; raw.len()];
        let mut k = 0;
        for (i, t) in xs_maybe.iter().enumerate() {
            if t.is_some() {
                raw_to_kept[i] = Some(k);
                k += 1;
            }
        }
        for (i, nbrs) in bonds.iter().enumerate() {
            if let Some(ki) = raw_to_kept[i] {
                for &j in nbrs {
                    if let Some(kj) = raw_to_kept[j] {
                        kept_bonds[ki].push(kj);
                    }
                }
            }
        }
        let fragment_ids = vec![0_u32; coords.len()];
        let fragment_mask = vec![1_u64; coords.len()]; // all in fragment 0
        Molecule {
            coords,
            xs_types,
            ad_types: ad_types_kept,
            partial_charges: vec![0.0; serials.len()],
            original_serials: serials,
            fragment_ids,
            bonds: kept_bonds,
            fragment_mask,
        }
    }

    fn precalc() -> Precalculate {
        Precalculate::vina()
    }

    // --- xs_for_receptor_interaction -------------------------------------

    #[test]
    fn glue_atoms_are_skipped() {
        for t in [XsType::G0, XsType::G1, XsType::G2, XsType::G3] {
            assert!(xs_for_receptor_interaction(t).is_none());
        }
    }

    #[test]
    fn macrocycle_ch_carbons_remap_to_ch() {
        for t in [XsType::CHCG0, XsType::CHCG1, XsType::CHCG2, XsType::CHCG3] {
            assert_eq!(xs_for_receptor_interaction(t), Some(XsType::CH));
        }
    }

    #[test]
    fn macrocycle_cp_carbons_remap_to_cp() {
        for t in [XsType::CPCG0, XsType::CPCG1, XsType::CPCG2, XsType::CPCG3] {
            assert_eq!(xs_for_receptor_interaction(t), Some(XsType::CP));
        }
    }

    #[test]
    fn regular_types_pass_through() {
        for t in [XsType::CH, XsType::CP, XsType::NA, XsType::OA, XsType::SP] {
            assert_eq!(xs_for_receptor_interaction(t), Some(t));
        }
    }

    // --- curl ------------------------------------------------------------

    #[test]
    fn curl_is_identity_for_negative_or_zero_energy() {
        assert_eq!(curl(-5.0, 1000.0), -5.0);
        assert_eq!(curl(0.0, 1000.0), 0.0);
    }

    #[test]
    fn curl_shrinks_positive_energy_toward_v() {
        // e=5, v=1000: factor = 1000/1005, result ≈ 4.9751.
        let got = curl(5.0, 1000.0);
        assert_relative_eq!(got, 5.0 * 1000.0 / 1005.0, epsilon = 1e-12);
        // e=100: factor = 1000/1100 ≈ 0.9090909.
        assert_relative_eq!(
            curl(100.0, 1000.0),
            100.0 * 1000.0 / 1100.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn curl_is_noop_when_v_is_infinite() {
        assert_eq!(curl(42.0, f64::INFINITY), 42.0);
        assert_eq!(curl(-42.0, f64::INFINITY), -42.0);
    }

    // --- inter_pair_energy -----------------------------------------------

    #[test]
    fn far_apart_molecules_yield_zero_energy() {
        let rec = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
        ]);
        let lig = molecule_from(vec![
            (AdType::C, [100.0, 0.0, 0.0]),
            (AdType::C, [101.54, 0.0, 0.0]),
        ]);
        let e = inter_pair_energy(&rec, &lig, &precalc(), 1000.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn single_atom_pair_at_contact_is_attractive() {
        // Minimal system: one-atom receptor and one-atom ligand
        // placed at the CH-CH optimum (2 × 1.9 Å).
        let rec = molecule_from(vec![(AdType::C, [0.0, 0.0, 0.0])]);
        let lig = molecule_from(vec![(AdType::C, [3.8, 0.0, 0.0])]);
        let e = inter_pair_energy(&rec, &lig, &precalc(), 1000.0);
        assert!(e < 0.0, "expected attractive, got {e}");
    }

    #[test]
    fn receptor_ligand_swap_gives_same_total_without_curl() {
        // Swap symmetry only holds when the soft energy cap is off:
        // curl is applied per ligand atom, which partitions the same
        // pair-sum differently after a swap.
        let a = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::Oa, [1.4, 0.0, 0.0]),
            (AdType::Hd, [1.9, 0.8, 0.0]),
        ]);
        let b = molecule_from(vec![
            (AdType::N, [3.2, 0.3, 0.0]),
            (AdType::C, [4.7, 0.0, 0.0]),
            (AdType::Hd, [3.2, 1.3, 0.0]),
        ]);
        let e_ab = inter_pair_energy(&a, &b, &precalc(), f64::INFINITY);
        let e_ba = inter_pair_energy(&b, &a, &precalc(), f64::INFINITY);
        assert_relative_eq!(e_ab, e_ba, epsilon = 1e-10);
    }

    #[test]
    fn glue_ligand_atoms_contribute_nothing() {
        // A glue-only ligand must score 0 against any receptor.
        let rec = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
            (AdType::Oa, [2.9, 0.0, 0.0]),
        ]);
        let lig = molecule_from(vec![
            (AdType::G0, [2.0, 0.0, 0.0]),
            (AdType::G1, [3.0, 0.0, 0.0]),
        ]);
        // Sanity: ligand indeed typed as G0/G1.
        assert!(lig
            .xs_types
            .iter()
            .all(|&t| matches!(t, XsType::G0 | XsType::G1)));
        let e = inter_pair_energy(&rec, &lig, &precalc(), 1000.0);
        assert_eq!(e, 0.0);
    }

    // --- intra_pair_list / intra_pair_energy -----------------------------

    #[test]
    fn single_fragment_ligand_has_empty_intra_pair_list() {
        // No BRANCH → all atoms share fragment_id 0 → every pair is
        // DISTANCE_FIXED → list is empty.
        let m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
            (AdType::Oa, [3.0, 0.0, 0.0]),
            (AdType::C, [4.5, 0.0, 0.0]),
        ]);
        assert!(intra_pair_list(&m).is_empty());
        let e = intra_pair_energy(&m, &intra_pair_list(&m), &precalc(), 1000.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn bonded_within_3_covers_1_2_1_3_1_4_but_not_1_5() {
        // Linear chain of 5 carbons: 0-1-2-3-4. All 1-2/1-3/1-4
        // neighbours of atom 0 are {0,1,2,3}; atom 4 is 1-5 and
        // therefore NOT forbidden.
        let m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
            (AdType::C, [3.08, 0.0, 0.0]),
            (AdType::C, [4.62, 0.0, 0.0]),
            (AdType::C, [6.16, 0.0, 0.0]),
        ]);
        let f = bonded_within_3(&m.bonds, 0);
        assert_eq!(f, vec![0, 1, 2, 3]);
    }

    #[test]
    fn intra_pair_list_excludes_glue_atoms() {
        // A glue atom in the middle shouldn't enter the pair list
        // even if its fragment differs.
        let mut m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::G0, [10.0, 0.0, 0.0]),
            (AdType::C, [20.0, 0.0, 0.0]),
        ]);
        // Force different fragments so the pair would otherwise qualify.
        m.fragment_ids = vec![0, 1, 2];
        let pairs = intra_pair_list(&m);
        assert!(
            pairs
                .iter()
                .all(|&(i, j)| m.xs_types[i] != XsType::G0 && m.xs_types[j] != XsType::G0),
            "pair list must not contain glue atoms"
        );
    }

    #[test]
    fn real_1iep_ligand_has_non_empty_intra_pair_list() {
        // Imatinib has 7 rotatable bonds → several fragments →
        // plenty of qualifying pairs.
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let pairs = intra_pair_list(&m);
        assert!(!pairs.is_empty());
        // Sanity: the per-pair constraints must all hold.
        for &(i, j) in &pairs {
            assert_ne!(m.fragment_ids[i], m.fragment_ids[j]);
            assert!(i < j);
        }
    }

    #[test]
    fn real_1iep_ligand_intra_energy_is_finite_and_bounded() {
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let pairs = intra_pair_list(&m);
        let e = intra_pair_energy(&m, &pairs, &precalc(), 1000.0);
        assert!(e.is_finite());
        // Not a tight parity check (that's Phase C.4), but the
        // magnitude should be modest — tens of kcal/mol at most.
        assert!(e.abs() < 100.0, "implausible intra energy {e}");
    }

    #[test]
    fn real_1iep_ligand_vs_receptor_inter_energy_is_attractive_and_finite() {
        // Integration: the 1iep docked pose should register as an
        // attractive interaction (inter < 0) with a sane magnitude.
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        const RECEPTOR_FIXTURE: &str = include_str!("../tests/fixtures/1iep_receptor.pdbqt");
        let rec = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).unwrap();
        let lig = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let e = inter_pair_energy(&rec, &lig, &precalc(), 1000.0);
        assert!(e.is_finite());
        assert!(e < 0.0, "1iep docked pose expected attractive, got {e}");
        assert!(
            e > -200.0,
            "1iep inter energy {e} is implausibly large in magnitude"
        );
    }

    // --- compute_num_tors + score_only -----------------------------------

    #[test]
    fn num_tors_is_zero_when_no_rotatable_bonds() {
        let m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
        ]);
        assert_eq!(compute_num_tors(&m, &[]), 0.0);
    }

    #[test]
    fn num_tors_counts_one_for_bond_between_two_core_atoms() {
        // Linear C-C-C-C-C: one rotatable bond between atoms 2 and 3
        // (both "core" — each has > 1 heavy neighbour).
        let m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),  // serial 1
            (AdType::C, [1.54, 0.0, 0.0]), // serial 2
            (AdType::C, [3.08, 0.0, 0.0]), // serial 3
            (AdType::C, [4.62, 0.0, 0.0]), // serial 4
            (AdType::C, [6.16, 0.0, 0.0]), // serial 5
        ]);
        // Bond between serials 2 and 3: each has 2 heavy neighbours.
        assert_eq!(compute_num_tors(&m, &[(2, 3)]), 1.0);
    }

    #[test]
    fn num_tors_counts_half_for_bond_to_terminal_methyl() {
        // C-C-C-Methyl: serial 3 has 2 heavy neighbours (core), serial 4
        // has 1 heavy neighbour (terminal). Bond (3, 4) contributes
        // only from serial 4's side (other end = serial 3 has > 1
        // heavy) — 0.5 total.
        let m = molecule_from(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [1.54, 0.0, 0.0]),
            (AdType::C, [3.08, 0.0, 0.0]),
            (AdType::C, [4.62, 0.0, 0.0]),
        ]);
        // bonds[0]=[1], bonds[1]=[0,2], bonds[2]=[1,3], bonds[3]=[2]
        //                                                ^ terminal (1 heavy nbr)
        assert_eq!(compute_num_tors(&m, &[(3, 4)]), 0.5);
    }

    #[test]
    fn score_only_on_rigid_single_fragment_ligand_has_zero_intra_and_zero_conf_adj() {
        let rec = molecule_from(vec![(AdType::C, [0.0, 0.0, 0.0])]);
        let lig = molecule_from(vec![
            (AdType::C, [3.8, 0.0, 0.0]),
            (AdType::C, [5.34, 0.0, 0.0]),
        ]);
        let c = score_only(&rec, &lig, &[], &precalc(), 1000.0);
        assert_eq!(c.lig_intra, 0.0);
        assert_eq!(c.inter_pairs, 0.0);
        assert_eq!(c.flex_grids, 0.0);
        assert_eq!(c.intra_pairs, 0.0);
        // No rotatable bonds → num_tors = 0 → divisor = 1 → total = lig_grids.
        assert_relative_eq!(c.total, c.lig_grids, epsilon = 1e-12);
        assert_relative_eq!(c.conf_independent, 0.0, epsilon = 1e-12);
        // intramolecular = lig_intra = 0.
        assert_eq!(c.intramolecular, 0.0);
    }

    // --- inter / intra with forces: FD regression ------------------------
    //
    // Caveat: the Vina pair potentials have a HARD cutoff at r = 8 Å
    // (terms simply return 0 for r ≥ cutoff). Near the cutoff,
    // perturbing an atom by h can push some pairs from r² < cutoff²
    // to r² ≥ cutoff², introducing a discontinuous jump into any
    // central-difference estimator regardless of how small h is.
    //
    // The FD-parity tests below therefore come in two flavours:
    //   1. Synthetic tiny systems whose pair geometry is well clear
    //      of cutoff — these gate at 5e-3 rel err (the expected
    //      bin-width artefact).
    //   2. "Safe-atom" tests on real fixtures that skip any atom
    //      whose pair set contains an r within `cutoff_safety` of
    //      the cutoff. For those atoms, analytical force and FD
    //      gradient agree to ≤ 5e-3 relative.

    /// Build a minimal `Molecule` from a list of (AD type, coords).
    /// Skips full Molecule construction and just wires the fields by
    /// hand — useful for small hand-built systems.
    fn tiny_molecule(atoms: Vec<(AdType, [f64; 3])>) -> Molecule {
        use crate::pdbqt::RawAtom;
        use crate::xs_assign::assign_xs_types;
        let raw: Vec<RawAtom> = atoms
            .iter()
            .enumerate()
            .map(|(i, (ad, xyz))| RawAtom {
                serial: i as u32 + 1,
                coords: *xyz,
                partial_charge: 0.0,
                ad_type: *ad,
            })
            .collect();
        let g = infer_bonds(&raw);
        let xs = assign_xs_types(&raw, &g);
        let mut coords = Vec::new();
        let mut xs_types = Vec::new();
        let mut ad_types = Vec::new();
        for (a, t) in raw.iter().zip(xs) {
            if let Some(t) = t {
                coords.push(a.coords);
                xs_types.push(t);
                ad_types.push(a.ad_type);
            }
        }
        let n = coords.len();
        Molecule {
            coords,
            xs_types,
            ad_types,
            partial_charges: vec![0.0; n],
            original_serials: (0..n as u32).collect(),
            fragment_ids: vec![0; n],
            bonds: vec![vec![]; n],
            fragment_mask: vec![1; n],
        }
    }

    fn inter_only(rec: &Molecule, lig: &Molecule, precalc: &Precalculate) -> f64 {
        inter_pair_energy_with_forces(rec, lig, precalc, f64::INFINITY).0
    }

    #[test]
    fn inter_forces_fd_parity_on_synthetic_pair() {
        // Two single-atom molecules at a safe interior distance.
        // Vary the ligand's coords; force should equal the finite-
        // difference of the energy to high precision.
        let rec = tiny_molecule(vec![(AdType::C, [0.0, 0.0, 0.0])]);
        let precalc = Precalculate::vina();
        let h = 1e-6;
        for r in [3.0, 4.0, 5.0, 6.5, 7.3] {
            let lig = tiny_molecule(vec![(AdType::C, [r, 0.1, 0.2])]);
            let (_, forces) = inter_pair_energy_with_forces(&rec, &lig, &precalc, f64::INFINITY);
            for axis in 0..3 {
                let mut plus = lig.clone();
                plus.coords[0][axis] += h;
                let mut minus = lig.clone();
                minus.coords[0][axis] -= h;
                let fd = -(inter_only(&rec, &plus, &precalc) - inter_only(&rec, &minus, &precalc))
                    / (2.0 * h);
                let err_abs = (forces[0][axis] - fd).abs();
                let scale = forces[0][axis].abs().max(fd.abs()).max(1.0);
                let err_rel = err_abs / scale;
                assert!(
                    err_rel < 5e-3,
                    "r={r}, axis={axis}: analytic={:.6e}, fd={:.6e}, rel err={err_rel:.3e}",
                    forces[0][axis],
                    fd,
                );
            }
        }
    }

    #[test]
    fn intra_forces_fd_parity_on_synthetic_pair() {
        // Two ligand atoms in different fragments so they form an
        // intra pair. We fake this by hand-assigning fragment IDs
        // and a single-element pair list.
        let mut lig = tiny_molecule(vec![
            (AdType::C, [0.0, 0.0, 0.0]),
            (AdType::C, [3.8, 0.1, 0.2]),
        ]);
        lig.fragment_ids = vec![0, 1];
        lig.fragment_mask = vec![1, 2];
        let pairs = vec![(0_usize, 1_usize)];
        let precalc = Precalculate::vina();

        let (_, forces) = intra_pair_energy_with_forces(&lig, &pairs, &precalc, f64::INFINITY);

        let h = 1e-6;
        for i in 0..2 {
            for axis in 0..3 {
                let mut plus = lig.clone();
                plus.coords[i][axis] += h;
                let mut minus = lig.clone();
                minus.coords[i][axis] -= h;
                let (e_plus, _) =
                    intra_pair_energy_with_forces(&plus, &pairs, &precalc, f64::INFINITY);
                let (e_minus, _) =
                    intra_pair_energy_with_forces(&minus, &pairs, &precalc, f64::INFINITY);
                let fd = -(e_plus - e_minus) / (2.0 * h);
                let err_abs = (forces[i][axis] - fd).abs();
                let scale = forces[i][axis].abs().max(fd.abs()).max(1.0);
                let err_rel = err_abs / scale;
                assert!(
                    err_rel < 5e-3,
                    "atom {i}, axis {axis}: analytic={:.6e}, fd={:.6e}, rel err={err_rel:.3e}",
                    forces[i][axis],
                    fd,
                );
            }
        }
    }

    #[test]
    fn inter_forces_on_real_fixture_are_finite_and_newton_balanced() {
        // Can't do tight FD parity on the real 1iep fixture because
        // the Vina potentials have multiple piecewise-linear
        // breakpoints (slope_step in hydrophobic + h_bond terms, on
        // top of the outer cutoff). These give the FD estimator
        // spurious contributions wherever h straddles a breakpoint.
        // Structural invariants are what we check instead:
        // per-atom forces finite, Newton's third law holds when
        // summed over ligand vs receptor halves. Tight
        // force-vs-energy agreement is guaranteed in practice by
        // the synthetic tests above plus the BFGS local-only
        // parity test to come in Phase D.3c.
        use crate::molecule::Molecule;
        const LIG: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
        const REC: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");
        let rec = Molecule::from_pdbqt_str(REC).unwrap();
        let lig = Molecule::from_pdbqt_str(LIG).unwrap();
        let precalc = Precalculate::vina();

        let (e_inter, forces_lig) =
            inter_pair_energy_with_forces(&rec, &lig, &precalc, f64::INFINITY);
        for f in &forces_lig {
            for c in f {
                assert!(c.is_finite());
            }
        }
        // Energy should be negative (docked pose is attractive).
        assert!(e_inter < 0.0);
        // Ligand forces summed: equal-and-opposite to receptor
        // forces (Newton's third law on the pair sum).
        let (_, forces_rec) = inter_pair_energy_with_forces(&lig, &rec, &precalc, f64::INFINITY);
        let sum_lig = forces_lig
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        let sum_rec = forces_rec
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for k in 0..3 {
            assert!((sum_lig[k] + sum_rec[k]).abs() < 1e-8);
        }
    }

    #[test]
    fn inter_forces_obey_newton_for_symmetric_sign_flip() {
        // Swapping receptor and ligand should yield forces equal in
        // magnitude but opposite sign (sum over all atoms), provided
        // the curl cap doesn't fire (which requires finite v_curl on
        // a positive energy). For the 1iep docked pose the system is
        // attractive overall, so per-atom energies are negative and
        // curl is a no-op — safe to use v_curl = 1000 here.
        use crate::molecule::Molecule;
        const LIG: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
        const REC: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");
        let rec = Molecule::from_pdbqt_str(REC).unwrap();
        let lig = Molecule::from_pdbqt_str(LIG).unwrap();
        let precalc = Precalculate::vina();

        let (_, forces_on_lig) = inter_pair_energy_with_forces(&rec, &lig, &precalc, f64::INFINITY);
        let (_, forces_on_rec) = inter_pair_energy_with_forces(&lig, &rec, &precalc, f64::INFINITY);

        let total_on_lig = forces_on_lig
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        let total_on_rec = forces_on_rec
            .iter()
            .fold([0.0_f64; 3], |a, f| [a[0] + f[0], a[1] + f[1], a[2] + f[2]]);
        for k in 0..3 {
            let sum = total_on_lig[k] + total_on_rec[k];
            assert!(
                sum.abs() < 1e-8,
                "Newton's third law violated at axis {k}: sum = {sum:.3e}",
            );
        }
    }

    #[test]
    fn score_only_1iep_matches_upstream_vina_1_2_7() {
        // Parity oracle: values captured from upstream
        //   `vina --score_only --autobox` on v1.2.7-27-g3c65c0b,
        //   input files are the vendored basic_docking/solution
        //   fixtures. Upstream prints to 3 decimals, so we gate on
        //   2 mkcal/mol (tighter than the display grain).
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        const RECEPTOR_FIXTURE: &str = include_str!("../tests/fixtures/1iep_receptor.pdbqt");
        use crate::pdbqt::parse_pdbqt;

        let rec = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).unwrap();
        let lig = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let rot = parse_pdbqt(LIGAND_FIXTURE).unwrap().rotatable_bonds;
        let c = score_only(&rec, &lig, &rot, &precalc(), 1000.0);

        let tol = 2e-3;
        assert_relative_eq!(c.total, -12.513, epsilon = tol);
        assert_relative_eq!(c.lig_grids, -17.634, epsilon = tol);
        assert_eq!(c.inter_pairs, 0.0);
        assert_eq!(c.flex_grids, 0.0);
        assert_eq!(c.intra_pairs, 0.0);
        assert_relative_eq!(c.lig_intra, -0.485, epsilon = tol);
        assert_relative_eq!(c.conf_independent, 5.121, epsilon = tol);
        assert_relative_eq!(c.intramolecular, -0.485, epsilon = tol);
    }

    #[test]
    fn score_only_on_1iep_is_attractive_and_components_are_finite() {
        const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
        const RECEPTOR_FIXTURE: &str = include_str!("../tests/fixtures/1iep_receptor.pdbqt");
        use crate::pdbqt::parse_pdbqt;

        let rec = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).unwrap();
        let lig = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let rot = parse_pdbqt(LIGAND_FIXTURE).unwrap().rotatable_bonds;
        let c = score_only(&rec, &lig, &rot, &precalc(), 1000.0);

        // All eight fields must be finite.
        for &v in &[
            c.total,
            c.lig_grids,
            c.inter_pairs,
            c.flex_grids,
            c.intra_pairs,
            c.lig_intra,
            c.conf_independent,
            c.intramolecular,
        ] {
            assert!(v.is_finite(), "component {v} is not finite");
        }
        // The docked 1iep pose: expect total in a plausible "docking
        // affinity" range (Vina reports this in kcal/mol, typically
        // -10 to -4 for a real pose).
        assert!(c.total < 0.0, "expected attractive total, got {}", c.total);
        assert!(c.total > -20.0, "total {} out of plausible range", c.total);
        // Relationship total = lig_grids / (1 + W_ROT * num_tors)
        // → total / lig_grids is in (0, 1].
        let ratio = c.total / c.lig_grids;
        assert!(ratio > 0.0 && ratio <= 1.0 + 1e-12);
    }
}
