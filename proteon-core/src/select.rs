//! Altloc / model SELECTORS for the label-safe repair layer.
//!
//! `has_altlocs` and `has_multiple_models` are hazards because prepare silently
//! works on the primary conformer / model 0, leaving the ambiguity in the
//! structure. The repair layer can `accept` that silent default — or use these
//! selectors to RESOLVE it: collapse each residue to one chosen conformer, and
//! keep one chosen model, so the output structure is unambiguous (and the
//! hazard clears on re-verify) rather than merely accepted.

use pdbtbx::PDB;

/// Keep only model `index`, dropping all others. No-op if there is one model or
/// `index` is out of range. Returns whether any model was dropped. After this,
/// `has_multiple_models` is False.
pub fn select_model(pdb: &mut PDB, index: usize) -> bool {
    if pdb.model_count() > 1 && index < pdb.model_count() {
        pdb.remove_models_except(&[index]);
        true
    } else {
        false
    }
}

#[inline]
fn mean_occupancy(conformer: &pdbtbx::Conformer) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for a in conformer.atoms() {
        sum += a.occupancy();
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Collapse every multi-conformer (alternate-location) residue to a single
/// conformer, and clear the surviving conformer's altloc id so the structure is
/// unambiguous. `by_occupancy` keeps the highest-mean-occupancy conformer;
/// otherwise the first one (altloc A by convention). Returns the number of
/// residues whose altloc state was modified (a multi-conformer residue
/// collapsed, OR a single conformer whose altloc id was cleared). After this,
/// `has_altlocs` is False.
pub fn collapse_altlocs(pdb: &mut PDB, by_occupancy: bool) -> usize {
    let mut modified = 0;
    for residue in pdb.residues_mut() {
        let n = residue.conformer_count();
        // A residue carries altloc ambiguity if it has multiple conformers OR a
        // single conformer that still has an altloc id. We "modify" (and so
        // count) such residues — including the single-conformer-with-id case, so
        // the repair action is recorded for it too (codex).
        let had_altloc = n > 1
            || residue
                .conformers()
                .any(|c| c.alternative_location().is_some());
        if n > 1 {
            let keep = if by_occupancy {
                residue
                    .conformers()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        mean_occupancy(a)
                            .partial_cmp(&mean_occupancy(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i)
            } else {
                0
            };
            // Remove every conformer except `keep` (reverse order: stable indices).
            for i in (0..n).rev() {
                if i != keep {
                    residue.remove_conformer(i);
                }
            }
        }
        // Clear any residual altloc id on the (now sole) conformer so the
        // structure no longer carries the ambiguity.
        for conformer in residue.conformers_mut() {
            conformer.remove_alternative_location();
        }
        if had_altloc {
            modified += 1;
        }
    }
    modified
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdbtbx::{Atom, Conformer, Residue};

    fn atom(name: &str, occ: f64, pos: f64) -> Atom {
        Atom::new(false, 1, name, name, pos, 0.0, 0.0, occ, 0.0, "C", 0).unwrap()
    }

    /// A residue with two alternate conformers A (occ 0.3) and B (occ 0.7).
    fn dual_conformer_pdb() -> PDB {
        let mut conf_a = Conformer::new("ALA", Some("A"), None).unwrap();
        conf_a.add_atom(atom("CB", 0.3, 1.0));
        let mut conf_b = Conformer::new("ALA", Some("B"), None).unwrap();
        conf_b.add_atom(atom("CB", 0.7, 2.0));
        let mut res = Residue::new(1, None, None).unwrap();
        res.add_conformer(conf_a);
        res.add_conformer(conf_b);
        let mut chain = pdbtbx::Chain::new("A").unwrap();
        chain.add_residue(res);
        let mut model = pdbtbx::Model::new(1);
        model.add_chain(chain);
        let mut pdb = PDB::new();
        pdb.add_model(model);
        pdb
    }

    #[test]
    fn collapse_keeps_highest_occupancy_and_clears_altloc() {
        let mut pdb = dual_conformer_pdb();
        let collapsed = collapse_altlocs(&mut pdb, true);
        assert_eq!(collapsed, 1);
        let res = pdb.residues().next().unwrap();
        assert_eq!(res.conformer_count(), 1);
        let conf = res.conformers().next().unwrap();
        assert!(
            conf.alternative_location().is_none(),
            "altloc id must be cleared"
        );
        // highest occupancy was conformer B (occ 0.7) at pos 2.0
        assert_eq!(conf.atoms().next().unwrap().x(), 2.0);
    }

    #[test]
    fn collapse_first_keeps_conformer_a() {
        let mut pdb = dual_conformer_pdb();
        collapse_altlocs(&mut pdb, false);
        let conf = pdb.residues().next().unwrap().conformers().next().unwrap();
        assert_eq!(conf.atoms().next().unwrap().x(), 1.0); // conformer A (first)
        assert!(conf.alternative_location().is_none());
    }

    #[test]
    fn collapse_noop_on_single_conformer_clears_altloc() {
        // A single conformer that still carries an altloc id -> id cleared, no remove.
        let mut conf = Conformer::new("ALA", Some("A"), None).unwrap();
        conf.add_atom(atom("CB", 0.5, 1.0));
        let mut res = Residue::new(1, None, None).unwrap();
        res.add_conformer(conf);
        let mut chain = pdbtbx::Chain::new("A").unwrap();
        chain.add_residue(res);
        let mut model = pdbtbx::Model::new(1);
        model.add_chain(chain);
        let mut pdb = PDB::new();
        pdb.add_model(model);
        // The single conformer HAD an altloc id (cleared) -> counts as modified,
        // so the repair action is recorded (codex).
        assert_eq!(collapse_altlocs(&mut pdb, true), 1);
        let conf = pdb.residues().next().unwrap().conformers().next().unwrap();
        assert!(conf.alternative_location().is_none());
        // A second pass: now no altloc id at all -> not modified.
        assert_eq!(collapse_altlocs(&mut pdb, true), 0);
    }

    #[test]
    fn select_model_keeps_one() {
        let mut pdb = PDB::new();
        pdb.add_model(pdbtbx::Model::new(1));
        pdb.add_model(pdbtbx::Model::new(2));
        pdb.add_model(pdbtbx::Model::new(3));
        assert_eq!(pdb.model_count(), 3);
        assert!(select_model(&mut pdb, 0));
        assert_eq!(pdb.model_count(), 1);
        assert!(!select_model(&mut pdb, 0)); // already single -> no-op
    }
}
