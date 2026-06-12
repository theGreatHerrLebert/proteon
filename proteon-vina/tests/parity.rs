//! Upstream Vina parity across a set of receptor/ligand fixtures.
//!
//! Each fixture lives at `tests/fixtures/pairs/<name>/` with
//! `receptor.pdbqt`, `ligand.pdbqt`, and `upstream.ref` (values
//! captured from `vina --score_only --autobox` on
//! v1.2.7-27-g3c65c0b). The reference file pins the eight energy
//! components plus a per-fixture tolerance and an optional list of
//! components to skip (used for the macrocycle fixture where v0 is
//! a known approximation).
//!
//! Re-generating the reference values: run
//!   `cargo run -p proteon-vina --example parity_check`
//! and paste the current upstream values back into the `.ref` files.

use std::collections::HashMap;
use std::path::PathBuf;

use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::parse_pdbqt;
use proteon_vina::precalculate::Precalculate;
use proteon_vina::score::{score_only, ScoreComponents};

/// Fixture directory names relative to `tests/fixtures/pairs/`.
const FIXTURES: &[&str] = &["1iep", "1fpu", "1s63", "bace1"];

struct Reference {
    values: HashMap<String, f64>,
    tolerance: f64,
    skip: Vec<String>,
}

fn parse_ref(path: &std::path::Path) -> Reference {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("missing reference file {}", path.display()));
    let mut values = HashMap::new();
    let mut tolerance = 0.002;
    let mut skip = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once('=').unwrap_or_else(|| {
            panic!("bad line in {}: {line:?}", path.display());
        });
        let k = k.trim();
        let v = v.trim();
        if k == "skip_components" {
            skip = v.split(',').map(|s| s.trim().to_string()).collect();
        } else if k == "tolerance_kcal" {
            tolerance = v.parse().unwrap();
        } else {
            values.insert(k.to_string(), v.parse().unwrap());
        }
    }
    Reference { values, tolerance, skip }
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pairs")
        .join(name)
}

fn score_fixture(dir: &std::path::Path) -> ScoreComponents {
    let rec_text = std::fs::read_to_string(dir.join("receptor.pdbqt")).unwrap();
    let lig_text = std::fs::read_to_string(dir.join("ligand.pdbqt")).unwrap();
    let receptor = Molecule::from_pdbqt_str(&rec_text).unwrap();
    let ligand = Molecule::from_pdbqt_str(&lig_text).unwrap();
    let rot = parse_pdbqt(&lig_text).unwrap().rotatable_bonds;
    score_only(&receptor, &ligand, &rot, &Precalculate::vina(), 1000.0)
}

fn component(c: &ScoreComponents, name: &str) -> f64 {
    match name {
        "total" => c.total,
        "lig_grids" => c.lig_grids,
        "inter_pairs" => c.inter_pairs,
        "flex_grids" => c.flex_grids,
        "intra_pairs" => c.intra_pairs,
        "lig_intra" => c.lig_intra,
        "conf_independent" => c.conf_independent,
        "intramolecular" => c.intramolecular,
        other => panic!("unknown component {other:?}"),
    }
}

#[test]
fn upstream_parity_all_fixtures() {
    let mut failures: Vec<String> = Vec::new();
    for &name in FIXTURES {
        let dir = fixture_dir(name);
        let r = parse_ref(&dir.join("upstream.ref"));
        let c = score_fixture(&dir);

        for (component_name, &expected) in &r.values {
            if r.skip.iter().any(|s| s == component_name) {
                continue;
            }
            let actual = component(&c, component_name);
            let diff = (actual - expected).abs();
            if diff > r.tolerance {
                failures.push(format!(
                    "{name}.{component_name}: actual={actual:.4}, upstream={expected:.4}, \
                     diff={diff:.4} (tolerance {:.4})",
                    r.tolerance
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "upstream parity failures:\n{}",
        failures.join("\n")
    );
}

/// Smoke test: every fixture must parse and produce finite scores.
#[test]
fn every_fixture_loads_and_scores_cleanly() {
    for &name in FIXTURES {
        let c = score_fixture(&fixture_dir(name));
        for v in &[
            c.total,
            c.lig_grids,
            c.inter_pairs,
            c.flex_grids,
            c.intra_pairs,
            c.lig_intra,
            c.conf_independent,
            c.intramolecular,
        ] {
            assert!(v.is_finite(), "{name}: non-finite component {v}");
        }
    }
}
