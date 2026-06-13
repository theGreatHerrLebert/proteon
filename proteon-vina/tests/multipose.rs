//! Multi-pose stress test: iterate every pose in a vina-output
//! PDBQT stream and verify per-pose component parity against
//! upstream `vina --score_only --autobox`. Covers three fixtures:
//!
//! * `1iep`: 4 native-Vina docked poses (drug-like kinase ligand).
//! * `1s63`: 9 AD4-docked poses of a halogenated zinc ligand
//!   (tests Zn/MetD handling across non-Vina-biased poses).
//! * `bace1`: 9 native-Vina poses of a macrocyclic BACE inhibitor
//!   (stress-tests is_closure_clash across rotamer variations).

use std::collections::HashMap;

use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::parse_pdbqt_models;
use proteon_vina::precalculate::Precalculate;
use proteon_vina::score::{score_only, ScoreComponents};

struct Reference {
    receptor: String,
    poses: HashMap<(usize, String), f64>,
    tolerance: f64,
}

fn parse_reference(text: &str) -> Reference {
    let mut receptor: Option<String> = None;
    let mut poses: HashMap<(usize, String), f64> = HashMap::new();
    let mut tolerance = 0.002;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once('=').unwrap();
        let (k, v) = (k.trim(), v.trim());
        match k {
            "receptor" => receptor = Some(v.to_string()),
            "tolerance_kcal" => tolerance = v.parse().unwrap(),
            other => {
                let (pose_tag, component) = other.split_once('.').unwrap();
                let i: usize = pose_tag.strip_prefix("pose").unwrap().parse().unwrap();
                poses.insert((i, component.to_string()), v.parse().unwrap());
            }
        }
    }
    Reference {
        receptor: receptor.expect("receptor name"),
        poses,
        tolerance,
    }
}

fn component(c: &ScoreComponents, name: &str) -> f64 {
    match name {
        "total" => c.total,
        "lig_grids" => c.lig_grids,
        "lig_intra" => c.lig_intra,
        "conf_independent" => c.conf_independent,
        "intramolecular" => c.intramolecular,
        other => panic!("unknown component {other:?}"),
    }
}

/// Each case: (multipose-dir name, poses text, receptor text, ref text).
struct Case {
    name: &'static str,
    poses_text: &'static str,
    receptor_text: &'static str,
    reference_text: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "1iep",
        poses_text: include_str!("fixtures/multipose/1iep/poses.pdbqt"),
        receptor_text: include_str!("fixtures/pairs/1iep/receptor.pdbqt"),
        reference_text: include_str!("fixtures/multipose/1iep/upstream.ref"),
    },
    Case {
        name: "1s63",
        poses_text: include_str!("fixtures/multipose/1s63/poses.pdbqt"),
        receptor_text: include_str!("fixtures/pairs/1s63/receptor.pdbqt"),
        reference_text: include_str!("fixtures/multipose/1s63/upstream.ref"),
    },
    Case {
        name: "bace1",
        poses_text: include_str!("fixtures/multipose/bace1/poses.pdbqt"),
        receptor_text: include_str!("fixtures/pairs/bace1/receptor.pdbqt"),
        reference_text: include_str!("fixtures/multipose/bace1/upstream.ref"),
    },
];

#[test]
fn every_pose_matches_upstream() {
    let mut failures: Vec<String> = Vec::new();
    for case in CASES {
        let r = parse_reference(case.reference_text);
        assert_eq!(r.receptor, case.name, "{} receptor mismatch", case.name);
        let receptor = Molecule::from_pdbqt_str(case.receptor_text).unwrap();
        let poses = parse_pdbqt_models(case.poses_text).unwrap();
        let precalc = Precalculate::vina();

        for (i, file) in poses.iter().enumerate() {
            let ligand = Molecule::from_pdbqt_file(file).unwrap();
            let c = score_only(&receptor, &ligand, &file.rotatable_bonds, &precalc, 1000.0);
            for name in [
                "total",
                "lig_grids",
                "lig_intra",
                "conf_independent",
                "intramolecular",
            ] {
                let Some(&expected) = r.poses.get(&(i, name.into())) else {
                    continue;
                };
                let actual = component(&c, name);
                let diff = (actual - expected).abs();
                if diff > r.tolerance {
                    failures.push(format!(
                        "{}.pose{i}.{name}: actual={actual:.4}, upstream={expected:.4}, diff={diff:.4}",
                        case.name
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "multi-pose parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn poses_are_distinct_in_score_per_case() {
    for case in CASES {
        let receptor = Molecule::from_pdbqt_str(case.receptor_text).unwrap();
        let poses = parse_pdbqt_models(case.poses_text).unwrap();
        let precalc = Precalculate::vina();

        let totals: Vec<f64> = poses
            .iter()
            .map(|p| {
                let lig = Molecule::from_pdbqt_file(p).unwrap();
                score_only(&receptor, &lig, &p.rotatable_bonds, &precalc, 1000.0).total
            })
            .collect();
        let min = totals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = totals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(
            max - min > 0.5,
            "{}: pose totals span only {min:.3}..{max:.3}",
            case.name
        );
    }
}

#[test]
fn total_pose_count_across_all_cases() {
    // 4 + 9 + 9 = 22 poses exercised by the multi-pose parity gate.
    let count: usize = CASES
        .iter()
        .map(|c| parse_pdbqt_models(c.poses_text).unwrap().len())
        .sum();
    assert_eq!(count, 22);
}
