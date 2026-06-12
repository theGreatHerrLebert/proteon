//! Parity check across multiple fixture pairs.
//!
//! Default mode: summarise proteon-vina vs upstream deltas for each
//! fixture.
//!
//! Diff mode: `cargo run -p proteon-vina --example parity_check --
//! diff <case> <upstream_log>` diffs the intra pair list atom-by-atom
//! against an upstream log captured with a temporary PARITY_DEBUG
//! patch in `model.cpp::evali`.

use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::{parse_pdbqt, PdbqtFile};
use proteon_vina::precalculate::Precalculate;
use proteon_vina::score::{intra_pair_list, score_only, ScoreComponents};
use std::collections::BTreeSet;

const UPSTREAM: &[(&str, [f64; 5])] = &[
    ("1iep", [-12.513, -17.634, -0.485, 5.121, -0.485]),
    ("1fpu", [-4.036, -5.687, -0.485, 1.652, -0.485]),
    ("1s63", [-8.266, -11.166, -1.488, 2.900, -1.488]),
    ("bace1", [-7.628, -17.216, -0.878, 9.588, -0.878]),
];

fn load(name: &str) -> (Molecule, Molecule, PdbqtFile, String) {
    let dir = format!("proteon-vina/tests/fixtures/pairs/{name}");
    let rec = std::fs::read_to_string(format!("{dir}/receptor.pdbqt")).unwrap();
    let lig = std::fs::read_to_string(format!("{dir}/ligand.pdbqt")).unwrap();
    let receptor = Molecule::from_pdbqt_str(&rec).unwrap();
    let ligand = Molecule::from_pdbqt_str(&lig).unwrap();
    let file = parse_pdbqt(&lig).unwrap();
    (receptor, ligand, file, lig)
}

fn run(name: &str) -> (ScoreComponents, usize) {
    let (rec, lig, file, _) = load(name);
    let c = score_only(&rec, &lig, &file.rotatable_bonds, &Precalculate::vina(), 1000.0);
    let pairs = intra_pair_list(&lig);
    (c, pairs.len())
}

fn summary() {
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}  max |Δ|",
        "case", "total", "inter", "intra", "conf", "unbound", "#pairs"
    );
    println!("{}", "-".repeat(88));
    for (name, up) in UPSTREAM {
        let (c, n_pairs) = run(name);
        let our = [c.total, c.lig_grids, c.lig_intra, c.conf_independent, c.intramolecular];
        let diffs: Vec<f64> = up.iter().zip(our.iter()).map(|(u, o)| o - u).collect();
        let max = diffs.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        println!(
            "{:<6} {:+10.4} {:+10.4} {:+10.4} {:+10.4} {:+10.4} {:>8}  {:.4}",
            name, our[0], our[1], our[2], our[3], our[4], n_pairs, max
        );
    }
}

fn diff(case: &str, log_path: &str) {
    use proteon_vina::ad_types::AdType;
    let (_, lig, file, _) = load(case);
    let ours_pairs = intra_pair_list(&lig);

    // Parse upstream log with AD types (needed to disambiguate
    // G0/CG0 atoms that share coordinates in macrocycles).
    let log_text = std::fs::read_to_string(log_path).unwrap();
    let mut upstream_records: Vec<([f64; 3], [f64; 3], u8, u8)> = Vec::new();
    for line in log_text.lines() {
        if let Some(rest) = line.strip_prefix("PARITY_DEBUG: pair ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }
            let xyz = |tag: &str| -> [f64; 3] {
                let v: Vec<f64> = tag
                    .split('=')
                    .nth(1)
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
                [v[0], v[1], v[2]]
            };
            let num = |tag: &str| -> u8 {
                tag.split('=').nth(1).unwrap().parse().unwrap()
            };
            upstream_records.push((xyz(parts[2]), xyz(parts[3]), num(parts[4]), num(parts[5])));
        }
    }

    // Match upstream atoms to PDBQT serials using (coords, AD type).
    let ad_to_u8 = |ad: AdType| -> u8 { ad.index() as u8 };
    let find_serial = |c: [f64; 3], ad: u8| -> Option<u32> {
        file.atoms
            .iter()
            .find(|x| {
                (x.coords[0] - c[0]).abs() < 1e-3
                    && (x.coords[1] - c[1]).abs() < 1e-3
                    && (x.coords[2] - c[2]).abs() < 1e-3
                    && ad_to_u8(x.ad_type) == ad
            })
            .map(|x| x.serial)
    };

    let theirs: BTreeSet<(u32, u32)> = upstream_records
        .iter()
        .filter_map(|&(a, b, ad_a, ad_b)| {
            let sa = find_serial(a, ad_a)?;
            let sb = find_serial(b, ad_b)?;
            Some(if sa < sb { (sa, sb) } else { (sb, sa) })
        })
        .collect();

    let ours: BTreeSet<(u32, u32)> = ours_pairs
        .iter()
        .map(|&(i, j)| {
            let a = lig.original_serials[i];
            let b = lig.original_serials[j];
            if a < b { (a, b) } else { (b, a) }
        })
        .collect();

    let only_ours: Vec<_> = ours.difference(&theirs).copied().collect();
    let only_theirs: Vec<_> = theirs.difference(&ours).copied().collect();

    println!("case: {case}");
    println!("ours {}, upstream {}", ours.len(), theirs.len());
    println!("only ours: {}, only upstream: {}", only_ours.len(), only_theirs.len());

    let describe = |pair: &(u32, u32)| -> String {
        let (sa, sb) = *pair;
        let describe_one = |s: u32| -> String {
            file.atoms
                .iter()
                .position(|x| x.serial == s)
                .map(|i| {
                    format!(
                        "{:?}/frag{}",
                        file.atoms[i].ad_type, file.fragment_ids[i]
                    )
                })
                .unwrap_or_default()
        };
        let ia = file.atoms.iter().position(|x| x.serial == sa).unwrap();
        let ib = file.atoms.iter().position(|x| x.serial == sb).unwrap();
        let ca = file.atoms[ia].coords;
        let cb = file.atoms[ib].coords;
        let r = {
            let dx = ca[0] - cb[0];
            let dy = ca[1] - cb[1];
            let dz = ca[2] - cb[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        format!(
            "({:3},{:3}) {} {}  r={:.3}",
            sa, sb, describe_one(sa), describe_one(sb), r
        )
    };

    if !only_ours.is_empty() {
        println!("\nExtra in ours:");
        for k in only_ours.iter().take(30) {
            println!("  {}", describe(k));
        }
    }
    if !only_theirs.is_empty() {
        println!("\nExtra in upstream:");
        for k in only_theirs.iter().take(30) {
            println!("  {}", describe(k));
        }
        println!("\nFragment tree:");
        for (f, ((p, ab), ae)) in file
            .fragment_parents
            .iter()
            .zip(file.fragment_axis_begin.iter())
            .zip(file.fragment_axis_end.iter())
            .enumerate()
        {
            println!(
                "  frag {f}: parent={p:?}  axis_begin={ab:?}  axis_end={ae:?}"
            );
        }
        println!("\nMasks for extra-pair atoms:");
        let mut shown = std::collections::BTreeSet::new();
        for pair in only_theirs.iter().take(10) {
            for s in [pair.0, pair.1] {
                if !shown.insert(s) {
                    continue;
                }
                if let Some(mi) = lig.original_serials.iter().position(|&x| x == s) {
                    println!(
                        "  serial {s} (mol {mi})  frag {}  mask {:b}",
                        lig.fragment_ids[mi], lig.fragment_mask[mi]
                    );
                }
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("diff") => {
            let case = args.next().expect("case name");
            let log = args.next().expect("upstream log path");
            diff(&case, &log);
        }
        _ => summary(),
    }
}
