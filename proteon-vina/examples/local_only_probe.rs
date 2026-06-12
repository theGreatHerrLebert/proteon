use proteon_vina::local_only::{local_only, LocalOnlyOptions};
use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::parse_pdbqt;
use proteon_vina::precalculate::Precalculate;

const UPSTREAM: &[(&str, [f64; 5])] = &[
    ("1iep",  [-13.241, -18.660, -0.387, 5.418, -0.387]),
    ("1fpu",  [-10.927, -15.398, -0.307, 4.471, -0.307]),
    ("1s63",  [ -8.993, -12.147, -1.585, 3.154, -1.585]),
    ("bace1", [ -7.628, -17.216, -0.878, 9.588, -0.878]),
];

fn main() {
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}  max |Δ|  steps",
        "case", "total", "inter", "intra", "conf", "unbound"
    );
    println!("{}", "-".repeat(82));
    let precalc = Precalculate::vina();
    for (name, up) in UPSTREAM {
        let lig = std::fs::read_to_string(format!(
            "proteon-vina/tests/fixtures/pairs/{name}/ligand.pdbqt"
        )).unwrap();
        let rec = std::fs::read_to_string(format!(
            "proteon-vina/tests/fixtures/pairs/{name}/receptor.pdbqt"
        )).unwrap();
        let receptor = Molecule::from_pdbqt_str(&rec).unwrap();
        let ligand = Molecule::from_pdbqt_str(&lig).unwrap();
        let file = parse_pdbqt(&lig).unwrap();
        let out = local_only(&receptor, &ligand, &file, &precalc, LocalOnlyOptions::default());
        let our = [
            out.components.total,
            out.components.lig_grids,
            out.components.lig_intra,
            out.components.conf_independent,
            out.components.intramolecular,
        ];
        let diffs: Vec<f64> = up.iter().zip(our.iter()).map(|(u, o)| o - u).collect();
        let max = diffs.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        println!(
            "{:<6} {:+10.4} {:+10.4} {:+10.4} {:+10.4} {:+10.4}  {:.4}  {}",
            name, our[0], our[1], our[2], our[3], our[4], max, out.bfgs.n_steps,
        );
    }
}
