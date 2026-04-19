/// Convert a one-letter amino acid code to its three-letter name.
///
/// Matches the C++ `AAmap(char)` function.
pub fn one_to_three(code: char) -> &'static str {
    match code {
        'A' => "ALA",
        'B' => "ASX",
        'C' => "CYS",
        'D' => "ASP",
        'E' => "GLU",
        'F' => "PHE",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'K' => "LYS",
        'L' => "LEU",
        'M' => "MET",
        'N' => "ASN",
        'O' => "PYL",
        'P' => "PRO",
        'Q' => "GLN",
        'R' => "ARG",
        'S' => "SER",
        'T' => "THR",
        'U' => "SEC",
        'V' => "VAL",
        'W' => "TRP",
        'Y' => "TYR",
        'Z' => "GLX",
        _ => "UNK",
    }
}

/// Convert a three-letter residue name to its one-letter code.
/// Handles D-amino acids (e.g., DAL→A) and selenomethionine (MSE→M).
///
/// Matches the C++ `AAmap(string)` function.
pub fn three_to_one(name: &str) -> char {
    match name {
        "ALA" | "DAL" => 'A',
        "ASX" => 'B',
        "CYS" | "DCY" => 'C',
        "ASP" | "DAS" => 'D',
        "GLU" | "DGL" => 'E',
        "PHE" | "DPN" => 'F',
        "GLY" => 'G',
        "HIS" | "DHI" => 'H',
        "ILE" | "DIL" => 'I',
        "LYS" | "DLY" => 'K',
        "LEU" | "DLE" => 'L',
        "MET" | "MED" | "MSE" => 'M',
        "ASN" | "DSG" => 'N',
        "PYL" => 'O',
        "PRO" | "DPR" => 'P',
        "GLN" | "DGN" => 'Q',
        "ARG" | "DAR" => 'R',
        "SER" | "DSN" => 'S',
        "THR" | "DTH" => 'T',
        "SEC" => 'U',
        "VAL" | "DVA" => 'V',
        "TRP" | "DTR" => 'W',
        "TYR" | "DTY" => 'Y',
        "GLX" => 'Z',
        _ => 'X',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_standard_amino_acids() {
        for code in "ACDEFGHIKLMNPQRSTVWY".chars() {
            let three = one_to_three(code);
            assert_eq!(three_to_one(three), code, "roundtrip failed for {code}");
        }
    }

    #[test]
    fn d_amino_acids() {
        assert_eq!(three_to_one("DAL"), 'A');
        assert_eq!(three_to_one("DCY"), 'C');
        assert_eq!(three_to_one("DAS"), 'D');
        assert_eq!(three_to_one("DPN"), 'F');
        assert_eq!(three_to_one("MSE"), 'M'); // selenomethionine
    }

    #[test]
    fn unknown_residues() {
        assert_eq!(three_to_one("HOH"), 'X');
        assert_eq!(three_to_one("ZZZ"), 'X');
        assert_eq!(one_to_three('X'), "UNK");
    }

    #[test]
    fn special_codes() {
        // B = ASX (asparagine or aspartic acid), Z = GLX (glutamine or glutamic acid)
        assert_eq!(three_to_one("ASX"), 'B');
        assert_eq!(three_to_one("GLX"), 'Z');
        // O = PYL (pyrrolysine), U = SEC (selenocysteine)
        assert_eq!(three_to_one("PYL"), 'O');
        assert_eq!(three_to_one("SEC"), 'U');
    }
}
