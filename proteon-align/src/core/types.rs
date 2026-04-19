/// A 3D coordinate [x, y, z].
pub type Coord3D = [f64; 3];

/// Rotation matrix (3x3) and translation vector (3) for superposition.
#[derive(Debug, Clone)]
pub struct Transform {
    pub t: [f64; 3],
    pub u: [[f64; 3]; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            t: [0.0; 3],
            u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl Transform {
    /// Apply rotation + translation to a single point.
    #[inline]
    pub fn apply(&self, x: &Coord3D) -> Coord3D {
        [
            self.t[0] + self.u[0][0] * x[0] + self.u[0][1] * x[1] + self.u[0][2] * x[2],
            self.t[1] + self.u[1][0] * x[0] + self.u[1][1] * x[1] + self.u[1][2] * x[2],
            self.t[2] + self.u[2][0] * x[0] + self.u[2][1] * x[1] + self.u[2][2] * x[2],
        ]
    }

    /// Apply rotation + translation to a batch of points.
    pub fn apply_batch(&self, x: &[Coord3D], out: &mut [Coord3D]) {
        for (xi, oi) in x.iter().zip(out.iter_mut()) {
            *oi = self.apply(xi);
        }
    }
}

/// Squared Euclidean distance between two 3D points.
#[inline]
pub fn dist_squared(x: &Coord3D, y: &Coord3D) -> f64 {
    let d0 = x[0] - y[0];
    let d1 = x[1] - y[1];
    let d2 = x[2] - y[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Molecule type: protein or RNA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolType {
    Protein,
    RNA,
}

/// TM-score normalization and distance parameters.
#[derive(Debug, Clone)]
pub struct TMParams {
    pub d0_min: f64,
    pub lnorm: f64,
    pub score_d8: f64,
    pub d0: f64,
    pub d0_search: f64,
    pub dcu0: f64,
}

impl TMParams {
    /// Parameter initialization for the search phase.
    /// Corresponds to C++ `parameter_set4search`.
    pub fn for_search(xlen: usize, ylen: usize) -> Self {
        let lnorm = xlen.min(ylen) as f64;
        let d0 = if lnorm <= 19.0 {
            0.168
        } else {
            1.24 * (lnorm - 15.0).powf(1.0 / 3.0) - 1.8
        };
        let d0_min = d0 + 0.8;
        let d0 = d0_min;
        let d0_search = d0.clamp(4.5, 8.0);
        let score_d8 = 1.5 * lnorm.powf(0.3) + 3.5;
        let dcu0 = 4.25;

        Self {
            d0_min,
            lnorm,
            score_d8,
            d0,
            d0_search,
            dcu0,
        }
    }

    /// Parameter initialization for final TM-score calculation (protein).
    /// Corresponds to C++ `parameter_set4final`.
    pub fn for_final(len: f64, mol_type: MolType) -> Self {
        if mol_type == MolType::RNA {
            return Self::for_final_c3prime(len);
        }
        let d0_min = 0.5;
        let lnorm = len;
        let d0 = if lnorm <= 21.0 {
            0.5
        } else {
            (1.24 * (lnorm - 15.0).powf(1.0 / 3.0) - 1.8).max(d0_min)
        };
        let d0_search = d0.clamp(4.5, 8.0);

        Self {
            d0_min,
            lnorm,
            score_d8: 0.0, // not used in final
            d0,
            d0_search,
            dcu0: 4.25,
        }
    }

    /// Parameter initialization for final TM-score calculation (RNA, C3' atom).
    /// Corresponds to C++ `parameter_set4final_C3prime`.
    pub fn for_final_c3prime(len: f64) -> Self {
        let d0_min = 0.3;
        let lnorm = len;
        let d0 = if lnorm <= 11.0 {
            0.3
        } else if lnorm <= 15.0 {
            0.4
        } else if lnorm <= 19.0 {
            0.5
        } else if lnorm <= 23.0 {
            0.6
        } else if lnorm < 30.0 {
            0.7
        } else {
            0.6 * (lnorm - 0.5).powf(1.0 / 2.0) - 2.5
        };
        let d0_search = d0.clamp(4.5, 8.0);

        Self {
            d0_min,
            lnorm,
            score_d8: 0.0,
            d0,
            d0_search,
            dcu0: 4.25,
        }
    }

    /// Parameter initialization for user-specified d0 scaling.
    /// Corresponds to C++ `parameter_set4scale`.
    pub fn for_scale(len: usize, d_s: f64) -> Self {
        let d0_search = d_s.clamp(4.5, 8.0);
        Self {
            d0_min: 0.5,
            lnorm: len as f64,
            score_d8: 0.0,
            d0: d_s,
            d0_search,
            dcu0: 4.25,
        }
    }
}

/// Result of a TM-align computation.
#[derive(Debug, Clone)]
pub struct AlignResult {
    /// TM-score normalized by chain2 (reference) length.
    pub tm_score_chain1: f64,
    /// TM-score normalized by chain1 length.
    pub tm_score_chain2: f64,
    /// TM-score normalized by average length of both chains.
    pub tm_score_avg: f64,
    /// TM-score normalized by user-specified length.
    pub tm_score_user: f64,
    /// TM-score scaled by user-specified d0.
    pub tm_score_scaled: f64,
    /// RMSD of aligned residues.
    pub rmsd: f64,
    /// Number of aligned residues (after distance filter).
    pub n_aligned: usize,
    /// Sequence identity (n_identical / n_aligned).
    pub seq_identity: f64,
    /// Optimal superposition transform (rotates chain1 onto chain2).
    pub transform: Transform,
    /// Aligned sequence of chain1 (with gaps as '-').
    pub aligned_seq_x: String,
    /// Aligned sequence of chain2 (with gaps as '-').
    pub aligned_seq_y: String,
    /// Alignment annotation: ':' = d < d0, '.' = other aligned, ' ' = gap.
    pub alignment_markers: String,
    /// d0 for chain1 normalization.
    pub d0a: f64,
    /// d0 for chain2 normalization.
    pub d0b: f64,
    /// d0 used for output annotation.
    pub d0_out: f64,
}

/// Options controlling the TM-align algorithm.
#[derive(Debug, Clone)]
pub struct AlignOptions {
    /// User-specified alignment mode:
    /// 0 = no user alignment, 1 = soft (-i), 2 = unused, 3 = strict (-I).
    pub i_opt: i32,
    /// Normalization by average length: 0=no, 1=yes (T), -1=shorter, -2=longer.
    pub a_opt: i32,
    /// Whether to normalize by user-specified length.
    pub u_opt: bool,
    /// User-specified normalization length.
    pub lnorm_ass: f64,
    /// Whether to use user-specified d0 scaling.
    pub d_opt: bool,
    /// User-specified d0 scale.
    pub d0_scale: f64,
    /// Use fast (fTM-align) algorithm.
    pub fast_opt: bool,
    /// Molecule type.
    pub mol_type: MolType,
    /// TM-score cutoff for early termination (-1 = disabled).
    pub tm_cut: f64,
    /// User-provided alignment sequences (for i_opt).
    pub user_alignment: Option<Vec<String>>,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            i_opt: 0,
            a_opt: 0,
            u_opt: false,
            lnorm_ass: 0.0,
            d_opt: false,
            d0_scale: 0.0,
            fast_opt: false,
            mol_type: MolType::Protein,
            tm_cut: -1.0,
            user_alignment: None,
        }
    }
}

/// Pre-allocated workspace for Needleman-Wunsch dynamic programming.
/// Avoids repeated allocation in hot loops.
pub struct DPWorkspace {
    pub path: Vec<Vec<bool>>,
    pub val: Vec<Vec<f64>>,
    cap_rows: usize,
    cap_cols: usize,
}

impl DPWorkspace {
    pub fn new(max_len1: usize, max_len2: usize) -> Self {
        let rows = max_len1 + 1;
        let cols = max_len2 + 1;
        Self {
            path: vec![vec![false; cols]; rows],
            val: vec![vec![0.0; cols]; rows],
            cap_rows: rows,
            cap_cols: cols,
        }
    }

    /// Ensure workspace is large enough for the given dimensions.
    /// Grows if necessary, never shrinks.
    pub fn ensure_size(&mut self, len1: usize, len2: usize) {
        let rows = len1 + 1;
        let cols = len2 + 1;
        if rows > self.cap_rows || cols > self.cap_cols {
            let new_rows = rows.max(self.cap_rows);
            let new_cols = cols.max(self.cap_cols);
            self.path = vec![vec![false; new_cols]; new_rows];
            self.val = vec![vec![0.0; new_cols]; new_rows];
            self.cap_rows = new_rows;
            self.cap_cols = new_cols;
        }
    }
}

/// Data extracted from a structure file for alignment.
#[derive(Debug, Clone)]
pub struct StructureData {
    /// CA (or C3') coordinates.
    pub coords: Vec<Coord3D>,
    /// One-letter amino acid or nucleotide codes.
    pub sequence: Vec<char>,
    /// Secondary structure assignment (H/E/T/C).
    pub sec_structure: Vec<char>,
    /// Residue index strings (for -byresi matching).
    pub resi_ids: Vec<String>,
    /// Chain identifier.
    pub chain_id: String,
    /// Molecule type.
    pub mol_type: MolType,
    /// Source file path.
    pub source_path: String,
    /// Original PDB-format lines (for superpose output).
    pub pdb_lines: Vec<String>,
}

impl StructureData {
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}
