//! Output formatting for TM-align results.
//!
//! Ported from C++ TMalign `output_results` (lines 3696-3785).
//! Matches the exact formatting of the C++ version.

use std::io::Write;

use proteon_align::core::types::AlignResult;

/// Options controlling output.
pub struct OutputOptions {
    pub xname: String,
    pub yname: String,
    pub chain_id1: String,
    pub chain_id2: String,
    pub xlen: usize,
    pub ylen: usize,
    pub outfmt: i32,
    pub a_opt: i32,
    pub u_opt: bool,
    pub d_opt: bool,
    pub i_opt: i32,
    pub mirror_opt: bool,
    pub lnorm_ass: f64,
    pub d0_scale: f64,
    pub tm_ali: f64,
    pub l_ali: usize,
    pub rmsd_ali: f64,
}

/// Write alignment results to a writer.
pub fn output_results<W: Write>(
    w: &mut W,
    result: &AlignResult,
    opts: &OutputOptions,
) -> std::io::Result<()> {
    match opts.outfmt {
        2 => output_tabular(w, result, opts),
        1 => output_fasta(w, result, opts),
        -1 => output_full(w, result, opts, false),
        _ => output_full(w, result, opts, true),
    }
}

fn output_full<W: Write>(
    w: &mut W,
    r: &AlignResult,
    o: &OutputOptions,
    show_version: bool,
) -> std::io::Result<()> {
    if show_version {
        writeln!(w)?;
        writeln!(
            w,
            " **************************************************************************"
        )?;
        writeln!(
            w,
            " *                        TM-align (Rust)                                 *"
        )?;
        writeln!(
            w,
            " * An algorithm for protein structure alignment and comparison             *"
        )?;
        writeln!(
            w,
            " * Based on: Y Zhang, J Skolnick. Nucl Acids Res 33, 2302-9 (2005)       *"
        )?;
        writeln!(
            w,
            " **************************************************************************"
        )?;
    }
    writeln!(w)?;
    writeln!(
        w,
        "Name of Chain_1: {}{} (to be superimposed onto Chain_2)",
        o.xname,
        if o.chain_id1.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id1)
        }
    )?;
    writeln!(
        w,
        "Name of Chain_2: {}{}",
        o.yname,
        if o.chain_id2.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id2)
        }
    )?;
    writeln!(w, "Length of Chain_1: {} residues", o.xlen)?;
    writeln!(w, "Length of Chain_2: {} residues", o.ylen)?;
    writeln!(w)?;

    if o.i_opt > 0 {
        writeln!(
            w,
            "User-specified initial alignment: TM/Lali/rmsd = {:.5}, {}, {:.2}",
            o.tm_ali, o.l_ali, o.rmsd_ali
        )?;
        writeln!(w)?;
    }

    writeln!(
        w,
        "Aligned length= {}, RMSD= {:6.2}, Seq_ID=n_identical/n_aligned= {:.3}",
        r.n_aligned, r.rmsd, r.seq_identity
    )?;
    writeln!(
        w,
        "TM-score= {:.5} (if normalized by length of Chain_1, i.e., LN={}, d0={:.2})",
        r.tm_score_chain2, o.xlen, r.d0b
    )?;
    writeln!(
        w,
        "TM-score= {:.5} (if normalized by length of Chain_2, i.e., LN={}, d0={:.2})",
        r.tm_score_chain1, o.ylen, r.d0a
    )?;

    if o.a_opt > 0 {
        writeln!(
            w,
            "TM-score= {:.5} (if normalized by average length of two structures, i.e., LN={:.1}, d0={:.2})",
            r.tm_score_avg,
            (o.xlen + o.ylen) as f64 / 2.0,
            r.d0_out
        )?;
    }
    if o.u_opt {
        writeln!(
            w,
            "TM-score= {:.5} (if normalized by user-specified LN={:.1}, d0={:.2})",
            r.tm_score_user, o.lnorm_ass, r.d0_out
        )?;
    }
    if o.d_opt {
        writeln!(
            w,
            "TM-score= {:.5} (if scaled by user-specified d0={:.2}, LN={:.1})",
            r.tm_score_scaled, o.d0_scale, o.ylen as f64
        )?;
    }

    writeln!(
        w,
        "(You should use TM-score normalized by length of the reference structure)"
    )?;
    writeln!(w)?;

    writeln!(
        w,
        "(\":\" denotes residue pairs of d < {:.2} Angstrom, \".\" denotes other aligned residues)",
        r.d0_out
    )?;
    writeln!(w, "{}", r.aligned_seq_x)?;
    writeln!(w, "{}", r.alignment_markers)?;
    writeln!(w, "{}", r.aligned_seq_y)?;

    Ok(())
}

fn output_fasta<W: Write>(w: &mut W, r: &AlignResult, o: &OutputOptions) -> std::io::Result<()> {
    writeln!(
        w,
        ">{}{}\tL={}\td0={:.2}\tseqID={:.3}\tTM-score={:.5}",
        o.xname,
        if o.chain_id1.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id1)
        },
        o.xlen,
        r.d0b,
        if o.xlen > 0 {
            r.seq_identity * r.n_aligned as f64 / o.xlen as f64
        } else {
            0.0
        },
        r.tm_score_chain2
    )?;
    writeln!(w, "{}", r.aligned_seq_x)?;
    writeln!(
        w,
        ">{}{}\tL={}\td0={:.2}\tseqID={:.3}\tTM-score={:.5}",
        o.yname,
        if o.chain_id2.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id2)
        },
        o.ylen,
        r.d0a,
        if o.ylen > 0 {
            r.seq_identity * r.n_aligned as f64 / o.ylen as f64
        } else {
            0.0
        },
        r.tm_score_chain1
    )?;
    writeln!(w, "{}", r.aligned_seq_y)?;
    writeln!(
        w,
        "# Lali={}\tRMSD={:.2}\tseqID_ali={:.3}",
        r.n_aligned, r.rmsd, r.seq_identity
    )?;
    writeln!(w, "$$$$")?;

    Ok(())
}

fn output_tabular<W: Write>(w: &mut W, r: &AlignResult, o: &OutputOptions) -> std::io::Result<()> {
    writeln!(
        w,
        "{}{}\t{}{}\t{:.4}\t{:.4}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}",
        o.xname,
        if o.chain_id1.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id1)
        },
        o.yname,
        if o.chain_id2.is_empty() {
            String::new()
        } else {
            format!(":{}", o.chain_id2)
        },
        r.tm_score_chain2,
        r.tm_score_chain1,
        r.rmsd,
        if o.xlen > 0 {
            r.seq_identity * r.n_aligned as f64 / o.xlen as f64
        } else {
            0.0
        },
        if o.ylen > 0 {
            r.seq_identity * r.n_aligned as f64 / o.ylen as f64
        } else {
            0.0
        },
        r.seq_identity,
        o.xlen,
        o.ylen,
        r.n_aligned,
    )
}

/// Write rotation matrix to a file.
pub fn output_rotation_matrix<W: Write>(w: &mut W, result: &AlignResult) -> std::io::Result<()> {
    writeln!(
        w,
        "------ The rotation matrix to rotate Chain_1 to Chain_2 ------"
    )?;
    writeln!(
        w,
        "m               t[m]            u[m][0]         u[m][1]         u[m][2]"
    )?;
    for m in 0..3 {
        writeln!(
            w,
            "{}    {:18.10}  {:14.10}  {:14.10}  {:14.10}",
            m,
            result.transform.t[m],
            result.transform.u[m][0],
            result.transform.u[m][1],
            result.transform.u[m][2]
        )?;
    }
    writeln!(w)?;
    writeln!(w, "Code for rotating Structure A from (x,y,z) to (X,Y,Z):")?;
    writeln!(w, "for(i=0; i<L; i++)")?;
    writeln!(w, "{{")?;
    writeln!(
        w,
        "  X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];"
    )?;
    writeln!(
        w,
        "  Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];"
    )?;
    writeln!(
        w,
        "  Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];"
    )?;
    writeln!(w, "}}")?;

    Ok(())
}
