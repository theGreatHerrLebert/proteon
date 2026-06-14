"""Structure-based AlphaFold template features (TM-align correspondence).

Builds `TemplateFeatures` from a *structural* query↔template alignment instead of a
sequence CIGAR — reaching the remote-homolog regime sequence search misses. See
`devdocs/STRUCTURE_TEMPLATES_PLAN.md`.

T1 (this module) is the **featurizer**: given a query structure and candidate
template structures, TM-align each, build an explicit `StructuralCorrespondence`
carrying original **atom37** residue indices — *not* inferred from the aligner's
letters, since TM-align filters non-CA residues — and gather the template's atom37
onto the query's residue rows (template-native frame; no superposition).

Scope: single-chain; the *featurizer only*. Retrieval, template-DB construction,
and date/leakage filtering are separate projects (plan §6). This is **not** de-novo
AF templating — structural retrieval needs the query structure (a known-structure /
refinement / iterative product, with a leakage caveat).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .align import tm_align
from .supervision_constants import ATOM_ORDER, AA_TO_INDEX, residue_to_one_letter
from .supervision_geometry import compute_torsion_angles_sin_cos, extract_atom37
from .templates import TEMPLATE_GAP_INDEX, TemplateFeatures

_X_INDEX = AA_TO_INDEX["X"]
_CA_IDX = ATOM_ORDER["CA"]
_CB_IDX = ATOM_ORDER["CB"]


def _pseudo_beta_from_atom37(positions, mask, resnames):
    """Per-residue pseudo-β (CB, or CA for glycine) from gathered atom37 — mirrors
    `supervision_geometry.compute_pseudo_beta` on `(L, 37)` arrays + resnames."""
    n = len(resnames)
    coords = np.zeros((n, 3), dtype=np.float32)
    pb_mask = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        idx = _CA_IDX if resnames[i] == "GLY" else _CB_IDX
        if mask[i, idx] > 0:
            coords[i] = positions[i, idx]
            pb_mask[i] = 1.0
    return coords, pb_mask


@dataclass
class StructuralCorrespondence:
    """Query↔template residue correspondence from a TM-align result, in **atom37
    indices** (the `is_amino_acid` residue order the supervision pipeline uses),
    *not* the aligner's CA-filtered order. Each `(query_idx[k], template_idx[k])`
    is one aligned residue pair; both index sequences are strictly increasing."""

    query_idx: NDArray[np.int32]
    template_idx: NDArray[np.int32]
    tm_score: float
    n_aligned: int
    query_len: int
    template_len: int


def _amino_acid_residues(structure, chain_id: Optional[str]):
    """The query/template residue list the supervision pipeline indexes against
    (`is_amino_acid`, single chain). Rejects accidental multi-chain inputs."""
    if chain_id is not None:
        chains = [c for c in structure.chains if c.id == chain_id]
        if not chains:
            raise ValueError(f"chain {chain_id!r} not found")
        chain = chains[0]
    else:
        chains = list(structure.chains)
        if len(chains) != 1:
            raise ValueError(
                "structure-based templates are single-chain; pass an explicit chain"
            )
        chain = chains[0]
    return [r for r in chain.residues if r.is_amino_acid]


def _ca_index_map(residues):
    """`(supervision_index, one_letter)` of each CA-bearing residue, in order — an
    approximation of the set TM-align aligns.

    **Known T1 limitation (codex review):** TM-align's Rust extractor takes any
    non-nucleotide residue with a *non-hetero* CA, whereas this reconstructs from
    the supervision `is_amino_acid` residues. The two coincide for standard protein
    structures (all test fixtures), but a structure with ATOM `UNK` / modified /
    HETATM amino acids can diverge — `structural_correspondence`'s sequence assert
    then *rejects* the candidate (safe: never a wrong index map) and
    `build_structure_template_features` skips it **with a warning**. The proper fix
    is the aligner exposing its own residue-index map; deferred (plan §5)."""
    idx: List[int] = []
    seq: List[str] = []
    for i, r in enumerate(residues):
        if any(a.name.strip() == "CA" for a in r.atoms):
            idx.append(i)
            seq.append(residue_to_one_letter(r.name))
    return idx, "".join(seq)


def structural_correspondence(
    align_result, query_residues, template_residues
) -> StructuralCorrespondence:
    """Build the explicit correspondence from a TM-align `AlignResult` and the two
    residue lists. The aligner's CA-filtered residue set is reconstructed and the
    ungapped aligned sequences are asserted equal to it — so the
    column→atom37-index map is self-checked, never inferred unsafely."""
    ax, ay = align_result.aligned_seq_x, align_result.aligned_seq_y
    if len(ax) != len(ay):
        raise ValueError(f"aligned strings differ in length: {len(ax)} vs {len(ay)}")
    qmap, qseq = _ca_index_map(query_residues)
    tmap, tseq = _ca_index_map(template_residues)
    if ax.replace("-", "") != qseq:
        raise ValueError("query aligned sequence != its CA residues — index map unsafe")
    if ay.replace("-", "") != tseq:
        raise ValueError("template aligned sequence != its CA residues — index map unsafe")

    qi: List[int] = []
    ti: List[int] = []
    kx = ky = 0
    for cx, cy in zip(ax, ay):
        if cx != "-" and cy != "-":
            qi.append(qmap[kx])
            ti.append(tmap[ky])
        kx += cx != "-"
        ky += cy != "-"

    qa = np.asarray(qi, dtype=np.int32)
    ta = np.asarray(ti, dtype=np.int32)
    if qa.size and not (np.all(np.diff(qa) > 0) and np.all(np.diff(ta) > 0)):
        raise ValueError("correspondence indices are not strictly increasing")
    # Confidence is the *query*-length-normalized TM-score. proteon's field
    # naming is inverted vs. what it measures (core/types.rs:165-168):
    # `tm_score_chain1` is normalized by chain2's length, `tm_score_chain2` by
    # chain1's. The query is TM-align chain1 (`tm_align(query, cand)`), so the
    # query-normalized score is `tm_score_chain2` — they diverge whenever query
    # and template lengths differ, and ranking by the wrong one can pick the
    # wrong top-K (codex catch).
    return StructuralCorrespondence(
        query_idx=qa,
        template_idx=ta,
        tm_score=float(align_result.tm_score_chain2),
        n_aligned=int(qa.size),
        query_len=len(query_residues),
        template_len=len(template_residues),
    )


def build_structure_template_features(
    query_structure,
    candidate_structures: Sequence,
    *,
    query_chain: Optional[str] = None,
    candidate_chains: Optional[Sequence[Optional[str]]] = None,
    top_k: int = 4,
    fast: bool = False,
) -> TemplateFeatures:
    """`TemplateFeatures` for `query_structure` from the structurally-aligned
    `candidate_structures`. Each candidate is TM-aligned to the query; its atom37
    is gathered onto the aligned query rows (template-native frame), unaligned rows
    stay zero/mask-0/`TEMPLATE_GAP_INDEX`. `template_sum_probs` is the **raw,
    query-length-normalized TM-score** (`tm_score_chain2`; see
    `structural_correspondence` for the field-naming caveat) — not a per-set
    max-normalization (a hit's confidence must not depend on its competitors).
    The top `top_k` by TM-score are kept. A candidate that can't be aligned is
    skipped."""
    q_res = _amino_acid_residues(query_structure, query_chain)
    length = len(q_res)
    chains = list(candidate_chains) if candidate_chains is not None else [None] * len(
        candidate_structures
    )
    if len(chains) != len(candidate_structures):
        raise ValueError("candidate_chains length must match candidate_structures")

    rows = []  # (tm_score, aatype, positions, masks)
    for cand_pos, (cand, cchain) in enumerate(zip(candidate_structures, chains)):
        try:
            t_res = _amino_acid_residues(cand, cchain)
            result = tm_align(
                query_structure, cand, chain1=query_chain, chain2=cchain, fast=fast
            )
            corr = structural_correspondence(result, q_res, t_res)
        except Exception as exc:
            # Skip an unusable candidate (failed alignment, or a residue-set
            # divergence that makes the index map unsafe — see `_ca_index_map`).
            # Visible, not silent, so a dropped template can be diagnosed.
            warnings.warn(
                f"structure template candidate {cand_pos} skipped: {exc}",
                stacklevel=2,
            )
            continue
        t37 = extract_atom37(t_res)
        aatype = np.full(length, TEMPLATE_GAP_INDEX, dtype=np.int32)
        positions = np.zeros((length, 37, 3), dtype=np.float32)
        masks = np.zeros((length, 37), dtype=np.float32)
        resnames = ["UNK"] * length
        # Per-query-row template residue *numbering* (serial_number), for the
        # torsion continuity check — matching what `build_structure_supervision_
        # example` uses, so a template residue-number gap / chain break correctly
        # masks pre_omega/phi (positional indices would stay consecutive across the
        # break and forge a bond — codex catch). Unaligned rows get strictly-
        # decreasing negative sentinels so they never read as peptide-bonded.
        tmpl_idx_row = -1 - 2 * np.arange(length, dtype=np.int64)
        for qi, ti in zip(corr.query_idx, corr.template_idx):
            positions[qi] = t37["positions"][ti]
            masks[qi] = t37["mask"][ti]
            rn = (t_res[ti].name or "UNK").strip().upper()
            resnames[qi] = rn
            aatype[qi] = AA_TO_INDEX.get(residue_to_one_letter(rn), _X_INDEX)
            tmpl_idx_row[qi] = int(t_res[ti].serial_number)

        # Derived geometry from the gathered template atom37. The torsion continuity
        # uses the *template* residue indices: query-adjacent rows mapping to
        # nonconsecutive template residues (a template insertion) must NOT form a
        # peptide bond, so pre_omega/phi mask there (codex catch).
        pb, pb_mask = _pseudo_beta_from_atom37(positions, masks, resnames)
        tors = compute_torsion_angles_sin_cos(positions, masks, resnames, tmpl_idx_row)
        rows.append((corr.tm_score, aatype, positions, masks, pb, pb_mask, tors))

    rows.sort(key=lambda r: -r[0])
    rows = rows[:top_k]
    n = len(rows)

    def _stack(items, shape, dtype):
        return np.stack(items) if items else np.zeros((0, *shape), dtype=dtype)

    return TemplateFeatures(
        template_aatype=_stack([r[1] for r in rows], (length,), np.int32),
        template_all_atom_positions=_stack(
            [r[2] for r in rows], (length, 37, 3), np.float32
        ),
        template_all_atom_masks=_stack([r[3] for r in rows], (length, 37), np.float32),
        template_sum_probs=np.asarray([r[0] for r in rows], dtype=np.float32),
        n_templates=n,
        query_len=length,
        template_pseudo_beta=_stack([r[4] for r in rows], (length, 3), np.float32),
        template_pseudo_beta_mask=_stack([r[5] for r in rows], (length,), np.float32),
        template_torsion_angles_sin_cos=_stack(
            [r[6]["torsion_angles_sin_cos"] for r in rows], (length, 7, 2), np.float32
        ),
        template_alt_torsion_angles_sin_cos=_stack(
            [r[6]["alt_torsion_angles_sin_cos"] for r in rows], (length, 7, 2), np.float32
        ),
        template_torsion_angles_mask=_stack(
            [r[6]["torsion_angles_mask"] for r in rows], (length, 7), np.float32
        ),
    )
