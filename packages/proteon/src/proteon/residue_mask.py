"""Per-residue label validity + structure coverage (phase 1: completeness).

The structure-level ``label_safe`` gate is all-or-nothing: a structure with one
missing loop is dropped whole, which caps diverse-PDB yield at ~15% of
*structures* even though ~87% of *residues* are complete coordinate labels.
Per-residue masking keeps the observed residues instead. This module computes,
per residue, whether it is a valid coordinate label, and the structure's
**coverage** (valid / exportable residues) — so a coverage gate can keep
mostly-complete structures and a downstream export can mask the incomplete ones.

**Phase 1 localizes the DOMINANT hazard only: missing atoms** (completeness, from
the atom37 presence-vs-exists mask — exact, and aligned to the supervision
residue order by construction). Severe-clash / chirality / altloc localization
(which need the Rust topology) and the per-label export masks (torsions, frames,
pseudo-beta, pairs — each combined along its own dependency, NOT one broadcast
mask) are phase 2. See ``devdocs/PER_RESIDUE_MASKING_SKETCH.md``.

The validity here is therefore **node validity for the missing-atoms hazard**: a
necessary condition for a coordinate label, not the full per-label mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .supervision_geometry import ATOM_ORDER, extract_atom37

#: Measured default coverage floor (974-PDB sample, devdocs sketch): keeps ~89%
#: of structures, cutting the sparse ~10% tail (p10 ≈ 0.79). A quality /
#: crop-efficiency knob — masking already handles the missing residues, so this
#: is NOT a corruption guard like the clashscore gate.
DEFAULT_COVERAGE = 0.8

#: atom37 backbone slots (N, CA, C, O) — NOT 0..3, which would include CB at 3.
_BB_SLOTS = [ATOM_ORDER[a] for a in ("N", "CA", "C", "O")]

#: Completeness levels a coverage gate can require.
_COVERAGE_PROFILES = ("heavy_coords", "backbone")


@dataclass
class ResidueCoverage:
    """Per-residue completeness validity and the structure's coverage fraction."""

    #: Completeness level required: ``"heavy_coords"`` (all expected heavy atoms)
    #: or ``"backbone"`` (N, CA, C, O).
    profile: str
    #: Exportable (amino-acid) residues considered — the coverage denominator.
    n_residues: int
    #: Valid (complete) residues — the coverage numerator.
    n_valid: int
    #: Per-residue boolean validity, aligned to the supervision ``residue_index``
    #: (0-based positional over amino-acid residues, model 0, in chain order).
    node_valid: NDArray  # bool[n_residues]

    @property
    def coverage(self) -> float:
        """Valid / exportable protein residues (0.0 when there are none)."""
        if self.n_residues == 0:
            return 0.0
        return self.n_valid / self.n_residues


@dataclass
class ComplexCoverage:
    """Per-chain coverage of a multi-chain structure, for interface labels."""

    #: chain id -> its :class:`ResidueCoverage` (one per protein chain).
    chains: "dict"

    @property
    def min_coverage(self) -> float:
        """The WEAKEST chain's coverage — the interface gate value.

        An interface label needs every partner usable, so one bad chain must drop
        the complex, not be averaged away by a pristine partner. 0.0 if empty.
        """
        if not self.chains:
            return 0.0
        return min(c.coverage for c in self.chains.values())

    @property
    def total_coverage(self) -> float:
        """Valid / all protein residues across all chains (reporting only)."""
        nv = sum(c.n_valid for c in self.chains.values())
        nr = sum(c.n_residues for c in self.chains.values())
        return nv / nr if nr else 0.0

    @property
    def n_protein_chains(self) -> int:
        return len(self.chains)


def _amino_acid_residues(structure, chain_id: Optional[str]) -> List:
    """Model-0 amino-acid residues of ONE chain — matching the supervision export.

    The export (``supervision._select_chain``) is chain-level: it supervises a
    single chain, and ``residue_index`` is that chain's positional index. So
    coverage must be single-chain too, or ``node_valid`` would not align with the
    exported chain's residue order/length (codex). Scoped to ``models[0]`` so a
    multi-model (NMR) input can't silently pull a different model's chain.

    ``chain_id=None`` is allowed only when there is exactly one protein chain (as
    the export requires); otherwise raise — the caller must say which chain.
    """
    chains = structure.models[0].chains
    if chain_id is not None:
        for ch in chains:
            if ch.id == chain_id:
                return [r for r in ch.residues if r.is_amino_acid]
        return []
    protein_chains = [ch for ch in chains if any(r.is_amino_acid for r in ch.residues)]
    if len(protein_chains) > 1:
        raise ValueError(
            "structure_coverage is chain-level (like the supervision export); "
            "pass chain_id for a multi-chain structure"
        )
    if not protein_chains:
        return []
    return [r for r in protein_chains[0].residues if r.is_amino_acid]


def residue_completeness(residues, profile: str = "heavy_coords") -> NDArray:
    """Per-residue boolean: are this residue's required atoms all present?

    ``"heavy_coords"`` requires every atom37 slot the residue type EXPECTS;
    ``"backbone"`` requires only N, CA, C, O. Empty input → empty array.
    """
    if profile not in _COVERAGE_PROFILES:
        raise ValueError(f"unknown coverage profile {profile!r}; {list(_COVERAGE_PROFILES)}")
    if not residues:
        return np.zeros((0,), dtype=bool)
    a = extract_atom37(residues)
    mask, exists = a["mask"], a["exists"]
    if profile == "backbone":
        return mask[:, _BB_SLOTS].sum(axis=1) == len(_BB_SLOTS)
    # heavy_coords: no EXPECTED atom is missing.
    missing = ((exists > 0) & (mask == 0)).sum(axis=1)
    return missing == 0


#: Trustworthiness hazards localizable per residue in phase 2. Unlike
#: completeness (a GATE signal — the presence masks already handle it per label,
#: see the sketch), these corrupt even the backbone, so they DO mask coordinate
#: labels. Phase 2 ships ``altloc``; ``severe_clash`` / ``chirality`` are the
#: next hazards into the same wiring (severe_clash needs the Rust topology).
_TRUST_HAZARDS = ("altloc",)


def residue_trustworthy(residues, hazards=_TRUST_HAZARDS) -> NDArray:
    """Per-residue boolean: is this residue a TRUSTWORTHY coordinate label?

    A residue is untrustworthy if any requested trustworthiness hazard touches
    it. Phase 2 localizes ``altloc`` (alternate locations → the chosen conformer
    is arbitrary; ``residue.conformer_count > 1``). This is the mask that feeds
    :func:`proteon.supervision_mask.apply_residue_trust_mask` — it corrupts the
    coordinate labels (not the sequence: identity is unambiguous). Empty input →
    empty array.

    Aligned to the supervision ``residue_index`` by construction (computed on the
    same amino-acid residue list the export uses).
    """
    unknown = set(hazards) - set(_TRUST_HAZARDS)
    if unknown:
        raise ValueError(f"unknown trust hazards {sorted(unknown)}; {list(_TRUST_HAZARDS)}")
    n = len(residues)
    trust = np.ones((n,), dtype=bool)
    if "altloc" in hazards:
        for i, r in enumerate(residues):
            if getattr(r, "conformer_count", 1) > 1:
                trust[i] = False
    return trust


def _residue_keep_mask(structure, global_indices, chain_id: str) -> NDArray:
    """Per-export-residue bool — True where the residue's GLOBAL ``residue_idx`` is
    NOT in ``global_indices``.

    ``global_indices`` are the topology's ``residue_idx``: 0-based over ALL model-0
    residues in chain→residue order. The export uses the AA-only residues of ONE
    chain, so this re-walks the same model-0 iteration to recover each export
    residue's global index — aligned by construction (the Rust scans build
    ``res_idx`` from the identical ``models[0].chains() → residues()`` walk).
    """
    bad = set(global_indices)
    out = []
    all_idx = 0
    for ch in structure.models[0].chains:
        in_export_chain = ch.id == chain_id
        for r in ch.residues:
            if in_export_chain and r.is_amino_acid:
                out.append(all_idx not in bad)
            all_idx += 1
    return np.array(out, dtype=bool)


def residue_clash_mask(structure, clash_residue_indices, chain_id: str) -> NDArray:
    """Per-residue bool — is each export residue CLASH-FREE? (See
    :func:`_residue_keep_mask`.) ANDs directly into the trust mask."""
    return _residue_keep_mask(structure, clash_residue_indices, chain_id)


def residue_chirality_mask(structure, chirality_residue_indices, chain_id: str) -> NDArray:
    """Per-residue bool — is each export residue L-chirality (not a D outlier)?

    Localizes the chirality hazard like :func:`residue_clash_mask` does clashes:
    a single D / mis-modelled CA centre masks just that residue's coordinate
    labels instead of dropping the whole structure."""
    return _residue_keep_mask(structure, chirality_residue_indices, chain_id)


def _resolve_export_chain_id(structure, chain_id: Optional[str]) -> Optional[str]:
    """The export chain id: the given one, or the single protein chain's id."""
    if chain_id is not None:
        return chain_id
    protein = [
        ch for ch in structure.models[0].chains if any(r.is_amino_acid for r in ch.residues)
    ]
    return protein[0].id if len(protein) == 1 else None


def structure_coverage(
    structure,
    *,
    profile: str = "heavy_coords",
    chain_id: Optional[str] = None,
    report=None,
) -> ResidueCoverage:
    """Per-residue label validity + coverage for one prepared structure.

    ``node_valid[i]`` is whether residue *i* is a USABLE coordinate label — and
    coverage is the fraction that are. With ``report=None`` that is COMPLETENESS
    only (phase 1). When a :class:`PrepReport` is given, ``node_valid`` also
    requires the residue to be TRUSTWORTHY — not an altloc pick, not in a severe
    clash (``report.clash_residue_indices``) — so a *pervasively* clashing or
    altloc-ridden structure gets a LOW coverage and is dropped, while a
    *localized* defect leaves coverage high. The export must then mask those same
    residues (``mask_untrustworthy_coords=True`` with the report) or the kept
    defects become labels.

    Aligned to the supervision ``residue_index`` (model-0 amino-acid residues of
    one chain) so the mask applies directly.
    """
    residues = _amino_acid_residues(structure, chain_id)
    node_valid = residue_completeness(residues, profile)
    if report is not None:
        node_valid = node_valid & residue_trustworthy(residues)  # altloc
        # Clash masking ONLY for SEVERE structures: mild clashes (clashscore ≤ 20)
        # are intentionally tolerated as heavy-coordinate labels, so they must not
        # reduce coverage and drop an otherwise-safe structure (codex). For a
        # severe structure the clashing residues ARE counted invalid, so a
        # localized severe clash stays high-coverage and a pervasive one drops.
        cid = _resolve_export_chain_id(structure, chain_id)
        if cid is not None:
            if getattr(report, "has_severe_clashes", False) and report.clash_residue_indices:
                node_valid = node_valid & residue_clash_mask(
                    structure, report.clash_residue_indices, cid
                )
            # D-chirality CA centres: a coordinate-geometry anomaly, localized like
            # clashes (a single outlier masks its residue, not the whole structure).
            # Gated on the ACTIVE hazard (like has_severe_clashes), so stale indices
            # on a report whose outlier was cleared don't mask (codex).
            if getattr(report, "has_chirality_outliers", False) and report.chirality_residue_indices:
                node_valid = node_valid & residue_chirality_mask(
                    structure, report.chirality_residue_indices, cid
                )
    return ResidueCoverage(
        profile=profile,
        n_residues=len(residues),
        n_valid=int(node_valid.sum()),
        node_valid=node_valid,
    )


def _protein_chain_ids(structure) -> List[str]:
    """Model-0 chain ids that contain at least one amino-acid residue."""
    return [
        ch.id for ch in structure.models[0].chains if any(r.is_amino_acid for r in ch.residues)
    ]


def complex_coverage(structure, *, profile: str = "heavy_coords", report=None) -> ComplexCoverage:
    """Per-chain :class:`ResidueCoverage` for every protein chain — for interface labels.

    One :func:`structure_coverage` per protein chain, sharing the whole-complex
    ``report``. The shared ``report.clash_residue_indices`` is the GLOBAL
    all-model-0-residue index namespace; each chain's :func:`residue_clash_mask`
    maps its residues back to that same global index (never chain-local), so a
    residue clashing only ACROSS an interface is correctly masked in its own
    chain. The clash scan is whole-complex, so inter-chain clashes are already
    included — no special interface handling needed.
    """
    chains = {
        cid: structure_coverage(structure, profile=profile, chain_id=cid, report=report)
        for cid in _protein_chain_ids(structure)
    }
    return ComplexCoverage(chains=chains)
