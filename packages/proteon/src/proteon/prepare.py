"""Structure preparation pipeline.

Composes fragment reconstruction, hydrogen placement, and energy
minimization into a single reliable workflow.

Functions:
    prepare              — full prep on a single structure
    batch_prepare        — parallel prep on many structures
    load_and_prepare     — load + prep in one call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence


class PrepStatus(str, Enum):
    """Outcome verdict for a prepared structure (see :attr:`PrepReport.status`)."""

    #: Usable structure, no hard failure. Does NOT imply the minimizer converged
    #: — pair with :attr:`PrepReport.converged` for energy-sensitive work.
    READY = "ready"
    #: >50% of non-water atoms have no force-field type (nucleic acid, ligand-only
    #: entry, exotic residues) — not a protein the FF could process.
    NOT_PROTEIN = "not_protein"
    #: Minimization hit a non-finite energy/force (the geometry blew up).
    MINIMIZE_FAILED = "minimize_failed"
    #: Mostly a protein, but a size-significant chunk of atoms lack FF types
    #: (>10 atoms AND >2%), so topology/energy is partially wrong.
    INCOMPLETE_FF = "incomplete_ff"

try:
    import proteon_connector
    _add_h = proteon_connector.py_add_hydrogens
    _ff = proteon_connector.py_forcefield
except ImportError:  # pragma: no cover
    _add_h = None
    _ff = None

# Re-use the same once-per-process AMBER96 warning machinery as forcefield.py.
from .forcefield import _maybe_warn_ff  # noqa: E402


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


# Rust batch_prepare returns energies in kcal/mol (the native unit of the
# AMBER96/CHARMM19 parameters). The rest of the proteon Python API defaults
# to kJ/mol via compute_energy / minimize_hydrogens, so the PrepReport
# dataclass also exposes kJ/mol by default and we convert on the way out.
_KCAL_TO_KJ = 4.184
_VALID_HYDROGEN_MODES = {"backbone", "all", "general", "none"}


def _convert_prep_result_to_kj(r: dict) -> dict:
    """Return a copy of the Rust batch_prepare result dict with energy fields
    converted from kcal/mol to kJ/mol.

    Touches: initial_energy, final_energy, and every component in
    ``components`` (bond_stretch, angle_bend, …). Leaves structural counts
    (minimizer_steps, n_unassigned_atoms, …) unchanged.
    """
    out = dict(r)
    for k in ("initial_energy", "final_energy"):
        if k in out and isinstance(out[k], (int, float)):
            out[k] = out[k] * _KCAL_TO_KJ
    comps = out.get("components")
    if isinstance(comps, dict):
        out["components"] = {
            k: (v * _KCAL_TO_KJ if isinstance(v, (int, float)) else v)
            for k, v in comps.items()
        }
    return out


@dataclass
class PrepReport:
    """Report from structure preparation.

    Attributes:
        atoms_reconstructed: Heavy atoms added by fragment reconstruction.
        hydrogens_added: Hydrogen atoms placed.
        hydrogens_skipped: Residues where H placement was skipped (e.g., missing backbone).
        initial_energy: Total energy before minimization (kJ/mol).
        final_energy: Total energy after minimization (kJ/mol).
        components: Per-component energy breakdown at the post-minimization
            geometry, in the same units as ``initial_energy`` /
            ``final_energy``. Keys: ``bond_stretch``, ``angle_bend``,
            ``torsion``, ``improper_torsion``, ``vdw``, ``electrostatic``,
            ``solvation``. All zero if ``minimize=False`` was passed, or if
            ``skipped_no_protein`` is True. Populated from the minimizer's
            final energy state — does NOT require a separate
            ``compute_energy`` call.
        minimizer_steps: Number of minimization steps taken.
        converged: Whether the minimizer converged. Only meaningful when
            ``skipped_no_protein`` is False — see that field's docs.
        n_unassigned_atoms: Atoms without force field type assignment.
        skipped_no_protein: True if the structure was skipped by the
            minimizer because more than half of its atoms have no protein
            force-field type assignment (e.g. nucleic acids, ligand-only
            entries, exotic non-standard residues). When True, ``converged``
            is always False because no minimization ran — distinguish this
            case from real convergence failures by checking
            ``skipped_no_protein`` first.
        warnings: List of warning messages.
    """
    atoms_reconstructed: int = 0
    hydrogens_added: int = 0
    hydrogens_skipped: int = 0
    initial_energy: float = 0.0
    final_energy: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    minimizer_steps: int = 0
    converged: bool = False
    #: Whether the minimization branch actually ran (vs skipped: no hydrogens
    #: added, ``minimize=False``, or ``skipped_no_protein``). ``relax_ok`` needs
    #: this to tell "did not relax" from "relaxed and converged".
    minimized: bool = False
    #: Optimizer termination status ("converged_gradient", "line_search_failed",
    #: "max_steps", "numerical_failure", "not_run", …). Empty when minimization
    #: did not run. Distinguishes a real relax from a stall that the bare
    #: ``converged`` bool conflates.
    minimizer_status: str = ""
    n_unassigned_atoms: int = 0
    skipped_no_protein: bool = False
    #: Mostly a protein, but a size-significant chunk of atoms lack FF types
    #: (>10 AND >2% of non-water). Drives :attr:`status` == ``INCOMPLETE_FF``.
    incomplete_ff: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> PrepStatus:
        """Single readiness verdict, derived from the per-step fields.

        A **hard-failure gate**: ``READY`` means a usable structure was produced
        and nothing failed — NOT that the minimizer converged. Convergence
        quality (``max_steps`` / line-search stalls) is a separate axis: a
        ``READY`` structure may not be at an energy minimum. For energy-sensitive
        work, gate on both::

            if report.ready and report.converged:
                ...
        """
        if self.skipped_no_protein:
            return PrepStatus.NOT_PROTEIN
        if self.minimizer_status == "numerical_failure":
            return PrepStatus.MINIMIZE_FAILED
        if self.incomplete_ff:
            return PrepStatus.INCOMPLETE_FF
        return PrepStatus.READY

    @property
    def ready(self) -> bool:
        """True iff prepare produced a usable structure with no hard failure.

        See :attr:`status`. Does not imply minimization convergence (check
        :attr:`converged`), protonation correctness, or chemical validity beyond
        the FF-coverage and numerical checks.
        """
        return self.status == PrepStatus.READY

    @property
    def reason(self) -> str:
        """Empty when :attr:`ready`; otherwise a short why-not explanation."""
        s = self.status
        if s == PrepStatus.READY:
            return ""
        if s == PrepStatus.NOT_PROTEIN:
            return (
                f"not a protein the force field can process "
                f"({self.n_unassigned_atoms} unassigned atoms, >50% of non-water)"
            )
        if s == PrepStatus.MINIMIZE_FAILED:
            return "minimization hit a non-finite energy/force (geometry blew up)"
        return (
            f"incomplete force-field coverage "
            f"({self.n_unassigned_atoms} unassigned atoms, >2% of non-water)"
        )

    def __repr__(self) -> str:
        lines = [
            f"PrepReport(",
            f"  reconstructed={self.atoms_reconstructed} heavy atoms",
            f"  hydrogens={self.hydrogens_added} added, {self.hydrogens_skipped} skipped",
        ]
        if self.skipped_no_protein:
            lines.append(
                f"  skipped_no_protein=True ({self.n_unassigned_atoms} unassigned atoms)"
            )
        else:
            lines.append(
                f"  energy={self.initial_energy:.1f} -> {self.final_energy:.1f} kJ/mol"
            )
            lines.append(
                f"  minimizer={self.minimizer_steps} steps, converged={self.converged}"
            )
            if self.n_unassigned_atoms > 0:
                lines.append(f"  unassigned_atoms={self.n_unassigned_atoms}")
        if self.warnings:
            lines.append(f"  warnings={self.warnings}")
        verdict = f"  ready={self.ready} (status={self.status.value})"
        if not self.ready:
            verdict += f": {self.reason}"
        lines.append(verdict)
        lines.append(")")
        return "\n".join(lines)


def _normalize_hydrogens_mode(hydrogens: str, report: PrepReport) -> str:
    mode = hydrogens.lower()
    if mode not in _VALID_HYDROGEN_MODES:
        report.warnings.append(f"Unknown hydrogens option '{hydrogens}', skipping")
        return "none"
    return mode


def prepare(
    structure,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    include_water: bool = False,
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
    gradient_tolerance: float = 0.1,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
) -> PrepReport:
    """Prepare a structure for downstream analysis or simulation.

    Pipeline: [strip H] -> reconstruct missing atoms -> place hydrogens
    -> minimize H positions.

    Args:
        structure: Proteon Structure object (modified in place).
        reconstruct: Add missing heavy atoms from fragment templates (default True).
        hydrogens: Hydrogen placement strategy:
            "backbone" — backbone amide N-H only
            "all"      — backbone + sidechain (standard AA, default)
            "general"  — all atoms including ligands and non-standard residues
            "none"     — skip hydrogen placement
        include_water: Place H on water molecules (only with hydrogens="general").
        minimize: Minimize hydrogen positions after placement (default True).
        minimize_method: Minimizer: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Maximum minimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
        strip_hydrogens: Remove all pre-existing H/D atoms before placement
            (default True). The default rescues structures with externally-
            placed hydrogens (NMR ensembles, deposited X-ray H, upstream
            protonators) whose positions are off the MM minimum and otherwise
            prevent LBFGS convergence within ``gradient_tolerance``. Set to
            False to retain experimental H positions when their provenance
            is trusted. See batch_prepare docstring for the rescue analysis.
        ff: Force field used by the topology builder and the minimizer.
            ``"charmm19_eef1"`` (default) is the **validated production
            path** — used by the 50K battle test and the fold-preservation
            benchmark. ``"amber96"`` has oracle-validated single-point
            energy parity against OpenMM at NoCutoff, but the preparation
            path uses proteon's default cutoff policy and is still the
            secondary workflow. Emits a UserWarning once per process.

    Returns:
        PrepReport with preparation statistics.

    Examples:
        >>> import proteon
        >>> s = proteon.load("structure.pdb")
        >>> report = proteon.prepare(s)
        >>> print(report)
    """
    ptr = _get_ptr(structure)
    report = PrepReport()
    _maybe_warn_ff(ff)
    hydrogens = _normalize_hydrogens_mode(hydrogens, report)

    # Step 0: Optionally strip existing hydrogens.
    if strip_hydrogens:
        # Use the batch path so strip+reconstruct+place+minimize all run in
        # one Rust call. Single-structure prepare is documented as a
        # hydrogen-placement / hydrogen-minimization workflow, so freeze
        # heavy atoms explicitly even on CHARMM.
        results = _add_h.batch_prepare(
            [ptr], reconstruct, hydrogens, include_water,
            minimize, minimize_method, minimize_steps, gradient_tolerance, None,
            True,
            ff,
            True,
        )
        if results:
            r = _convert_prep_result_to_kj(results[0])
            report.atoms_reconstructed = r["atoms_reconstructed"]
            report.hydrogens_added = r["hydrogens_added"]
            report.hydrogens_skipped = r["hydrogens_skipped"]
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.components = dict(r.get("components", {}))
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]
            report.minimized = r.get("minimized", False)
            report.minimizer_status = r.get("minimizer_status", "")
            report.n_unassigned_atoms = r["n_unassigned_atoms"]
            report.skipped_no_protein = r["skipped_no_protein"]
            report.incomplete_ff = r.get("incomplete_ff", False)
            if report.skipped_no_protein:
                report.warnings.append(
                    f"skipped: {report.n_unassigned_atoms} atoms have no "
                    "protein force-field type (likely nucleic acid, ligand, "
                    "or non-standard residue)"
                )
            elif report.n_unassigned_atoms > 10:
                report.warnings.append(
                    f"{report.n_unassigned_atoms} atoms without force field type "
                    "(non-standard residues or ligands)"
                )
        return report

    # Step 1: Reconstruct missing heavy atoms
    if reconstruct:
        report.atoms_reconstructed = _add_h.reconstruct_fragments(ptr)

    # Step 2: Place hydrogens
    if hydrogens == "backbone":
        added, skipped = _add_h.place_peptide_hydrogens(ptr)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped
    elif hydrogens == "all":
        added, skipped = _add_h.place_all_hydrogens(ptr)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped
    elif hydrogens == "general":
        added, skipped = _add_h.place_general_hydrogens(ptr, include_water)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped

    # Step 3: Minimize hydrogen positions
    # Use batch_prepare for a single structure so coords are applied back in Rust.
    if minimize and report.hydrogens_added > 0:
        # Run minimization via the Rust batch path (applies coords in-place).
        # Keep this hydrogen-only to match the public prepare() contract.
        results = _add_h.batch_prepare(
            [ptr], False, "none", False,
            True, minimize_method, minimize_steps, gradient_tolerance, None,
            False,
            ff,
            True,
        )
        if results:
            r = _convert_prep_result_to_kj(results[0])
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.components = dict(r.get("components", {}))
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]
            report.minimized = r.get("minimized", False)
            report.minimizer_status = r.get("minimizer_status", "")
            report.skipped_no_protein = r["skipped_no_protein"]
            report.incomplete_ff = r.get("incomplete_ff", False)

    # Step 4: FF coverage + readiness flags. Source them from the SAME Rust
    # prepare path the default (strip_hydrogens) branch uses, via a coverage-only
    # no-op call (no strip / reconstruct / placement / minimize — just build
    # topology and compute the flags). This keeps skipped_no_protein /
    # incomplete_ff correct on this legacy path too, instead of leaving the
    # readiness verdict blind when minimization is skipped.
    cov = _add_h.batch_prepare(
        [ptr], False, "none", False, False, "lbfgs", 0, gradient_tolerance,
        None, False, ff, None,
    )
    if cov:
        c = cov[0]
        report.n_unassigned_atoms = c.get("n_unassigned_atoms", 0)
        report.skipped_no_protein = c["skipped_no_protein"]
        report.incomplete_ff = c.get("incomplete_ff", False)
    if report.n_unassigned_atoms > 10:
        report.warnings.append(
            f"{report.n_unassigned_atoms} atoms without force field type "
            "(non-standard residues or ligands)"
        )

    return report


def batch_prepare(
    structures: Sequence,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    include_water: bool = False,
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
    gradient_tolerance: float = 0.1,
    n_threads: Optional[int] = None,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
    constrain_heavy: Optional[bool] = None,
) -> List[PrepReport]:
    """Prepare many structures in parallel (Rust + rayon, zero GIL).

    Each structure is modified in place. Full pipeline runs in Rust:
    [optional strip H] -> reconstruct -> place H -> minimize H,
    parallelized across structures.

    Args:
        structures: List of proteon Structure objects.
        reconstruct: Add missing heavy atoms (default True).
        hydrogens: "backbone", "all", "general", or "none" (default "all").
        include_water: Place H on water (only with hydrogens="general").
        minimize: Minimize H positions (default True).
        minimize_method: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Max minimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.
        strip_hydrogens: Remove all pre-existing H/D atoms before placement
            (default True). The default rescues structures with externally-
            placed hydrogens (NMR ensembles, deposited X-ray H, upstream
            protonators) whose positions are off the MM force-field minimum
            and would otherwise prevent LBFGS from converging within
            ``gradient_tolerance``. On the 50K benchmark this raised the
            convergence rate from 169/200 to 199/200 and cut wall time ~3x
            (stragglers stop burning the LBFGS step cap). Set to False to
            retain experimental H positions when their provenance is trusted.
        ff: Force field used by the topology builder and the minimizer.
            ``"charmm19_eef1"`` (default) is the **validated production
            path** — used by the 50K battle test and the fold-preservation
            benchmark. ``"amber96"`` has oracle-validated single-point
            energy parity against OpenMM at NoCutoff, but the preparation
            path uses proteon's default cutoff policy and is still the
            secondary workflow. Emits a UserWarning once per process.
        constrain_heavy: Whether to freeze heavy atoms during minimization.
            ``None`` (default) uses the FF-aware default: True for AMBER96
            (H-only minimization is the intended pattern — all-atom AMBER
            in vacuum has unscreened electrostatic issues, so full minimization
            gives meaningless numbers), False for CHARMM19+EEF1 (polar-H
            united-atom with inflated carbon radii needs heavy-atom relaxation
            for correctly-signed totals). Pass ``True`` or ``False`` to
            override the default explicitly. Primarily useful for testing,
            profiling, or when you specifically want to preserve
            experimentally-determined heavy-atom geometry.

    Returns:
        List of PrepReport, one per structure.
    """
    _maybe_warn_ff(ff)
    ptrs = [_get_ptr(s) for s in structures]
    raw_results = _add_h.batch_prepare(
        ptrs, reconstruct, hydrogens, include_water,
        minimize, minimize_method, minimize_steps, gradient_tolerance, n_threads,
        strip_hydrogens,
        ff,
        constrain_heavy,
    )
    reports = []
    for raw in raw_results:
        r = _convert_prep_result_to_kj(raw)
        report = PrepReport(
            atoms_reconstructed=r["atoms_reconstructed"],
            hydrogens_added=r["hydrogens_added"],
            hydrogens_skipped=r["hydrogens_skipped"],
            initial_energy=r["initial_energy"],
            final_energy=r["final_energy"],
            components=dict(r.get("components", {})),
            minimizer_steps=r["minimizer_steps"],
            converged=r["converged"],
            minimized=r.get("minimized", False),
            minimizer_status=r.get("minimizer_status", ""),
            n_unassigned_atoms=r["n_unassigned_atoms"],
            skipped_no_protein=r["skipped_no_protein"],
            incomplete_ff=r.get("incomplete_ff", False),
        )
        if report.skipped_no_protein:
            report.warnings.append(
                f"skipped: {report.n_unassigned_atoms} atoms have no protein "
                "force-field type (likely nucleic acid, ligand, or non-standard residue)"
            )
        elif report.n_unassigned_atoms > 10:
            report.warnings.append(
                f"{report.n_unassigned_atoms} atoms without force field type"
            )
        reports.append(report)
    return reports


def load_and_prepare(
    path: str,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
) -> "tuple[object, PrepReport]":
    """Load a structure file and prepare it in one call.

    Args:
        path: Path to PDB or mmCIF file.
        reconstruct: Add missing heavy atoms (default True).
        hydrogens: "backbone", "all", "general", or "none" (default "all").
        minimize: Minimize H positions (default True).
        minimize_method: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Max minimization steps (default 500).

    Returns:
        (structure, PrepReport) tuple.
    """
    from .io import load
    structure = load(path)
    report = prepare(
        structure,
        reconstruct=reconstruct,
        hydrogens=hydrogens,
        minimize=minimize,
        minimize_method=minimize_method,
        minimize_steps=minimize_steps,
    )
    return structure, report


@dataclass
class LoadPrepResult:
    """One aligned record per input path from :func:`batch_load_and_prepare`.

    Collapses the pipeline's two failure modes — a file that never loaded and a
    structure that loaded but did not prepare cleanly — into a single
    :attr:`ready` decision, so the load->prepare loop is one branch::

        for res in proteon.batch_load_and_prepare(paths):
            if res.ready:
                use(res.structure)
            else:
                log(res.path, res.reason)

    Attributes:
        path: The input path this record corresponds to (always set; the result
            list is the same length and order as the input ``paths``).
        structure: The loaded (and prepared) Structure, or ``None`` if the file
            failed to load.
        report: The :class:`PrepReport`, or ``None`` if the file failed to load
            (prepare never ran).
        error: A short load-failure message, or ``None`` if the file loaded.
    """

    path: str
    structure: Optional[object] = None
    report: Optional[PrepReport] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        """True iff the file parsed into a Structure (prepare may still have failed)."""
        return self.error is None

    @property
    def ready(self) -> bool:
        """True iff the file loaded AND prepared with no hard failure.

        The single trust gate across both pipeline stages. Does not imply
        minimizer convergence — see :attr:`PrepReport.converged`.
        """
        return self.error is None and self.report is not None and self.report.ready

    @property
    def reason(self) -> str:
        """Empty when :attr:`ready`; else why-not, spanning both stages.

        A load failure reports ``"load failed: ..."``; a structure that loaded
        but did not prepare cleanly forwards :attr:`PrepReport.reason`.
        """
        if self.error is not None:
            return f"load failed: {self.error}"
        if self.report is not None:
            return self.report.reason
        return ""


def batch_load_and_prepare(
    paths: Sequence,
    *,
    n_threads: Optional[int] = None,
    **prepare_kwargs,
) -> List[LoadPrepResult]:
    """Load many files and prepare them, returning one verdict per input path.

    Both stages run in parallel in Rust (rayon, GIL released): a tolerant
    parallel load, then :func:`batch_prepare` over the survivors. Files that
    fail to load are not dropped silently — they come back as a
    :class:`LoadPrepResult` with ``loaded=False`` and a load error, so the
    returned list is always the same length and order as ``paths``.

    This is the batch counterpart of :func:`load_and_prepare`, and the
    recommended entry point for archive-scale ingestion: ``res.ready`` is a
    single decision covering both load and prepare failures.

    Args:
        paths: File paths (.pdb, .cif, .mmcif).
        n_threads: Thread count for both the load and the prepare stage.
            ``None`` / ``-1`` / ``0`` = all cores (default); a positive integer
            = exactly that many threads.
        **prepare_kwargs: Forwarded verbatim to :func:`batch_prepare`
            (``reconstruct``, ``hydrogens``, ``minimize``, ``minimize_method``,
            ``minimize_steps``, ``gradient_tolerance``, ``strip_hydrogens``,
            ``ff``, ``constrain_heavy``, …).

    Returns:
        List of :class:`LoadPrepResult`, one per input path, in input order.

    Examples:
        >>> results = proteon.batch_load_and_prepare(glob.glob("pdbs/*.pdb"))
        >>> ready = [r.structure for r in results if r.ready]
        >>> for r in results:
        ...     if not r.ready:
        ...         print(r.path, "->", r.reason)
    """
    from .io import batch_load_tolerant

    str_paths = [str(p) for p in paths]
    # Stage 1: tolerant parallel load. Survivors carry their original index;
    # absent indices are load failures.
    results = [
        LoadPrepResult(path=p, error="could not parse file as PDB or mmCIF")
        for p in str_paths
    ]
    loaded = batch_load_tolerant(str_paths, n_threads=n_threads)  # [(idx, Structure)]
    structures = []
    positions = []
    for orig_idx, structure in loaded:
        results[orig_idx].structure = structure
        results[orig_idx].error = None
        structures.append(structure)
        positions.append(orig_idx)

    # Stage 2: parallel prepare over the survivors only.
    if structures:
        reports = batch_prepare(structures, n_threads=n_threads, **prepare_kwargs)
        for pos, report in zip(positions, reports):
            results[pos].report = report

    return results


# ---------------------------------------------------------------------------
# AMBER96 histidine tautomer normalisation
# ---------------------------------------------------------------------------
#
# Background: AMBER96 in OpenMM ships three histidine residue templates with
# different per-atom partial charges:
#   HID  δ-tautomer  (Hδ1 only,  neutral)
#   HIE  ε-tautomer  (Hε2 only,  neutral)
#   HIP  protonated  (both Hs,   +1 charge)
#
# PDBFixer's `addMissingHydrogens(7.0)` keeps the residue name "HIS" but
# adds the H atoms in the geometrically/electrostatically appropriate
# positions. Without renaming, proteon string-matches "HIS" against the
# (single, HIP-charge) template — every histidine in every input gets
# the wrong charges, producing a systematic 7-12% AMBER96-vs-OpenMM
# energy drift on every PDB containing histidines (issue #60).
#
# `normalize_histidine_tautomers` reads a PDBFixer-prepared PDB,
# inspects which of HD1 / HE2 each HIS residue has, and writes a
# new PDB with the residue name updated. proteon's existing
# residue-name-based template lookup then picks up the correct
# AMBER96 charges automatically (per the data added in PR #62).
#
# The function operates on PDB-text in/out (no proteon Structure
# mutation API needed, no Rust changes). The renamed PDB is fed
# back to `proteon.load` exactly like the original.

import re as _re
from pathlib import Path as _Path

# Column slices (PDB ATOM record, fixed-width per the PDB v3.30 spec):
#   cols  1-6   record name   ("ATOM  " / "HETATM")
#   cols 13-16  atom name
#   cols 17     altloc indicator
#   cols 18-20  residue name (3 chars, right-justified into the 4-char field 18-21)
#   cols 22     chain id
#   cols 23-26  residue sequence number
# Python 0-indexed slices (end-exclusive):
_ATOM_NAME_SLICE = slice(12, 16)
_RESNAME_SLICE = slice(17, 20)
_CHAIN_SLICE = slice(21, 22)
_RESSEQ_SLICE = slice(22, 26)


def _residue_key(line: str) -> tuple:
    """(chain, residue_seq, insertion_code) — uniquely identifies a residue
    across all of its ATOM lines."""
    return (
        line[_CHAIN_SLICE],
        line[_RESSEQ_SLICE].strip(),
        line[26:27] if len(line) > 26 else " ",
    )


def _is_atom(line: str) -> bool:
    return line.startswith(("ATOM  ", "HETATM"))


def _is_his(line: str) -> bool:
    return _is_atom(line) and line[_RESNAME_SLICE].strip() == "HIS"


def _classify_histidine(atom_names: set[str]) -> str:
    """Return target residue name based on which Hs are present.

    HD1 + HE2 → HIP   (+1 charge tautomer)
    HD1 only  → HID   (δ tautomer)
    HE2 only  → HIE   (ε tautomer)
    neither   → HIS   (no rename — caller should warn)
    """
    has_hd1 = "HD1" in atom_names
    has_he2 = "HE2" in atom_names
    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    if has_he2:
        return "HIE"
    return "HIS"


def normalize_histidine_tautomers(
    in_path: str | _Path,
    out_path: str | _Path | None = None,
) -> dict[str, int]:
    """Rewrite a PDB so each HIS residue carries the correct AMBER96 tautomer name.

    Walks the ATOM records, groups by (chain, resseq, icode), and for each
    HIS residue inspects which of HD1 / HE2 are present. The residue name
    is then updated in place (still 3 characters, fits the PDB column
    layout) so downstream proteon.load picks up the correct AMBER96
    template via the data added in PR #62.

    Args:
        in_path:  Path to a PDB file (typically PDBFixer-prepared with
                  `addMissingHydrogens(7.0)` so HD1/HE2 are present).
        out_path: Where to write the renamed PDB. If None, write to a sibling
                  file with `.histaut.pdb` suffix.

    Returns:
        Dict of {original_name: count_after_rename} aggregated across all
        HIS residues, e.g. {"HIS": 0, "HID": 3, "HIE": 1, "HIP": 0}. The
        count for "HIS" reports residues that could NOT be classified
        (no HD1, no HE2 — caller should warn or skip).

    The function is idempotent: running it on a PDB that already has
    HID/HIE/HIP residue names is a no-op for those residues and only
    affects remaining HIS residues.
    """
    in_path = _Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(".histaut.pdb")
    else:
        out_path = _Path(out_path)

    text = in_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # First pass: walk ATOM lines, group HIS atoms by residue key,
    # collect atom names per residue.
    his_atoms: dict[tuple, set[str]] = {}
    for line in lines:
        if _is_his(line):
            key = _residue_key(line)
            his_atoms.setdefault(key, set()).add(
                line[_ATOM_NAME_SLICE].strip()
            )

    # Decide tautomer per residue.
    his_targets: dict[tuple, str] = {
        key: _classify_histidine(atoms) for key, atoms in his_atoms.items()
    }

    # Second pass: rewrite ATOM lines with new resname where applicable.
    out_lines: list[str] = []
    counts = {"HIS": 0, "HID": 0, "HIE": 0, "HIP": 0}
    for line in lines:
        if _is_his(line):
            key = _residue_key(line)
            target = his_targets[key]
            if target != "HIS":
                # Replace the 3-char resname in cols 17-19 (slice 17:20).
                # New name is exactly 3 chars too — no width change, no
                # column drift downstream.
                new_line = line[:_RESNAME_SLICE.start] + target + line[_RESNAME_SLICE.stop:]
                out_lines.append(new_line)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    # Final per-residue counts (one entry per HIS residue, not per atom).
    for target in his_targets.values():
        counts[target] += 1

    out_path.write_text("".join(out_lines), encoding="utf-8")
    return counts
