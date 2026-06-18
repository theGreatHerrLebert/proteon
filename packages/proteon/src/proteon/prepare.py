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

    #: Fully FF-covered, usable structure, no hard failure. Does NOT imply the
    #: minimizer converged — pair with :attr:`PrepReport.converged` for
    #: energy-sensitive work.
    READY = "ready"
    #: Usable protein, but it carries untyped het-groups (heme, other cofactors,
    #: ligands, ions, modified residues) the protein-only FF doesn't cover. The
    #: protein chain itself is well covered, so the structure is still
    #: ``ready`` — but it is NOT :attr:`PrepReport.fully_typed`, so energy-grade
    #: callers that need every atom parameterised should gate on that instead.
    READY_WITH_LIGANDS = "ready_with_ligands"
    #: >50% of non-water atoms have no force-field type (nucleic acid, ligand-only
    #: entry, exotic residues) — not a protein the FF could process.
    NOT_PROTEIN = "not_protein"
    #: Minimization hit a non-finite energy/force (the geometry blew up).
    MINIMIZE_FAILED = "minimize_failed"
    #: A size-significant chunk of atoms IN A POLYMER CHAIN — amino-acid OR
    #: nucleic-acid residues — lack FF types (>10 atoms AND >2% of non-water):
    #: a macromolecule is under-parameterised, so topology/energy is partially
    #: wrong. Catches protein-chain gaps and protein–nucleic-acid complexes
    #: whose nucleic acid the protein-only FF can't type. (Untyped small
    #: cofactors/ligands are :attr:`READY_WITH_LIGANDS`, not this.)
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


def _warn_unrelaxed_reconstruct(report) -> None:
    """Warn when reconstruct added heavy atoms that were not relaxed.

    The rebuilt atoms sit at their template positions and may clash. Harmless
    for global/geometric analysis, but it matters if they fall in active sites,
    interfaces, or feed energy/supervision labels (claudex). Distinguish *why*
    they are unrelaxed (codex): minimization was skipped, vs it ran H-only.
    """
    if report.atoms_reconstructed <= 0 or report.heavy_relaxed:
        return
    if not report.minimized:
        # No minimization ran at all (minimize=False, or it was skipped) — the
        # fix is to enable minimization, not to change constrain_heavy.
        report.warnings.append(
            f"{report.atoms_reconstructed} reconstructed heavy atoms are "
            "unrelaxed (minimization did not run): they sit at template "
            "positions. Enable minimize=True if these atoms matter downstream."
        )
    else:
        # Minimization ran but froze heavy atoms (H-only) — relax them.
        report.warnings.append(
            f"{report.atoms_reconstructed} reconstructed heavy atoms left "
            "unrelaxed (H-only minimization): they sit at template positions. "
            "Use constrain_heavy=False (CHARMM19+EEF1) or an external solvated "
            "minimization if these atoms matter downstream."
        )


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
    #: Whether the minimizer moved HEAVY atoms (vs the default H-only). ``False``
    #: under the default ``constrain_heavy=True``: heavy atoms keep their
    #: experimental coordinates, so :attr:`final_energy` is NOT a heavy-atom
    #: energy minimum — it still carries crystal strain (and any reconstructed
    #: heavy atoms / clashes are unrelaxed). ``True`` only when minimization ran
    #: with heavy atoms free (``constrain_heavy=False``/``None``-for-CHARMM): an
    #: equilibrated structure suitable for energy/MD work. Check this before
    #: trusting ``final_energy``/``components`` as an equilibrium quantity.
    heavy_relaxed: bool = False
    #: Optimizer termination status ("converged_gradient", "line_search_failed",
    #: "max_steps", "numerical_failure", "not_run", …). Empty when minimization
    #: did not run. Distinguishes a real relax from a stall that the bare
    #: ``converged`` bool conflates.
    minimizer_status: str = ""
    n_unassigned_atoms: int = 0
    #: Untyped atoms EXCLUDING water (``n_unassigned_atoms`` counts waters, which
    #: are always untyped under a protein-only FF). Zero iff every protein and
    #: het atom got a force-field type — the basis for :attr:`fully_typed`.
    n_unassigned_nonwater: int = 0
    skipped_no_protein: bool = False
    #: A size-significant chunk of atoms IN A POLYMER CHAIN (amino-acid or
    #: nucleic-acid residues) lack FF types (>10 AND >2% of non-water). A
    #: macromolecule is under-covered — a hard defect. Drives :attr:`status` ==
    #: ``INCOMPLETE_FF``.
    incomplete_ff: bool = False
    #: The protein chain is well covered, but untyped small het-groups
    #: (cofactors, ligands, ions, modified residues) are present. A soft signal:
    #: the structure is still :attr:`ready`, but not :attr:`fully_typed`. Drives
    #: :attr:`status` == ``READY_WITH_LIGANDS``.
    untyped_cofactors: bool = False
    # --- label-safety hazards (for geometric-DL supervision) ---
    #: Heavy-atom steric clashes on the final geometry. A clash left by H-only
    #: minimization (or introduced by a reconstructed atom) silently poisons a
    #: training label, so it is surfaced as a first-class count. See
    #: :attr:`has_heavy_clashes` and :attr:`label_safe`.
    n_heavy_clashes: int = 0
    #: True if the clash count is APPROXIMATE because the topology used the
    #: distance-inferred bond fallback for un-templated residues (ligands /
    #: non-standard). Intra-ligand clashes there can't be told from bonds.
    clash_count_inferred: bool = False
    #: Number of models in the input. Only model 0 is prepared; ``> 1`` (e.g. an
    #: NMR ensemble) means a silent model choice was made — see
    #: :attr:`has_multiple_models`.
    n_models: int = 1
    #: True if any residue carries alternate locations (a conformer was silently
    #: chosen). An arbitrary label decision for the affected residues.
    has_altlocs: bool = False
    #: True if any residue carries a PDB insertion code — a residue-identity /
    #: numbering hazard for ``(chain, resnum)``-keyed (sequence-indexed) labels.
    has_insertion_codes: bool = False
    #: HEAVY atoms missing from standard residues on the final structure (vs
    #: templates). Nonzero only when residues are incomplete AND reconstruction
    #: did not fill them (the supervision default) — a partial coordinate label.
    n_missing_heavy_atoms: int = 0
    #: A non-standard / modified amino-acid residue is present (selenomethionine,
    #: a PTM, the 21st/22nd amino acids) — a residue-identity / typing hazard for
    #: sequence-indexed and energy labels (the heavy coordinates are still real).
    has_nonstandard_residues: bool = False
    #: A metal atom is present — coordination chemistry the protein-only force
    #: field does not model (an energy-label hazard).
    has_metals: bool = False
    #: Broken peptide bonds between consecutive amino acids (missing residues /
    #: physical breaks) — a FALSE sequential-adjacency hazard for graph /
    #: sequence-indexed labels (the present residues' coordinates are fine).
    n_chain_gaps: int = 0
    #: CA centres with non-L (D) chirality — a D-amino acid or a modeling error;
    #: a coordinate-geometry anomaly a standard L-protein pipeline should see.
    n_chirality_outliers: int = 0
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
        if self.untyped_cofactors:
            return PrepStatus.READY_WITH_LIGANDS
        return PrepStatus.READY

    #: Statuses that count as a usable structure (the :attr:`ready` gate).
    _READY_STATUSES = frozenset({PrepStatus.READY, PrepStatus.READY_WITH_LIGANDS})

    @property
    def ready(self) -> bool:
        """True iff prepare produced a usable structure with no hard failure.

        Includes :attr:`PrepStatus.READY_WITH_LIGANDS` — a protein with untyped
        cofactors/ligands is still usable. For an energy-grade gate that also
        requires every atom to be parameterised, use :attr:`fully_typed`.

        Does not imply minimization convergence (check :attr:`converged`),
        protonation correctness, or chemical validity beyond the FF-coverage and
        numerical checks.
        """
        return self.status in self._READY_STATUSES

    @property
    def fully_typed(self) -> bool:
        """True iff :attr:`ready` AND every non-water atom got a force-field type.

        The strict, energy-grade gate. Excludes ``READY_WITH_LIGANDS`` (untyped
        cofactors/ligands) and also a plain ``READY`` structure that still has a
        few untyped atoms below the ``incomplete_ff`` threshold — any untyped
        non-water atom makes the topology partial. Use this when that would
        corrupt an energy or MD calculation; use :attr:`ready` when you just need
        a usable protein structure.
        """
        return self.status == PrepStatus.READY and self.n_unassigned_nonwater == 0

    # --- label-safety contract (geometric-DL supervision) ---
    #
    # `prepare` is step 0 of supervision pipelines; the prepared coordinates and
    # FF assignment BECOME the training labels, so a silent corruption (guessed
    # atom, residual clash, arbitrary altloc/model pick) poisons every example
    # invisibly. These derived signals make that class of error a structured,
    # impossible-to-ignore decision rather than text in `warnings`.

    @property
    def has_heavy_clashes(self) -> bool:
        """Any heavy-atom steric clash on the final geometry (see :attr:`n_heavy_clashes`)."""
        return self.n_heavy_clashes > 0

    @property
    def has_reconstructed_atoms(self) -> bool:
        """Any heavy atoms were fabricated from templates (model-derived, not observed)."""
        return self.atoms_reconstructed > 0

    @property
    def has_chain_gaps(self) -> bool:
        """Broken peptide bonds between consecutive residues (false adjacency)."""
        return self.n_chain_gaps > 0

    @property
    def has_chirality_outliers(self) -> bool:
        """CA centres with non-L (D) chirality — a D-amino acid or modeling error."""
        return self.n_chirality_outliers > 0

    @property
    def has_missing_atoms(self) -> bool:
        """Standard residues are missing heavy atoms (an incomplete coordinate label).

        Nonzero only with reconstruction off (the supervision default); with it
        on, those atoms are filled and flagged as :attr:`has_reconstructed_atoms`
        instead. Either way the structure is not a complete *observed* label.
        """
        return self.n_missing_heavy_atoms > 0

    @property
    def has_untyped_atoms(self) -> bool:
        """Any non-water atom lacks a force-field type."""
        return self.n_unassigned_nonwater > 0

    @property
    def has_multiple_models(self) -> bool:
        """Input had >1 model; only model 0 was prepared (silent selection)."""
        return self.n_models > 1

    @property
    def label_hazards(self) -> List[str]:
        """The label-corrupting hazards that fired, by name (empty iff none).

        The structured replacement for parsing :attr:`warnings`. A consumer that
        tolerates some hazards for a specific label type can inspect this instead
        of the all-or-nothing :attr:`label_safe`.
        """
        h: List[str] = []
        if not self.ready:
            h.append(self.status.value)
        if self.has_reconstructed_atoms:
            h.append("reconstructed_atoms")
        if self.has_missing_atoms:
            h.append("missing_atoms")
        if self.heavy_relaxed:
            # Heavy atoms were minimized: the coordinates are no longer the
            # deposited experimental ones (moved ~0.5 Å). A provenance hazard for
            # coordinate labels — like reconstructed_atoms, it must be explicitly
            # accepted (the repair layer's `relax` records the drift).
            h.append("relaxed_coords")
        if self.has_heavy_clashes:
            h.append("heavy_clashes")
        if self.has_untyped_atoms:
            h.append("untyped_atoms")
        if self.has_nonstandard_residues:
            h.append("nonstandard_residues")
        if self.has_metals:
            h.append("metals")
        if self.has_chirality_outliers:
            h.append("chirality_outliers")
        if self.has_chain_gaps:
            h.append("chain_gaps")
        if self.hydrogens_added == 0:
            # No hydrogens placed (e.g. hydrogens="none"): all-atom / energy
            # labels are unavailable. Not a hazard for heavy-coordinate labels,
            # but it IS why `label_safe` (the all-types gate) is False, so it must
            # be listed — otherwise `only_safe` drops the record with no reason.
            h.append("no_hydrogens")
        if self.has_altlocs:
            h.append("altlocs")
        if self.has_multiple_models:
            h.append("multiple_models")
        if self.has_insertion_codes:
            h.append("insertion_codes")
        return h

    @property
    def label_safe_heavy_coords(self) -> bool:
        """Safe for backbone / heavy-atom coordinate labels.

        Experimentally-observed heavy atoms, sterically sane, with an unambiguous
        conformer and a single model. Does NOT require full FF typing — an
        untyped heme does not corrupt the protein backbone coordinates.
        """
        return (
            self.status != PrepStatus.NOT_PROTEIN
            and self.minimizer_status != "numerical_failure"
            and not self.has_reconstructed_atoms
            and not self.has_missing_atoms
            and not self.has_chirality_outliers
            and not self.heavy_relaxed
            and not self.has_heavy_clashes
            and not self.has_altlocs
            and not self.has_multiple_models
        )

    @property
    def label_safe_all_atom_coords(self) -> bool:
        """:attr:`label_safe_heavy_coords` AND hydrogens were placed.

        Does NOT require ``hydrogens_skipped == 0``: skips are dominated by
        legitimate chemistry (proline has no backbone amide H, chain termini),
        which the count cannot tell from a genuine failure — so it is not a
        label hazard on its own.
        """
        return self.label_safe_heavy_coords and self.hydrogens_added > 0

    @property
    def label_safe_energy(self) -> bool:
        """All-atom coords + full FF typing + clean chemistry — for energy labels.

        Also excludes non-standard residues and metals: their force-field typing
        / coordination chemistry is not modelled, so the energy is unreliable.
        NOTE: protonation / histidine-tautomer correctness is not yet verified
        (a later phase).
        """
        return (
            self.label_safe_all_atom_coords
            and self.fully_typed
            and not self.has_nonstandard_residues
            and not self.has_metals
        )

    @property
    def label_safe_sequence_indexed(self) -> bool:
        """Safe for ``(chain, resnum)``-keyed labels: no residue-identity ambiguity.

        Excludes non-standard / modified residues — they are not a canonical
        sequence token. NOTE: chain-gap false-adjacency detection is a later phase.
        """
        return (
            self.status != PrepStatus.NOT_PROTEIN
            and not self.has_insertion_codes
            and not self.has_multiple_models
            and not self.has_nonstandard_residues
            and not self.has_chain_gaps
        )

    @property
    def label_safe(self) -> bool:
        """The strict gate: safe to use as a training label of ANY supported type.

        The conjunction of every label profile — experimentally-observed,
        sterically sane, fully typed, unambiguous conformer/model, and free of
        residue-identity ambiguity. Use a specific ``label_safe_*`` profile to
        relax this for one label type; inspect :attr:`label_hazards` to see why a
        structure is excluded.
        """
        return self.label_safe_energy and self.label_safe_sequence_indexed

    @property
    def reason(self) -> str:
        """Empty when :attr:`ready`; otherwise a short why-not explanation."""
        s = self.status
        if s in self._READY_STATUSES:
            return ""
        if s == PrepStatus.NOT_PROTEIN:
            return (
                f"not a protein the force field can process "
                f"({self.n_unassigned_atoms} unassigned atoms, >50% of non-water)"
            )
        if s == PrepStatus.MINIMIZE_FAILED:
            return "minimization hit a non-finite energy/force (geometry blew up)"
        return (
            f"incomplete force-field coverage in a polymer chain "
            f"(protein or nucleic acid; {self.n_unassigned_atoms} unassigned "
            f"atoms, >2% of non-water)"
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
    gradient_tolerance: float = 1.0,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
    constrain_heavy: Optional[bool] = True,
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
        gradient_tolerance: Convergence criterion — max per-atom force in
            kcal/mol/A (default 1.0). This is the achievable band for the
            default heavy-atom relaxation: L-BFGS plateaus with a few strained
            atoms keeping the max force in 0.1-1.0, so a tighter 0.1 never
            converges (it burns all 500 steps at the same fold). 1.0 reports
            honest convergence at the same structure (measured: CA-RMSD and
            energy within 0.3% of the 0.1 result, converged 0% -> 93%). Lower it
            for stricter, slower minimization.
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
        constrain_heavy: Whether to freeze heavy atoms during minimization
            (move only hydrogens). Unified default across ``prepare`` /
            ``batch_prepare`` / ``load_and_prepare``. Pick by what you need:

            ``True`` (**H-only**, the default) — freeze heavy atoms, relax only
            the placed hydrogens. Preserves the deposited coordinates exactly
            (CA-RMSD 0), fast, converges. The right choice for ANALYSIS
            (alignment, SASA, DSSP, contacts, ML/supervision features) where
            faithfulness to the experimental structure matters. NOTE this does
            NOT equilibrate the structure: ``final_energy`` still carries crystal
            strain (``heavy_relaxed`` is False), and if ``reconstruct`` added
            heavy atoms they stay at their template positions, unrelaxed.

            ``False`` (**heavy relax**) — also relax heavy atoms. A deeper,
            clash-reduced minimum that settles reconstructed atoms; for
            energy/minimized-structure work. Moves the backbone ~0.5 Å off the
            deposited coordinates. Most appropriate under CHARMM19+EEF1 (its
            implicit solvent screens electrostatics); under AMBER96 (vacuum +
            cutoff) it can distort charged/polar geometry — prefer H-only here,
            or do a restrained/solvated minimization in an MD engine.

            ``None`` (**FF-aware**) — heavy relax for CHARMM19+EEF1, H-only for
            AMBER96 (the pre-unification batch behaviour). A reasonable
            "do the right thing per force field" shortcut for energy work.

            See the prepare subsystem docs for the full decision tree and the
            NMR/membrane/crystal-packing caveats. Check ``report.heavy_relaxed``
            to know which you got, and gate any trust in ``final_energy`` on it.

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
        # one Rust call. `constrain_heavy` (default True = H-only) controls
        # whether heavy atoms move; see the function docstring for the trade-off.
        results = _add_h.batch_prepare(
            [ptr], reconstruct, hydrogens, include_water,
            minimize, minimize_method, minimize_steps, gradient_tolerance, None,
            True,
            ff,
            constrain_heavy,
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
            report.heavy_relaxed = r.get("heavy_relaxed", False)
            report.minimizer_status = r.get("minimizer_status", "")
            report.n_unassigned_atoms = r["n_unassigned_atoms"]
            report.n_unassigned_nonwater = r.get("n_unassigned_nonwater", 0)
            report.skipped_no_protein = r["skipped_no_protein"]
            report.incomplete_ff = r.get("incomplete_ff", False)
            report.untyped_cofactors = r.get("untyped_cofactors", False)
            report.n_heavy_clashes = r.get("n_heavy_clashes", 0)
            report.clash_count_inferred = r.get("clash_count_inferred", False)
            report.n_models = r.get("n_models", 1)
            report.has_altlocs = r.get("has_altlocs", False)
            report.has_insertion_codes = r.get("has_insertion_codes", False)
            report.n_missing_heavy_atoms = r.get("n_missing_heavy_atoms", 0)
            report.has_nonstandard_residues = r.get("has_nonstandard_residues", False)
            report.has_metals = r.get("has_metals", False)
            report.n_chain_gaps = r.get("n_chain_gaps", 0)
            report.n_chirality_outliers = r.get("n_chirality_outliers", 0)
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
            _warn_unrelaxed_reconstruct(report)
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
        # `constrain_heavy` (default True = H-only) controls whether heavy atoms
        # move; see the function docstring for the trade-off.
        results = _add_h.batch_prepare(
            [ptr], False, "none", False,
            True, minimize_method, minimize_steps, gradient_tolerance, None,
            False,
            ff,
            constrain_heavy,
        )
        if results:
            r = _convert_prep_result_to_kj(results[0])
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.components = dict(r.get("components", {}))
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]
            report.minimized = r.get("minimized", False)
            report.heavy_relaxed = r.get("heavy_relaxed", False)
            report.minimizer_status = r.get("minimizer_status", "")
            report.skipped_no_protein = r["skipped_no_protein"]
            report.incomplete_ff = r.get("incomplete_ff", False)
            report.untyped_cofactors = r.get("untyped_cofactors", False)
            report.n_heavy_clashes = r.get("n_heavy_clashes", 0)
            report.clash_count_inferred = r.get("clash_count_inferred", False)
            report.n_models = r.get("n_models", 1)
            report.has_altlocs = r.get("has_altlocs", False)
            report.has_insertion_codes = r.get("has_insertion_codes", False)
            report.n_missing_heavy_atoms = r.get("n_missing_heavy_atoms", 0)
            report.has_nonstandard_residues = r.get("has_nonstandard_residues", False)
            report.has_metals = r.get("has_metals", False)
            report.n_chain_gaps = r.get("n_chain_gaps", 0)
            report.n_chirality_outliers = r.get("n_chirality_outliers", 0)

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
        report.n_unassigned_nonwater = c.get("n_unassigned_nonwater", 0)
        report.skipped_no_protein = c["skipped_no_protein"]
        report.incomplete_ff = c.get("incomplete_ff", False)
        report.untyped_cofactors = c.get("untyped_cofactors", False)
        report.n_heavy_clashes = c.get("n_heavy_clashes", 0)
        report.clash_count_inferred = c.get("clash_count_inferred", False)
        report.n_models = c.get("n_models", 1)
        report.has_altlocs = c.get("has_altlocs", False)
        report.has_insertion_codes = c.get("has_insertion_codes", False)
        report.n_missing_heavy_atoms = c.get("n_missing_heavy_atoms", 0)
        report.has_nonstandard_residues = c.get("has_nonstandard_residues", False)
        report.has_metals = c.get("has_metals", False)
        report.n_chain_gaps = c.get("n_chain_gaps", 0)
        report.n_chirality_outliers = c.get("n_chirality_outliers", 0)
    if report.n_unassigned_atoms > 10:
        report.warnings.append(
            f"{report.n_unassigned_atoms} atoms without force field type "
            "(non-standard residues or ligands)"
        )
    _warn_unrelaxed_reconstruct(report)

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
    gradient_tolerance: float = 1.0,
    n_threads: Optional[int] = None,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
    constrain_heavy: Optional[bool] = True,
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
        gradient_tolerance: Convergence criterion — max per-atom force in
            kcal/mol/A (default 1.0). This is the achievable band for the
            default heavy-atom relaxation: L-BFGS plateaus with a few strained
            atoms keeping the max force in 0.1-1.0, so a tighter 0.1 never
            converges (it burns all 500 steps at the same fold). 1.0 reports
            honest convergence at the same structure (measured: CA-RMSD and
            energy within 0.3% of the 0.1 result, converged 0% -> 93%). Lower it
            for stricter, slower minimization.
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
        constrain_heavy: Whether to freeze heavy atoms during minimization
            (move only hydrogens). Unified with ``prepare`` /
            ``load_and_prepare`` — same default and meaning:
            ``True`` (**H-only**, the default) preserves the experimental
            heavy-atom coordinates exactly (perfect fold, CA-RMSD 0) and relaxes
            only the placed hydrogens — fast, converges, energy correctly-signed.
            The right default for "load and prepare a usable structure".
            ``False`` (**heavy relax**) also relaxes heavy atoms — a deeper
            energy minimum (better for energy/MD work) but moves the backbone
            ~0.5 Å off the deposited structure. NOTE: before unification this was
            the CHARMM19+EEF1 default; pass ``False`` to restore it.
            ``None`` (**FF-aware**) picks per force field: True for AMBER96
            (all-atom AMBER in vacuum has unscreened electrostatics, so full
            minimization gives meaningless numbers), False for CHARMM19+EEF1.

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
            heavy_relaxed=r.get("heavy_relaxed", False),
            minimizer_status=r.get("minimizer_status", ""),
            n_unassigned_atoms=r["n_unassigned_atoms"],
            n_unassigned_nonwater=r.get("n_unassigned_nonwater", 0),
            skipped_no_protein=r["skipped_no_protein"],
            incomplete_ff=r.get("incomplete_ff", False),
            untyped_cofactors=r.get("untyped_cofactors", False),
            n_heavy_clashes=r.get("n_heavy_clashes", 0),
            clash_count_inferred=r.get("clash_count_inferred", False),
            n_models=r.get("n_models", 1),
            has_altlocs=r.get("has_altlocs", False),
            has_insertion_codes=r.get("has_insertion_codes", False),
            n_missing_heavy_atoms=r.get("n_missing_heavy_atoms", 0),
            has_nonstandard_residues=r.get("has_nonstandard_residues", False),
            has_metals=r.get("has_metals", False),
            n_chain_gaps=r.get("n_chain_gaps", 0),
            n_chirality_outliers=r.get("n_chirality_outliers", 0),
        )
        if report.skipped_no_protein:
            report.warnings.append(
                f"skipped: {report.n_unassigned_atoms} atoms have no protein "
                "force-field type (likely nucleic acid, ligand, or non-standard residue)"
            )
        elif report.untyped_cofactors:
            report.warnings.append(
                f"{report.n_unassigned_atoms} atoms without force field type "
                "(untyped cofactors/ligands; protein chain is covered)"
            )
        elif report.n_unassigned_atoms > 10:
            report.warnings.append(
                f"{report.n_unassigned_atoms} atoms without force field type"
            )
        _warn_unrelaxed_reconstruct(report)
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
    constrain_heavy: Optional[bool] = True,
) -> "tuple[object, PrepReport]":
    """Load a structure file and prepare it in one call.

    Args:
        path: Path to PDB or mmCIF file.
        reconstruct: Add missing heavy atoms (default True).
        hydrogens: "backbone", "all", "general", or "none" (default "all").
        minimize: Minimize H positions (default True).
        minimize_method: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Max minimization steps (default 500).
        constrain_heavy: Freeze heavy atoms during minimization. Default
            ``True`` (H-only) preserves experimental coordinates; ``False``
            relaxes heavy atoms; ``None`` is FF-aware. See :func:`prepare`.

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
        constrain_heavy=constrain_heavy,
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
    #: The :class:`~proteon.repair.RepairOutcome` when a ``repair`` policy was
    #: applied via :func:`prepare_for_supervision`; ``None`` otherwise.
    repair: Optional[object] = None

    @property
    def loaded(self) -> bool:
        """True iff the file parsed into a Structure (prepare may still have failed)."""
        return self.error is None

    @property
    def passes_policy(self) -> bool:
        """True iff a repair policy was applied and the structure passes it.

        ``False`` for a load failure or when no ``repair`` policy was used (check
        :attr:`label_safe` in that case).
        """
        return self.repair is not None and self.repair.passes_policy

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

    @property
    def label_safe(self) -> bool:
        """True iff loaded AND safe to use as a geometric-DL training label.

        The label-safety gate across both stages: a load failure is never label
        safe; otherwise forwards :attr:`PrepReport.label_safe` (clean coords,
        fully typed, unambiguous conformer/model, no fabricated atoms). Use a
        ``PrepReport.label_safe_*`` profile via :attr:`report` to relax this per
        label type.
        """
        return self.error is None and self.report is not None and self.report.label_safe

    @property
    def label_hazards(self) -> List[str]:
        """The label-corrupting hazards that fired (empty iff :attr:`label_safe`).

        ``["load_failed"]`` for a parse failure; otherwise forwards
        :attr:`PrepReport.label_hazards`.
        """
        if self.error is not None:
            return ["load_failed"]
        if self.report is not None:
            return self.report.label_hazards
        return []


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


def prepare_for_supervision(
    paths: Sequence,
    *,
    n_threads: Optional[int] = None,
    only_safe: bool = False,
    repair=None,
    **prepare_kwargs,
) -> List[LoadPrepResult]:
    """Load + prepare structures for geometric-DL supervision, label-safe by default.

    The same pipeline as :func:`batch_load_and_prepare`, but with the
    conservative default that matters for training data: **``reconstruct=False``**.
    A reconstructed heavy atom is a model-derived guess (the network would learn
    the reconstruction algorithm's rotamer/loop priors, not the experiment), so
    by default missing atoms are NOT fabricated into labels — incomplete residues
    are left as-is and surfaced via the report. Override with
    ``reconstruct=True`` if you explicitly want completed structures (then those
    atoms are flagged by :attr:`PrepReport.has_reconstructed_atoms` and excluded
    from :attr:`LoadPrepResult.label_safe`).

    Gate each result on :attr:`LoadPrepResult.label_safe` (or a
    ``PrepReport.label_safe_*`` profile for a specific label type) and inspect
    :attr:`LoadPrepResult.label_hazards` for why a structure is excluded. Silent
    label hazards — heavy clashes, altloc ambiguity, multi-model inputs (NMR
    ensembles, only model 0 prepared), insertion codes — all flip ``label_safe``
    to False rather than quietly corrupting a label.

    Args:
        paths: File paths (.pdb, .cif, .mmcif).
        n_threads: Thread count for both stages (``None``/``-1``/``0`` = all).
        only_safe: If True, return ONLY the passing results (label_safe, or — with
            a ``repair`` policy — passes_policy). Default False — every input path
            comes back so you can log/inspect the excluded ones and their hazards.
        repair: An optional :class:`~proteon.repair.RepairPolicy`. When given, the
            structure is prepared with the policy's FIX flags (currently
            ``reconstruct`` for ``missing_atoms``), then re-verified relative to
            the policy's label profile; the outcome is attached as ``res.repair``
            and ``res.passes_policy`` is the gate. Without a policy, ``reconstruct``
            defaults to False (don't fabricate atoms into labels).
        **prepare_kwargs: Forwarded to :func:`batch_prepare`. ``reconstruct``
            defaults to False here (vs True elsewhere); a ``repair`` policy sets
            the fix flags.

    Returns:
        List of :class:`LoadPrepResult`. Same length/order as ``paths`` unless
        ``only_safe=True``.

    Examples:
        >>> for res in proteon.prepare_for_supervision(glob.glob("pdbs/*.pdb")):
        ...     if res.label_safe:
        ...         add_training_example(res.structure)
        ...     else:
        ...         log.info("skip %s: %s", res.path, res.label_hazards)

        >>> policy = proteon.RepairPolicy.coords_only()   # fix missing, accept ligands
        >>> results = proteon.prepare_for_supervision(paths, repair=policy)
        >>> ready = [r.structure for r in results if r.passes_policy]
        >>> print(proteon.RepairSummary.from_results(results))

    Note:
        Per-atom provenance masks (observed / reconstructed) are produced by the
        supervision tensor export, where atoms are indexed (atom37/atom14); this
        preset is the structure-level gate that precedes it.
    """
    if repair is not None:
        return _prepare_with_repair(
            paths, repair, n_threads=n_threads, only_safe=only_safe, **prepare_kwargs
        )
    prepare_kwargs.setdefault("reconstruct", False)
    results = batch_load_and_prepare(paths, n_threads=n_threads, **prepare_kwargs)
    if only_safe:
        return [r for r in results if r.label_safe]
    return results


def _prepare_with_repair(paths, policy, *, n_threads, only_safe, **prepare_kwargs):
    """Apply a RepairPolicy: prepare with its FIX flags, then re-verify per profile.

    Pass 1 prepares H-only (reconstruct per the policy) so clashes are detected
    on faithful coordinates. Pass 2 — only if the policy has ``relax`` — re-runs
    heavy-atom minimization on JUST the structures that actually clash (not the
    whole batch), records the CA drift off the deposited coordinates, and
    re-detects. The verdict is then computed per structure relative to the
    policy's label profile.
    """
    from .repair import RepairOutcome, evaluate

    prepare_kwargs["reconstruct"] = policy.reconstruct
    # The repair layer owns heavy-atom relaxation (via the per-structure `relax`
    # action), so pass 1 is ALWAYS H-only — override any caller `constrain_heavy`
    # so a forwarded `constrain_heavy=False` can't relax the whole batch (codex).
    prepare_kwargs["constrain_heavy"] = True
    # Pass 1: H-only — faithful coords + clash detection.
    results = batch_load_and_prepare(
        [str(p) for p in paths], n_threads=n_threads, **prepare_kwargs
    )

    # Pass 2: relax ONLY the clashy structures (lossy, so never the whole batch).
    drift = {}
    relaxed = set()
    if policy.relax:
        clashy = [
            res for res in results
            if res.loaded and res.report is not None and res.report.has_heavy_clashes
        ]
        if clashy:
            import numpy as np
            from .analysis import batch_extract_ca

            structs = [res.structure for res in clashy]
            ca_before = batch_extract_ca(structs, n_threads=n_threads)
            relax_kwargs = dict(prepare_kwargs)
            relax_kwargs["reconstruct"] = False  # already done in pass 1
            relax_kwargs["constrain_heavy"] = False  # heavy-atom relaxation
            relax_kwargs["minimize"] = True  # the relax pass IS a minimization
            # (override any forwarded minimize=False — codex)
            new_reports = batch_prepare(structs, n_threads=n_threads, **relax_kwargs)
            ca_after = batch_extract_ca(structs, n_threads=n_threads)
            for res, rep, cb, ca in zip(clashy, new_reports, ca_before, ca_after):
                # Pass 2 ran with reconstruct=False, so its report has
                # atoms_reconstructed=0 — but the reconstructed atoms from pass 1
                # are STILL in the structure. Carry the provenance forward so a
                # reconstruct+relax policy still requires reconstructed_atoms to
                # be accepted (codex).
                rep.atoms_reconstructed = res.report.atoms_reconstructed
                res.report = rep
                relaxed.add(id(res))
                if cb.shape == ca.shape and cb.shape[0] > 0:
                    drift[id(res)] = float(
                        np.sqrt(np.mean(np.sum((cb - ca) ** 2, axis=1)))
                    )

    for res in results:
        if not res.loaded or res.report is None:
            # A load/parse failure is a dropped structure — account for it in the
            # outcome (and the corpus summary), not silently skipped (codex).
            res.repair = RepairOutcome(
                profile=policy.profile,
                passes_policy=False,
                dropped_for=["load_failed"],
                remaining_hazards=["load_failed"],
            )
            continue
        res.repair = evaluate(
            res.report, policy,
            reconstruct_applied=policy.reconstruct,
            relax_applied=id(res) in relaxed,
            coords_drift=drift.get(id(res)),
        )

    if only_safe:
        return [r for r in results if r.passes_policy]
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
