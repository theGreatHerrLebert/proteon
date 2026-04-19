"""Canonical failure classes for corpus prep + export pipelines.

Roadmap Section 7 spells out 10 stable classes that every failure
record should use. Stable names matter: dataset quality trends are
only measurable if "missing_required_atoms" means the same thing
today and six months from now.

One shared `classify_exception(exc)` function maps a raised
exception to its class so every layer emits consistent labels.
Individual emitters may also pass `failure_class=...` explicitly
when they already know (e.g. a forcefield parameter lookup miss
is unambiguously `forcefield_parameterization_error` — no need to
re-derive from the exception message).

Keep the classifier boring and string-pattern driven. Don't wire it
into exception type hierarchies — that couples the taxonomy to
proteon internals and to upstream library exception classes we
don't control.
"""

from __future__ import annotations

from typing import Literal

# Canonical values. Update the `FailureClass` type alias if this list
# changes; downstream consumers rely on it as a closed enumeration.
PARSE_ERROR: str = "parse_error"
UNSUPPORTED_CHEMISTRY: str = "unsupported_chemistry"
MISSING_REQUIRED_ATOMS: str = "missing_required_atoms"
RESIDUE_MAPPING_ERROR: str = "residue_mapping_error"
HYDROGEN_PLACEMENT_ERROR: str = "hydrogen_placement_error"
FORCEFIELD_PARAMETERIZATION_ERROR: str = "forcefield_parameterization_error"
MINIMIZATION_NONCONVERGENCE: str = "minimization_nonconvergence"
NUMERICAL_INSTABILITY: str = "numerical_instability"
POSTPREP_QUALITY_FAILURE: str = "postprep_quality_failure"
INTERNAL_PIPELINE_ERROR: str = "internal_pipeline_error"

FailureClass = Literal[
    "parse_error",
    "unsupported_chemistry",
    "missing_required_atoms",
    "residue_mapping_error",
    "hydrogen_placement_error",
    "forcefield_parameterization_error",
    "minimization_nonconvergence",
    "numerical_instability",
    "postprep_quality_failure",
    "internal_pipeline_error",
]

ALL_FAILURE_CLASSES: tuple[str, ...] = (
    PARSE_ERROR,
    UNSUPPORTED_CHEMISTRY,
    MISSING_REQUIRED_ATOMS,
    RESIDUE_MAPPING_ERROR,
    HYDROGEN_PLACEMENT_ERROR,
    FORCEFIELD_PARAMETERIZATION_ERROR,
    MINIMIZATION_NONCONVERGENCE,
    NUMERICAL_INSTABILITY,
    POSTPREP_QUALITY_FAILURE,
    INTERNAL_PIPELINE_ERROR,
)


def classify_exception(exc: BaseException) -> str:
    """Map an exception to one of the canonical failure classes.

    The classifier is deliberately conservative: if the message
    doesn't strongly suggest one of the specific classes, it falls
    back to `internal_pipeline_error`. That keeps false positives
    down — a misclassified failure is worse than a "bucket unknown"
    one because it skews quality reports.

    Pattern matching uses the lowercased exception message plus
    selected exception-type hints. The full class list + rationale
    lives in devdocs/GEOMETRIC_DL_INFRA_ROADMAP.md Section 7.
    """
    message = str(exc).lower()

    # Parse-layer errors usually surface before this ever runs (raw
    # intake filters), but if a caller passes pre-parsed garbage
    # structures we catch the common signatures here.
    if any(
        s in message
        for s in (
            "malformed",
            "unexpected token",
            "invalid pdb",
            "failed to parse",
            "mmcif parse",
        )
    ):
        return PARSE_ERROR

    # Missing required atoms: proteon's supervision extractors raise
    # messages referencing CA/N/C/CB when the geometry step can't
    # proceed. Also covers the chain-level protein-requirement guard.
    if (
        "missing" in message
        and any(atom in message for atom in ("ca", "n", "c", "cb", "backbone", "atom"))
    ):
        return MISSING_REQUIRED_ATOMS
    if "requires a protein chain" in message:
        return MISSING_REQUIRED_ATOMS

    # Residue identity / mapping issues. KeyError into AA tables or
    # unknown three-letter codes land here.
    if (
        isinstance(exc, KeyError)
        or "unknown residue" in message
        or "residue name" in message
        or "residue_mapping" in message
        or "three-letter" in message
    ):
        return RESIDUE_MAPPING_ERROR

    # Explicit markers from the prep / hydrogen / forcefield / min
    # stages. Exact substrings are worth more than fuzzy keyword
    # matches since the emitters control them.
    if "hydrogen" in message and ("place" in message or "failed" in message):
        return HYDROGEN_PLACEMENT_ERROR
    if "force field" in message or "forcefield" in message or "parameteriz" in message:
        return FORCEFIELD_PARAMETERIZATION_ERROR
    if (
        "minimization" in message
        and ("did not converge" in message or "nonconvergence" in message or "diverged" in message)
    ):
        return MINIMIZATION_NONCONVERGENCE

    # Numerical issues: NaN / Inf / overflow in the geometry or
    # supervision paths. numpy raises FloatingPointError when seterr
    # is active; also detect via message.
    if isinstance(exc, FloatingPointError):
        return NUMERICAL_INSTABILITY
    if "nan" in message or "inf" in message or "overflow" in message:
        return NUMERICAL_INSTABILITY

    # Post-prep quality failures: set explicitly by the QC layer.
    if "quality" in message and ("rejected" in message or "failure" in message):
        return POSTPREP_QUALITY_FAILURE

    # Unsupported chemistry (non-standard residues, unresolved
    # modified residues, etc.) — explicit marker expected.
    if "unsupported" in message or "non-standard" in message or "modified residue" in message:
        return UNSUPPORTED_CHEMISTRY

    return INTERNAL_PIPELINE_ERROR
