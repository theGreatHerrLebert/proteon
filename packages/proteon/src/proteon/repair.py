"""The remediation layer: decide, per label hazard, how to act — then re-verify.

P1's `label_safe` / `label_hazards` are a *gate* (is this structure safe as a
training label, and why not). `RepairPolicy` is the *decision* layer on top: a
declarative, per-hazard rule set — ``fix`` / ``accept`` / ``drop`` — applied to
the structures that don't pass, with the result re-verified relative to a chosen
label profile.

    detect (P1) -> decide (rules) -> repair -> re-verify -> report

Design (claudex-reviewed, see devdocs/REPAIR_POLICY_DESIGN.md):

- A policy targets a label PROFILE (``heavy_coords`` / ``all_atom_coords`` /
  ``energy`` / ``sequence_indexed``). Hazards only matter relative to the
  intended label, so ``accept`` is scoped and can't leak into a context where it
  is invalid.
- The first cut's only FIX is ``reconstruct`` (fill missing atoms). Clash
  relaxation (``relax``) is a deliberate follow-on: doing it safely needs
  per-structure application (relax only clashy inputs, not the whole batch) and
  explicit acceptance of the moved-off-experiment coordinates. For now
  ``heavy_clashes`` can only be accepted or dropped.
- A FIX does NOT implicitly accept the provenance hazard it creates:
  ``reconstruct`` requires ``reconstructed_atoms="accept"`` explicitly.
- ``altlocs`` / ``multiple_models`` are ``accept_selected`` — the selection
  (primary conformer / model 0) already happened in prepare; the policy accepts
  that lossy choice (real selectors are a later primitive).
- Single-pass: collect the FIX flags, re-prepare once, re-verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# --- hazard vocabulary & profile blockers ---------------------------------
#
# The hazards that *block* each label profile. Kept in lockstep with the
# PrepReport.label_safe_* properties by test_repair's consistency test.

_HEAVY_COORDS_BLOCKERS = {
    "not_protein",
    "minimize_failed",
    "reconstructed_atoms",
    "missing_atoms",
    "relaxed_coords",
    "heavy_clashes",
    "altlocs",
    "multiple_models",
}
_ALL_ATOM_BLOCKERS = _HEAVY_COORDS_BLOCKERS | {"no_hydrogens"}
_ENERGY_BLOCKERS = _ALL_ATOM_BLOCKERS | {"incomplete_ff", "untyped_atoms"}
_SEQ_INDEXED_BLOCKERS = {"not_protein", "insertion_codes", "multiple_models"}

PROFILE_BLOCKERS: Dict[str, frozenset] = {
    "heavy_coords": frozenset(_HEAVY_COORDS_BLOCKERS),
    "all_atom_coords": frozenset(_ALL_ATOM_BLOCKERS),
    "energy": frozenset(_ENERGY_BLOCKERS),
    "sequence_indexed": frozenset(_SEQ_INDEXED_BLOCKERS),
    # "all" == the strict label_safe gate (energy AND sequence_indexed).
    "all": frozenset(_ENERGY_BLOCKERS | _SEQ_INDEXED_BLOCKERS),
}

# Every hazard name a policy rule may target (the union of all profile blockers).
# A typo'd rule must be rejected, not silently ignored — otherwise it falls back
# to `default`, which is dangerous with `default="accept"` (codex).
KNOWN_HAZARDS = frozenset().union(*PROFILE_BLOCKERS.values())

# Valid actions, and which hazards each FIX action applies to.
#   reconstruct (missing_atoms): fill from templates -> reconstructed_atoms.
#   relax (heavy_clashes): heavy-atom minimize to resolve clashes. LOSSY — it
#     moves the deposited coordinates, so it is applied PER-STRUCTURE (only the
#     clashy inputs), records a CA-drift metric, and the resulting
#     `relaxed_coords` provenance must be explicitly accepted.
ACTIONS = {"reconstruct", "relax", "accept", "accept_selected", "drop"}
_FIX_FOR = {"reconstruct": "missing_atoms", "relax": "heavy_clashes"}
# A FIX creates a provenance hazard that must be decided EXPLICITLY (a rule,
# never via `default`) so a broad default="accept" can't silently let
# fabricated / moved-off-experiment coordinates pass as observed labels (codex).
_FIX_PROVENANCE = {"reconstruct": "reconstructed_atoms", "relax": "relaxed_coords"}
_SELECTION_HAZARDS = {"altlocs", "multiple_models"}


@dataclass(frozen=True)
class RepairPolicy:
    """A per-hazard remediation rule set, targeting one label profile.

    Build with :meth:`for_profile`. Each hazard maps to an action:
    ``"reconstruct"`` / ``"relax"`` (FIX), ``"accept"`` / ``"accept_selected"``
    (tolerate), or ``"drop"`` (exclude). Unlisted hazards use ``default``.
    """

    profile: str
    rules: Dict[str, str]
    default: str = "drop"

    @classmethod
    def for_profile(cls, profile: str, *, default: str = "drop", **rules: str) -> "RepairPolicy":
        if profile not in PROFILE_BLOCKERS:
            raise ValueError(
                f"unknown profile {profile!r}; choose one of {sorted(PROFILE_BLOCKERS)}"
            )
        for hazard, action in rules.items():
            if hazard not in KNOWN_HAZARDS:
                raise ValueError(
                    f"unknown hazard {hazard!r}; known hazards: {sorted(KNOWN_HAZARDS)}"
                )
            if action not in ACTIONS:
                raise ValueError(f"unknown action {action!r} for {hazard!r}; {sorted(ACTIONS)}")
            if action in _FIX_FOR and _FIX_FOR[action] != hazard:
                raise ValueError(
                    f"action {action!r} only applies to hazard {_FIX_FOR[action]!r}, not {hazard!r}"
                )
            if action == "accept_selected" and hazard not in _SELECTION_HAZARDS:
                raise ValueError(
                    f"'accept_selected' only applies to {sorted(_SELECTION_HAZARDS)}, not {hazard!r}"
                )
        # `default` applies to every unlisted hazard, so only the universal
        # tolerate/exclude actions are valid — a FIX or accept_selected is
        # hazard-specific and would silently not be applied (codex).
        if default not in ("accept", "drop"):
            raise ValueError(f"default action must be 'accept' or 'drop', not {default!r}")
        # A FIX's provenance hazard must be decided by an EXPLICIT rule, not the
        # default — otherwise default="accept" silently lets lossy / fabricated
        # coordinates pass (codex).
        for action, provenance in _FIX_PROVENANCE.items():
            if any(a == action for a in rules.values()) and provenance not in rules:
                raise ValueError(
                    f"action {action!r} requires an explicit rule for its provenance "
                    f"hazard {provenance!r} (e.g. {provenance}='accept' or 'drop'); "
                    "it must not be left to `default`"
                )
        return cls(profile=profile, rules=dict(rules), default=default)

    def action_for(self, hazard: str) -> str:
        return self.rules.get(hazard, self.default)

    @property
    def reconstruct(self) -> bool:
        """Whether the policy fixes missing atoms by reconstruction."""
        return self.action_for("missing_atoms") == "reconstruct"

    @property
    def relax(self) -> bool:
        """Whether the policy resolves clashes by per-structure heavy relaxation (lossy)."""
        return self.action_for("heavy_clashes") == "relax"

    # Convenience presets ---------------------------------------------------

    @classmethod
    def coords_only(cls) -> "RepairPolicy":
        """Heavy-coordinate labels: fix missing atoms, accept typing/cofactor and
        conformer/model selections, drop identity/geometry hazards."""
        return cls.for_profile(
            "heavy_coords",
            missing_atoms="reconstruct",
            reconstructed_atoms="accept",
            altlocs="accept_selected",
            multiple_models="accept_selected",
            heavy_clashes="drop",
            default="drop",
        )

    @classmethod
    def strict(cls, profile: str = "all") -> "RepairPolicy":
        """Drop anything with any blocking hazard (no fixes, no accepts)."""
        return cls.for_profile(profile, default="drop")


@dataclass
class RepairOutcome:
    """What the policy did to one structure, and whether it passes."""

    profile: str
    passes_policy: bool = False
    actions_taken: List[str] = field(default_factory=list)
    remaining_hazards: List[str] = field(default_factory=list)
    accepted_hazards: List[str] = field(default_factory=list)
    dropped_for: List[str] = field(default_factory=list)
    #: CA-RMSD (Å) of the relaxed vs deposited coordinates when `relax` moved
    #: heavy atoms; None when no relaxation happened. The "loud" drift signal.
    coords_drift: Optional[float] = None


def evaluate(report, policy: RepairPolicy, *, reconstruct_applied: bool,
             relax_applied: bool = False, coords_drift: Optional[float] = None) -> RepairOutcome:
    """Decide the policy verdict for an already-prepared `report`.

    `report` must reflect any FIXes already applied (the caller re-prepares with
    the policy's fix flags first). Computes the blocking hazards for the target
    profile that are still present, resolves each against the policy, and reports
    pass/accept/drop.
    """
    out = RepairOutcome(profile=policy.profile)
    # Record an action only when it actually did work (codex): the policy may
    # enable reconstruction globally, but a clean structure has nothing to fill,
    # so it should not be counted as repaired.
    if reconstruct_applied and report.atoms_reconstructed > 0:
        out.actions_taken.append(f"reconstruct(missing_atoms; +{report.atoms_reconstructed})")
    if relax_applied and report.heavy_relaxed:
        out.coords_drift = coords_drift
        d = f"{coords_drift:.3f}A" if coords_drift is not None else "?"
        out.actions_taken.append(f"relax(heavy_clashes; CA-drift {d})")

    present = set(report.label_hazards)
    # The hazards this policy cares about: the profile's blockers PLUS any the
    # caller explicitly ruled on. The latter lets a policy be STRICTER than the
    # profile (e.g. heavy_coords + untyped_atoms="drop") — a non-blocker rule
    # must not be silently ignored (codex).
    considered = PROFILE_BLOCKERS[policy.profile] | set(policy.rules)
    present_considered = present & considered

    passes = True
    for hazard in sorted(present_considered):
        action = policy.action_for(hazard)
        if action in ("accept", "accept_selected"):
            out.accepted_hazards.append(hazard)
        elif action in _FIX_FOR and _FIX_FOR[action] == hazard:
            # The FIX was applied but the hazard PERSISTS (reconstruction could
            # not fill, or relaxation could not clear every clash) -> does not pass.
            out.dropped_for.append(f"{action}_failed:{hazard}")
            passes = False
        else:
            out.dropped_for.append(hazard)  # "drop"
            passes = False
    out.remaining_hazards = sorted(present_considered)
    out.passes_policy = passes
    return out


@dataclass
class RepairSummary:
    """Corpus-level aggregation: how many fixed / accepted / dropped, by hazard."""

    total: int = 0
    passed: int = 0
    dropped: int = 0
    by_action: Dict[str, int] = field(default_factory=dict)
    dropped_by_hazard: Dict[str, int] = field(default_factory=dict)
    accepted_by_hazard: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results) -> "RepairSummary":
        s = cls()
        for r in results:
            outcome = getattr(r, "repair", None)
            if outcome is None:
                continue
            s.total += 1
            if outcome.passes_policy:
                s.passed += 1
            else:
                s.dropped += 1
            for a in outcome.actions_taken:
                key = a.split("(")[0]
                s.by_action[key] = s.by_action.get(key, 0) + 1
            for h in outcome.dropped_for:
                s.dropped_by_hazard[h] = s.dropped_by_hazard.get(h, 0) + 1
            for h in outcome.accepted_hazards:
                s.accepted_by_hazard[h] = s.accepted_by_hazard.get(h, 0) + 1
        return s
