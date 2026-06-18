"""RepairPolicy — the remediation layer (per-hazard fix/accept/drop + re-verify).

The decision layer on top of the P1 label-safety gate: declaratively decide,
per hazard, what to do with structures that don't pass, then re-verify relative
to a label profile.
"""

import os

import pytest

import proteon
from proteon import PrepReport, RepairPolicy, RepairSummary
from proteon.repair import PROFILE_BLOCKERS, evaluate

PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _pdb(name):
    return os.path.join(PDBS, name)


# --- the blocker map must stay in lockstep with the label_safe_* properties ---


# (hazard name -> a field assignment that triggers ONLY that hazard on a clean
# baseline). The consistency test below asserts each profile property is False
# iff the hazard is in PROFILE_BLOCKERS[profile].
_HAZARD_TRIGGER = {
    "not_protein": dict(skipped_no_protein=True, n_unassigned_atoms=400),
    "minimize_failed": dict(minimizer_status="numerical_failure"),
    "incomplete_ff": dict(incomplete_ff=True, n_unassigned_nonwater=40),
    "reconstructed_atoms": dict(atoms_reconstructed=3),
    "missing_atoms": dict(n_missing_heavy_atoms=2),
    "relaxed_coords": dict(heavy_relaxed=True),
    "heavy_clashes": dict(n_heavy_clashes=4),
    "untyped_atoms": dict(n_unassigned_nonwater=5),
    "no_hydrogens": dict(hydrogens_added=0),
    "altlocs": dict(has_altlocs=True),
    "multiple_models": dict(n_models=3),
    "insertion_codes": dict(has_insertion_codes=True),
}

_PROFILE_PROP = {
    "heavy_coords": "label_safe_heavy_coords",
    "all_atom_coords": "label_safe_all_atom_coords",
    "energy": "label_safe_energy",
    "sequence_indexed": "label_safe_sequence_indexed",
}


class TestBlockerConsistency:
    @pytest.mark.parametrize("hazard", sorted(_HAZARD_TRIGGER))
    def test_blockers_match_profile_properties(self, hazard):
        # Clean baseline (label_safe everywhere), then trigger exactly one hazard.
        base = dict(hydrogens_added=50, minimizer_status="converged_gradient")
        base.update(_HAZARD_TRIGGER[hazard])
        r = PrepReport(**base)
        assert hazard in r.label_hazards, f"{hazard} should appear in label_hazards"
        for profile, prop in _PROFILE_PROP.items():
            blocks = hazard in PROFILE_BLOCKERS[profile]
            assert getattr(r, prop) is (not blocks), (
                f"{hazard}: profile {profile} property={getattr(r, prop)} but "
                f"blockers say blocks={blocks}"
            )


# --- policy construction / validation ---


class TestPolicyValidation:
    def test_rejects_unknown_profile(self):
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("nonsense")

    def test_rejects_unknown_action(self):
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", heavy_clashes="teleport")

    def test_fix_action_must_match_hazard(self):
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", altlocs="reconstruct")

    def test_accept_selected_only_for_selections(self):
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("energy", untyped_atoms="accept_selected")

    def test_rejects_unknown_hazard(self):
        # A typo'd rule must be rejected, not silently fall back to default
        # (dangerous with default="accept") — codex.
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("energy", untyped_atom="drop")  # typo

    def test_default_action(self):
        p = RepairPolicy.for_profile("heavy_coords", default="accept")
        assert p.action_for("heavy_clashes") == "accept"

    def test_fix_action_rejected_as_default(self):
        # A FIX/accept_selected as default would be honored by action_for but not
        # applied at prepare time (codex) -> reject it.
        for bad in ("reconstruct", "relax", "accept_selected"):
            with pytest.raises(ValueError):
                RepairPolicy.for_profile("heavy_coords", default=bad)


# --- evaluate() verdict logic ---


def _report(**kw):
    base = dict(hydrogens_added=50, minimizer_status="converged_gradient")
    base.update(kw)
    return PrepReport(**base)


class TestEvaluate:
    def test_clean_passes(self):
        r = _report()
        out = evaluate(r, RepairPolicy.coords_only(),
                       reconstruct_applied=True)
        assert out.passes_policy is True
        assert out.dropped_for == []

    def test_clash_drops_under_coords_only(self):
        r = _report(n_heavy_clashes=3)
        out = evaluate(r, RepairPolicy.coords_only(),
                       reconstruct_applied=True)
        assert out.passes_policy is False
        assert "heavy_clashes" in out.dropped_for

    def test_accepted_hazard_passes(self):
        # An untyped cofactor blocks energy, but coords_only targets heavy_coords
        # (untyped not a blocker there) -> passes.
        r = _report(untyped_cofactors=True, n_unassigned_nonwater=43)
        out = evaluate(r, RepairPolicy.coords_only(),
                       reconstruct_applied=True)
        assert out.passes_policy is True

    def test_reconstruct_requires_explicit_provenance_rule(self):
        # reconstruct without an explicit reconstructed_atoms rule is rejected at
        # construction — default (even default="accept") must not decide it (codex).
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", missing_atoms="reconstruct",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", missing_atoms="reconstruct",
                                     default="accept")

    def test_reconstruct_dropped_provenance(self):
        # reconstruct with reconstructed_atoms="drop" -> the rebuilt structure
        # is excluded for the provenance (explicit, not via default).
        r = _report(atoms_reconstructed=4)
        p = RepairPolicy.for_profile("heavy_coords", missing_atoms="reconstruct",
                                     reconstructed_atoms="drop",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        out = evaluate(r, p, reconstruct_applied=True)
        assert out.passes_policy is False
        assert "reconstructed_atoms" in out.dropped_for

    def test_explicit_drop_outside_profile_honored(self):
        # untyped_atoms is NOT a heavy_coords blocker, but an explicit drop rule
        # (stricter than the profile) must be honored, not silently ignored (codex).
        r = _report(untyped_cofactors=True, n_unassigned_nonwater=43)
        p = RepairPolicy.for_profile("heavy_coords", untyped_atoms="drop",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        out = evaluate(r, p, reconstruct_applied=False)
        assert out.passes_policy is False
        assert "untyped_atoms" in out.dropped_for

    def test_clash_can_be_accepted(self):
        # heavy_clashes can be accepted (or dropped or relaxed).
        r = _report(n_heavy_clashes=2)
        p = RepairPolicy.for_profile("heavy_coords", heavy_clashes="accept",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        out = evaluate(r, p, reconstruct_applied=False)
        assert out.passes_policy is True
        assert "heavy_clashes" in out.accepted_hazards

    def test_relax_clears_clashes_with_accepted_provenance(self):
        # After relax: clashes gone, relaxed_coords present and accepted -> passes,
        # drift recorded.
        r = _report(heavy_relaxed=True, minimized=True)  # no clashes remain
        p = RepairPolicy.for_profile("heavy_coords", heavy_clashes="relax",
                                     relaxed_coords="accept",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        out = evaluate(r, p, reconstruct_applied=False, relax_applied=True, coords_drift=0.5)
        assert out.coords_drift == 0.5
        assert any("relax" in a for a in out.actions_taken)
        assert "relaxed_coords" in out.accepted_hazards
        assert out.passes_policy is True

    def test_relax_requires_explicit_provenance_rule(self):
        # relax without an explicit relaxed_coords rule is rejected at
        # construction — even with default="accept" (codex: broad accept must not
        # silently pass moved-off-experiment coordinates).
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", heavy_clashes="relax",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        with pytest.raises(ValueError):
            RepairPolicy.for_profile("heavy_coords", heavy_clashes="relax", default="accept")

    def test_relax_failed_when_clashes_persist(self):
        r = _report(n_heavy_clashes=2, heavy_relaxed=True, minimized=True)
        p = RepairPolicy.for_profile("heavy_coords", heavy_clashes="relax",
                                     relaxed_coords="accept",
                                     altlocs="accept_selected", multiple_models="accept_selected")
        out = evaluate(r, p, reconstruct_applied=False, relax_applied=True, coords_drift=0.6)
        assert any("relax_failed" in d for d in out.dropped_for)
        assert out.passes_policy is False


# --- integration through prepare_for_supervision ---


class TestRepairIntegration:
    def test_coords_only_on_corpus(self):
        paths = [_pdb("1crn.pdb"), _pdb("4hhb.pdb")]
        results = proteon.prepare_for_supervision(paths, repair=RepairPolicy.coords_only())
        assert all(r.repair is not None for r in results)
        by_path = {os.path.basename(r.path): r for r in results}
        assert by_path["1crn.pdb"].passes_policy is True   # clean
        assert by_path["4hhb.pdb"].passes_policy is False  # clashy old structure

    def test_summary_aggregates(self):
        # Include a structure with missing atoms so the reconstruct action
        # actually fires (it is only counted when atoms were added).
        paths = [
            _pdb("1crn.pdb"),
            _pdb("4hhb.pdb"),
            os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb"),
        ]
        results = proteon.prepare_for_supervision(paths, repair=RepairPolicy.coords_only())
        s = RepairSummary.from_results(results)
        assert s.total == 3
        assert s.passed + s.dropped == 3
        assert s.by_action.get("reconstruct", 0) >= 1

    def test_load_failure_counted_in_summary(self, tmp_path):
        junk = tmp_path / "x.pdb"
        junk.write_text("not a structure\n")
        paths = [_pdb("1crn.pdb"), str(junk)]
        results = proteon.prepare_for_supervision(paths, repair=RepairPolicy.coords_only())
        junk_res = [r for r in results if r.path == str(junk)][0]
        assert junk_res.repair is not None
        assert junk_res.passes_policy is False
        assert "load_failed" in junk_res.repair.dropped_for
        s = RepairSummary.from_results(results)
        assert s.total == 2  # load failure is accounted, not skipped
        assert s.dropped_by_hazard.get("load_failed", 0) == 1

    def test_relax_applied_only_to_clashy_structures(self):
        # Two-pass: a clean structure (1crn, 0 clashes) is NOT relaxed; a clashy
        # one (4hhb) IS, with the CA-drift recorded. relax never touches the
        # whole batch.
        policy = RepairPolicy.for_profile(
            "heavy_coords",
            heavy_clashes="relax", relaxed_coords="accept",
            altlocs="accept_selected", multiple_models="accept_selected",
        )
        results = proteon.prepare_for_supervision(
            [_pdb("1crn.pdb"), _pdb("4hhb.pdb")], repair=policy
        )
        by = {os.path.basename(r.path): r for r in results}
        # clean: not relaxed (coords preserved)
        assert by["1crn.pdb"].repair.coords_drift is None
        assert not any("relax" in a for a in by["1crn.pdb"].repair.actions_taken)
        # clashy: relaxed, drift measured
        assert by["4hhb.pdb"].repair.coords_drift is not None
        assert any("relax" in a for a in by["4hhb.pdb"].repair.actions_taken)

    def test_reconstruct_then_relax_preserves_provenance(self):
        # A structure that needs BOTH reconstruct (missing atoms) and relax (the
        # rebuilt atoms clash): after the relax pass, the reconstructed_atoms
        # provenance must survive — not be erased by the reconstruct=False relax
        # pass (codex). Here it is NOT accepted, so it must drop the structure.
        fixture = os.path.join(CORPUS, "missing_atoms", "missing_cb.pdb")
        policy = RepairPolicy.for_profile(
            "heavy_coords",
            missing_atoms="reconstruct", reconstructed_atoms="drop",  # explicit: drop rebuilt
            heavy_clashes="relax", relaxed_coords="accept",
            altlocs="accept_selected", multiple_models="accept_selected",
        )
        r = proteon.prepare_for_supervision([fixture], repair=policy)[0]
        assert r.report.atoms_reconstructed > 0      # provenance survived the relax pass
        assert "reconstructed_atoms" in r.repair.dropped_for
        assert r.passes_policy is False

    def test_only_safe_filters_by_policy(self):
        paths = [_pdb("1crn.pdb"), _pdb("4hhb.pdb")]
        kept = proteon.prepare_for_supervision(
            paths, repair=RepairPolicy.coords_only(), only_safe=True
        )
        assert all(r.passes_policy for r in kept)
        assert _pdb("4hhb.pdb") not in [r.path for r in kept]
