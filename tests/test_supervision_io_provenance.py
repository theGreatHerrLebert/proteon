"""I/O drop-visibility provenance on structure supervision examples.

These tests cover the *visibility half* of the #150 follow-up: the silent
supervision reductions (models 2..N dropped, alternate conformers dropped,
parse warnings discarded) are now RECORDED as structured provenance on
``StructureQualityMetadata`` — without changing the selection behaviour
(model 1 / primary conformer stay).

They also pin the model-1 selection bug fix: ``PyPDB.chains()`` flattens
across all models, so chain selection must be scoped to model 1 or it can
silently return a different model's chain.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import proteon
from proteon import supervision_export as sup_export
from proteon.supervision import _io_provenance

_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
_MULTIMODEL = os.path.join(_CORPUS, "multimodel", "two_models.pdb")
_MODEL1_MISSING = os.path.join(_CORPUS, "multimodel", "model1_missing_chain.pdb")
_ALTLOC = os.path.join(_CORPUS, "altloc", "dual_conformer.pdb")

# atom37 index 1 is CA.
_CA = 1


# -- models provenance -------------------------------------------------------

def test_models_present_records_multimodel_reduction():
    ex = proteon.build_structure_supervision_example(
        proteon.load(_MULTIMODEL), chain_id="A", record_id="mm:A"
    )
    assert ex.quality.models_present == 2
    assert ex.quality.model_selected_index == 0


def test_models_present_is_one_for_single_model():
    # dual_conformer.pdb is a single model.
    ex = proteon.build_structure_supervision_example(
        proteon.load(_ALTLOC), chain_id="A", record_id="ac:A"
    )
    assert ex.quality.models_present == 1
    assert ex.quality.model_selected_index == 0


# -- altloc provenance -------------------------------------------------------

def test_altloc_provenance_counts_reduced_residues():
    # VAL 2 has CA altloc A/B (one residue with >1 conformer); ALA 1 and GLY 3
    # are single-conformer.
    ex = proteon.build_structure_supervision_example(
        proteon.load(_ALTLOC), chain_id="A", record_id="ac:A"
    )
    assert ex.quality.conformer_reduced_residue_count == 1
    assert ex.quality.altloc_policy == "primary"


def test_no_altloc_provenance_is_zero():
    ex = proteon.build_structure_supervision_example(
        proteon.load(_MULTIMODEL), chain_id="A", record_id="mm:A"
    )
    assert ex.quality.conformer_reduced_residue_count == 0
    assert ex.quality.altloc_policy == "primary"


# -- model-1 selection bug fix ----------------------------------------------

def test_selection_picks_model1_chain_single():
    # Chain A exists in both models; model 1's CA(ALA) is at x=2.0, model 2's at
    # x=12.0. The selection must come from model 1.
    ex = proteon.build_structure_supervision_example(
        proteon.load(_MODEL1_MISSING), chain_id="A", record_id="m1:A"
    )
    assert float(ex.all_atom_positions[0, _CA, 0]) == pytest.approx(2.0)


def test_selection_picks_model1_chain_batch():
    (ex,) = proteon.batch_build_structure_supervision_examples(
        [proteon.load(_MODEL1_MISSING)], chain_ids=["A"]
    )
    assert float(ex.all_atom_positions[0, _CA, 0]) == pytest.approx(2.0)


def test_selection_raises_for_cross_model_chain_single():
    # Chain B is absent from model 1 (only present in model 2). Selection must
    # raise, NOT silently fall through to model 2's chain B.
    with pytest.raises(ValueError, match="not found in model 1"):
        proteon.build_structure_supervision_example(
            proteon.load(_MODEL1_MISSING), chain_id="B", record_id="m1:B"
        )


def test_selection_raises_for_cross_model_chain_batch():
    with pytest.raises(ValueError, match="not found in model 1"):
        proteon.batch_build_structure_supervision_examples(
            [proteon.load(_MODEL1_MISSING)], chain_ids=["B"]
        )


# -- parse warnings ----------------------------------------------------------

def test_parse_warnings_propagate_to_quality():
    s = proteon.load(_MODEL1_MISSING)
    assert s.parse_warnings, "fixture should emit a non-fatal parse warning"
    ex = proteon.build_structure_supervision_example(s, chain_id="A", record_id="m1:A")
    assert ex.quality.parse_warnings == list(s.parse_warnings)


def test_parse_warnings_empty_for_clean_structure():
    s = proteon.load(_ALTLOC)
    assert s.parse_warnings == []  # empty list, not None
    ex = proteon.build_structure_supervision_example(s, chain_id="A", record_id="ac:A")
    assert ex.quality.parse_warnings == []


def test_parse_warnings_survive_prepare():
    s = proteon.load(_MODEL1_MISSING)
    before = list(s.parse_warnings)
    assert before
    proteon.prepare(s)
    # prepare mutates pdb.inner in place; the sibling parse_warnings field
    # (immutable initial-parse diagnostics) survives unchanged.
    assert list(s.parse_warnings) == before
    ex = proteon.build_structure_supervision_example(s, chain_id="A", record_id="m1:A")
    assert ex.quality.parse_warnings == before


def test_fake_structure_provenance_uses_getattr_fallbacks():
    # A minimal structure without models/model_count/parse_warnings/conformer_count
    # must still yield well-formed provenance via the getattr fallbacks.
    chain = SimpleNamespace(
        id="A",
        residues=[SimpleNamespace(name="ALA", is_amino_acid=True)],
    )
    structure = SimpleNamespace(chains=[chain], chain_count=1)
    prov = _io_provenance(structure, chain)
    assert prov["models_present"] is None
    assert prov["model_selected_index"] == 0
    assert prov["conformer_reduced_residue_count"] == 0
    assert prov["altloc_policy"] == "primary"
    assert prov["parse_warnings"] == []


# -- round-trips / parity ----------------------------------------------------

def test_single_batch_provenance_parity():
    single = proteon.build_structure_supervision_example(
        proteon.load(_ALTLOC), chain_id="A", record_id="ac:A"
    )
    (batch,) = proteon.batch_build_structure_supervision_examples(
        [proteon.load(_ALTLOC)], chain_ids=["A"]
    )
    for field_name in (
        "models_present",
        "model_selected_index",
        "conformer_reduced_residue_count",
        "altloc_policy",
        "parse_warnings",
    ):
        assert getattr(single.quality, field_name) == getattr(batch.quality, field_name)


def test_quality_json_round_trips_new_fields(tmp_path):
    pytest.importorskip("pyarrow")
    ex = proteon.build_structure_supervision_example(
        proteon.load(_MULTIMODEL), chain_id="A", record_id="mm:A"
    )
    out = tmp_path / "examples"
    sup_export.export_structure_supervision_examples([ex], out)
    (back,) = sup_export.load_structure_supervision_examples(out)
    assert back.quality.models_present == ex.quality.models_present == 2
    assert back.quality.model_selected_index == 0
    assert back.quality.conformer_reduced_residue_count == ex.quality.conformer_reduced_residue_count
    assert back.quality.altloc_policy == "primary"
    assert back.quality.parse_warnings == ex.quality.parse_warnings


def test_old_quality_json_without_new_keys_reloads_via_defaults():
    # Backward-compat (plan §5): a v2 quality_json predating these fields must
    # still reload through the dataclass defaults, not KeyError.
    from proteon.supervision import StructureQualityMetadata

    legacy = json.dumps({"prep_success": True, "source_format": "pdb"})
    q = StructureQualityMetadata(**json.loads(legacy))
    assert q.models_present is None
    assert q.model_selected_index is None
    assert q.conformer_reduced_residue_count is None
    assert q.altloc_policy is None
    assert q.parse_warnings == []
