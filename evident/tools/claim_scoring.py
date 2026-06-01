#!/usr/bin/env python3
"""Score EVIDENT claims against their evidence artifacts.

The manifest validator (``validate_manifest.py``) only schema-checks a
claim's ``tolerances:`` block — metric in vocabulary, op valid, value
numeric. It never opens the artifact. This module closes that gap: given
a tolerance that carries a declarative ``scoring:`` spec, it loads the
named artifact, computes the tolerance's metric over it, applies the
recorded ``op``/``value``, and emits a PASS/FAIL verdict.

The point is that the recorded band is enforced by code rather than by a
human transcribing a number off a report (issue #85). A claim that stops
meeting its own tolerance fails loudly — at replay time, at release-lock
time, or whenever ``score_claim.py`` is run.

The ``scoring:`` spec
=====================

Each ``tolerances[]`` entry MAY carry a ``scoring:`` mapping. When it
does, this module can compute the metric from the artifact. When it does
not (CI-tier claims whose evidence is "pytest console output", say), the
tolerance is reported as ``unscored`` and skipped — the test passing is
the gate there, not an artifact.

Fields::

    scoring:
      artifact: validation/results.json   # path, resolved from repo root
      format: jsonl | json                # how the artifact is read
      records_path: results               # (json only) dotted path to the
                                           #   list of records inside the file
      select:                             # (optional) flatten a nested record:
        list: tests                       #   for each record, scan this list,
        match: {test: sasa}               #   pick the element matching these
        take: details                     #   key/values, score its `take` dict
      value: relative_diff                # per-record numeric: a dotted field
                                           #   path, OR a computed form (below)
      where: {field: relative_diff, present: true}   # (optional) in-scope filter
      aggregate: median | mean | max | min | value | fraction
                                           # (optional) defaults from `metric`
      pass_when: {op: "<", value: 0.05}   # (fraction only) per-record numerator
      reference: 0                        # (drift only) subtract before |.|

The per-record ``value`` is normally a dotted field path
(``rel_diff.bond_stretch``, ``agreement_rate``, ``tm_diff``). It may also
be a computed relative error from two raw fields, which lets a claim
recompute its own rel-diff (and make a denominator floor explicit)::

    value: {rel: [proteon.bond_stretch, ball.bond_stretch], floor: 1.0}
            # |a - b| / max(|b|, floor)
    value: {abs: [e_total_proteon, e_total_openmm]}
            # |a - b|

Default aggregate per metric:

==========================  =========
metric                      aggregate
==========================  =========
median_relative_error       median
relative_error              value
absolute_error              value
pass_rate                   fraction
drift                       median   (of |value - reference|)
==========================  =========

``aggregate: value`` requires exactly one in-scope record (an n=1 CI
fixture); more than one is a spec error.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

VALID_FORMATS = {"jsonl", "json"}
VALID_AGGREGATES = {"median", "mean", "max", "min", "value", "fraction"}

# Default aggregation for each tolerance metric. Overridable via
# scoring.aggregate.
METRIC_DEFAULT_AGGREGATE = {
    "median_relative_error": "median",
    "relative_error": "value",
    "absolute_error": "value",
    "pass_rate": "fraction",
    "drift": "median",
    # recall/precision/f1 are in the manifest vocabulary but no claim uses
    # them yet and they have no artifact convention; left unmapped so a
    # scoring block on one raises a clear error rather than guessing.
}

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}


class ScoringError(ValueError):
    """A scoring spec is malformed or cannot be applied to its artifact."""


@dataclass
class ToleranceScore:
    """Result of scoring one tolerance entry."""

    metric: str
    op: str
    threshold: float
    output: str | None
    scored: bool                 # False => no scoring: block, skipped
    observed: float | None = None
    passed: bool | None = None
    n_in_scope: int | None = None
    n_records: int | None = None
    aggregate: str | None = None
    reason: str | None = None    # why unscored / how computed, for the report

    @property
    def status(self) -> str:
        if not self.scored:
            return "unscored"
        return "pass" if self.passed else "fail"


@dataclass
class ClaimScore:
    """Result of scoring every tolerance on a claim."""

    claim_id: str
    tolerances: list[ToleranceScore] = field(default_factory=list)

    @property
    def scored(self) -> list[ToleranceScore]:
        return [t for t in self.tolerances if t.scored]

    @property
    def any_failed(self) -> bool:
        return any(t.passed is False for t in self.tolerances)

    @property
    def all_passed(self) -> bool:
        scored = self.scored
        return bool(scored) and all(t.passed for t in scored)

    @property
    def status(self) -> str:
        if self.any_failed:
            return "fail"
        if not self.scored:
            return "unscored"
        return "pass"


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #

def _dig(obj: Any, dotted: str) -> Any:
    """Walk a dotted path into nested dicts. Returns None on any miss."""
    cur = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_records(artifact: pathlib.Path, fmt: str, records_path: str | None) -> list[dict]:
    """Load an artifact into a flat list of record dicts.

    ``jsonl``: one JSON object per line (malformed lines skipped).
    ``json``:  a single JSON document; ``records_path`` is a dotted path to
               the list of records inside it (e.g. ``results``).
    """
    if fmt not in VALID_FORMATS:
        raise ScoringError(f"unknown scoring.format {fmt!r}; allowed: {sorted(VALID_FORMATS)}")
    if not artifact.exists():
        raise ScoringError(f"artifact does not exist: {artifact}")

    if fmt == "jsonl":
        records: list[dict] = []
        with artifact.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    # fmt == "json"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    node = _dig(doc, records_path) if records_path else doc
    if records_path and node is None:
        raise ScoringError(f"records_path {records_path!r} not found in {artifact.name}")
    if not isinstance(node, list):
        raise ScoringError(
            f"records_path {records_path!r} must point at a list in {artifact.name}; "
            f"got {type(node).__name__}"
        )
    return [r for r in node if isinstance(r, dict)]


def _apply_select(records: list[dict], select: dict) -> list[dict]:
    """Flatten each record by picking a matching element from a nested list.

    ``select.list`` names a list field on each record; ``select.match`` is a
    {key: value} filter applied to its elements; ``select.take`` (optional)
    names the sub-dict of the matched element to use as the scored record.
    Records with no matching element are dropped.
    """
    list_key = select.get("list")
    match = select.get("match") or {}
    take = select.get("take")
    if not isinstance(list_key, str):
        raise ScoringError("scoring.select.list must be a field name")
    if not isinstance(match, dict):
        raise ScoringError("scoring.select.match must be a mapping")

    out: list[dict] = []
    for rec in records:
        items = rec.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if all(item.get(k) == v for k, v in match.items()):
                sub = item.get(take) if take else item
                if isinstance(sub, dict):
                    out.append(sub)
                break
    return out


# --------------------------------------------------------------------------- #
# Predicates and value extraction
# --------------------------------------------------------------------------- #

def _eval_predicate(record: dict, pred: dict, *, default_value: Any = None) -> bool:
    """Evaluate a record predicate.

    A predicate may carry ``field`` (a dotted path; if omitted, the supplied
    ``default_value`` is tested — used for pass_when against the extracted
    per-record value) plus exactly one test:
      present: true   -> field exists and is not None
      eq: <any>       -> equals
      in: [..]        -> membership
      op + value      -> numeric comparison
    """
    if "field" in pred:
        val = _dig(record, pred["field"])
    else:
        val = default_value

    if pred.get("present") is True:
        return val is not None
    if "eq" in pred:
        return val == pred["eq"]
    if "in" in pred:
        return val in pred["in"]
    if "op" in pred and "value" in pred:
        if val is None:
            return False
        try:
            return _OPS[pred["op"]](float(val), float(pred["value"]))
        except (TypeError, ValueError):
            return False
    raise ScoringError(f"predicate has no recognised test: {pred!r}")


def _extract_value(record: dict, value_spec: Any) -> float | None:
    """Pull the per-record numeric value per the ``value`` spec.

    String spec  -> dotted field path.
    {rel: [a,b], floor?: f} -> |a-b| / max(|b|, floor)   (floor default 0 = no floor)
    {abs: [a,b]}            -> |a-b|
    Returns None when a needed field is absent (record is out of scope).
    """
    if isinstance(value_spec, str):
        v = _dig(record, value_spec)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if isinstance(value_spec, dict):
        if "rel" in value_spec:
            fields = value_spec["rel"]
            if not (isinstance(fields, list) and len(fields) == 2):
                raise ScoringError("scoring.value.rel must be [numerator_field, oracle_field]")
            a, b = _dig(record, fields[0]), _dig(record, fields[1])
            if a is None or b is None:
                return None
            floor = float(value_spec.get("floor", 0.0))
            denom = max(abs(float(b)), floor)
            if denom == 0.0:
                return None
            return abs(float(a) - float(b)) / denom
        if "abs" in value_spec:
            fields = value_spec["abs"]
            if not (isinstance(fields, list) and len(fields) == 2):
                raise ScoringError("scoring.value.abs must be [field_a, field_b]")
            a, b = _dig(record, fields[0]), _dig(record, fields[1])
            if a is None or b is None:
                return None
            return abs(float(a) - float(b))
        raise ScoringError(f"scoring.value mapping must use 'rel' or 'abs': {value_spec!r}")

    raise ScoringError(f"scoring.value must be a field path or mapping: {value_spec!r}")


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def _aggregate(values: list[float], how: str) -> float:
    if not values:
        raise ScoringError(f"no in-scope values to aggregate with {how!r}")
    if how == "median":
        return statistics.median(values)
    if how == "mean":
        return statistics.fmean(values)
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "value":
        if len(values) != 1:
            raise ScoringError(
                f"aggregate=value expects exactly one in-scope record, got {len(values)}"
            )
        return values[0]
    raise ScoringError(f"unknown aggregate {how!r}")


def compute_metric(records: list[dict], scoring: dict, metric: str) -> tuple[float, int, int, str]:
    """Compute the observed metric value.

    Returns (observed, n_in_scope, n_total, aggregate_used).
    """
    select = scoring.get("select")
    scoped_source = _apply_select(records, select) if select else records
    n_total = len(scoped_source)

    where = scoring.get("where")
    value_spec = scoring.get("value")
    aggregate = scoring.get("aggregate") or METRIC_DEFAULT_AGGREGATE.get(metric)
    if aggregate is None:
        raise ScoringError(
            f"metric {metric!r} has no default aggregate; set scoring.aggregate explicitly"
        )
    if aggregate not in VALID_AGGREGATES:
        raise ScoringError(f"unknown scoring.aggregate {aggregate!r}")

    # In-scope = records passing `where` (the denominator for fraction, and
    # the population every other aggregate reduces over).
    in_scope = [r for r in scoped_source if (where is None or _eval_predicate(r, where))]
    n_in_scope = len(in_scope)

    if aggregate == "fraction":
        pass_when = scoring.get("pass_when")
        if not isinstance(pass_when, dict):
            raise ScoringError("aggregate=fraction requires a scoring.pass_when predicate")
        if n_in_scope == 0:
            raise ScoringError("aggregate=fraction has an empty denominator (no in-scope records)")
        hits = 0
        for r in in_scope:
            # pass_when may test a named field, or the extracted value when
            # no `field` is given.
            dv = None if "field" in pass_when else _extract_value(r, value_spec)
            if _eval_predicate(r, pass_when, default_value=dv):
                hits += 1
        return hits / n_in_scope, n_in_scope, n_total, aggregate

    # Numeric aggregates reduce the extracted per-record value.
    if value_spec is None:
        raise ScoringError(f"aggregate={aggregate} requires scoring.value")
    values: list[float] = []
    for r in in_scope:
        v = _extract_value(r, value_spec)
        if v is not None:
            values.append(v)
    observed = _aggregate(values, aggregate)
    # drift measures bias: aggregate the *signed* per-record values, then take
    # the absolute deviation from the reference. This is |median(signed)|, not
    # median(|signed|) — signed errors cancel, which is the intent (a claim
    # gating |median(openmm − proteon)| < band).
    if metric == "drift":
        observed = abs(observed - float(scoring.get("reference", 0.0)))
    return observed, len(values), n_total, aggregate


# --------------------------------------------------------------------------- #
# Tolerance / claim scoring
# --------------------------------------------------------------------------- #

def score_tolerance(tolerance: dict, repo_root: pathlib.Path) -> ToleranceScore:
    """Score a single tolerance entry. No ``scoring:`` block => unscored."""
    metric = tolerance.get("metric")
    op = tolerance.get("op")
    threshold = tolerance.get("value")
    output = tolerance.get("output")

    scoring = tolerance.get("scoring")
    if scoring is None:
        return ToleranceScore(
            metric=metric, op=op, threshold=threshold, output=output,
            scored=False, reason="no scoring: block",
        )
    if metric is None or op is None or threshold is None:
        raise ScoringError("scoring: present but metric/op/value missing on tolerance")
    if op not in _OPS:
        raise ScoringError(f"unknown tolerance op {op!r}")

    artifact_rel = scoring.get("artifact")
    if not isinstance(artifact_rel, str) or not artifact_rel.strip():
        raise ScoringError("scoring.artifact must be a non-empty path")
    artifact = (repo_root / artifact_rel).resolve()
    fmt = scoring.get("format")
    records = load_records(artifact, fmt, scoring.get("records_path"))

    observed, n_in_scope, n_total, aggregate = compute_metric(records, scoring, metric)
    passed = _OPS[op](observed, float(threshold))
    return ToleranceScore(
        metric=metric, op=op, threshold=float(threshold), output=output,
        scored=True, observed=observed, passed=passed,
        n_in_scope=n_in_scope, n_records=n_total, aggregate=aggregate,
        reason=f"{aggregate} over {n_in_scope}/{n_total} records",
    )


def tolerance_bands(claim: dict, metric: str | None = None) -> dict[str, float]:
    """Map each tolerance's ``output`` to its threshold ``value``.

    Lets report renderers read their band lines from the claim YAML instead
    of hardcoding their own constants — so the number a figure draws and the
    number the scorer enforces cannot drift apart. Optionally restrict to a
    single ``metric`` (e.g. only the median bands). Tolerances without an
    ``output`` or ``value`` are skipped.
    """
    bands: dict[str, float] = {}
    for tol in claim.get("tolerances") or []:
        if not isinstance(tol, dict):
            continue
        if metric is not None and tol.get("metric") != metric:
            continue
        output = tol.get("output")
        value = tol.get("value")
        if isinstance(output, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
            bands[output] = float(value)
    return bands


def first_claim(doc: dict) -> dict:
    """Return the claim mapping from a claim YAML document.

    Claim files wrap a single claim in a ``claims:`` list (the manifest
    merges them); a few bare docs are the claim mapping directly.
    """
    if isinstance(doc.get("claims"), list) and doc["claims"]:
        return doc["claims"][0]
    return doc


def score_claim(claim: dict, repo_root: pathlib.Path) -> ClaimScore:
    """Score every tolerance on a claim against its artifacts."""
    result = ClaimScore(claim_id=str(claim.get("id", "<unknown>")))
    for tol in claim.get("tolerances") or []:
        if not isinstance(tol, dict):
            continue
        result.tolerances.append(score_tolerance(tol, repo_root))
    return result
