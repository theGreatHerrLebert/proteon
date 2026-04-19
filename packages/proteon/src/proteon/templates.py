"""AF2-style template features from sequence-search hits.

Templates are structurally-similar proteins whose atom-level
coordinates are projected into the query coordinate frame as an
auxiliary input to the model. Per roadmap Section 8 the canonical
template tensors are:

- `template_aatype`: target amino-acid identity at each query position
- `template_all_atom_positions`: (N_templates, L, 37, 3) coords
- `template_all_atom_masks`: (N_templates, L, 37) atom-existence mask
- `template_sum_probs`: (N_templates,) per-template confidence score

This v0 builds sequence-based templates: for each of the top-K
search hits, the hit's alignment (query_start / target_start / CIGAR)
maps query positions to target positions. At each aligned position,
the template tensors inherit the target's atom37 coordinates and
aatype; elsewhere they're masked out.

Structure-based template retrieval (TM-align / FoldSeek-style) is
a separate pathway deferred post-v0. The public interface here
accepts any engine that exposes `.search(query_seq) -> list[dict]`
with the same hit-dict schema as `MsaSearch.search`, so swapping
in a structure-search backend later is drop-in on the data side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .supervision import StructureSupervisionExample
from .supervision_constants import AA_TO_INDEX


# AF2 uses 22 classes (20 AA + X + gap). Our AA_TO_INDEX has X at 20
# already; we reserve index 21 as the "template gap" marker for query
# positions where the template has no residue.
TEMPLATE_GAP_INDEX: int = 21


@dataclass
class TemplateFeatures:
    """AF2-style template tensor bundle.

    Shapes:
        template_aatype:              (N_templates, L) int32
        template_all_atom_positions:  (N_templates, L, 37, 3) float32
        template_all_atom_masks:      (N_templates, L, 37) float32
        template_sum_probs:           (N_templates,) float32

    `n_templates` can be 0 — callers that padded to a fixed slot
    count should handle empty bundles. See the roadmap's "Raw input
    features" list (Section 8) for the AF2 feature meanings.
    """

    template_aatype: NDArray[np.int32]
    template_all_atom_positions: NDArray[np.float32]
    template_all_atom_masks: NDArray[np.float32]
    template_sum_probs: NDArray[np.float32]
    n_templates: int
    query_len: int


def build_template_features(
    query_length: int,
    engine,
    target_supervisions: dict[int, StructureSupervisionExample],
    *,
    query_sequence: str,
    max_templates: int = 4,
    exclude_target_ids: Optional[Iterable[int]] = None,
) -> TemplateFeatures:
    """Build template features from the top sequence hits.

    Args:
        query_length: Number of residues in the query (= L dimension).
        engine: Search engine with `.search(query_seq) -> list[hit_dict]`.
            Accepts `MsaSearch` or raw connector `SearchEngine`.
        target_supervisions: Lookup from target_id (as used by the
            engine) to the target's `StructureSupervisionExample`.
            The template path needs the target's atom37 coordinates
            + aatype, which the supervision example already carries.
        query_sequence: Query amino-acid sequence (single-letter).
        max_templates: Cap on how many hits become templates. 4 is
            AF2's default for the "template embedding" branch.
        exclude_target_ids: target_ids to drop from the hit list —
            typically the query's own ID, to avoid self-templating
            collapsing into an identity copy of the query.

    Returns:
        `TemplateFeatures` with `n_templates` rows (may be 0 if the
        search returned nothing usable after exclusion).

    Contract:
        - Unaligned / out-of-matrix query positions have all-zero
          coords, mask 0.0, and aatype = TEMPLATE_GAP_INDEX.
        - Aligned positions inherit the target's atom37 tensors
          verbatim — no reindexing, no rotation (sequence-based
          templates; structural alignment is a separate phase).
    """
    exclude = set(exclude_target_ids or [])
    hits = engine.search(query_sequence)

    # Filter and clip.
    usable = [h for h in hits if h["target_id"] not in exclude][:max_templates]

    n = len(usable)
    L = int(query_length)

    aatype = np.full((n, L), TEMPLATE_GAP_INDEX, dtype=np.int32)
    positions = np.zeros((n, L, 37, 3), dtype=np.float32)
    masks = np.zeros((n, L, 37), dtype=np.float32)
    sum_probs = np.zeros((n,), dtype=np.float32)

    for i, hit in enumerate(usable):
        target = target_supervisions.get(hit["target_id"])
        if target is None:
            # Hit target_id we can't resolve — skip. Callers typically
            # keep target_supervisions in sync with the engine's corpus,
            # so this is a diagnostic rather than a hard error.
            continue
        _fill_template_from_alignment(
            aatype[i],
            positions[i],
            masks[i],
            target=target,
            query_start=int(hit["query_start"]),
            target_start=int(hit["target_start"]),
            cigar=str(hit["cigar"]),
            query_length=L,
        )
        sum_probs[i] = float(hit.get("score", 0))

    # Normalize sum_probs to [0, 1]: AF2 uses HHsearch probabilities,
    # which are already in [0, 1]. We don't have those; we expose
    # min-max-normalized gapped scores so the downstream model can
    # still use them as a relative-confidence ranking without being
    # sensitive to absolute score magnitude (which drifts with
    # substitution matrix choice).
    if n > 0 and sum_probs.max() > 0:
        sum_probs = sum_probs / sum_probs.max()

    return TemplateFeatures(
        template_aatype=aatype,
        template_all_atom_positions=positions,
        template_all_atom_masks=masks,
        template_sum_probs=sum_probs,
        n_templates=n,
        query_len=L,
    )


def _parse_cigar(cigar: str) -> list[tuple[int, str]]:
    """Parse a CIGAR string like '42M2I8M' into `[(count, op), ...]`.

    Matches the format produced by `gapped.GappedAlignment.cigar_string()`
    (ops are in {M, I, D}, count is a positive int).
    """
    out: list[tuple[int, str]] = []
    num_start = 0
    for i, ch in enumerate(cigar):
        if ch.isalpha():
            count = int(cigar[num_start:i])
            out.append((count, ch))
            num_start = i + 1
    return out


def _fill_template_from_alignment(
    aatype_out: NDArray[np.int32],
    positions_out: NDArray[np.float32],
    masks_out: NDArray[np.float32],
    *,
    target: StructureSupervisionExample,
    query_start: int,
    target_start: int,
    cigar: str,
    query_length: int,
) -> None:
    """Walk one template's CIGAR and copy target atoms into query coords.

    For each CIGAR op:
        - `M`: aligned pair. Copy target's atom37 tensors at `t_pos`
          into template slot at `q_pos`. Advance both.
        - `I`: query insertion (target has gap). Template has no
          residue for these query positions — leave masked. Advance
          query only.
        - `D`: target insertion (query has gap). Skip target residues
          — they don't appear in the template output because
          `positions_out` / `masks_out` / `aatype_out` are indexed
          by query position.

    Conventions match `gapped.CigarOp`:
        - Match: both consumed.
        - Insert: query consumed, target not.
        - Delete: target consumed, query not.
    """
    q_pos = int(query_start)
    t_pos = int(target_start)

    target_positions = np.asarray(target.all_atom_positions)
    target_mask = np.asarray(target.all_atom_mask)
    target_aatype = np.asarray(target.aatype)

    for count, op in _parse_cigar(cigar):
        if op == "M":
            for _ in range(count):
                if 0 <= q_pos < query_length and 0 <= t_pos < target.length:
                    aatype_out[q_pos] = int(target_aatype[t_pos])
                    positions_out[q_pos] = target_positions[t_pos]
                    masks_out[q_pos] = target_mask[t_pos]
                q_pos += 1
                t_pos += 1
        elif op == "I":
            # Query residues with no target counterpart — template is
            # silent at these positions (mask stays 0, aatype stays
            # TEMPLATE_GAP_INDEX).
            q_pos += count
        elif op == "D":
            # Target residues with no query counterpart — advance the
            # target walker, but don't emit (no query position to
            # write to).
            t_pos += count
        else:
            raise ValueError(f"unexpected CIGAR op {op!r} in {cigar!r}")
