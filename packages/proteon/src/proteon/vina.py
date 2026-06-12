"""AutoDock-Vina scoring + local optimization.

Rust port of Trott & Olson's scoring function + upstream's BFGS local
minimiser. PDBQT inputs only (both receptor and ligand); no external
Vina binary required.

Functions:
    score_only        — 8-component upstream score for one (receptor,
                        ligand) pose. Matches `vina --score_only` to
                        ≤ 1 mkcal/mol on our parity fixtures.
    local_only        — BFGS to the ligand's nearest local minimum,
                        returning (refined pose, score, BFGS stats).
    batch_score_only  — virtual-screening primitive: one receptor ×
                        N ligands, parsed + scored in a rayon pool.
                        The receptor and the ~2 MB pair-potential
                        table are built once and reused.
    batch_local_only  — same parallelism on the refined pose.
    dock              — full Monte-Carlo global search (flexible-ligand
                        docking): random placement + Metropolis MC +
                        BFGS, over parallel replicates, returning ranked
                        binding modes.

Classes:
    VinaScoreComponents   — 8 energy-component getters + as_dict().
    BfgsOutcome           — initial/final energy, step/eval count,
                            converged flag.
    VinaLocalOnlyOutcome  — components + bfgs + (N, 3) coords +
                            original_serials.
    VinaDockPose          — one docked mode: components + (N, 3) coords +
                            original_serials + search_energy + center.

Examples:

    >>> import proteon
    >>> rec = open("1iep_receptor.pdbqt").read()
    >>> lig = open("1iep_ligand.pdbqt").read()
    >>> s = proteon.vina.score_only(rec, lig)
    >>> s.total
    -12.5131...
    >>> r = proteon.vina.local_only(rec, lig)
    >>> r.bfgs.n_steps
    20
    >>> r.components.total
    -13.2039...
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

try:
    from proteon_connector import py_vina as _v  # type: ignore
except ImportError:  # pragma: no cover
    _v = None


# Re-export the Rust classes so users can type-hint against them.
VinaScoreComponents = getattr(_v, "VinaScoreComponents", None) if _v else None
BfgsOutcome = getattr(_v, "BfgsOutcome", None) if _v else None
VinaLocalOnlyOutcome = getattr(_v, "VinaLocalOnlyOutcome", None) if _v else None
VinaDockPose = getattr(_v, "VinaDockPose", None) if _v else None


def score_only(receptor_pdbqt: str, ligand_pdbqt: str):
    """Score a ligand pose against a receptor.

    Parameters
    ----------
    receptor_pdbqt
        PDBQT text for the rigid receptor (not a file path).
    ligand_pdbqt
        PDBQT text for the ligand at its pose.

    Returns
    -------
    VinaScoreComponents
        The 8 components upstream Vina reports. Access fields directly
        (`.total`, `.lig_grids`, `.lig_intra`, ...) or call
        `as_dict()` for a plain mapping.

    Matches `vina --score_only` at v1.2.7-27 to ≤ 1 mkcal/mol on the
    proteon-vina fixture suite (drug-like + zinc + macrocycle).
    """
    if _v is None:  # pragma: no cover
        raise ImportError("proteon_connector is not installed")
    return _v.score_only(receptor_pdbqt, ligand_pdbqt)


def local_only(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    *,
    max_steps: Optional[int] = None,
    v_curl: float = 1000.0,
):
    """BFGS to the ligand's nearest local minimum + rescored pose.

    Parameters
    ----------
    receptor_pdbqt, ligand_pdbqt
        PDBQT text (not file paths).
    max_steps
        BFGS iteration cap. `None` picks upstream's default of
        `(25 + N_movable) / 3` — roughly 20 steps for drug-like
        ligands.
    v_curl
        Soft energy-cap. Upstream's default (1000 kcal/mol) is
        effectively no cap for reasonable geometries.

    Returns
    -------
    VinaLocalOnlyOutcome
        `.components` — 8-component score at the refined pose.
        `.bfgs`       — BfgsOutcome with step/eval counts.
        `.coords`     — (N, 3) numpy float64, refined pose coords.
        `.original_serials` — (N,) u32, matching PDBQT atom serials.
        `.total`      — shortcut for `.components.total`.
    """
    if _v is None:  # pragma: no cover
        raise ImportError("proteon_connector is not installed")
    return _v.local_only(
        receptor_pdbqt,
        ligand_pdbqt,
        max_steps=max_steps,
        v_curl=v_curl,
    )


def batch_score_only(
    receptor_pdbqt: str,
    ligands_pdbqt: Sequence[str],
    *,
    n_threads: Optional[int] = None,
) -> List:
    """Virtual-screening primitive: score N ligands against one receptor.

    Parses the receptor and builds the pair-potential table ONCE,
    then iterates the ligand list on a rayon pool. Output order
    matches input order.

    Parameters
    ----------
    receptor_pdbqt
        PDBQT text for the receptor (shared across every ligand).
    ligands_pdbqt
        Sequence of PDBQT texts, one per ligand pose to score.
    n_threads
        `None` (or 0) uses every core; positive values cap the
        pool size. Matches proteon's `batch_*` convention.

    Returns
    -------
    list[VinaScoreComponents]
        One per input ligand, in input order.
    """
    if _v is None:  # pragma: no cover
        raise ImportError("proteon_connector is not installed")
    return _v.batch_score_only(
        receptor_pdbqt,
        list(ligands_pdbqt),
        n_threads=n_threads,
    )


def batch_local_only(
    receptor_pdbqt: str,
    ligands_pdbqt: Sequence[str],
    *,
    n_threads: Optional[int] = None,
    max_steps: Optional[int] = None,
    v_curl: float = 1000.0,
) -> List:
    """Batch BFGS refinement: minimise N ligands against one receptor.

    Same parallelism model as `batch_score_only`. Each element of
    the returned list carries the refined pose, score components,
    and BFGS statistics for one ligand.
    """
    if _v is None:  # pragma: no cover
        raise ImportError("proteon_connector is not installed")
    return _v.batch_local_only(
        receptor_pdbqt,
        list(ligands_pdbqt),
        n_threads=n_threads,
        max_steps=max_steps,
        v_curl=v_curl,
    )


def dock(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    *,
    center: Optional[Tuple[float, float, float]] = None,
    size: Optional[Tuple[float, float, float]] = None,
    padding: float = 6.0,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    seed: int = 0,
    global_steps: int = 2500,
    n_threads: Optional[int] = None,
) -> List:
    """Dock a flexible ligand into a receptor (Monte-Carlo global search).

    Runs upstream Vina's outer search — random placement, then
    `global_steps` of *mutate → BFGS → Metropolis* per replicate, over
    `exhaustiveness` parallel replicates — and returns distinct binding
    modes ranked best-first.

    Parameters
    ----------
    receptor_pdbqt, ligand_pdbqt
        PDBQT text (not file paths).
    center, size
        Search-box center and side lengths in Å, as 3-tuples. Give both
        or neither: if both are omitted the box is the ligand's bounding
        box grown by `padding` Å (i.e. redocking around the input pose).
    padding
        Autobox padding in Å, used only when `center`/`size` are omitted.
    exhaustiveness
        Number of independent MC replicates (parallel). Higher is more
        thorough and slower.
    n_poses
        Maximum number of distinct modes to return.
    seed
        RNG seed; a given `(seed, params)` reproduces the run exactly.
    global_steps
        MC steps per replicate (upstream default 2500). Lower it for
        quick exploratory runs — docking is much heavier than
        `score_only` / `local_only`.
    n_threads
        `None`/0 uses every core; positive caps the pool.

    Returns
    -------
    list[VinaDockPose]
        Ranked modes (best search energy first). Each has `.components`
        (8-component score), `.total`, `.search_energy`, `.coords`
        ((N, 3) float64), `.original_serials`, and `.center`.

    Docking parity is statistical, not bit-exact: on the 1iep fixture
    the top modes recover the crystal pose to < 1 Å.
    """
    if _v is None:  # pragma: no cover
        raise ImportError("proteon_connector is not installed")
    return _v.dock(
        receptor_pdbqt,
        ligand_pdbqt,
        center=center,
        size=size,
        padding=padding,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        seed=seed,
        global_steps=global_steps,
        n_threads=n_threads,
    )


__all__ = [
    "VinaScoreComponents",
    "BfgsOutcome",
    "VinaLocalOnlyOutcome",
    "VinaDockPose",
    "score_only",
    "local_only",
    "batch_score_only",
    "batch_local_only",
    "dock",
]
