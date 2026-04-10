"""SASA runners.

Per-op payload schema (`payload` field of RunnerResult):

    {
        "total_sasa": float,            # Total SASA in Å²
        "per_residue": [                # One entry per residue, ordered by
            {                           # (chain, resi, icode) — but JOIN ON
                "chain": "A",           # the (chain, resi, icode) tuple in
                "resi": 1,              # the aggregator, never positional.
                "icode": "",
                "name": "THR",          # 3-letter residue name
                "sasa": 12.34,          # Å²
            },
            ...
        ],
        "n_residues": int,
        "radii": "protor",              # which radii table was used
    }
"""

from __future__ import annotations

import importlib.metadata as _metadata
from typing import Optional

from ._base import (
    RunnerResult,
    register,
    register_batch,
    time_call,
)
from time import perf_counter as _time_perf
from typing import List as _List

# ---------------------------------------------------------------------------
# ferritin baseline
# ---------------------------------------------------------------------------

try:
    import ferritin as _ferritin
    _FERRITIN_OK = True
    try:
        _FERRITIN_VERSION = "ferritin " + _metadata.version("ferritin")
    except Exception:
        _FERRITIN_VERSION = "ferritin (unknown version)"
except ImportError as e:
    _FERRITIN_OK = False
    _FERRITIN_VERSION = ""
    _FERRITIN_IMPORT_ERROR = str(e)


if _FERRITIN_OK:

    @register("sasa", "ferritin")
    def ferritin(pdb_path: str) -> RunnerResult:
        """Compute SASA via ferritin.total_sasa + ferritin.residue_sasa.

        Uses radii="protor" to match FreeSASA's NACCESS-style table. Probe
        radius defaults to 1.4 Å (water), n_points defaults to 960 (matches
        ferritin's batch_total_sasa default).
        """
        s, _ = time_call(_ferritin.load, pdb_path)
        total, elapsed_total = time_call(
            _ferritin.total_sasa, s, radii="protor"
        )
        per_res_arr, elapsed_res = time_call(
            _ferritin.residue_sasa, s, radii="protor"
        )

        # Build the per-residue list keyed for downstream join.
        residues = list(s.residues)
        per_residue = []
        if len(residues) != len(per_res_arr):
            # Length mismatch — should not happen if ferritin's API is consistent.
            return RunnerResult(
                op="sasa",
                impl="ferritin",
                impl_version=_FERRITIN_VERSION,
                pdb_id="",
                pdb_path=pdb_path,
                elapsed_s=elapsed_total + elapsed_res,
                status="error",
                error=(
                    f"residue count mismatch: structure.residues has "
                    f"{len(residues)}, residue_sasa has {len(per_res_arr)}"
                ),
                payload={},
            )

        for r, val in zip(residues, per_res_arr):
            per_residue.append({
                "chain": r.chain_id,
                "resi": int(r.serial_number),
                "icode": r.insertion_code or "",
                "name": r.name or "",
                "sasa": float(val),
            })

        return RunnerResult(
            op="sasa",
            impl="ferritin",
            impl_version=_FERRITIN_VERSION,
            pdb_id="",  # filled in by the driver
            pdb_path=pdb_path,
            elapsed_s=elapsed_total + elapsed_res,
            status="ok",
            error=None,
            payload={
                "total_sasa": float(total),
                "per_residue": per_residue,
                "n_residues": len(per_residue),
                "radii": "protor",
            },
        )

    @register_batch("sasa", "ferritin")
    def ferritin_batch(pdb_paths: _List[str]) -> _List[RunnerResult]:
        """Batched ferritin SASA runner.

        Loads all structures in parallel via ferritin.batch_load_tolerant,
        then calls batch_total_sasa once across the whole list. This gives
        the in-Rust rayon parallelism a whole chunk of work rather than
        spinning up a pool for every structure. Per-residue SASA is still
        computed per structure because ferritin.residue_sasa is
        single-structure.

        Wall time for the batched path should scale as
        ~O(total_work / num_cores) + O(per_structure overhead * N).
        """
        t_total = _time_perf()
        # Phase 1: parallel load (indices of successful loads returned in
        # (i, structure) tuples by batch_load_tolerant)
        loaded = _ferritin.batch_load_tolerant(pdb_paths, n_threads=-1)
        load_index = {i: s for i, s in loaded}

        # Phase 2: batch total SASA across all successfully-loaded structs
        # in one Rust call
        successful_idx = [i for i in range(len(pdb_paths)) if i in load_index]
        successful_structs = [load_index[i] for i in successful_idx]
        if successful_structs:
            total_sasa_arr = _ferritin.batch_total_sasa(
                successful_structs, n_threads=-1, radii="protor"
            )
        else:
            total_sasa_arr = []

        # Phase 3: per-residue SASA still single-structure
        results: _List[RunnerResult] = []
        t_phase3 = _time_perf()
        for i, pdb_path in enumerate(pdb_paths):
            if i not in load_index:
                results.append(RunnerResult(
                    op="sasa", impl="ferritin", impl_version=_FERRITIN_VERSION,
                    pdb_id="", pdb_path=pdb_path, elapsed_s=0.0,
                    status="error",
                    error="batch_load_tolerant failed for this file",
                    payload={},
                ))
                continue
            s = load_index[i]
            # Find the index in successful_structs to get the total
            pos = successful_idx.index(i)
            total = float(total_sasa_arr[pos])
            # Per-residue (still single-structure call, cheap)
            per_res_arr = _ferritin.residue_sasa(s, radii="protor")
            residues = list(s.residues)
            per_residue = []
            if len(residues) != len(per_res_arr):
                results.append(RunnerResult(
                    op="sasa", impl="ferritin", impl_version=_FERRITIN_VERSION,
                    pdb_id="", pdb_path=pdb_path, elapsed_s=0.0,
                    status="error",
                    error=(f"residue count mismatch: {len(residues)} vs {len(per_res_arr)}"),
                    payload={},
                ))
                continue
            for r, val in zip(residues, per_res_arr):
                per_residue.append({
                    "chain": r.chain_id,
                    "resi": int(r.serial_number),
                    "icode": r.insertion_code or "",
                    "name": r.name or "",
                    "sasa": float(val),
                })
            # Split the total time evenly for per-structure attribution
            # (reasonable approximation; sub-phase timing is complicated
            # and the user cares about total wall anyway)
            results.append(RunnerResult(
                op="sasa", impl="ferritin", impl_version=_FERRITIN_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=0.0,  # per-structure not meaningful in batched mode
                status="ok", error=None,
                payload={
                    "total_sasa": total,
                    "per_residue": per_residue,
                    "n_residues": len(per_residue),
                    "radii": "protor",
                },
            ))
        # Stamp aggregate wall time on the first result for reporting
        if results and results[0].status == "ok":
            results[0].elapsed_s = _time_perf() - t_total
        return results


# ---------------------------------------------------------------------------
# FreeSASA
# ---------------------------------------------------------------------------

try:
    import freesasa as _freesasa  # noqa: F401  (only used inside the runner)
    _FREESASA_OK = True
    try:
        _FREESASA_VERSION = "freesasa " + _metadata.version("freesasa")
    except Exception:
        _FREESASA_VERSION = "freesasa (unknown version)"
except ImportError:
    _FREESASA_OK = False
    _FREESASA_VERSION = ""


if _FREESASA_OK:

    @register("sasa", "freesasa")
    def freesasa(pdb_path: str) -> RunnerResult:
        """Compute SASA via the freesasa Python package.

        Matches ferritin's defaults as closely as possible:

        - Include HETATM atoms (ferritin loads them; freesasa default skips).
          This is the single biggest contributor to the agreement: 1pgb alone
          has 24 waters that ferritin counts and stock freesasa doesn't.
        - Use the ProtOr classifier explicitly (FreeSASA default is OONS-like
          Lee-Richards; ferritin's `radii="protor"` should match ProtOr).
        - Use the Shrake-Rupley algorithm with 960 points (matches ferritin's
          default).
        - Probe radius 1.4 Å (matches ferritin default).

        Per-residue results are keyed by (chain, resi, icode) tuples and
        normalized to the same shape as the ferritin runner so the aggregator
        can join them directly.
        """
        options = {
            "hetatm": True,          # include HETATM to match ferritin
            "hydrogen": False,       # ferritin-loaded crystal files have no H
            "join-models": False,    # first model only, same as ferritin
            # skip-unknown: skip atoms that freesasa can't classify instead
            # of trying to include them as generic atoms. Without this,
            # freesasa segfaults in C on PDBs containing e.g. cobalt,
            # uranium, or exotic heme variants (we hit this on 1 of 100
            # sampled PDBs from the 50K corpus). skip-unknown causes a
            # small atom-count drift vs ferritin on HETATM-heavy
            # structures, but the per-residue agreement on standard
            # protein atoms stays intact.
            "skip-unknown": True,
            "halt-at-unknown": False,
        }
        classifier = _freesasa.Classifier.getStandardClassifier("protor")
        structure, elapsed_load = time_call(
            _freesasa.Structure, pdb_path, classifier, options
        )
        params = _freesasa.Parameters(
            {
                "algorithm": _freesasa.ShrakeRupley,
                "n-points": 960,
                "probe-radius": 1.4,
            }
        )
        result, elapsed_calc = time_call(_freesasa.calc, structure, params)
        total = result.totalArea()

        # Per-residue: residueAreas() returns dict[chain_id -> dict[res_num_str -> ResidueArea]]
        # The res_num_str may include insertion code as e.g. "65A".
        per_residue = []
        residue_areas = result.residueAreas()
        for chain_id, by_resnum in residue_areas.items():
            for resnum_str, ra in by_resnum.items():
                # Split off insertion code if present.
                icode = ""
                resi_str = resnum_str
                if resi_str and not (resi_str[-1].isdigit() or resi_str[-1] == "-"):
                    icode = resi_str[-1]
                    resi_str = resi_str[:-1]
                try:
                    resi = int(resi_str)
                except ValueError:
                    # Can't parse — skip but record.
                    continue
                per_residue.append({
                    "chain": chain_id,
                    "resi": resi,
                    "icode": icode,
                    "name": ra.residueType,
                    "sasa": float(ra.total),
                })

        return RunnerResult(
            op="sasa",
            impl="freesasa",
            impl_version=_FREESASA_VERSION,
            pdb_id="",
            pdb_path=pdb_path,
            elapsed_s=elapsed_load + elapsed_calc,
            status="ok",
            error=None,
            payload={
                "total_sasa": float(total),
                "per_residue": per_residue,
                "n_residues": len(per_residue),
                "radii": "freesasa-default",
            },
        )

    # ------------------------------------------------------------------
    # Process-isolated batched FreeSASA runner (P1b)
    # ------------------------------------------------------------------
    #
    # Why this exists: on the 100-PDB scaling demo, FreeSASA segfaulted in
    # C (EXIT=139) somewhere in the per-structure loop. The same loop ran
    # cleanly in a standalone freesasa-only script, so the suspect is a
    # C-state / GIL interaction with ferritin's rayon pool inherited via
    # the same Python process. `skip-unknown=True` does not fix it.
    #
    # Fix: wrap each freesasa() call in its own short-lived worker process
    # via multiprocessing.Pool with maxtasksperchild=1. Each task spawns
    # a fresh process, runs once, dies. A SIGSEGV in a worker is contained
    # to that worker — the parent records the crash as a per-structure
    # error and the next task gets a clean process. Spawn context (not
    # fork) so the child does NOT inherit the parent's rayon-touched
    # C state in the first place.
    #
    # Cost: ~50-100 ms of process spawn overhead per call vs ~10-50 ms
    # for the FreeSASA call itself on small structures. Worth it: the
    # alternative is "the driver crashes on PDB #47 of 10000".

    def _freesasa_worker(pdb_path: str) -> RunnerResult:
        """Top-level worker called inside a child process by freesasa_batch.

        Must be defined at module top-level (within the `if _FREESASA_OK:`
        guard, which is module-level under spawn) so multiprocessing can
        pickle it by qualified name. Returns a RunnerResult dataclass —
        dataclasses pickle cleanly.
        """
        try:
            return freesasa(pdb_path)
        except Exception as e:
            return RunnerResult(
                op="sasa", impl="freesasa", impl_version=_FREESASA_VERSION,
                pdb_id="", pdb_path=pdb_path, elapsed_s=0.0,
                status="error",
                error=f"{type(e).__name__}: {e}",
                payload={},
            )

    @register_batch("sasa", "freesasa")
    def freesasa_batch(pdb_paths: _List[str]) -> _List[RunnerResult]:
        """Process-isolated batched FreeSASA runner.

        Submits one apply_async per path against a Pool with
        maxtasksperchild=1, then collects results with .get(timeout=...).
        Worker crashes (SIGSEGV / signal kill) raise on the parent side
        and are recorded as per-structure errors; the next task spawns
        a fresh worker because of maxtasksperchild=1.

        Pool size is capped at 8 because (a) FreeSASA is fast enough that
        the Amdahl's bottleneck is process spawn overhead, not concurrent
        SASA work, and (b) the driver is typically running in parallel
        with the ferritin batch on the same node and we don't want to
        oversubscribe a 120-core box across runners.
        """
        import multiprocessing as _mp
        t_total = _time_perf()
        ctx = _mp.get_context("spawn")
        n_workers = min(8, max(1, len(pdb_paths)))
        out: _List[RunnerResult] = []
        with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            async_results = [
                pool.apply_async(_freesasa_worker, (p,)) for p in pdb_paths
            ]
            for p, ar in zip(pdb_paths, async_results):
                try:
                    # 5 min per structure is a generous ceiling — FreeSASA
                    # finishes in well under a second on everything we've
                    # seen. The cap exists so a hung worker can't stall
                    # the whole batch.
                    out.append(ar.get(timeout=300))
                except Exception as e:
                    out.append(RunnerResult(
                        op="sasa", impl="freesasa",
                        impl_version=_FREESASA_VERSION,
                        pdb_id="", pdb_path=p, elapsed_s=0.0,
                        status="error",
                        error=f"freesasa worker crashed: {type(e).__name__}: {e}",
                        payload={},
                    ))
        # Stamp aggregate wall time on the first ok result for reporting,
        # mirroring the ferritin_batch convention.
        if out and out[0].status == "ok":
            out[0].elapsed_s = _time_perf() - t_total
        return out
