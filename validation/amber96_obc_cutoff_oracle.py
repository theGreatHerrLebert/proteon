"""AMBER96+OBC *CutoffNonPeriodic* oracle: proteon vs OpenMM.

Companion to `amber96_obc_oracle.py` (which validates the exact NoCutoff GB).
This validates proteon's opt-in `CutoffNonPeriodic` GB method — Born-radius
integral and GB pair sum truncated at the cutoff, plus OpenMM's reaction-field
energy shift — against OpenMM's `GBSAOBCForce` under
`nonbondedMethod=CutoffNonPeriodic` at the SAME cutoff.

Contract (matched conditions):
  - both: AMBER96 force field
  - both: OBC implicit solvent
  - both: CutoffNonPeriodic at CUTOFF_NM (GB truncation + reaction-field shift)
  - both: PDBFixer's H placement (identical atoms fed to both)

We compare the **GB component** (total − vacuum) in isolation, so the
LJ/Coulomb cutoff-treatment differences (switching vs reaction field) don't
contaminate the GB parity check. Pass: |ΔGB| / |GB_openmm| < 5% (the same
tolerance the NoCutoff oracle meets).

Usage:
    cd /scratch/TMAlign/proteon
    .venv/bin/python validation/amber96_obc_cutoff_oracle.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAMBIN = REPO_ROOT / "test-pdbs" / "1crn.pdb"

# Cutoff used on BOTH sides. 1.2 nm = 12 Å (> the 1.0 nm CutoffNonPeriodic
# default so the GB descreening is reasonably converged while still truncating).
CUTOFF_NM = 1.2
CUTOFF_ANG = CUTOFF_NM * 10.0


def pdbfixer_prepped(pdb_path: Path):
    from openmm import app
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        raise RuntimeError(
            f"{Path(str(pdb_path)).name}: {len(fixer.missingAtoms)} missing "
            "heavy atoms; pre-resolve the structure or pick a different PDB"
        )
    fixer.addMissingHydrogens(7.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp, keepIds=True)
    tmp.close()
    return Path(tmp.name), fixer.topology, fixer.positions


def openmm_gb_cutoff(topology, positions) -> dict:
    """GB component (total − vacuum) under CutoffNonPeriodic at CUTOFF_NM."""
    from openmm import app, openmm, unit

    def total(with_gb: bool) -> float:
        ff = app.ForceField("amber96.xml", "amber96_obc.xml")
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=CUTOFF_NM * unit.nanometer,
            constraints=None,
            rigidWater=False,
        )
        if not with_gb:
            for i in range(system.getNumForces() - 1, -1, -1):
                name = type(system.getForce(i)).__name__
                if "GBSA" in name or "GeneralizedBorn" in name:
                    system.removeForce(i)
        sim = app.Simulation(
            topology, system, openmm.VerletIntegrator(0.001 * unit.picosecond)
        )
        sim.context.setPositions(positions)
        return float(
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    e_total = total(with_gb=True)
    e_vac = total(with_gb=False)
    return {"total_kj": e_total, "vacuum_kj": e_vac, "gb_kj": e_total - e_vac}


def proteon_gb_cutoff(prepped_pdb: Path) -> dict:
    import proteon

    s = proteon.load(str(prepped_pdb))
    # ff="amber96_obc_cutoff" selects the CutoffNonPeriodic GB method; the GB
    # cutoff follows nonbonded_cutoff. nbl_threshold huge → all-pairs cutoff
    # path (the O(N) neighbor-list path is cross-checked in Rust unit tests).
    result = proteon.compute_energy(
        s,
        ff="amber96_obc_cutoff",
        units="kJ/mol",
        nbl_threshold=10**9,
        nonbonded_cutoff=CUTOFF_ANG,
    )
    return {
        "total_kj": float(result["total"]),
        "vacuum_kj": float(result["total"] - result.get("solvation", 0.0)),
        "gb_kj": float(result.get("solvation", 0.0)),
    }


def main() -> int:
    if not CRAMBIN.exists():
        print(f"missing {CRAMBIN}", file=sys.stderr)
        return 1

    print(f"=== AMBER96+OBC CutoffNonPeriodic oracle: crambin @ {CUTOFF_ANG:.1f} Å ===\n")
    prepped, topology, positions = pdbfixer_prepped(CRAMBIN)
    try:
        print("OpenMM AMBER96+OBC (CutoffNonPeriodic)…")
        om = openmm_gb_cutoff(topology, positions)
        print(f"  GB:      {om['gb_kj']:>14.3f} kJ/mol")

        print("\nProteon AMBER96+OBC_cutoff…")
        fr = proteon_gb_cutoff(prepped)
        print(f"  GB:      {fr['gb_kj']:>14.3f} kJ/mol")
    finally:
        prepped.unlink(missing_ok=True)

    delta_gb = abs(fr["gb_kj"] - om["gb_kj"])
    rel_gb = delta_gb / max(abs(om["gb_kj"]), 1.0)

    print("\n=== Comparison (GB component) ===")
    print(f"  Δ GB:        {delta_gb:>10.3f} kJ/mol  ({rel_gb*100:.2f} %)")

    if abs(fr["gb_kj"]) < 1e-6 and abs(om["gb_kj"]) > 1.0:
        print("\nproteon GB = 0.0 — cutoff GB method not wired?")
        return 1

    if rel_gb < 5e-2:
        print("\nPASS — proteon CutoffNonPeriodic GB matches OpenMM to <5%.")
        return 0
    print("\nFAIL — GB gap exceeds 5%. Re-check the reaction-field shift + truncation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
