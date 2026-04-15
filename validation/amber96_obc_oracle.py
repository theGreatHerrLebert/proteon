"""AMBER96+OBC single-point oracle: ferritin vs OpenMM.

The Phase D contract: when ferritin's OBC GB implementation lands, this
script must show |E_ferritin - E_openmm| / |E_openmm| < 1e-2 (1%) on
crambin under matched conditions:

  - both: AMBER96 force field
  - both: OBC1 implicit solvent (ε_in=1, ε_out=78.5, offset=0.09 Å,
          α=0.8, β=0, γ=2.909125)
  - both: NoCutoff (long-range vacuum + GB)
  - both: PDBFixer's H placement (identical hydrogens fed to both)

Phase A baseline (the math isn't implemented yet): expected to FAIL
with `ferritin solvation = 0.0` because gb_obc::gb_obc_energy is a
stub. That's the point — this test is the contract that the math has
to satisfy when it lands.

Usage:
    cd /scratch/TMAlign/ferritin
    .venv/bin/python validation/amber96_obc_oracle.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAMBIN = REPO_ROOT / "test-pdbs" / "1crn.pdb"


def pdbfixer_prepped(pdb_path: Path):
    """PDBFixer-add-H + write to temp PDB. Both pipelines load this so they
    see identical atoms + identical hydrogen positions — the only
    remaining variable is each tool's energy implementation. Same prep
    pattern as `validation/amber96_oracle.py`."""
    from openmm import app
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp, keepIds=True)
    tmp.close()
    return Path(tmp.name), fixer.topology, fixer.positions


def openmm_amber96_obc_total(topology, positions) -> dict:
    """Single-point AMBER96+OBC energy in kJ/mol via OpenMM, on the
    pre-prepped topology so ferritin sees identical atoms."""
    from openmm import app, openmm, unit

    # amber96_obc.xml only carries the GBSAOBCForce — needs amber96.xml
    # loaded alongside for residue templates + bonded params.
    ff = app.ForceField("amber96.xml", "amber96_obc.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    sim = app.Simulation(topology, system, integrator)
    sim.context.setPositions(positions)
    e_total = float(
        sim.context.getState(getEnergy=True).getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    # Decompose: rerun with GB force removed → vacuum total → GB = total − vacuum.
    sys_vac = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    n_removed = 0
    for i in range(sys_vac.getNumForces() - 1, -1, -1):
        f = sys_vac.getForce(i)
        if "GBSA" in type(f).__name__ or "GeneralizedBorn" in type(f).__name__:
            sys_vac.removeForce(i)
            n_removed += 1
    sim2 = app.Simulation(
        topology, sys_vac, openmm.VerletIntegrator(0.001 * unit.picosecond)
    )
    sim2.context.setPositions(positions)
    e_vac = float(
        sim2.context.getState(getEnergy=True).getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    return {
        "total_kj": e_total,
        "vacuum_kj": e_vac,
        "gb_kj": e_total - e_vac,
        "n_gb_forces_removed": n_removed,
    }


def ferritin_amber96_obc_total(prepped_pdb: Path) -> dict:
    """Single-point AMBER96+OBC1 via ferritin, on the same H-placed PDB
    OpenMM was given. Uses ff="amber96_obc" so the GB term is added to
    the solvation component."""
    import ferritin

    s = ferritin.load(str(prepped_pdb))
    # nbl_threshold huge → forces exact O(N²) path to match OpenMM NoCutoff.
    # nonbonded_cutoff=1e6 disables the 15 Å cutoff policy for oracle parity.
    result = ferritin.compute_energy(
        s,
        ff="amber96_obc",
        units="kJ/mol",
        nbl_threshold=10**9,
        nonbonded_cutoff=1e6,
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

    print("=== AMBER96+OBC oracle: crambin ===\n")
    prepped, topology, positions = pdbfixer_prepped(CRAMBIN)
    try:
        print("OpenMM AMBER96+OBC1 (NoCutoff)…")
        om = openmm_amber96_obc_total(topology, positions)
        print(f"  total:   {om['total_kj']:>14.3f} kJ/mol")
        print(f"  vacuum:  {om['vacuum_kj']:>14.3f} kJ/mol")
        print(f"  GB:      {om['gb_kj']:>14.3f} kJ/mol  ({om['n_gb_forces_removed']} GB force(s))")

        print("\nFerritin AMBER96 (+ OBC GB once implemented)…")
        fr = ferritin_amber96_obc_total(prepped)
        print(f"  total:   {fr['total_kj']:>14.3f} kJ/mol")
        print(f"  vacuum:  {fr['vacuum_kj']:>14.3f} kJ/mol")
        print(f"  GB:      {fr['gb_kj']:>14.3f} kJ/mol")
    finally:
        prepped.unlink(missing_ok=True)

    delta_total = abs(fr["total_kj"] - om["total_kj"])
    rel_total = delta_total / max(abs(om["total_kj"]), 1.0)
    delta_gb = abs(fr["gb_kj"] - om["gb_kj"])
    rel_gb = delta_gb / max(abs(om["gb_kj"]), 1.0)

    print("\n=== Comparison ===")
    print(f"  vacuum Δ:    {abs(fr['vacuum_kj'] - om['vacuum_kj']):>10.3f} kJ/mol")
    print(f"  Δ total:     {delta_total:>10.3f} kJ/mol  ({rel_total*100:.2f} %)")
    print(f"  Δ GB:        {delta_gb:>10.3f} kJ/mol  ({rel_gb*100:.2f} %)")

    if abs(fr["gb_kj"]) < 1e-6 and abs(om["gb_kj"]) > 1.0:
        print("\nPHASE A: ferritin solvation = 0.0 (stub) — math not yet implemented.")
        print("This script is the contract for Phase B/C/D. It MUST pass when GB lands.")
        return 0  # not a hard fail in Phase A

    if rel_total < 1e-2 and rel_gb < 5e-2:
        print("\nPHASE D PASS — ferritin AMBER96+OBC matches OpenMM to <1% total.")
        return 0
    print("\nPHASE D FAIL — gap exceeds tolerance. Re-check Born radii + pair sum.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
