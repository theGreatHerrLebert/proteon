"""Triangulated AMBER96 single-point oracle: proteon vs OpenMM vs GROMACS.

Demonstrates that proteon's AMBER96 energy agrees with OpenMM to 0.03% and
OpenMM agrees with GROMACS to 0.03% on crambin — transitively validating
proteon's force-field math against two independent reference implementations.

Requires:
  * gmx binary on PATH or at GMX env var (default: gromacs-2026.1 build tree)
  * openmm, pdbfixer installed
  * proteon installed with the nonbonded_cutoff override (2026-04-13 onward)
  * amber96.ff available in the gmx distribution

Usage:
  python validation/amber96_oracle_triangulate.py <crambin.pdb>

Produces the three-way single-point comparison on both PDBFixer-prepped
and GROMACS-pdb2gmx-prepped inputs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import openmm
import openmm.app as app
from openmm import unit
from pdbfixer import PDBFixer

import proteon


GMX = os.environ.get(
    "GMX",
    "/scratch/TMAlign/gromacs-2026.1/build/bin/gmx",
)


def pdbfixer_prep(pdb_path: str, out_path: str) -> None:
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(out_path, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)


def gromacs_prep(pdb_path: str, work_dir: str) -> tuple[str, str]:
    """Run gmx pdb2gmx + editconf; return (prepped .pdb, topol .top)."""
    subprocess.run(
        [GMX, "pdb2gmx", "-f", pdb_path, "-o", f"{work_dir}/out.gro",
         "-p", f"{work_dir}/topol.top", "-ff", "amber96",
         "-water", "none", "-ignh"],
        input=b"1\n", cwd=work_dir, check=True, capture_output=True,
    )
    subprocess.run(
        [GMX, "editconf", "-f", f"{work_dir}/out.gro",
         "-o", f"{work_dir}/out.pdb"],
        cwd=work_dir, check=True, capture_output=True,
    )
    subprocess.run(
        [GMX, "editconf", "-f", f"{work_dir}/out.gro",
         "-o", f"{work_dir}/big.gro", "-box", "30", "30", "30", "-c"],
        cwd=work_dir, check=True, capture_output=True,
    )
    return f"{work_dir}/out.pdb", f"{work_dir}/topol.top"


def gromacs_energy(work_dir: str) -> float:
    mdp = f"""\
integrator     = steep
nsteps         = 0
nstenergy      = 1
nstlog         = 1
pbc            = xyz
cutoff-scheme  = Verlet
coulombtype    = Cut-off
rcoulomb       = 14.0
rvdw           = 14.0
vdw-type       = Cut-off
nstlist        = 1
verlet-buffer-tolerance = -1
rlist          = 14.0
continuation   = yes
"""
    (Path(work_dir) / "em.mdp").write_text(mdp)
    subprocess.run(
        [GMX, "grompp", "-f", "em.mdp", "-c", "big.gro", "-p", "topol.top",
         "-o", "em.tpr", "-maxwarn", "5"],
        cwd=work_dir, check=True, capture_output=True,
    )
    subprocess.run(
        [GMX, "mdrun", "-s", "em.tpr", "-c", "em.gro", "-e", "em.edr",
         "-o", "em.trr", "-g", "em.log", "-nt", "1"],
        cwd=work_dir, check=True, capture_output=True,
    )
    # Potential is entry 9 in gmx energy's enumeration.
    proc = subprocess.run(
        [GMX, "energy", "-f", "em.edr"],
        input=b"9\n", cwd=work_dir, check=True, capture_output=True,
    )
    for line in proc.stdout.decode().splitlines():
        if line.strip().startswith("Potential"):
            return float(line.split()[1])
    raise RuntimeError("could not parse Potential from gmx energy output")


def openmm_energy(pdb_path: str) -> float:
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber96.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                             constraints=None)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    plat = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(pdb.topology, system, integrator, plat,
                         {"Threads": "1"})
    sim.context.setPositions(pdb.positions)
    return sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def proteon_energy(pdb_path: str) -> float:
    s = proteon.load(pdb_path)
    r = proteon.compute_energy(
        s, ff="amber96", units="kJ/mol",
        nbl_threshold=10**9, nonbonded_cutoff=1e6,
    )
    return r["total"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdb", nargs="?",
                   default="/scratch/TMAlign/proteon/validation/pdbs/1crn.pdb")
    args = p.parse_args()
    pdb_path = args.pdb

    with tempfile.TemporaryDirectory() as work:
        # Path X: PDBFixer → OpenMM + proteon.
        pdbfixer_out = f"{work}/pdbfixer.pdb"
        pdbfixer_prep(pdb_path, pdbfixer_out)
        e_omm_x = openmm_energy(pdbfixer_out)
        e_fer_x = proteon_energy(pdbfixer_out)

        # Path Y: GROMACS pdb2gmx -ignh → GROMACS + OpenMM.
        gmx_pdb, _gmx_top = gromacs_prep(pdb_path, work)
        e_omm_y = openmm_energy(gmx_pdb)
        e_gmx_y = gromacs_energy(work)

    print(f"AMBER96 single-point on {Path(pdb_path).name} (kJ/mol):\n")
    print(f"  {'input prep':<18s} {'openmm':>10s} {'proteon':>10s} {'gromacs':>10s}")
    print(f"  {'PDBFixer':<18s} {e_omm_x:10.2f} {e_fer_x:10.2f} {'—':>10s}")
    print(f"  {'gmx pdb2gmx':<18s} {e_omm_y:10.2f} {'—':>10s} {e_gmx_y:10.2f}")
    print()
    d_fo = abs(e_fer_x - e_omm_x)
    d_og = abs(e_omm_y - e_gmx_y)
    print(f"  proteon vs OpenMM (PDBFixer):  {d_fo:7.3f} kJ/mol  "
          f"({100*d_fo/max(abs(e_omm_x),1):.3f}%)")
    print(f"  OpenMM vs GROMACS (pdb2gmx):    {d_og:7.3f} kJ/mol  "
          f"({100*d_og/max(abs(e_gmx_y),1):.3f}%)")


if __name__ == "__main__":
    main()
