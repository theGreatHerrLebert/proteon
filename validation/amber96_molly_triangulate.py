"""Triangulated AMBER96 (+OBC GB) single-point oracle: proteon vs OpenMM vs Molly.jl.

Why a third engine. Proteon's AMBER96 numbers are pinned against OpenMM, and
its OBC GB numbers (Phase B, 2026-04-15) are pinned against OpenMM *alone*.
A single oracle cannot distinguish "proteon is correct" from "proteon and
OpenMM share a convention". Molly.jl is an independent Julia implementation
that parses the very same `amber96.xml`, so it breaks that tie per component.

One asymmetry worth stating up front: Molly **ignores** `GBSAOBCForce` when
reading OpenMM XML (it emits a warning) and sources OBC radii and screen
factors from its own element table instead. That makes the GB comparison
genuinely independent in its parameters, not merely in its arithmetic — and
it means a GB delta here may be a radii-table difference rather than a math
error. Read a GB disagreement as a prompt to compare radii first.

Requires:
  * Julia (>= 1.11.5) at $JULIA or on PATH
  * The Molly oracle project instantiated — see docs/ORACLE_SETUP.md
  * openmm, pdbfixer installed
  * proteon installed with the nonbonded_cutoff override (2026-04-13 onward)

Usage:
  python validation/amber96_molly_triangulate.py [structure.pdb ...]
  python validation/amber96_molly_triangulate.py --solvent none

Defaults to test-pdbs/1crn.pdb (crambin), the same structure the BALL and
OpenMM oracles report on, so the numbers line up with devdocs/ORACLE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import openmm
import openmm.app as app
from openmm import unit
from pdbfixer import PDBFixer

import proteon

REPO_ROOT = Path(__file__).resolve().parent.parent
MOLLY_PROJ = REPO_ROOT / "tests" / "oracle" / "julia" / "molly"
MOLLY_SCRIPT = MOLLY_PROJ / "molly_energy_oracle.jl"
CRAMBIN = REPO_ROOT / "test-pdbs" / "1crn.pdb"

# Both OpenMM and Molly must be handed the *same* parameter file, otherwise
# this is a comparison of parameter sets rather than of implementations.
OPENMM_DATA = Path(app.__file__).parent / "data"
AMBER96_XML = OPENMM_DATA / "amber96.xml"

# Map each engine's component naming onto one shared vocabulary. proteon's
# keys are the canonical ones (they are what compute_energy returns).
OPENMM_COMPONENT_NAMES = {
    "HarmonicBondForce": "bond_stretch",
    "HarmonicAngleForce": "angle_bend",
    "ProperTorsionForce": "torsion",
    "ImproperTorsionForce": "improper_torsion",
    "NonbondedForce": "vdw_plus_electrostatic",
    "GBSAOBCForce": "solvation",
    "CMMotionRemover": "cm_motion",
}

COMPONENTS = [
    "bond_stretch",
    "angle_bend",
    "torsion",
    "improper_torsion",
    "vdw",
    "electrostatic",
    "solvation",
    "total",
]


def julia_bin() -> str:
    """Resolve the Julia binary, honouring $JULIA the way ORACLE_SETUP.md does."""
    j = os.environ.get("JULIA")
    if j and Path(j).exists():
        return j
    found = shutil.which("julia")
    if found:
        return found
    raise RuntimeError(
        "Julia not found. Set $JULIA to the binary or put it on PATH — "
        "see docs/ORACLE_SETUP.md (pinned: 1.11.5)."
    )


def pdbfixer_prepped(pdb_path: Path) -> tuple[Path, object, object]:
    """PDBFixer-add-H, written to a temp PDB that every engine loads.

    Identical prep to validation/amber96_oracle.py: all three tools then see
    the same atoms in the same positions, so the only remaining variable is
    each engine's energy implementation. Missing heavy atoms are a hard skip
    rather than a repair — addMissingAtoms() hangs deterministically on a
    non-trivial fraction of wwPDB inputs (PR #47).
    """
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        raise RuntimeError(
            f"{pdb_path.name}: {len(fixer.missingAtoms)} missing heavy atoms; "
            "pre-resolve the structure or pick a different PDB"
        )
    fixer.addMissingHydrogens(7.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp, keepIds=True)
    tmp.close()
    return Path(tmp.name), fixer.topology, fixer.positions


def _split_torsion_force(system, topology) -> None:
    """Split OpenMM's PeriodicTorsionForce into separate proper/improper forces.

    OpenMM reports proper and improper torsions in a single force group, so a
    naive comparison can only see their sum — which hides exactly the term the
    other two engines disagree on. Classify each 4-tuple by topology: a proper
    torsion is a bonded chain i-j-k-l, whereas an improper has one atom bonded
    to the other three. This is a measurement, not an inference from the sum.
    """
    bonded = {i: set() for i in range(system.getNumParticles())}
    for a, b in topology.bonds():
        bonded[a.index].add(b.index)
        bonded[b.index].add(a.index)

    idx = next(
        (i for i in range(system.getNumForces())
         if type(system.getForce(i)).__name__ == "PeriodicTorsionForce"),
        None,
    )
    if idx is None:
        return
    src = system.getForce(idx)

    proper = openmm.PeriodicTorsionForce()
    improper = openmm.PeriodicTorsionForce()
    for t in range(src.getNumTorsions()):
        i, j, k, l, per, phase, k_ = src.getTorsionParameters(t)
        is_chain = (j in bonded[i]) and (k in bonded[j]) and (l in bonded[k])
        (proper if is_chain else improper).addTorsion(i, j, k, l, per, phase, k_)

    system.removeForce(idx)
    system.addForce(proper)
    system.addForce(improper)
    # Tag them so the caller can name the groups without relying on class name,
    # which is identical for both.
    proper.setName("ProperTorsionForce")
    improper.setName("ImproperTorsionForce")


def openmm_components(topology, positions, solvent: str) -> dict:
    """Per-force-group AMBER96 breakdown at NoCutoff, in kJ/mol.

    NoCutoff is mandatory for oracle-grade comparison: proteon's production
    default is a 15 Å cutoff with switching, worth ~1.4% of total energy on
    crambin. That is a perf-vs-accuracy policy choice, not a bug, and it has
    to be held constant to isolate the force-field math.
    """
    xmls = ["amber96.xml"]
    if solvent != "none":
        xmls.append("amber96_obc.xml")
    ff = app.ForceField(*xmls)
    system = ff.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    _split_torsion_force(system, topology)
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    plat = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(topology, system, integrator, plat, {"Threads": "1"})
    sim.context.setPositions(positions)

    out = {
        "total": float(
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    }
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        name = force.getName() if force.getName() in (
            "ProperTorsionForce", "ImproperTorsionForce"
        ) else type(force).__name__
        e = (
            sim.context.getState(getEnergy=True, groups={i})
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        out[OPENMM_COMPONENT_NAMES.get(name, name)] = float(e)
    return out


def molly_components(prepped_pdb: Path, solvent: str) -> dict:
    """Shell out to the Julia oracle and parse its JSON."""
    if not MOLLY_SCRIPT.exists():
        raise RuntimeError(f"Molly oracle script missing: {MOLLY_SCRIPT}")
    cmd = [
        julia_bin(),
        f"--project={MOLLY_PROJ}",
        str(MOLLY_SCRIPT),
        "--ff", str(AMBER96_XML),
        "--solvent", solvent,
        str(prepped_pdb),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Molly oracle failed:\n{proc.stderr[-2000:]}")
    records = json.loads(proc.stdout)
    rec = records[0]
    if "error" in rec:
        raise RuntimeError(f"Molly oracle error: {rec['error']}")
    return rec


def proteon_components(prepped_pdb: Path, solvent: str) -> dict:
    """Single-point proteon energy on the same H-placed PDB.

    Histidine tautomers are normalised first: proteon's residue-name-driven
    typer needs HIS renamed to HID/HIE/HIP based on which Hδ1/Hε2 PDBFixer
    placed (issue #60), whereas OpenMM's amber96.xml detects this internally.
    """
    tmp = tempfile.NamedTemporaryFile(suffix="_proteon.pdb", delete=False).name
    proteon.normalize_histidine_tautomers(str(prepped_pdb), tmp)
    s = proteon.load(tmp)
    ff = "amber96" if solvent == "none" else "amber96_obc"
    # nbl_threshold forces the exact O(N²) path; nonbonded_cutoff=1e6 disables
    # the 15 Å default to match NoCutoff.
    res = proteon.compute_energy(
        s, ff=ff, units="kJ/mol", nbl_threshold=10**9, nonbonded_cutoff=1e6
    )
    os.unlink(tmp)
    return {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}


def _fmt(v) -> str:
    return f"{'—':>14}" if v is None else f"{v:>14.3f}"


def report(name: str, pro: dict, omm: dict, mol: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"{'component':<22}{'proteon':>14}{'OpenMM':>14}{'Molly':>14}"
        f"{'P-vs-M %':>11}{'O-vs-M %':>11}"
    )
    print("-" * 86)

    # OpenMM lumps proper+improper into PeriodicTorsionForce and vdW+Coulomb
    # into NonbondedForce, so it can only be compared on the sums for those.
    derived_omm = dict(omm)
    if "torsion_plus_improper" in omm:
        derived_omm["torsion"] = None
        derived_omm["improper_torsion"] = None
    if "vdw_plus_electrostatic" in omm:
        derived_omm["vdw"] = None
        derived_omm["electrostatic"] = None

    for c in COMPONENTS:
        p, o, m = pro.get(c), derived_omm.get(c), mol.get(c)
        pm = (
            f"{abs(p - m) / max(abs(m), 1.0) * 100:>10.3f}"
            if (p is not None and m is not None)
            else f"{'—':>10}"
        )
        om = (
            f"{abs(o - m) / max(abs(m), 1.0) * 100:>10.3f}"
            if (o is not None and m is not None)
            else f"{'—':>10}"
        )
        print(f"{c:<22}{_fmt(p)}{_fmt(o)}{_fmt(m)}{pm} {om}")

    # The lumped OpenMM groups still carry information — compare them against
    # the sum of Molly's split components.
    print("-" * 86)
    for lumped, parts in (
        ("torsion_plus_improper", ("torsion", "improper_torsion")),
        ("vdw_plus_electrostatic", ("vdw", "electrostatic")),
    ):
        if lumped not in omm:
            continue
        o = omm[lumped]
        m = sum(mol[p] for p in parts)
        p = sum(pro[p] for p in parts if p in pro)
        print(
            f"{lumped:<22}{_fmt(p)}{_fmt(o)}{_fmt(m)}"
            f"{abs(p - m) / max(abs(m), 1.0) * 100:>10.3f} "
            f"{abs(o - m) / max(abs(m), 1.0) * 100:>10.3f}"
        )

    print(
        f"\ncounts (Molly): bonds={mol['n_bonds']} angles={mol['n_angles']} "
        f"torsions={mol['n_torsions']} impropers={mol['n_impropers']} "
        f"atoms={mol['n_atoms']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdbs", nargs="*", default=[str(CRAMBIN)])
    ap.add_argument(
        "--prepped",
        action="store_true",
        help="inputs are already hydrogen-prepared; skip PDBFixer. Required for "
             "reproducible absolute numbers, because PDBFixer's hydrogen "
             "placement is not deterministic across runs (see "
             "tests/oracle/data/1crn_prepped_amber96.pdb).",
    )
    ap.add_argument(
        "--solvent",
        default="obc1",
        choices=["none", "obc1", "obc2", "gbn2"],
        help="implicit solvent model (default: obc1, matching proteon's amber96_obc)",
    )
    args = ap.parse_args()

    pdbs = args.pdbs or [str(CRAMBIN)]
    failures = 0
    for pdb in pdbs:
        prepped = None
        try:
            if args.prepped:
                pdbfile = app.PDBFile(str(pdb))
                prepped, topology, positions = None, pdbfile.topology, pdbfile.positions
                molly_input = Path(pdb)
            else:
                prepped, topology, positions = pdbfixer_prepped(Path(pdb))
                molly_input = prepped
            omm = openmm_components(topology, positions, args.solvent)
            mol = molly_components(molly_input, args.solvent)
            pro = proteon_components(molly_input, args.solvent)
            report(Path(pdb).name, pro, omm, mol)
        except Exception as e:
            failures += 1
            print(f"\n=== {Path(pdb).name} ===\nFAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
        finally:
            if prepped and prepped.exists():
                prepped.unlink()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
