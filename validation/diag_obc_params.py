"""Diagnose the 16% GB gap between proteon and OpenMM on crambin.

Dumps per-atom (OBC radius, OBC scale, charge) from both tools on
identical PDBFixer-prepped structures and prints the atoms whose
parameters diverge. Atom order is preserved across tools because both
operate on the same `fixer.positions` coordinate vector and topology.
"""

from __future__ import annotations

import configparser
import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAMBIN = REPO_ROOT / "test-pdbs" / "1crn.pdb"
OBC_INI = REPO_ROOT / "proteon-connector/data/amber96_obc.ini"


def load_proteon_class_to_obc() -> dict[str, tuple[float, float]]:
    """Parse amber96_obc.ini -> {amber_class: (radius_A, scale)}."""
    table: dict[str, tuple[float, float]] = {}
    section = None
    for raw in OBC_INI.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "@")):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        if line.startswith(("ver:", "key:", "value:")):
            continue
        if section != "OBCSolvation":
            continue
        payload = line.split(";", 1)[0].strip()
        fields = payload.split()
        if len(fields) < 4:
            continue
        _ver, cls, r, s = fields[0], fields[1], float(fields[2]), float(fields[3])
        table[cls] = (r, s)
    return table


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
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp, keepIds=True)
    tmp.close()
    return Path(tmp.name), fixer.topology, fixer.positions


def proteon_per_atom(prepped_pdb: Path, cls_to_obc):
    import proteon
    import proteon_connector

    s = proteon.load(str(prepped_pdb))
    # Reuse proteon's shared _get_ptr trick (same as compute_energy does).
    from proteon.align import _get_ptr  # type: ignore
    topo = proteon_connector.py_forcefield.dump_topology(_get_ptr(s), "amber96")
    idents = topo["atom_identities"]
    types = topo["atom_types"]
    charges = topo["atom_charges"]
    out = []
    for (residx, resname, atom_name), atype, q in zip(idents, types, charges):
        r_scale = cls_to_obc.get(atype, (None, None))
        out.append({
            "residx": residx,
            "resname": resname,
            "atom_name": atom_name,
            "amber_type": atype,
            "charge": q,
            "radius_A": r_scale[0],
            "scale": r_scale[1],
        })
    return out


def openmm_per_atom(topology, positions):
    """Extract (charge, radius_A, scale) per atom from a freshly-built
    OpenMM GBSAOBCForce, in the same atom order as the topology."""
    from openmm import app, openmm, unit

    ff = app.ForceField("amber96.xml", "amber96_obc.xml")
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

    gbforce = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        name = type(f).__name__
        if "GBSAOBCForce" in name:
            gbforce = f
            break
    if gbforce is None:
        raise RuntimeError("No GBSAOBCForce found in AMBER96+OBC system")

    # Iterate atoms in topology order (same order OpenMM used).
    atoms = list(topology.atoms())
    out = []
    for i, atom in enumerate(atoms):
        charge, radius_nm, scale = gbforce.getParticleParameters(i)
        charge = charge.value_in_unit(unit.elementary_charge)
        radius_nm = radius_nm.value_in_unit(unit.nanometer)
        # radius_nm may or may not be a Quantity; guard both.
        if hasattr(scale, "value_in_unit"):
            scale = scale.value_in_unit(unit.dimensionless)
        out.append({
            "residx": atom.residue.index,
            "resname": atom.residue.name,
            "atom_name": atom.name,
            "charge": float(charge),
            "radius_A": float(radius_nm) * 10.0,
            "scale": float(scale),
        })
    return out


def main() -> int:
    cls_to_obc = load_proteon_class_to_obc()
    prepped, topology, positions = pdbfixer_prepped(CRAMBIN)
    try:
        ferr = proteon_per_atom(prepped, cls_to_obc)
        om = openmm_per_atom(topology, positions)
    finally:
        prepped.unlink(missing_ok=True)

    print(f"proteon atoms: {len(ferr)}   openmm atoms: {len(om)}")
    if len(ferr) != len(om):
        print("WARNING: atom counts differ — cannot align 1:1")
        # Fall back to resname+atom_name lookup
        om_by_key = {(a["residx"], a["resname"], a["atom_name"]): a for a in om}
        ferr_by_key = {(a["residx"], a["resname"], a["atom_name"]): a for a in ferr}
        common = sorted(set(ferr_by_key) & set(om_by_key))
        pairs = [(ferr_by_key[k], om_by_key[k]) for k in common]
        only_ferr = sorted(set(ferr_by_key) - set(om_by_key))
        only_om = sorted(set(om_by_key) - set(ferr_by_key))
        print(f"in both: {len(pairs)}   proteon-only: {len(only_ferr)}   openmm-only: {len(only_om)}")
        if only_ferr[:5]:
            print(f"  sample proteon-only: {only_ferr[:5]}")
        if only_om[:5]:
            print(f"  sample openmm-only: {only_om[:5]}")
    else:
        pairs = list(zip(ferr, om))

    n_total = len(pairs)
    n_radius_diff = 0
    n_scale_diff = 0
    n_charge_diff = 0
    class_mismatch_by_class: dict[str, int] = {}
    sample_rows: list[str] = []
    # Aggregate class -> (n_diffs, n_total)
    class_stats: dict[str, dict] = {}

    for fr, om_row in pairs:
        cls = fr["amber_type"]
        st = class_stats.setdefault(cls, {"n": 0, "r_same": 0, "s_same": 0})
        st["n"] += 1
        if fr["radius_A"] is None:
            continue
        r_match = abs(fr["radius_A"] - om_row["radius_A"]) < 1e-4
        s_match = abs(fr["scale"] - om_row["scale"]) < 1e-6
        q_match = abs(fr["charge"] - om_row["charge"]) < 1e-4
        if r_match:
            st["r_same"] += 1
        if s_match:
            st["s_same"] += 1
        if not r_match:
            n_radius_diff += 1
            class_mismatch_by_class[cls] = class_mismatch_by_class.get(cls, 0) + 1
        if not s_match:
            n_scale_diff += 1
        if not q_match:
            n_charge_diff += 1
        if (not r_match or not s_match) and len(sample_rows) < 25:
            sample_rows.append(
                f"  res {fr['resname']}:{fr['atom_name']} (class={cls})  "
                f"proteon r={fr['radius_A']:.4f}/s={fr['scale']:.3f}  "
                f"openmm r={om_row['radius_A']:.4f}/s={om_row['scale']:.3f}"
            )

    # Build the per-atom charge delta table
    charge_rows = []
    tot_q_ferr = 0.0
    tot_q_om = 0.0
    ss_ferr = 0.0
    ss_om = 0.0
    for fr, om_row in pairs:
        tot_q_ferr += fr["charge"]
        tot_q_om += om_row["charge"]
        ss_ferr += fr["charge"] ** 2
        ss_om += om_row["charge"] ** 2
        if abs(fr["charge"] - om_row["charge"]) >= 1e-4:
            charge_rows.append((
                fr["residx"], fr["resname"], fr["atom_name"], fr["amber_type"],
                fr["charge"], om_row["charge"], fr["charge"] - om_row["charge"]
            ))

    print(f"\n=== Diff summary ({n_total} atom pairs) ===")
    print(f"  radius mismatches: {n_radius_diff}")
    print(f"  scale  mismatches: {n_scale_diff}")
    print(f"  charge mismatches: {n_charge_diff}")
    print(f"\nCharge totals:  proteon Σq={tot_q_ferr:+.4f}  openmm Σq={tot_q_om:+.4f}")
    print(f"Charge² totals: proteon Σq²={ss_ferr:.4f}  openmm Σq²={ss_om:.4f}"
          f"  Δ(Σq²)={ss_ferr-ss_om:+.4f}  ({(ss_ferr/ss_om - 1)*100:+.1f}%)")
    if charge_rows:
        print("\nCharge mismatches (top 30 by |Δq|):")
        charge_rows.sort(key=lambda r: -abs(r[6]))
        print(f"  {'resid':>5} {'resn':<4} {'atom':<5} {'class':<5}  "
              f"{'proteon q':>10} {'openmm q':>10}  {'Δq':>9}")
        for r in charge_rows[:30]:
            print(f"  {r[0]:>5} {r[1]:<4} {r[2]:<5} {r[3]:<5}  "
                  f"{r[4]:>+10.4f} {r[5]:>+10.4f}  {r[6]:>+9.4f}")
        # Histogram by residue+atom-name to see if it's a whole residue or
        # a single atom type.
        by_pair = {}
        for r in charge_rows:
            k = (r[1], r[2])
            by_pair[k] = by_pair.get(k, 0) + 1
        print("\nMismatch counts by (resname, atom_name):")
        for k, n in sorted(by_pair.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {k[0]:<4} {k[1]:<5}  {n}")
    if n_radius_diff + n_scale_diff:
        print("\nMismatches grouped by proteon amber class (n_diffs / n_total):")
        for cls, n in sorted(class_mismatch_by_class.items(), key=lambda kv: -kv[1]):
            total = class_stats[cls]["n"]
            print(f"  {cls:<6}  {n}/{total}")
        print("\nSample mismatched rows (max 25):")
        for row in sample_rows:
            print(row)
    else:
        print("No OBC parameter differences detected — gap comes from elsewhere.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
