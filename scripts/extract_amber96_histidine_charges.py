"""One-shot extractor: pull AMBER96 HID/HIE/HIP partial charges + atom-type
classes from OpenMM's vendored amber96.xml and emit them in proteon's
amber96.ini format.

Source-of-truth: whatever OpenMM ships in
``site-packages/openmm/app/data/amber96.xml`` for the version of OpenMM
proteon's EVIDENT image is built against. The ini file should pin the
OpenMM version in a comment header at the new HID/HIE/HIP sections.

This script is **not run in CI** — it's run once when proteon's amber96.ini
needs to be (re-)synced with OpenMM. The output is committed as static
data.

Run:
    python scripts/extract_amber96_histidine_charges.py

Emits:
- amber96.ini delta to stdout (proteon residue-name × atom-name × charge × class lines)
- proteon-connector/data/fragments/HID.json, HIE.json, HIP.json (overwrites)

Naming map (OpenMM → proteon):
- HB2 → 1HB, HB3 → 2HB        (methylene Hs)
- H1 → 1H, H2 → 2H, H3 → 3H   (N-terminal Hs)
- HD1, HD2, HE1, HE2          unchanged (imidazole ring + ε)
- everything else             unchanged (N, CA, HA, CB, CG, ND1, CE1, NE2, CD2, C, O, OXT)
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import openmm.app  # noqa: F401  (used to locate the data dir)


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = REPO_ROOT / "proteon-connector" / "data" / "fragments"
HIS_TEMPLATE = FRAGMENT_DIR / "HIS.json"


# OpenMM atom-name → proteon atom-name remap.
ATOM_NAME_MAP = {
    "HB2": "1HB",
    "HB3": "2HB",
    "H1":  "1H",
    "H2":  "2H",
    "H3":  "3H",
}


# Mapping: (proteon residue prefix, OpenMM residue name)
# Mid-chain, N-terminal, C-terminal × HID/HIE/HIP.
# proteon convention: "<base>" for mid-chain, "<base>-N" for N-term, "<base>-C" for C-term.
RESIDUE_MAP = [
    ("HID",   "HID"),
    ("HID-N", "NHID"),
    ("HID-C", "CHID"),
    ("HIE",   "HIE"),
    ("HIE-N", "NHIE"),
    ("HIE-C", "CHIE"),
    ("HIP",   "HIP"),
    ("HIP-N", "NHIP"),
    ("HIP-C", "CHIP"),
]


def _load_amber96_xml() -> ET.Element:
    xml_path = Path(openmm.app.__file__).parent / "data" / "amber96.xml"
    if not xml_path.is_file():
        raise SystemExit(f"OpenMM amber96.xml not found at {xml_path}")
    return ET.parse(xml_path).getroot(), xml_path


def _build_type_lookup(root: ET.Element) -> tuple[dict[str, str], dict[str, float]]:
    """Return (type_id -> atom_class, type_id -> partial_charge)."""
    type_classes: dict[str, str] = {}
    atomtypes = root.find(".//AtomTypes")
    if atomtypes is not None:
        for at in atomtypes.findall("Type"):
            type_classes[at.get("name")] = at.get("class")
    type_charges: dict[str, float] = {}
    nbforce = root.find(".//NonbondedForce")
    if nbforce is not None:
        for atom in nbforce.findall("Atom"):
            tid = atom.get("type")
            q = atom.get("charge")
            if tid and q is not None:
                type_charges[tid] = float(q)
    return type_classes, type_charges


def _emit_ini_section(
    root: ET.Element,
    proteon_prefix: str,
    openmm_resname: str,
    type_classes: dict[str, str],
    type_charges: dict[str, float],
) -> list[str]:
    """One ini section for the (proteon_prefix, openmm_resname) pair.

    Output line shape mirrors existing proteon AMBER96 entries:
        '   1.0 HID:N      -0.41570 N    '
    """
    residue = None
    for r in root.findall(".//Residue"):
        if r.get("name") == openmm_resname:
            residue = r
            break
    if residue is None:
        raise SystemExit(f"residue {openmm_resname} not found in amber96.xml")

    lines: list[str] = []
    for atom in residue.findall("Atom"):
        openmm_name = atom.get("name") or ""
        proteon_name = ATOM_NAME_MAP.get(openmm_name, openmm_name)
        type_id = atom.get("type") or ""
        cls = type_classes.get(type_id, "?")
        q = type_charges.get(type_id)
        if q is None:
            raise SystemExit(f"no charge for {openmm_resname}:{openmm_name} (type {type_id})")
        # Match the existing AMBER96 ini formatting exactly:
        #   '   1.0 HIS:N      -0.41570 N    '
        lines.append(
            f"   1.0 {proteon_prefix}:{proteon_name:<6}{q:>9.5f} {cls:<5}"
        )
    return lines


def _is_hd1(name: str) -> bool:
    return name == "HD1"


def _is_he2(name: str) -> bool:
    return name == "HE2"


def _build_fragment(his_template: dict, target_residue: str) -> dict:
    """Generate HID/HIE/HIP fragment template from HIS.json.

    The strategy mirrors HIS.json: keep the atom-superset (which already
    contains HD1, HE2, OXT, 1H/2H/3H), and adjust each variant's `delete`
    list to drop the ε or δ H that doesn't belong in this tautomer.

      target_residue == 'HID' → drop HE2 from every variant
      target_residue == 'HIE' → drop HD1 from every variant
      target_residue == 'HIP' → drop neither (both Hs present)
    """
    new = json.loads(json.dumps(his_template))  # deep copy
    new["name"] = target_residue
    new["names"] = [target_residue]

    # Update each variant's `delete` list and rename the variant id.
    drop_extra: list[str] = []
    if target_residue == "HID":
        drop_extra = ["HE2"]
    elif target_residue == "HIE":
        drop_extra = ["HD1"]
    elif target_residue == "HIP":
        drop_extra = []
    else:
        raise SystemExit(f"unknown target_residue {target_residue}")

    for v in new["variants"]:
        old_name = v["name"]  # 'HIS', 'HIS-M', 'HIS-N', 'HIS-C'
        v["name"] = old_name.replace("HIS", target_residue)
        existing_delete = list(v.get("delete") or [])
        for atom in drop_extra:
            if atom not in existing_delete:
                existing_delete.append(atom)
        if existing_delete:
            v["delete"] = existing_delete
    return new


def main() -> int:
    root, xml_path = _load_amber96_xml()
    type_classes, type_charges = _build_type_lookup(root)

    print(
        "; AMBER96 HID/HIE/HIP charges + atom types,\n"
        f"; extracted from {xml_path}\n"
        f"; via scripts/extract_amber96_histidine_charges.py\n"
        ";\n"
        "; HID = δ-tautomer (Hδ1 only,    neutral)\n"
        "; HIE = ε-tautomer (Hε2 only,    neutral)\n"
        "; HIP = both Hs    (Hδ1 + Hε2,   +1 charge)\n"
        ";\n"
        "; Add these blocks to [ChargesAndTypeNames] in amber96.ini.\n"
        "; Existing 'HIS' entries are untouched (backward-compat alias).\n"
    )
    for proteon_prefix, openmm_resname in RESIDUE_MAP:
        for line in _emit_ini_section(
            root, proteon_prefix, openmm_resname, type_classes, type_charges
        ):
            print(line)
        print()

    his = json.loads(HIS_TEMPLATE.read_text(encoding="utf-8"))
    for target in ("HID", "HIE", "HIP"):
        out = FRAGMENT_DIR / f"{target}.json"
        new = _build_fragment(his, target)
        out.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        print(f"# wrote {out.relative_to(REPO_ROOT)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
