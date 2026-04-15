"""Regenerate `ferritin-connector/data/amber96_obc.ini` from OpenMM's XMLs.

Joins `amber96.xml` (type-id -> class) with `amber96_obc.xml`
(type-id -> radius nm + HCT scale) and writes an INI file keyed by
AMBER atom-type class name. Radii are converted nm -> Å.

Invariant enforced: every AMBER class must map to exactly one
(radius, scale) pair across all type instances. The extraction
aborts if this is violated, because the INI format assumes it.

Usage:
    .venv/bin/python validation/extract_amber96_obc_params.py
"""

from __future__ import annotations

import collections
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OM_DATA = REPO_ROOT / ".venv/lib/python3.10/site-packages/openmm/app/data"
FF_XML = OM_DATA / "amber96.xml"
OBC_XML = OM_DATA / "amber96_obc.xml"
OUT_INI = REPO_ROOT / "ferritin-connector/data/amber96_obc.ini"


def main() -> int:
    ff = ET.parse(FF_XML).getroot()
    obc = ET.parse(OBC_XML).getroot()

    id_to_class = {t.get("name"): (t.get("class"), t.get("element")) for t in ff.findall(".//Type")}
    id_to_obc = {a.get("type"): (a.get("radius"), a.get("scale")) for a in obc.findall(".//Atom")}

    klass_to_rs: dict[str, set] = collections.defaultdict(set)
    for tid, (r_nm, s) in id_to_obc.items():
        cls, elem = id_to_class[tid]
        klass_to_rs[cls].add((float(r_nm) * 10.0, float(s), elem))

    offenders = {k: v for k, v in klass_to_rs.items() if len(v) > 1}
    if offenders:
        print(f"ERROR: non-unique OBC params for {len(offenders)} classes:", file=sys.stderr)
        for k, vs in offenders.items():
            print(f"  {k}: {sorted(vs)}", file=sys.stderr)
        return 1

    rows = sorted(
        (k, next(iter(v))) for k, v in klass_to_rs.items()
    )

    lines = [
        "; AMBER96 OBC GB per-atom parameters.",
        "; Source: OpenMM amber96.xml (type->class map) + amber96_obc.xml (type->radius nm + scale).",
        "; Joined and converted nm->A by validation/extract_amber96_obc_params.py.",
        "; Regenerate with:  .venv/bin/python validation/extract_amber96_obc_params.py",
        ";",
        "; Radii in Angstroms. Scales are dimensionless HCT overlap factors.",
        "",
        "[OBCSolvation]",
        "ver:version key:class value:radius value:scale",
        "@unit_radius=Angstrom",
        ";",
        ";  Rev  class    radius  scale   element",
        ";  --- -------- ------- ------  -------",
    ]
    for cls, (r_ang, scale, elem) in rows:
        lines.append(f"   1.0  {cls:<8} {r_ang:>6.4f} {scale:>6.3f}    ; {elem}")
    lines.append("")

    OUT_INI.write_text("\n".join(lines))
    print(f"wrote {OUT_INI}  ({len(rows)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
