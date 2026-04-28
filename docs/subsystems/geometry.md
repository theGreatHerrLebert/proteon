# Geometry

Coordinate transforms, RMSD, and bulk geometric analyses (dihedrals, contact
maps, distance matrices, radius of gyration).

The split between `proteon.geometry` and `proteon.analysis` is historical:
`geometry` holds rigid-body transforms and primitive operations; `analysis`
holds derived per-structure quantities. Both are stable.

## Example

```python
import proteon

s = proteon.load("1crn.pdb")

phi, psi, omega = proteon.backbone_dihedrals(s)
ca = proteon.extract_ca_coords(s)
cm = proteon.contact_map(ca, cutoff=8.0)
rg = proteon.radius_of_gyration(s)
```

## API reference

### `proteon.geometry`

::: proteon.geometry
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### `proteon.analysis`

::: proteon.analysis
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
