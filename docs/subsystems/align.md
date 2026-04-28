# Alignment

Structural alignment over the TM-align / US-align family. Four flavors:

- **TM-align** — sequence-independent, sequence-order-preserving, single-chain.
- **SOI-align** — sequence-order independent (handles circular permutations,
  domain swaps).
- **FlexAlign** — flexible hinge-based; allows two rigid bodies to align
  independently.
- **MM-align** — multi-chain / complex alignment.

Each comes in three shapes:

| Shape | Function suffix | Returns |
|-------|-----------------|---------|
| Single pair | `tm_align` | `AlignResult` |
| One vs many | `tm_align_one_to_many` | `list[AlignResult]` |
| Many vs many | `tm_align_many_to_many` | matrix-shaped result |

Substitute `soi_align`, `flex_align`, `mm_align` for the other flavors.

## Example

```python
import proteon

a = proteon.load("1crn.pdb")
b = proteon.load("2igh.pdb")

result = proteon.tm_align(a, b)
print(f"TM-score (chain1): {result.tm_score_chain1:.4f}")
print(f"TM-score (chain2): {result.tm_score_chain2:.4f}")
print(f"RMSD:              {result.rmsd:.2f} Å")
print(f"Aligned length:    {result.aligned_length}")
```

## Choosing a method

| If... | Use |
|-------|-----|
| Two single-chain proteins, sequence order intact | `tm_align` |
| Possible circular permutation or domain swap | `soi_align` |
| Two-domain proteins with a flexible hinge | `flex_align` |
| Multi-chain complexes (chain assignment matters) | `mm_align` |

## API reference

The mkdocstrings handler reads the live module below and renders the curated
public surface. Function order matches `proteon/align.py`.

::: proteon.align
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
