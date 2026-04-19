# Foldseek vs proteon Retrieval Diagnostics

## Headline

- Queries: 50
- Primary threshold: TM >= 0.7
- Candidate traces present: False
- Skipped truth candidates: 61 across 22 queries
- Skipped proteon queries: 1

## Recall By Threshold

| TM threshold | Foldseek mean | proteon mean | proteon - Foldseek |
|---:|---:|---:|---:|
| 0.5 | 0.4127 | 0.2627 | -0.15 |
| 0.7 | 0.6578 | 0.5035 | -0.1543 |
| 0.9 | 0.9094 | 0.8761 | -0.0333 |

## Top-1 Classes

- both_exact: 14
- foldseek_only: 6
- proteon_only: 2
- both_miss: 27
- no_truth: 1

## Primary Recall Classes

- foldseek_better: 20
- proteon_better: 6
- tie: 24
- missing: 0

## Worst proteon Deltas

| Query | Truth | Truth TM | Foldseek top | proteon top | Foldseek recall | proteon recall | Delta | proteon truth rank |
|---|---|---:|---|---|---:|---:|---:|---:|
| 2l19.pdb | 1d2a.pdb | 0.7786 | 1d2a.pdb | 2h3i.pdb | 1.0 | 0.0 | -1.0 |  |
| 2mf6.pdb | 1nqn.pdb | 0.9373 | 2a8g.pdb | 2mqn.pdb | 1.0 | 0.0 | -1.0 |  |
| 6hx7.pdb | 2w7j.pdb | 0.7192 | 2oat.pdb | 2oat.pdb | 1.0 | 0.0 | -1.0 |  |
| 7fcc.pdb | 8qlo.pdb | 0.7487 | 8qlo.pdb | 8ju8.pdb | 1.0 | 0.0 | -1.0 |  |
| 7p8y.pdb | 1lcw.pdb | 0.8117 | 7alx.cif | 7alx.cif | 1.0 | 0.25 | -0.75 |  |
| 5b3k.pdb | 1f4p.pdb | 0.8466 | 1f4p.pdb | 1f4p.pdb | 0.6667 | 0.1667 | -0.5 |  |
| 6iiu.pdb | 9p1t.pdb | 0.8022 | 6dyj.pdb | 6dyg.pdb | 0.5 | 0.0 | -0.5 |  |
| 2i0i.pdb | 1gm1.pdb | 0.7641 | 1gm1.pdb | 6xa6.pdb | 0.6667 | 0.3333 | -0.3334 |  |
| 6kgn.pdb | 2dw4.pdb | 0.9764 | 7e0g.pdb | 7e0g.pdb | 1.0 | 0.6667 | -0.3333 |  |
| 8jif.cif | 1h9q.pdb | 0.891 | 5jnc.pdb | 5jnc.pdb | 0.2333 | 0.0 | -0.2333 |  |

## Best proteon Deltas

| Query | Truth | Truth TM | Foldseek top | proteon top | Foldseek recall | proteon recall | Delta | proteon truth rank |
|---|---|---:|---|---|---:|---:|---:|---:|
| 9iqt.pdb | 9ppy.pdb | 0.9448 | 9iy8.pdb | 9iy8.pdb | 0.3 | 0.5 | 0.2 |  |
| 3sdx.pdb | 6c6a.pdb | 0.9662 | 3vwk.pdb | 3vwk.pdb | 0.2105 | 0.3158 | 0.1053 |  |
| 3g0e.pdb | 2i0v.pdb | 0.9396 | 8jot.pdb | 2i0v.pdb | 0.1489 | 0.1915 | 0.0426 |  |
| 9p1t.pdb | 2j4y.pdb | 0.8557 | 2y02.pdb | 6i9k.pdb | 0.2069 | 0.2414 | 0.0345 |  |
| 9ob3.cif | 9ob2.cif | 0.9991 | 9ob2.cif | 9ob2.cif | 0.1951 | 0.2195 | 0.0244 |  |
| 3bz3.pdb | 6gcx.pdb | 0.9704 | 6gcx.pdb | 6gcx.pdb | 0.1429 | 0.1607 | 0.0178 |  |
| 9yfm.cif | 3lfo.pdb | 0.7929 | 9e7f.cif | 5ywf.pdb | 0.0 | 0.0 | 0.0 |  |
| 9gs4.cif | 8ov3.pdb | 0.9984 | 8ov3.pdb | 8ov3.pdb | 1.0 | 1.0 | 0.0 |  |
| 9g34.pdb | 5oz8.pdb | 0.9989 | 5p2l.pdb | 5lwu.pdb | 0.36 | 0.36 | 0.0 |  |
| 9fqa.cif | 8jcm.pdb | 0.9985 | 8gtv.pdb | 8z46.cif | 0.1552 | 0.1552 | 0.0 |  |
