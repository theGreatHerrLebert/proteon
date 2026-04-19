# Fold preservation comparison — summary

Sample: 1000 PDBs (seed=42) from the 50K corpus.

|                          | Proteon CHARMM19+EEF1 | OpenMM CHARMM36+OBC2 |
|--------------------------|------------------------|----------------------|
| n success                | 949/1000 (94.9%) | 928/1000 (92.8%) |
| TM mean                  | 0.9878 | 0.9948 |
| TM median                | 0.9945 | 0.9991 |
| TM p01 / p05             | 0.8538 / 0.9722 | 0.9179 / 0.9882 |
| TM min                   | 0.4666 | 0.6536 |
| RMSD mean (Å)            | 0.581 | 0.340 |
| RMSD median (Å)          | 0.585 | 0.213 |
| RMSD max (Å)             | 1.379 | 2.974 |
| Wall total (min)         | 14.9 | 449.7 |
| Throughput (struct/s)    | 1.06 | 0.03 |

Paired (same structure, both tools): n=886