# Third-Party Notices

proteon incorporates derivative work from, and paper-level inspiration
from, the upstream projects listed below. This file consolidates the
license notices required by those upstreams.

proteon itself is distributed under the MIT License; see `LICENSE`.

The `proteon-vina` crate is distributed under the Apache License,
Version 2.0, matching its upstream (see `proteon-vina/LICENSE` and the
AutoDock Vina entry below).

---

## 1. TM-align and US-align — Zhang group (University of Michigan)

**Used in:** `proteon-align/src/core/**` (TM-align port) and
`proteon-align/src/ext/**` (US-align port including `SOIalign`,
`MMalign`, `NWalign`, `se`, `HwRMSD`, `flexalign`, `BLOSUM`).

**Upstream:** https://zhanggroup.org/US-align/

**License terms (reproduced from TMalign.cpp and USalign.cpp):**

> DISCLAIMER: Permission to use, copy, modify, and distribute the
> Software for any purpose, with or without fee, is hereby granted,
> provided that the notices on the head, the reference information,
> and this copyright notice appear in all copies or substantial
> portions of the Software. It is provided "as is" without express or
> implied warranty.

**Primary references:**
- Zhang & Skolnick. *Nucleic Acids Research* 33, 2302-2309 (2005).
- Zhang, Shine, Pyle & Zhang. *Nature Methods* 19(9), 1109-1115 (2022).
- Zhang & Pyle. *iScience* 25(10), 105218 (2022).

---

## 2. MMseqs2 — Steinegger & Söding

**Used in:** `proteon-search/src/**` (k-mer prefilter, ungapped and
gapped Smith-Waterman, PSSM, reduced alphabet, DB I/O).

**Upstream:** https://github.com/soedinglab/MMseqs2

**License (reproduced from MMseqs2 `LICENSE.md`):**

> The MIT License (MIT)
>
> Copyright © 2024 The MMseqs2 Development Team
>
> Permission is hereby granted, free of charge, to any person
> obtaining a copy of this software and associated documentation
> files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use,
> copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
> OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
> HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
> WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
> OTHER DEALINGS IN THE SOFTWARE.

**Primary reference:**
- Steinegger & Söding. *Nature Biotechnology* 35(11), 1026-1028 (2017).

---

## 3. BiochemicalAlgorithms.jl — Hildebrandt et al.

**Used in:** `proteon-connector/src/reconstruct.rs` (`match_points`,
`_get_two_reference_atoms`, `_reconstruct_fragment!`).

**Upstream:** https://github.com/hildebrandtlab/BiochemicalAlgorithms.jl

**License (reproduced from BiochemicalAlgorithms.jl `LICENSE`):**

> MIT License
>
> Copyright (c) 2022 Andreas Hildebrandt <andreas.hildebrandt@uni-mainz.de>
> and contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

---

## 4. OpenMM — Stanford University and the Authors

**Used in:** `proteon-connector/src/forcefield/gb_obc.rs` (OBC GB
Born-radii rescaling, pair-integral assembly, and force chain-rule).
Derived specifically from the Reference Platform
(`platforms/reference/src/SimTKReference/ReferenceObc.cpp`), which is
the MIT-licensed portion of OpenMM.

**Upstream:** https://openmm.org / https://github.com/openmm/openmm

**License (reproduced from OpenMM `docs-source/licenses/Licenses.txt`,
MIT portion covering the API, Reference Platform, and CPU Platform):**

> OpenMM was originally developed by Simbios, the NIH National Center for
> Physics-Based Simulation of Biological Structures at Stanford, funded under
> the NIH Roadmap for Medical Research, grant U54 GM072970. Currently, OpenMM
> is developed and maintained by various researchers and developers: for more
> information, see <https://openmm.org/development>.
>
> Portions copyright © 2008-2025 Stanford University and the Authors.
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject
> to the following conditions:
>
> The above copyright notice and this permission notice shall be included
> in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
> OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
> LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
> OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
> WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

proteon does not incorporate or link any portion of OpenMM's CUDA Platform,
OpenCL Platform, or other non-MIT-licensed components.

---

## 5. pdbtbx — Douwe Schulte and contributors

**Used in:** `proteon-io` (PDB / mmCIF I/O) and transitively across
proteon-align, proteon-connector. Linked as a Rust dependency.

**Upstream:** https://github.com/douweschulte/pdbtbx

**License (reproduced from pdbtbx `LICENSE`):**

> MIT License
>
> Copyright (c) 2020 Douwe Schulte and contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

---

## 6. AutoDock Vina — Trott / Forli Lab, The Scripps Research Institute

**Used in:** `proteon-vina/src/**` (XS atom types, Vina scoring-function
terms, precalculated pairwise potentials, receptor grid cache,
`score_only` entry point).

**Upstream:** https://github.com/ccsb-scripps/AutoDock-Vina

**License:** Apache License, Version 2.0. The full license text is
reproduced at `proteon-vina/LICENSE`. The `proteon-vina` crate is itself
distributed under Apache-2.0; the rest of proteon remains MIT.

**Copyright notice (reproduced from upstream source headers):**

> Copyright (c) 2006-2010, The Scripps Research Institute
>
> Licensed under the Apache License, Version 2.0 (the "License"); you
> may not use this file except in compliance with the License. You may
> obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
> implied. See the License for the specific language governing
> permissions and limitations under the License.
>
> Author: Dr. Oleg Trott <ot14@columbia.edu>, The Olson Lab, The
> Scripps Research Institute.

**Primary references:**
- Trott & Olson. *Journal of Computational Chemistry* 31(2), 455-461 (2010).
- Eberhardt, Santos-Martins, Tillack & Forli. *Journal of Chemical
  Information and Modeling* 61(8), 3891-3898 (2021).

---

## 7. Paper-inspired components (no source-code lineage)

The following components are implemented from primary literature. No
source code from their reference implementations is incorporated; no
upstream license notice is required.

- **3Di structural alphabet** — `proteon-align/src/search/alphabet.rs`.
  Paper-inspired by van Kempen et al., *Nature Biotechnology* 42(2),
  243-246 (2024) (Foldseek). proteon's encoder weights and centroids
  are trained independently; no GPL-licensed Foldseek code is reused.
- **libmarv GPU Smith-Waterman kernel design** —
  `proteon-search/src/gpu/pssm_sw*.rs` and the accompanying `.cu`
  sources. Paper-inspired by Kallenborn et al., *Nature Methods* 22,
  2024-2027 (2025).
- **AlphaFold MSA feature tensor shapes** — `proteon-search/src/msa.rs`.
  Paper-inspired by Jumper et al., *Nature* 596, 583-589 (2021).
- **Force-field functional forms and parameter sets** — CHARMM19
  (Neria, Fischer & Karplus 1996), EEF1 (Lazaridis & Karplus 1999),
  AMBER94/96 (Cornell et al. 1995), OBC Generalized Born (Onufriev,
  Bashford & Case 2004). Parameter files are reproduced from the
  standard distribution of each force field.

---

## 8. Correctness oracles (not incorporated)

The following projects are used at development time to verify proteon's
numerical and algorithmic correctness. They are NOT linked into
proteon, NOT distributed with it, and their licenses do not attach to
the proteon distribution.

OpenMM, BALL, Foldseek, MMseqs2, US-align, Biopython, Gemmi, FreeSASA.

proteon's test suite makes the tolerance of agreement with each oracle
explicit; see the `tests/oracle/` directories in the respective crates.
