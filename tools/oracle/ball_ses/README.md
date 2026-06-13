# ball_ses — BALL analytic SES oracle

A minimal C++ driver that computes **BALL's analytic Solvent-Excluded Surface**
area + volume for a set of spheres, used to anchor proteon's SES mesher against the
reference Connolly implementation.

`ball-py` (the wheel at `ball-zomball/dist/`) does **not** expose SES — only
`sasa`/energies — so this drives `libBALL` (`BALL/STRUCTURE/analyticalSES.h`)
directly. If SES is later bound in `ball-zomball/python/src/module.cpp`, prefer that.

## Input

An `xyzr` file: one sphere per line, `x y z radius` (Å). Get proteon's *exact*
spheres (so radii match) by running `ses_diag` with `SES_DUMP_SPHERES=<dir>`:

```bash
SES_DUMP_SPHERES=/tmp/spheres ./target/release/ses_diag validation/corpus/1ijp.pdb
```

## Build

```bash
BZ=/scratch/TMAlign/ball-zomball
VENV=/scratch/TMAlign/proteon/.venv/lib/python3.12/site-packages   # bundles libBALL.so
g++ -std=c++17 -O2 ball_ses.C \
  -I"$BZ/include" -I"$BZ/build/include" -I/usr/include/eigen3 \
  -L"$VENV" -lBALL -Wl,-rpath,"$VENV" -o ball_ses
```

## Run

```bash
./ball_ses /tmp/spheres/1ijp.xyzr 1.4     # probe radius 1.4 Å
# OK area=4933.6 volume=9722.68
```

Validated against BALL's own unit-test cases (two r=1.0 atoms, probe 1.5):
area 25.1327 (10 Å apart) and 18.7218 (1 Å apart).

## Note on degeneracies

BALL's *analytic* area (used here) is robust to the near-tangency configs that make
proteon's *triangulation* fail. BALL's own *triangulation* path
(`source/STRUCTURE/surfaceProcessor.C`) escapes them by bounded probe-radius
perturbation retry (±0.01 Å, ≤10×) — the same family of remedy as proteon's
atom-centre jitter. See `devdocs/archive/SES_CDT_CROSSING.md`.
