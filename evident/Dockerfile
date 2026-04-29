# Proteon EVIDENT reproducibility image.
#
# Bundles proteon + the oracle stack that backs the claims in
# proteon/evident/evident.yaml, plus the `evident` CLI for manifest
# validation and listing. Designed to be the single artifact a
# reviewer can pull (`docker pull ghcr.io/thegreatherrlebert/proteon-evident:<tag>`)
# and replay claims against, anchored to the immutable image digest.
#
# v1 oracle scope:
#   - pip-installable: Biopython, Gemmi, FreeSASA, pydssp, OpenMM, pdbfixer
#   - source-built:    USAlign, reduce, mkdssp (libmcfp + libcifpp + dssp)
#   - binary release:  MMseqs2
#   - apt:             GROMACS (~2023.x; the manifest's release-tier
#                      median-TM number was produced against custom-built
#                      2026.1 — that divergence is honest and queryable
#                      through last_verified.value once replay populates it)
#
# Companion image evident/Dockerfile.cuda adds GPU support (cudarc +
# proteon-search/cuda kernels) on the same builder + oracle stages.
#
# Out of scope: live BALL Julia regeneration. The forcefield_amber_ball
# claim's test (tests/oracle/test_ball_energy.py) carries a frozen
# BALL_CRAMBIN_RAW reference dict from a 2026-04-18 BALL run, so the
# claim IS replayable in this image — it imports only proteon. What's
# out of scope is regenerating that reference (BALL Python bindings are
# dead; ball_energy_raw.jl needs Julia + BiochemicalAlgorithms.jl);
# staleness of the frozen numbers is documented in the claim's
# failure_modes and tracked manually until a v1.1 image carries Julia.
#
# Build (from proteon repo root):
#   docker build -f evident/Dockerfile -t proteon-evident:dev .
# Run:
#   docker run --rm proteon-evident:dev evident list evident/evident.yaml
#   docker run --rm proteon-evident:dev evident validate evident/evident.yaml

# ---------------------------------------------------------------------------
# Stage 1: Rust builder — compile the proteon Python wheel via maturin.
# ---------------------------------------------------------------------------
# Rust 1.88+ required because cudarc's transitive dep libloading@0.9.0
# bumped its MSRV to 1.88; the :slim image doesn't pull cudarc and
# would compile fine on 1.85, but keeping the toolchain version in
# lockstep with :cuda lets a single rebuild flow cover both tags.
FROM rust:1.88-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        build-essential pkg-config \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Build env: maturin (for the Rust+PyO3 connector wheel) and `build`
# (for the pure-Python wrapper wheel). Two wheels because proteon ships
# as `proteon-connector` (Rust cdylib) + `proteon` (Python wrapper that
# imports from the connector).
RUN python3 -m venv /opt/build-venv && \
    /opt/build-venv/bin/pip install --no-cache-dir maturin build

WORKDIR /src
# Copy only what the wheel builds need. Cache-friendly: Rust crate
# manifests change far less often than source files.
COPY Cargo.toml Cargo.lock ./
COPY proteon-align       ./proteon-align/
COPY proteon-io          ./proteon-io/
COPY proteon-arrow       ./proteon-arrow/
COPY proteon-search      ./proteon-search/
COPY proteon-bin         ./proteon-bin/
COPY proteon-connector   ./proteon-connector/
COPY packages            ./packages/
# proteon-align includes a small validation/ data file at compile time
# (alphabet_vqvae_rust.txt) via include!(..). The builder stage needs
# it; the runtime stage doesn't (the data is baked into the Rust .rlib).
COPY validation          ./validation/
COPY README.md LICENSE THIRD_PARTY_NOTICES.md VERSION ./

# Wheel 1: proteon-connector (Rust + PyO3 via maturin). Built first
# because the pure-Python wrapper depends on this exact version.
RUN /opt/build-venv/bin/maturin build \
        --release \
        --out /wheels \
        --manifest-path proteon-connector/Cargo.toml

# Wheel 2: proteon (pure-Python wrapper, setuptools).
RUN /opt/build-venv/bin/python -m build --wheel \
        --outdir /wheels packages/proteon \
    && ls -la /wheels

# ---------------------------------------------------------------------------
# Stage 2: Oracle binaries — build USAlign + reduce + fetch MMseqs2.
# Kept separate so a runtime-stage cache miss doesn't force a Rust rebuild.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS oracles

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        wget tar xz-utils \
        libboost-program-options-dev libboost-iostreams-dev libboost-date-time-dev \
        zlib1g-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# USAlign — single-file C++, fast build.
RUN git clone --depth 1 https://github.com/pylelab/USAlign.git && \
    cd USAlign && make USalign && \
    install -m 0755 USalign /usr/local/bin/USalign && \
    cd .. && rm -rf USAlign

# reduce — cmake build, ships its own dictionary.
RUN git clone --depth 1 https://github.com/rlabduke/reduce.git && \
    cmake -S reduce -B reduce/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build reduce/build -j"$(nproc)" && \
    install -m 0755 reduce/build/reduce_src/reduce /usr/local/bin/reduce && \
    install -m 0644 reduce/reduce_wwPDB_het_dict.txt /usr/local/share/reduce_wwPDB_het_dict.txt && \
    rm -rf reduce

# MMseqs2 — pinned static binary release for byte-stable behaviour.
ARG MMSEQS_VERSION=15-6f452
RUN wget -q "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_VERSION}/mmseqs-linux-avx2.tar.gz" -O /tmp/mmseqs.tar.gz && \
    tar -xzf /tmp/mmseqs.tar.gz -C /opt && \
    ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs && \
    rm /tmp/mmseqs.tar.gz

# Newer CMake. libmcfp's CMakeLists requires CMake 3.28+; bookworm
# ships 3.25.1 which fails configure. Pin to a Kitware binary release
# rather than apt repos — same shape as the MMseqs2 download above.
ARG CMAKE_VERSION=3.30.5
RUN wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -O /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /opt && \
    ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake && \
    ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/ctest" /usr/local/bin/ctest && \
    rm /tmp/cmake.tar.gz

# mkdssp (DSSP C++ reference, PDB-REDO/dssp). Closes the 8-class DSSP
# gap claim's oracle install; the asserting test that uses it is
# tracked separately. Build chain: libmcfp → libcifpp → dssp. CCD
# download is disabled (-DCIFPP_DOWNLOAD_CCD=OFF) because PDB input
# does not need the chemistry-component dictionary; mmCIF input would.
RUN git clone --depth 1 https://github.com/mhekkel/libmcfp.git && \
    cmake -S libmcfp -B libmcfp/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build libmcfp/build -j"$(nproc)" && \
    cmake --install libmcfp/build && \
    rm -rf libmcfp

RUN git clone --depth 1 https://github.com/PDB-REDO/libcifpp.git && \
    cmake -S libcifpp -B libcifpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCIFPP_DOWNLOAD_CCD=OFF \
        -DCIFPP_INSTALL_UPDATE_SCRIPT=OFF \
        -DBUILD_TESTING=OFF && \
    cmake --build libcifpp/build -j"$(nproc)" && \
    cmake --install libcifpp/build && \
    rm -rf libcifpp

RUN git clone --depth 1 https://github.com/PDB-REDO/dssp.git && \
    cmake -S dssp -B dssp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    cmake --build dssp/build -j"$(nproc)" && \
    install -m 0755 dssp/build/mkdssp /usr/local/bin/mkdssp && \
    rm -rf dssp

# ---------------------------------------------------------------------------
# Stage 3: Runtime — slim Python image with proteon + oracles + CLI.
# ---------------------------------------------------------------------------
# Python 3.11 matches the builder stage (rust:1.85-slim-bookworm
# ships Debian's default Python 3.11). proteon's requires-python
# is >=3.11, so this is fully supported. To move to Python 3.12,
# the builder stage must also be 3.12 (rebuild wheels) — not just
# the runtime.
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REDUCE_BIN=/usr/local/bin/reduce \
    REDUCE_DB=/usr/local/share/reduce_wwPDB_het_dict.txt \
    USALIGN_BIN=/usr/local/bin/USalign \
    PROTEON_SEARCH_MMSEQS_BIN=/usr/local/bin/mmseqs \
    GMX=/usr/bin/gmx

LABEL org.opencontainers.image.title="proteon-evident" \
      org.opencontainers.image.description="Proteon + oracle stack + EVIDENT CLI for replaying manifest claims" \
      org.opencontainers.image.source="https://github.com/theGreatHerrLebert/proteon" \
      org.opencontainers.image.licenses="MIT"

# Runtime libraries the source-built oracles need at execution time,
# plus gcc + python3-dev so freesasa (no manylinux wheel) can build
# from sdist during pip install, plus apt-installed gromacs for the
# fold-preservation-amber-vs-gromacs claim.
#
# Version note: apt's gromacs on bookworm is ~2023.x. The manifest's
# release-tier claim was originally produced against a custom-built
# GROMACS 2026.1 at /scratch/TMAlign/gromacs-2026.1/build/bin/gmx.
# When this image's `evident replay` runs the gromacs claim, the
# manifest's pinned_versions[GROMACS] PENDING-PIN placeholder will
# resolve to apt's version, and the cited median-TM number may
# shift slightly. That divergence is honest and queryable through
# last_verified.value once replay populates it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 \
        gcc python3-dev \
        gromacs \
        libboost-program-options1.74.0 libboost-iostreams1.74.0 libboost-date-time1.74.0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy oracle binaries from stage 2.
COPY --from=oracles /usr/local/bin/USalign        /usr/local/bin/USalign
COPY --from=oracles /usr/local/bin/reduce         /usr/local/bin/reduce
COPY --from=oracles /usr/local/share/reduce_wwPDB_het_dict.txt \
                                                  /usr/local/share/reduce_wwPDB_het_dict.txt
COPY --from=oracles /opt/mmseqs                   /opt/mmseqs
COPY --from=oracles /usr/local/bin/mmseqs         /usr/local/bin/mmseqs
COPY --from=oracles /usr/local/bin/mkdssp         /usr/local/bin/mkdssp
# libcifpp shared library — mkdssp links against it dynamically, so
# the runtime needs both the .so and the libcifpp data files. Copying
# the whole /usr/local/{lib,share}/cifpp prefix keeps the install
# layout consistent with the cmake --install that ran in stage 2.
COPY --from=oracles /usr/local/lib/libcifpp.so*   /usr/local/lib/
COPY --from=oracles /usr/local/share/libcifpp     /usr/local/share/libcifpp
RUN ldconfig

# Pip oracles + the proteon wheel from stage 1.
COPY --from=builder /wheels /tmp/wheels
# Pip oracle stack. Versions intentionally NOT pinned here — the
# manifest's pinned_versions field is what tracks the actual oracle
# versions used to produce a cited result (PENDING-PIN-* placeholders
# get resolved at first replay). Layer-cache invalidation on PyPI
# version drift is acceptable; pinning here without a sync mechanism
# to the manifest would be precision theatre.
RUN pip install --no-cache-dir \
        biopython \
        gemmi \
        freesasa \
        pydssp \
        openmm \
        pdbfixer \
        numpy \
        pyyaml \
        pytest \
    && pip install --no-cache-dir \
        /tmp/wheels/proteon_connector-*.whl \
        /tmp/wheels/proteon-*.whl \
    && rm -rf /tmp/wheels

# Proteon source tree — needed so pytest can find tests/ and
# validation/ scripts that the manifest's evidence.command points at.
WORKDIR /workspace/proteon
COPY tests          ./tests/
COPY test-pdbs      ./test-pdbs/
COPY validation     ./validation/
COPY evident        ./evident/
COPY benchmark      ./benchmark/
COPY devdocs        ./devdocs/
COPY docs           ./docs/
COPY README.md THIRD_PARTY_NOTICES.md LICENSE VERSION ./

# Install the EVIDENT CLI from the vendored copy.
RUN install -m 0755 /workspace/proteon/evident/tools/evident.py \
                    /usr/local/bin/evident && \
    install -m 0644 /workspace/proteon/evident/tools/validate_manifest.py \
                    /usr/local/lib/python3.11/site-packages/validate_manifest.py

# Default invocation: list all claims in the proteon manifest.
ENTRYPOINT ["evident"]
CMD ["list", "evident/evident.yaml"]
