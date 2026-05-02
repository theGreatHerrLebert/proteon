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
#   - apt:             GROMACS (the manifest's release-tier median-TM
#                      number was produced against custom-built 2026.1 —
#                      that divergence is honest and queryable through
#                      last_verified.value once replay populates it)
#
# Base distribution: Debian trixie (debian 13) across builder, oracles,
# and runtime stages. The earlier bookworm-everywhere story collapsed
# under libcifpp HEAD's C++20 <format> need, which requires libstdc++-13+
# — backports gymnastics on bookworm proved unreliable (gcc-13 is not
# in bookworm-backports). Trixie ships gcc-14 / libstdc++-14 / cmake
# 3.31 by default, so the oracle build chain works without any toolchain
# workarounds. glibc 2.41 throughout — keeps the in-image cross-stage
# story consistent; consumers running the image see a self-contained
# environment regardless of host glibc.
#
# Companion image evident/Dockerfile.cuda adds GPU support (cudarc +
# proteon-search/cuda kernels). It stays on jammy / glibc 2.35 because
# NVIDIA's CUDA 12.5 runtime tags do not include trixie at the time of
# writing — see that file's own header for the divergence rationale.
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
# rust:1-slim-trixie tracks Rust stable on the trixie base — proteon's
# MSRV is 1.75+ so any Rust >= that works. The slim image does not
# pull cudarc, so the cudarc/libloading 1.88 MSRV constraint that
# applies to the :cuda image does not apply here. The cuda Dockerfile
# pins its own Rust version separately for that reason.
FROM rust:1-slim-trixie AS builder

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
FROM debian:trixie-slim AS oracles

ENV DEBIAN_FRONTEND=noninteractive

# trixie ships gcc-14, libstdc++-14, and cmake 3.31 by default — all
# the toolchain workarounds the bookworm build accumulated (gcc-13
# from backports, Kitware cmake tarball, libstdc++6 backports in
# runtime) collapse to a single apt install on trixie.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        wget tar xz-utils \
        python3 python3-dev \
        libboost-program-options-dev libboost-iostreams-dev libboost-date-time-dev \
        libboost-python-dev \
        libsqlite3-dev \
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
# python:3.11-slim-trixie matches the builder + oracles stages on the
# trixie base. proteon's requires-python is >=3.11, so this is fully
# supported. The builder stage uses trixie's default python3 (3.13),
# producing cp313-cp313 wheels — the runtime base must match the
# builder's wheel ABI tag, hence python:3.13.
FROM python:3.13-slim-trixie AS runtime

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
# Version note: apt's gromacs on trixie is newer than bookworm's
# (~2024.x vs ~2023.x). The manifest's release-tier claim was
# produced against a custom-built GROMACS 2026.1 at
# /scratch/TMAlign/gromacs-2026.1/build/bin/gmx; the apt version will
# differ from that, and the cited median-TM number may shift
# slightly when `evident replay` runs the gromacs claim. The
# divergence is honest and queryable through last_verified.value
# once replay populates it.
#
# trixie's default libstdc++6 is libstdc++-14, which carries the
# GLIBCXX_3.4.33+ symbols mkdssp's libcifpp.so needs — no backports
# install required here, unlike on the previous bookworm image.
# Boost runtime libraries land at the trixie 1.83 series rather than
# bookworm's 1.74; the package names track that.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 \
        gcc python3-dev \
        gromacs \
        libboost-program-options1.83.0 libboost-iostreams1.83.0 libboost-date-time1.83.0 \
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
                    /usr/local/lib/python3.13/site-packages/validate_manifest.py

# Default invocation: list all claims in the proteon manifest.
ENTRYPOINT ["evident"]
CMD ["list", "evident/evident.yaml"]
