# Vendored EVIDENT CLI

`evident.py` and `validate_manifest.py` here are vendored copies of
the upstream EVIDENT framework
(https://github.com/.../evident — `workflow/`).

This vendor exists so the proteon Dockerfile and the CI manifest
check do not need a cross-repo fetch at build time.

Proteon's copy currently carries one local hardening patch: the
`--strict-release-pins` validator option rejects placeholder release
corpus hashes, pinned versions, and `last_verified` pins during release
prep.

## Re-syncing

When the framework's `validate_manifest.py` schema or `evident.py`
CLI evolves:

```bash
cp <FRAMEWORK_REPO>/workflow/{evident.py,validate_manifest.py} \
   proteon/evident/tools/
python3 proteon/evident/tools/validate_manifest.py \
   proteon/evident/evident.yaml   # confirm proteon's manifest still passes
```

Then reapply or upstream proteon's local `--strict-release-pins` patch
before committing the updated files. Keep any local drift explicit:
untracked divergence between proteon's vendored copy and the framework
breaks composability across projects using EVIDENT.

## Why vendor

- Docker build is hermetic; no network fetch of an external repo.
- proteon's CI gates on this exact validator version; a framework
  change cannot silently break or loosen proteon's gate.
- The framework repo and proteon repo are independent in proteon's
  release flow — the vendor decouples them at the right boundary.
