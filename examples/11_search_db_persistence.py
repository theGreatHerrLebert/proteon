#!/usr/bin/env python3
"""Example 11: Persist and Reuse a Search DB.

Demonstrates:
    - Building and saving a structural-alphabet search DB
    - Querying the default compiled serving layout
    - Opting into Parquet-only storage
    - Auto-compiling an older Parquet-only DB in place on first reuse

Usage:
    python examples/11_search_db_persistence.py
"""

from pathlib import Path
import shutil
import tempfile

import proteon


paths = [
    "test-pdbs/1crn.pdb",
    "test-pdbs/1ubq.pdb",
    "test-pdbs/1bpi.pdb",
]
query = proteon.load(paths[0])

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    print("=== Default persisted path: compiled serving layout included ===")
    compiled_root = tmp / "search_db_compiled"
    proteon.build_search_db(paths, out=compiled_root, k=6, n_threads=-1)
    hits = proteon.search(query, compiled_root, top_k=3, rerank=False)
    print(f"  Compiled manifest exists: {(compiled_root / 'compiled' / 'manifest.json').exists()}")
    print(f"  Top hit: {hits[0].id} score={hits[0].score:.3f}")
    print()

    print("=== Explicit Parquet-only path ===")
    db = proteon.build_search_db(paths, k=6, n_threads=-1)
    lazy_root = tmp / "search_db_lazy"
    proteon.save_search_db(db, lazy_root, write_compiled=False)
    lazy_db = proteon.load_search_db(lazy_root, prefer_compiled=False)
    print(f"  Compiled manifest exists: {(lazy_root / 'compiled' / 'manifest.json').exists()}")
    print(f"  Lazy entries materialized eagerly: {lazy_db.entries is not None}")
    print()

    print("=== Upgrade an older Parquet-only DB in place ===")
    upgrade_root = tmp / "search_db_upgrade"
    shutil.copytree(lazy_root, upgrade_root)
    upgraded = proteon.load_search_db(upgrade_root, auto_compile_missing=True)
    print(f"  Compiled manifest exists after upgrade: {(upgrade_root / 'compiled' / 'manifest.json').exists()}")
    print(f"  Upgraded entries materialized eagerly: {upgraded.entries is not None}")
    hits = proteon.search(query, upgrade_root, top_k=3, rerank=False)
    print(f"  Top hit after upgrade: {hits[0].id} score={hits[0].score:.3f}")
