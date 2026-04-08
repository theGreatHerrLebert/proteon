#!/usr/bin/env python3
"""Download random PDB files for benchmarking.

Downloads N random protein structures from the PDB archive.

Usage:
    python download_pdbs.py --n 50000 --out pdbs_50k/ --workers 64
"""

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


def get_all_protein_pdb_ids():
    """Fetch all protein PDB IDs from RCSB search API."""
    import json
    from urllib.request import Request, urlopen

    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein"
            }
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }

    req = Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    ids = [hit["identifier"] for hit in data.get("result_set", [])]
    print(f"Found {len(ids)} protein PDB IDs")
    return ids


def download_pdb(pdb_id, out_dir):
    """Download a single PDB file."""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        return out_path, True
    try:
        urlretrieve(url, out_path)
        return out_path, True
    except (URLError, Exception):
        # Try mmCIF
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            out_path = os.path.join(out_dir, f"{pdb_id}.cif")
            urlretrieve(url, out_path)
            return out_path, True
        except Exception:
            return pdb_id, False


def main():
    parser = argparse.ArgumentParser(description="Download PDB files for benchmarking")
    parser.add_argument("--n", type=int, default=50000, help="Number of structures")
    parser.add_argument("--out", default="pdbs_50k", help="Output directory")
    parser.add_argument("--workers", type=int, default=64, help="Download threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Get all protein PDB IDs
    print("Fetching PDB ID list...")
    all_ids = get_all_protein_pdb_ids()

    # Random sample
    random.seed(args.seed)
    sample = random.sample(all_ids, min(args.n, len(all_ids)))
    print(f"Downloading {len(sample)} structures to {args.out}/")

    # Download in parallel
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_pdb, pid, args.out): pid for pid in sample}
        for future in as_completed(futures):
            path, ok = future.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (done + failed) % 1000 == 0:
                print(f"  {done + failed}/{len(sample)} ({done} ok, {failed} failed)")

    print(f"Done: {done} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
