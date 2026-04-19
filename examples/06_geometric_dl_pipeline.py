#!/usr/bin/env python3
"""Example 06: From PDB Files to Geometric Deep Learning.

Demonstrates how proteon collapses the structural biology data pipeline
from ~200 lines of glue code (Biopython + Gemmi + MDTraj + FreeSASA)
into 5 lines of proteon, then feeds directly into PyTorch Geometric.

This is the "missing link" between PDB files and GNN training.

Requirements:
    pip install torch torch-geometric

Usage:
    python examples/06_geometric_dl_pipeline.py
"""

import os
import time
import glob

import numpy as np

# ============================================================================
# STEP 1: Data preparation with proteon (the hard part, now easy)
# ============================================================================

import proteon

print("=" * 60)
print("PDB → Features → GNN: The Proteon Pipeline")
print("=" * 60)

# Grab some structures: prefer the bigger validation/ corpus if present,
# otherwise fall back to the small test-pdbs/ set bundled in the repo.
pdb_dir = "validation/pdbs/"
if not os.path.exists(pdb_dir):
    pdb_dir = "test-pdbs/"

pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))[:100]
print(f"\n{len(pdb_files)} PDB files")

# --- THE PROTEON WAY: one call, everything you need ---
print("\n--- Proteon: load + analyze in one call ---")
t0 = time.time()
results = proteon.load_and_analyze(pdb_files, cutoff=8.0, n_threads=-1)
t_proteon = time.time() - t0
print(f"  {len(results)} structures processed in {t_proteon:.2f}s")
print(f"  ({len(results)/t_proteon:.0f} structures/second)")

# Now add SASA and DSSP (need loaded structures for these)
structures = proteon.batch_load_tolerant(pdb_files, n_threads=-1)
struct_map = {i: s for i, s in structures}

t0 = time.time()
for r in results:
    idx = r["index"]
    if idx in struct_map:
        s = struct_map[idx]
        r["sasa"] = np.asarray(proteon.residue_sasa(s))
        r["rsa"] = np.asarray(proteon.relative_sasa(s))
        r["dssp"] = proteon.dssp(s)
        r["hbond_count"] = np.asarray(proteon.hbond_count(s))
t_features = time.time() - t0
print(f"  SASA + DSSP + H-bonds: {t_features:.2f}s")
print(f"  Total data prep: {t_proteon + t_features:.2f}s")

# ============================================================================
# STEP 2: Build graph dataset for PyTorch Geometric
# ============================================================================

print("\n--- Building graph dataset ---")

import torch
from torch_geometric.data import Data, InMemoryDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# Encode secondary structure as one-hot
SS_CODES = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "C": 7}


def ss_to_onehot(ss_string, n_residues):
    """Convert DSSP string to 8-class one-hot encoding."""
    oh = np.zeros((n_residues, 8), dtype=np.float32)
    for i, c in enumerate(ss_string):
        if i < n_residues:
            oh[i, SS_CODES.get(c, 7)] = 1.0
    return oh


def structure_to_graph(result):
    """Convert proteon analysis result to a PyG Data object.

    Node features (per residue):
        - phi, psi angles (sin/cos encoded, 4 features)
        - RSA (relative solvent accessibility, 1 feature)
        - H-bond count (1 feature)
        - SS one-hot (8 features)
        = 14 features total

    Edges: from contact map (CA-CA distance < 8 Å)
    Edge features: CA-CA distance
    """
    n_ca = result["n_ca"]
    if n_ca < 5:
        return None

    # --- Node features ---
    # phi/psi may differ in length from n_ca (multi-chain, missing atoms)
    # Safely pad/truncate to match n_ca
    phi = np.nan_to_num(result["phi"], nan=0.0)
    psi = np.nan_to_num(result["psi"], nan=0.0)

    # Pad or truncate to n_ca
    if len(phi) < n_ca:
        phi = np.pad(phi, (0, n_ca - len(phi)))
        psi = np.pad(psi, (0, n_ca - len(psi)))
    else:
        phi = phi[:n_ca]
        psi = psi[:n_ca]

    phi_rad = np.deg2rad(phi)
    psi_rad = np.deg2rad(psi)

    angle_features = np.stack([
        np.sin(phi_rad), np.cos(phi_rad),
        np.sin(psi_rad), np.cos(psi_rad),
    ], axis=1).astype(np.float32)  # (n_ca, 4)

    # RSA, H-bond count, DSSP may have different lengths than n_ca
    # (residue count includes non-AA, n_ca is only amino acids with CA)
    # Safely extract n_ca values, padding/truncating as needed
    def fit_to_n(arr, n, fill=0.0):
        arr = np.asarray(arr).flatten()
        if len(arr) >= n:
            return arr[:n].astype(np.float32).reshape(-1, 1)
        return np.pad(arr, (0, n - len(arr)), constant_values=fill).astype(np.float32).reshape(-1, 1)

    rsa = fit_to_n(np.nan_to_num(result.get("rsa", np.zeros(n_ca)), nan=0.0), n_ca)
    hbc = fit_to_n(result.get("hbond_count", np.zeros(n_ca)), n_ca)
    dssp = result.get("dssp", "C" * n_ca)
    ss_oh = ss_to_onehot(dssp, n_ca)

    # Combine: (n_ca, 14)
    x = np.concatenate([angle_features, rsa, hbc, ss_oh], axis=1)

    # --- Edges from contact map ---
    contact_map = result["contact_map"]
    # Get edge indices (upper triangle, excluding self-loops)
    rows, cols = np.where(np.triu(contact_map, k=1))
    edge_index = np.stack([
        np.concatenate([rows, cols]),  # bidirectional
        np.concatenate([cols, rows]),
    ], axis=0)

    # Edge features: distances
    dm = result["distance_matrix"]
    edge_dist = np.concatenate([dm[rows, cols], dm[cols, rows]]).astype(np.float32)

    # --- Target: classify by dominant SS type ---
    from collections import Counter
    ss_counts = Counter(dssp[:n_ca])
    if ss_counts.get("H", 0) > n_ca * 0.3:
        label = 0  # mostly helical
    elif ss_counts.get("E", 0) > n_ca * 0.2:
        label = 1  # has significant sheet
    else:
        label = 2  # mixed/coil

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_dist, dtype=torch.float32).unsqueeze(1),
        y=torch.tensor([label], dtype=torch.long),
        pos=torch.tensor(result["ca_coords"], dtype=torch.float32),
        num_nodes=n_ca,
    )


# Build dataset
graphs = []
for r in results:
    g = structure_to_graph(r)
    if g is not None:
        graphs.append(g)

print(f"  {len(graphs)} graphs built")
print(f"  Node features: {graphs[0].x.shape[1]} per residue")
print(f"  Example: {graphs[0].num_nodes} nodes, {graphs[0].edge_index.shape[1]} edges")

# Class distribution
labels = [g.y.item() for g in graphs]
from collections import Counter
label_counts = Counter(labels)
print(f"  Classes: helical={label_counts[0]}, sheet={label_counts[1]}, mixed={label_counts[2]}")

# ============================================================================
# STEP 3: Train a simple GCN
# ============================================================================

print("\n--- Training GCN ---")

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# Split
n_train = int(0.8 * len(graphs))
train_graphs = graphs[:n_train]
test_graphs = graphs[n_train:]
print(f"  Train: {len(train_graphs)}, Test: {len(test_graphs)}")

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)


class ProteinGCN(torch.nn.Module):
    """Simple 3-layer GCN for protein fold classification."""

    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


model = ProteinGCN(in_channels=14, hidden=64, out_channels=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    if (epoch + 1) % 10 == 0:
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch).argmax(dim=1)
                correct += (pred == batch.y.squeeze()).sum().item()
                total += batch.num_graphs

        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1:3d}: loss={total_loss/len(train_graphs):.3f}, "
              f"test_acc={acc:.3f} ({correct}/{total})")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"""
  Data preparation:
    {len(pdb_files)} PDB files → {len(graphs)} protein graphs
    Features per residue: phi/psi (4) + RSA (1) + H-bonds (1) + SS (8) = 14
    Edges: CA-CA contacts at 8Å cutoff
    Time: {t_proteon + t_features:.1f}s (proteon, {len(results)/t_proteon:.0f} structs/sec)

  Without proteon, this would require:
    - Biopython for loading + SASA (~{t_proteon * 24:.0f}s at 24x slower)
    - External DSSP binary + subprocess parsing
    - MDTraj or manual code for dihedrals
    - Custom contact map code
    - ~200 lines of glue code

  With proteon: 5 function calls, {t_proteon + t_features:.1f}s, zero glue.

  Model: 3-layer GCN, {sum(p.numel() for p in model.parameters())} parameters
  Device: {DEVICE}
""")

print("Done!")
