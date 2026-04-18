#!/usr/bin/env python3
"""Example 07: Predict B-factors (Flexibility) from Structure.

Self-supervised task: predict crystallographic B-factors (temperature factors)
from structural features. B-factors correlate with atomic flexibility and are
recorded in every PDB file — free labels, no external annotation needed.

This is a regression task at the residue level, commonly used to benchmark
geometric DL models for protein dynamics prediction.

Requirements:
    pip install torch torch-geometric

Usage:
    python examples/07_bfactor_prediction.py
"""

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import ferritin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Task: Per-Residue B-factor Prediction")
print("=" * 60)

# --- Load structures ---
pdb_dir = "validation/pdbs/"
if not os.path.exists(pdb_dir):
    pdb_dir = "test-pdbs/"

pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))[:200]
print(f"\n{len(pdb_files)} PDB files")

t0 = time.time()
loaded = ferritin.batch_load_tolerant(pdb_files, n_threads=-1)
print(f"  Loaded {len(loaded)} structures in {time.time()-t0:.2f}s")


def structure_to_bfactor_graph(s):
    """Build a graph for B-factor prediction.

    Node features: phi/psi, RSA, H-bond count, SS one-hot (14 features)
    Target: normalized CA B-factors
    """
    ca_coords = ferritin.extract_ca_coords(s)
    n = len(ca_coords)
    if n < 10:
        return None

    # Features
    phi, psi, _ = ferritin.backbone_dihedrals(s)
    rsa = ferritin.relative_sasa(s)
    hbc = ferritin.hbond_count(s)
    ss = ferritin.dssp(s)

    # Pad/truncate to n
    def fit(arr, n):
        arr = np.nan_to_num(np.asarray(arr).flatten(), nan=0.0).astype(np.float32)
        if len(arr) >= n: return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

    phi_r = np.deg2rad(fit(phi, n))
    psi_r = np.deg2rad(fit(psi, n))
    rsa_v = fit(rsa, n)
    hbc_v = fit(hbc, n)

    ss_oh = np.zeros((n, 8), dtype=np.float32)
    ss_map = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "C": 7}
    for i, c in enumerate(ss[:n]):
        ss_oh[i, ss_map.get(c, 7)] = 1.0

    x = np.column_stack([
        np.sin(phi_r), np.cos(phi_r), np.sin(psi_r), np.cos(psi_r),
        rsa_v, hbc_v, ss_oh
    ])

    # Contact graph
    cm = ferritin.contact_map(ca_coords, cutoff=10.0)
    rows, cols = np.where(np.triu(cm, k=1))
    edge_index = np.stack([
        np.concatenate([rows, cols]),
        np.concatenate([cols, rows])
    ], axis=0)

    # Target: CA B-factors, normalized per structure
    mask = ferritin.select(s, "CA")
    bfactors = s.b_factors[mask][:n].astype(np.float32)
    # Z-score normalize
    mean_b = bfactors.mean()
    std_b = bfactors.std()
    if std_b < 0.01:
        return None
    bfactors_norm = (bfactors - mean_b) / std_b

    return Data(
        x=torch.tensor(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(bfactors_norm),
        pos=torch.tensor(ca_coords, dtype=torch.float32),
    )


print("\n--- Building graphs ---")
t0 = time.time()
graphs = []
for _, s in loaded:
    g = structure_to_bfactor_graph(s)
    if g is not None:
        graphs.append(g)
print(f"  {len(graphs)} graphs in {time.time()-t0:.2f}s")

# Split
n_train = int(0.8 * len(graphs))
train_loader = DataLoader(graphs[:n_train], batch_size=8, shuffle=True)
test_loader = DataLoader(graphs[n_train:], batch_size=8)
print(f"  Train: {n_train}, Test: {len(graphs)-n_train}")


# --- Model: node-level regression ---
class BFactorGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(14, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.head = torch.nn.Linear(32, 1)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        return self.head(x).squeeze(-1)


print("\n--- Training ---")
model = BFactorGCN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(30):
    model.train()
    total_loss = 0
    n_nodes = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.y.shape[0]
        n_nodes += batch.y.shape[0]

    if (epoch + 1) % 10 == 0:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                targets.append(batch.y.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        corr = np.corrcoef(preds, targets)[0, 1]
        mse = np.mean((preds - targets) ** 2)
        print(f"  Epoch {epoch+1:3d}: train_mse={total_loss/n_nodes:.4f}, "
              f"test_mse={mse:.4f}, test_corr={corr:.3f}")

print(f"\n  Final Pearson correlation: {corr:.3f}")
print(f"  (Random baseline: ~0.0, good models: >0.5)")
print(f"  Device: {DEVICE}")
print("\nDone!")
