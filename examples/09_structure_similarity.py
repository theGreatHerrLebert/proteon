#!/usr/bin/env python3
"""Example 09: Learn Structure Similarity with Siamese GNN.

Contrastive learning: train a model that maps protein structures to
an embedding space where structurally similar proteins are close.

Uses ferritin's TM-align to compute pairwise TM-scores as ground truth
similarity labels, then trains a Siamese GCN to predict similarity
from graph features alone.

This is the foundation for structure-based retrieval, clustering,
and transfer learning in geometric DL.

Requirements:
    pip install torch torch-geometric

Usage:
    python examples/09_structure_similarity.py
"""

import glob
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import ferritin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Task: Structure Similarity Learning (Siamese GNN)")
print("=" * 60)

# --- Load structures ---
pdb_dir = "validation/pdbs/"
if not os.path.exists(pdb_dir):
    pdb_dir = "test-pdbs/"

pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))[:50]
print(f"\n{len(pdb_files)} PDB files")

t0 = time.time()
loaded = ferritin.batch_load_tolerant(pdb_files, n_threads=-1)
structures = {i: s for i, s in loaded}
print(f"  {len(structures)} loaded in {time.time()-t0:.2f}s")


def structure_to_graph(s):
    """Build graph for embedding."""
    ca = ferritin.extract_ca_coords(s)
    n = len(ca)
    if n < 10:
        return None

    phi, psi, _ = ferritin.backbone_dihedrals(s)
    rsa = ferritin.relative_sasa(s)
    ss = ferritin.dssp(s)

    def fit(arr, n):
        arr = np.nan_to_num(np.asarray(arr).flatten(), nan=0.0).astype(np.float32)
        if len(arr) >= n: return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

    phi_r = np.deg2rad(fit(phi, n))
    psi_r = np.deg2rad(fit(psi, n))

    ss_oh = np.zeros((n, 8), dtype=np.float32)
    ss_map = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "C": 7}
    for i, c in enumerate(ss[:n]):
        ss_oh[i, ss_map.get(c, 7)] = 1.0

    x = np.column_stack([
        np.sin(phi_r), np.cos(phi_r), np.sin(psi_r), np.cos(psi_r),
        fit(rsa, n), ss_oh
    ])  # 13 features

    cm = ferritin.contact_map(ca, cutoff=10.0)
    rows, cols = np.where(np.triu(cm, k=1))
    edge_index = np.stack([
        np.concatenate([rows, cols]),
        np.concatenate([cols, rows])
    ], axis=0)

    return Data(
        x=torch.tensor(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=torch.tensor(ca, dtype=torch.float32),
    )


# Build graphs
print("\n--- Building graphs ---")
t0 = time.time()
idx_to_graph = {}
struct_list = []
for idx, s in structures.items():
    g = structure_to_graph(s)
    if g is not None:
        idx_to_graph[idx] = g
        struct_list.append((idx, s))
print(f"  {len(idx_to_graph)} graphs in {time.time()-t0:.2f}s")

# --- Compute pairwise TM-scores with ferritin ---
print("\n--- Computing pairwise TM-scores ---")
n_structs = min(len(struct_list), 30)  # limit for speed
struct_subset = struct_list[:n_structs]
subset_structures = [s for _, s in struct_subset]
subset_indices = [i for i, _ in struct_subset]

t0 = time.time()
all_results = ferritin.tm_align_many_to_many(
    subset_structures, subset_structures, n_threads=-1, fast=True
)
t_align = time.time() - t0

# Build TM-score matrix
tm_matrix = np.zeros((n_structs, n_structs))
for qi, ti, r in all_results:
    tm_matrix[qi, ti] = max(r.tm_score_chain1, r.tm_score_chain2)

n_pairs = n_structs * n_structs
print(f"  {n_pairs} pairs in {t_align:.2f}s ({n_pairs/t_align:.0f} pairs/s)")
print(f"  TM-score range: [{tm_matrix[tm_matrix > 0].min():.3f}, {tm_matrix.max():.3f}]")

# --- Build training pairs ---
print("\n--- Building training pairs ---")
pairs = []
for i in range(n_structs):
    for j in range(i + 1, n_structs):
        idx_i = subset_indices[i]
        idx_j = subset_indices[j]
        if idx_i in idx_to_graph and idx_j in idx_to_graph:
            tm = tm_matrix[i, j]
            pairs.append((idx_i, idx_j, tm))

random.shuffle(pairs)
n_train = int(0.8 * len(pairs))
train_pairs = pairs[:n_train]
test_pairs = pairs[n_train:]
print(f"  {len(pairs)} pairs total, {n_train} train, {len(test_pairs)} test")


# --- Siamese GCN ---
class StructureEncoder(torch.nn.Module):
    """Encode a protein graph into a fixed-size embedding."""

    def __init__(self, in_channels=13, hidden=64, embed_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, embed_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.conv3(x, data.edge_index)
        # Global mean pool → fixed-size embedding
        return global_mean_pool(x, data.batch)


print("\n--- Training Siamese GNN ---")
encoder = StructureEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)

for epoch in range(50):
    encoder.train()
    random.shuffle(train_pairs)
    total_loss = 0

    for idx_i, idx_j, tm_score in train_pairs[:200]:  # mini-batches
        g_i = idx_to_graph[idx_i].to(DEVICE)
        g_j = idx_to_graph[idx_j].to(DEVICE)

        # Add batch index for single graphs
        g_i.batch = torch.zeros(g_i.num_nodes, dtype=torch.long, device=DEVICE)
        g_j.batch = torch.zeros(g_j.num_nodes, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()
        emb_i = encoder(g_i)
        emb_j = encoder(g_j)

        # Cosine similarity → should match TM-score
        cos_sim = F.cosine_similarity(emb_i, emb_j)
        target = torch.tensor([tm_score], dtype=torch.float32, device=DEVICE)
        loss = F.mse_loss(cos_sim, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        # Evaluate
        encoder.eval()
        preds, targets = [], []
        with torch.no_grad():
            for idx_i, idx_j, tm_score in test_pairs:
                g_i = idx_to_graph[idx_i].to(DEVICE)
                g_j = idx_to_graph[idx_j].to(DEVICE)
                g_i.batch = torch.zeros(g_i.num_nodes, dtype=torch.long, device=DEVICE)
                g_j.batch = torch.zeros(g_j.num_nodes, dtype=torch.long, device=DEVICE)
                emb_i = encoder(g_i)
                emb_j = encoder(g_j)
                cos_sim = F.cosine_similarity(emb_i, emb_j).item()
                preds.append(cos_sim)
                targets.append(tm_score)

        preds = np.array(preds)
        targets = np.array(targets)
        corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 2 else 0
        mse = np.mean((preds - targets) ** 2)

        print(f"  Epoch {epoch+1:3d}: train_loss={total_loss/min(len(train_pairs),200):.4f}, "
              f"test_corr={corr:.3f}, test_mse={mse:.4f}")

print(f"\n  Final correlation between predicted and true TM-scores: {corr:.3f}")
print(f"  (Random baseline: ~0.0, good models: >0.5)")
print(f"  Device: {DEVICE}")
print("\nDone!")
