"""Parse SWC morphology files and turn them into PyTorch Geometric graphs.

An SWC file lists one sample point per line:
    id  type  x  y  z  radius  parent_id
Each point becomes a node; each child-parent link becomes an undirected edge.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

# SWC structure codes (Cannon et al. 1998 convention, used by NeuroMorpho.Org)
SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE = 1, 2, 3, 4

# Order of the per-node feature columns produced by `node_features`
FEATURE_NAMES = [
    "x", "y", "z",          # coordinates, centred on the neuron's mean position
    "radius",
    "swc_type",             # structure code: see note in README on label leakage
    "degree",               # number of neighbours
    "path_dist",            # hop distance from the root node, scaled 0-1
    "is_branch_point",      # 1 if degree > 2
]


@dataclass
class SWC:
    xyz: np.ndarray        # (n, 3) float32
    radius: np.ndarray     # (n,)  float32
    types: np.ndarray      # (n,)  int64
    edges: np.ndarray      # (2, e) int64, both directions


def parse_swc(path: str | Path) -> SWC:
    """Read an SWC file. Malformed lines and links to missing parents are skipped."""
    ids, types, xyz, radius, parents = [], [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                ids.append(int(p[0]))
                types.append(int(p[1]))
                xyz.append([float(p[2]), float(p[3]), float(p[4])])
                radius.append(float(p[5]))
                parents.append(int(p[6]))
            except ValueError:
                continue

    # Map SWC ids to 0-based row indices; build the index first so a parent
    # listed after its child is still found.
    index = {node_id: i for i, node_id in enumerate(ids)}
    edges = []
    for i, parent in enumerate(parents):
        if parent in index:
            j = index[parent]
            edges += [[i, j], [j, i]]

    return SWC(
        xyz=np.asarray(xyz, np.float32).reshape(-1, 3),
        radius=np.asarray(radius, np.float32),
        types=np.asarray(types, np.int64),
        edges=np.asarray(edges, np.int64).T if edges else np.zeros((2, 0), np.int64),
    )


def node_features(swc: SWC) -> np.ndarray:
    """Per-node features, columns in the order of FEATURE_NAMES."""
    n = len(swc.xyz)
    xyz = swc.xyz - swc.xyz.mean(axis=0, keepdims=True)

    adj = defaultdict(list)
    for a, b in swc.edges.T:
        adj[int(a)].append(int(b))
    degree = np.array([len(adj[i]) for i in range(n)], np.float32)

    # Breadth-first hop distance from node 0 (the root, normally the soma)
    dist = np.full(n, -1.0, np.float32)
    dist[0] = 0.0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    # Disconnected fragments get the maximum distance
    dist[dist < 0] = dist.max() if (dist >= 0).any() else 0.0
    dist /= dist.max() + 1e-6

    return np.concatenate(
        [
            xyz,
            swc.radius[:, None],
            swc.types[:, None].astype(np.float32),
            degree[:, None],
            dist[:, None],
            (degree > 2).astype(np.float32)[:, None],
        ],
        axis=1,
    )


def swc_to_graph(
    path: str | Path,
    label: int | None = None,
    min_nodes: int = 5,
    max_nodes: int = 10_000,
) -> Data | None:
    """Build a graph from one SWC file. Returns None if the node count is out of range."""
    swc = parse_swc(path)
    n = len(swc.xyz)
    if n < min_nodes or n > max_nodes:
        return None
    data = Data(
        x=torch.tensor(node_features(swc), dtype=torch.float),
        edge_index=torch.tensor(swc.edges, dtype=torch.long),
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    data.name = Path(path).name.replace(".CNG.swc", "").replace(".swc", "")
    return data
