"""Spatial neighbor graph construction, two-hop control rings, and contiguous candidate tiles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(coords: np.ndarray, k: int = 6) -> tuple[csr_matrix, np.ndarray]:
    """Build directed k-NN indices and their symmetrized undirected adjacency graph.

    Explicitly excludes self-edges even when duplicate spatial coordinates exist,
    then breaks ties lexicographically by (distance, target_index).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be finite with shape (n_spots, 2)")
    n = len(coords)
    if n == 0 or k < 1:
        raise ValueError("At least one spot and k >= 1 are required")
    k_eff = min(int(k), n - 1)
    if k_eff == 0:
        return csr_matrix((n, n), dtype=np.uint8), np.empty((n, 0), dtype=int)
    model = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    distances, indices = model.kneighbors(coords)
    rows = []
    for i, (ds, js) in enumerate(zip(distances, indices)):
        keep = js != i
        ds, js = ds[keep], js[keep]
        order = np.lexsort((js, ds))
        rows.append(js[order][:k_eff])
    neighbor_indices = np.asarray(rows, dtype=int)
    src = np.repeat(np.arange(n, dtype=int), k_eff)
    dst = neighbor_indices.reshape(-1)
    graph = csr_matrix((np.ones(len(src), dtype=np.uint8), (src, dst)), shape=(n, n))
    graph = graph.maximum(graph.T)
    return graph, neighbor_indices


def compute_two_hop_ring(members: np.ndarray, graph: csr_matrix, hops: int = 2) -> np.ndarray:
    """Return sorted spot indices in the disjoint two-hop spatial ring outside ``members``."""
    member_set = set(np.asarray(members, dtype=int).tolist())
    frontier = set(member_set)
    ring: set[int] = set()
    for _ in range(int(hops)):
        nxt: set[int] = set()
        for node in frontier:
            nxt.update(int(j) for j in graph.getrow(node).indices)
        nxt -= member_set
        ring.update(nxt)
        frontier = nxt
    return np.asarray(sorted(ring), dtype=int)


def partition_spatial_tiles(
    coords: np.ndarray,
    reference_labels: np.ndarray,
    graph: csr_matrix,
    *,
    target_tile_size: int = 20,
    seed: int = 20261111,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Partition spatial coordinates into connected candidate tiles stratified by departure."""
    n = len(coords)
    n_clusters = max(1, int(math.floor(n / float(target_tile_size) + 0.5)))
    km_labels = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit_predict(coords)

    raw_components: list[np.ndarray] = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(km_labels == cluster)
        subgraph = graph[members][:, members]
        n_comp, comp_labels = connected_components(subgraph, directed=False)
        for comp in range(n_comp):
            raw_components.append(members[comp_labels == comp])
    raw_components.sort(key=lambda arr: int(arr.min()))
    if any(len(arr) < 5 for arr in raw_components):
        bad = [int(len(arr)) for arr in raw_components if len(arr) < 5]
        raise ValueError(f"Candidate tile partitioning produced component(s) smaller than 5: {bad}")

    tile_of = np.full(n, -1, dtype=np.int32)
    members_by_tile: list[np.ndarray] = []
    for tid, members in enumerate(raw_components):
        tile_of[members] = tid
        members_by_tile.append(members)
    if np.any(tile_of < 0):
        raise ValueError("Tile assignment omitted at least one spot")

    records: list[dict[str, Any]] = []
    for tid, members in enumerate(members_by_tile):
        _, counts = np.unique(reference_labels[members], return_counts=True)
        dep = float(1.0 - counts.max() / len(members))
        category = "LOW" if dep <= 0.20 else ("HIGH" if dep >= 0.35 else "MID")
        adj: set[int] = set()
        for spot in members:
            for nb in graph.getrow(int(spot)).indices:
                other = int(tile_of[int(nb)])
                if other != tid:
                    adj.add(other)
        ring = compute_two_hop_ring(members, graph, hops=2)
        scorable = len(ring) >= max(len(members), 4)
        records.append(
            {
                "tile_id": f"T{tid:04d}",
                "n_points": int(len(members)),
                "dep": dep,
                "category": category,
                "adjacent_tiles": ";".join(f"T{x:04d}" for x in sorted(adj)),
                "scorable": bool(scorable),
            }
        )
    return tile_of, pd.DataFrame.from_records(records)
