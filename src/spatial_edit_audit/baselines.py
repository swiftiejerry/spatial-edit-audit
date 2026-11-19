"""Spatial smoothing, neighbor-augmented clustering, and optional GNN baseline evaluators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import KMeans

from .contrast import budget_tile_count, build_contrast_operator, score_candidate_pool_fast
from .preprocessing import SectionBaseline


def normalized_adjacency(graph: csr_matrix) -> csr_matrix:
    """Return row-normalized undirected spatial adjacency matrix ``D^{-1} A``."""
    adj = graph.astype(np.float64)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 0.0, 1.0 / deg, 0.0)
    return diags(inv_deg) @ adj


def apply_graph_diffusion(features: np.ndarray, a_norm: csr_matrix, alpha: float) -> np.ndarray:
    """Apply linear one-step spatial neighbor smoothing ``(1 - alpha) X + alpha A_norm X``."""
    if alpha <= 0.0:
        return features.copy()
    return (1.0 - float(alpha)) * features + float(alpha) * (a_norm @ features)


def evaluate_neighbor_augmented_clustering(
    baseline: SectionBaseline,
    x_p: np.ndarray,
    x_q: np.ndarray,
    target_indices: np.ndarray,
    *,
    neighbor_weight: float = 0.0,
    n_clusters: int = 12,
    seed: int = 20261111,
) -> float:
    """Run BANKSY-style spatial neighbor-augmented clustering and return best-matching target F1."""
    a_norm = normalized_adjacency(baseline.graph)
    feat = np.hstack([x_p, x_q])
    if neighbor_weight > 0.0:
        feat = np.hstack([feat, float(neighbor_weight) * (a_norm @ feat)])
    preds = KMeans(n_clusters=int(n_clusters), n_init=3, random_state=seed).fit_predict(feat)
    t_mask = np.zeros(baseline.n_spots, dtype=bool)
    t_mask[np.asarray(target_indices, dtype=int)] = True
    n_t = int(t_mask.sum())
    best_f1 = 0.0
    for cid in np.unique(preds):
        c_mask = preds == cid
        tp = int((c_mask & t_mask).sum())
        if tp > 0:
            f1 = 2.0 * tp / float(int(c_mask.sum()) + n_t)
            if f1 > best_f1:
                best_f1 = f1
    return float(best_f1)


def score_smoothed_views(
    baseline: SectionBaseline,
    x_p: np.ndarray,
    x_q: np.ndarray,
    *,
    alpha: float = 0.0,
) -> dict[str, Any]:
    """Smooth both modality views via spatial graph diffusion and compute 3% fixed-budget min ranking."""
    a_norm = normalized_adjacency(baseline.graph)
    sp = apply_graph_diffusion(x_p, a_norm, alpha)
    sq = apply_graph_diffusion(x_q, a_norm, alpha)
    candidates = [(f"T{i:04d}", m) for i, m in enumerate(baseline.members_by_tile)]
    op, scorable = build_contrast_operator(candidates, baseline.graph, baseline.n_spots)
    ep, eq = score_candidate_pool_fast(op, scorable, sp, sq)
    s_min = np.minimum(ep, eq)
    m_cap = budget_tile_count(0.03, baseline.n_tiles)
    idx = np.flatnonzero(scorable)
    chosen = idx[np.lexsort((idx, -s_min[idx]))][:m_cap]
    return {"e_min": s_min, "selected_indices": chosen.tolist()}
