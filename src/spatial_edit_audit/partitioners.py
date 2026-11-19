"""Candidate partition geometry builders and target spot-coverage evaluation."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans

from .contrast import RULES, budget_tile_count, rank_scorable_candidates
from .preprocessing import SectionBaseline


def partition_ward_graph(
    baseline: SectionBaseline,
    x_p: np.ndarray,
    *,
    min_patch_size: int = 1,
) -> list[tuple[str, np.ndarray]]:
    """Build spatially constrained Ward agglomerative clusters (D1 raw or size-filtered).

    Setting ``min_patch_size=1`` yields the unfiltered D1 pool matching historical runs;
    setting ``min_patch_size=5`` enforces the same lower patch bound used elsewhere.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        labels = AgglomerativeClustering(
            n_clusters=baseline.n_tiles,
            linkage="ward",
            connectivity=baseline.graph,
        ).fit_predict(x_p)
    out: list[tuple[str, np.ndarray]] = []
    idx = 0
    for lb in sorted(np.unique(labels).tolist()):
        members = np.flatnonzero(labels == lb)
        if len(members) >= int(min_patch_size):
            prefix = "D1F" if min_patch_size > 1 else "D1"
            out.append((f"{prefix}_{idx:04d}", members))
            idx += 1
    return out


def partition_kmeans_components(
    baseline: SectionBaseline,
    x_p: np.ndarray,
    *,
    n_clusters: int = 12,
    min_patch_size: int = 5,
    max_area_fraction: float = 0.05,
    seed: int = 20261111,
) -> list[tuple[str, np.ndarray]]:
    """Build admissible connected components from k-means clustering on ``x_p`` (D2)."""
    km = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit_predict(x_p)
    max_size = int(math.floor(float(max_area_fraction) * baseline.n_spots))
    raw: list[np.ndarray] = []
    for lb in range(n_clusters):
        members = np.flatnonzero(km == lb)
        if len(members) == 0:
            continue
        sub = baseline.graph[members][:, members]
        n_comp, comp_labels = connected_components(sub, directed=False)
        for c_id in range(n_comp):
            group = members[comp_labels == c_id]
            if int(min_patch_size) <= len(group) <= max_size:
                raw.append(group)
    raw.sort(key=lambda arr: int(arr.min()))
    return [(f"D2_{i:04d}", m) for i, m in enumerate(raw)]


def build_grown_patch_candidates(
    baseline: SectionBaseline,
    x_p: np.ndarray,
    *,
    include_filtered_ward: bool = True,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Return fixed-tile (F), Ward (D1, optionally D1_filtered), and KMeans-CC (D2) pools."""
    fixed = [(f"T{i:04d}", m) for i, m in enumerate(baseline.members_by_tile)]
    out: dict[str, list[tuple[str, np.ndarray]]] = {
        "F": fixed,
        "D1": partition_ward_graph(baseline, x_p, min_patch_size=1),
        "D2": partition_kmeans_components(baseline, x_p),
    }
    if include_filtered_ward:
        out["D1_filtered"] = partition_ward_graph(baseline, x_p, min_patch_size=5)
    return out


def evaluate_partition_coverage(
    candidates: list[tuple[str, np.ndarray]],
    scores: pd.DataFrame,
    truth: dict[str, Any],
    *,
    budget_fraction: float = 0.03,
) -> list[dict[str, Any]]:
    """Evaluate 3% top-k target spot coverage and post hoc Oracle ceiling for one candidate source."""
    target = set(int(v) for v in truth["target_members"])
    decoy_a = set(int(v) for v in truth["decoy_A_members"])
    decoy_b = set(int(v) for v in truth["decoy_B_members"])
    lookup = {cid: set(int(v) for v in members) for cid, members in candidates}
    m = budget_tile_count(budget_fraction, len(candidates))

    upper = sorted(candidates, key=lambda pair: (-len(target & set(pair[1])), pair[0]))[:m]
    upper_union = set().union(*(set(int(v) for v in members) for _, members in upper)) if upper else set()
    rows: list[dict[str, Any]] = []
    for rule in RULES:
        selected = rank_scorable_candidates(scores, rule)[:m]
        union = set().union(*(lookup[cid] for cid in selected)) if selected else set()
        rows.append(
            {
                "rule": rule,
                "m_requested": m,
                "candidate_count": len(candidates),
                "n_scorable_candidates": int(scores["scorable"].astype(bool).sum()),
                "selected_point_count": len(union),
                "target_covered_points": len(target & union),
                "target_coverage": len(target & union) / float(len(target)),
                "outside_points": len(union - target),
                "decoy_A_overlap_points": len(decoy_a & union),
                "decoy_B_overlap_points": len(decoy_b & union),
                "oracle_upper_coverage": len(target & upper_union) / float(len(target)),
            }
        )
    return rows
