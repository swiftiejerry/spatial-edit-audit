"""Stratified PCG64 simulation generators across Blocks 1-3 and raw count-level checks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from .preprocessing import SectionBaseline
from .spatial_graph import compute_two_hop_ring

BLOCK_SEED_ROOT = 20261111
MAIN_DOSES = {4: 1.0, 5: 2.0, 6: 3.0, 7: 4.0, 8: 3.0, 9: None}


def stream_rng(block: int, section_index: int, stratum: int, rep: int) -> np.random.Generator:
    """Create the deterministic PCG64 generator keyed by (root, block, section, stratum, replicate)."""
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([BLOCK_SEED_ROOT, int(block), int(section_index), int(stratum), int(rep)]))
    )


def _isolated_tiles(baseline: SectionBaseline, a: int, b: int) -> bool:
    return bool(baseline.isolated_matrix[a, b])


def _sample_positions(rng: np.random.Generator, stratum: int, baseline: SectionBaseline) -> dict[str, int]:
    categories = baseline.tile_table["category"].astype(str).to_numpy()
    pos: dict[str, int] = {}
    if 4 <= stratum <= 9:
        cls = "HIGH" if stratum == 8 else "LOW"
        candidates = sorted(np.flatnonzero(categories == cls).tolist())
        pos["T"] = int(rng.choice(candidates))
    if stratum in (2, 4, 5, 6, 7, 8, 9):
        if stratum == 2:
            pos["A"] = int(rng.choice(np.arange(baseline.n_tiles)))
            pool_b = [j for j in range(baseline.n_tiles) if j != pos["A"] and _isolated_tiles(baseline, pos["A"], j)]
        else:
            t_id = pos["T"]
            pool_a = [j for j in range(baseline.n_tiles) if j != t_id and _isolated_tiles(baseline, t_id, j)]
            pos["A"] = int(rng.choice(sorted(pool_a)))
            pool_b = [
                j
                for j in range(baseline.n_tiles)
                if j not in (t_id, pos["A"])
                and _isolated_tiles(baseline, t_id, j)
                and _isolated_tiles(baseline, pos["A"], j)
            ]
        pos["B"] = int(rng.choice(sorted(pool_b)))
    elif stratum == 3:
        pool_ap = [i for i, nbrs in enumerate(baseline.adjacent_tiles) if nbrs]
        ap = int(rng.choice(sorted(pool_ap)))
        pos["A_prime"] = ap
        pos["B_prime"] = int(rng.choice(sorted(baseline.adjacent_tiles[ap])))
    return pos


def _inject_four_axis_shift(
    rng: np.random.Generator,
    label: str,
    channel: str,
    members: np.ndarray,
    delta: float,
    x_p: np.ndarray,
    x_q: np.ndarray,
) -> dict[str, Any]:
    axes = np.sort(rng.choice(16, size=4, replace=False)).astype(int)
    signs = (2 * rng.integers(0, 2, size=4) - 1).astype(int)
    target = x_p if channel == "P" else x_q
    target[np.asarray(members, dtype=int)[:, None], axes[None, :]] += float(delta) * signs[None, :]
    return {"label": label, "channel": channel, "J": axes.tolist(), "s": signs.tolist(), "delta": float(delta)}


def generate_main_case(
    baseline: SectionBaseline,
    block: int,
    stratum: int,
    rep: int,
    *,
    calibration_draw: bool = False,
    fit_i_case: bool = True,
) -> dict[str, Any]:
    """Generate one Block-0, Block-1 calibration, or Block-2 main realization on SVD baselines."""
    rng = stream_rng(block, baseline.section_index, stratum, rep)
    actual_stratum = int(rng.integers(1, 4)) if calibration_draw else int(stratum)
    n = baseline.n_spots
    x_p = baseline.p0.copy() + 0.5 * rng.standard_normal((n, 16))
    x_q = baseline.q0.copy() + 0.5 * rng.standard_normal((n, 16))

    positions = _sample_positions(rng, actual_stratum, baseline)
    dose = MAIN_DOSES.get(actual_stratum)
    asym_variant = int(rng.integers(0, 2)) if actual_stratum == 9 else None
    injections: list[dict[str, Any]] = []

    if 4 <= actual_stratum <= 9:
        t_idx = positions["T"]
        p_delta, q_delta = (
            ((3.0, 1.0) if asym_variant == 0 else (1.0, 3.0))
            if actual_stratum == 9
            else (float(dose), float(dose))
        )
        injections.append(_inject_four_axis_shift(rng, "T", "P", baseline.members_by_tile[t_idx], p_delta, x_p, x_q))
        injections.append(_inject_four_axis_shift(rng, "T", "Q", baseline.members_by_tile[t_idx], q_delta, x_p, x_q))
    if actual_stratum == 2 or 4 <= actual_stratum <= 9:
        injections.append(_inject_four_axis_shift(rng, "A", "P", baseline.members_by_tile[positions["A"]], 3.0, x_p, x_q))
        injections.append(_inject_four_axis_shift(rng, "B", "Q", baseline.members_by_tile[positions["B"]], 3.0, x_p, x_q))
    elif actual_stratum == 3:
        injections.append(_inject_four_axis_shift(rng, "A_prime", "P", baseline.members_by_tile[positions["A_prime"]], 3.0, x_p, x_q))
        injections.append(_inject_four_axis_shift(rng, "B_prime", "Q", baseline.members_by_tile[positions["B_prime"]], 3.0, x_p, x_q))

    i_case = None
    if fit_i_case:
        clusters = KMeans(n_clusters=8, n_init=10, random_state=BLOCK_SEED_ROOT).fit_predict(x_q)
        i_case = np.asarray([f"I{int(v)}" for v in clusters], dtype=str)

    truth = {
        "block": int(block),
        "background": baseline.name,
        "section_index": baseline.section_index,
        "stratum": actual_stratum,
        "rep": int(rep),
        "positions": {k: f"T{v:04d}" for k, v in positions.items()},
        "injections": injections,
        "asym_variant": asym_variant,
        "dose": dose,
    }
    return {"X_P": x_p, "X_Q": x_q, "I_case": i_case, "truth": truth, "positions": positions}


def _grow_contiguous_patch(
    rng: np.random.Generator,
    graph: Any,
    n_spots: int,
    blocked: set[int],
    size: int = 20,
) -> np.ndarray:
    for _ in range(100):
        start = int(rng.integers(0, n_spots))
        if start in blocked:
            continue
        chosen = {start}
        while len(chosen) < size:
            frontier: set[int] = set()
            for node in chosen:
                frontier.update(int(v) for v in graph.getrow(node).indices)
            frontier.difference_update(chosen)
            frontier.difference_update(blocked)
            if not frontier:
                break
            chosen.add(int(rng.choice(sorted(frontier))))
        if len(chosen) == size:
            return np.asarray(sorted(chosen), dtype=int)
    raise RuntimeError("Failed to grow disjoint contiguous 20-spot patch after 100 starts")


def generate_secondary_case(
    baseline: SectionBaseline,
    rep: int,
    *,
    block: int = 3,
    dose: float = 6.0,
) -> dict[str, Any]:
    """Generate one Block-3 secondary grown-patch case (disjoint 20-spot T, A, B regions)."""
    rng = stream_rng(block, baseline.section_index, 10, rep)
    x_p = baseline.p0.copy() + 0.5 * rng.standard_normal((baseline.n_spots, 16))
    x_q = baseline.q0.copy() + 0.5 * rng.standard_normal((baseline.n_spots, 16))

    regions: dict[str, np.ndarray] = {}
    blocked: set[int] = set()
    for label in ("T", "A", "B"):
        members = _grow_contiguous_patch(rng, baseline.graph, baseline.n_spots, blocked, size=20)
        regions[label] = members
        blocked.update(members.tolist())
        blocked.update(compute_two_hop_ring(members, baseline.graph, hops=2).tolist())

    injections = [
        _inject_four_axis_shift(rng, "T", "P", regions["T"], dose, x_p, x_q),
        _inject_four_axis_shift(rng, "T", "Q", regions["T"], dose, x_p, x_q),
        _inject_four_axis_shift(rng, "A", "P", regions["A"], dose, x_p, x_q),
        _inject_four_axis_shift(rng, "B", "Q", regions["B"], dose, x_p, x_q),
    ]
    truth = {
        "block": int(block),
        "background": baseline.name,
        "stratum": 10,
        "rep": int(rep),
        "target_members": regions["T"].tolist(),
        "decoy_A_members": regions["A"].tolist(),
        "decoy_B_members": regions["B"].tolist(),
        "injections": injections,
        "dose": float(dose),
    }
    return {"X_P": x_p, "X_Q": x_q, "truth": truth, "regions": regions}


def inject_poisson_count_shift(
    counts: csr_matrix,
    tile_spots: np.ndarray,
    feature_indices: np.ndarray,
    rng: np.random.Generator,
    fold_multiplier: float,
    *,
    affected_fraction: float = 0.20,
) -> csr_matrix:
    """Inject additive Poisson count elevation into raw UMI/fragment counts for pipeline robustness checks."""
    n_pick = max(4, int(round(affected_fraction * len(feature_indices))))
    chosen = rng.choice(feature_indices, size=min(n_pick, len(feature_indices)), replace=False)
    dense = counts.toarray().astype(np.float64)
    bg_means = np.maximum(dense[:, chosen].mean(axis=0), 0.1)
    lam = max(0.0, float(fold_multiplier) - 1.0) * bg_means[None, :]
    added = rng.poisson(np.broadcast_to(lam, (len(tile_spots), len(chosen))))
    dense[np.ix_(tile_spots, chosen)] += added
    return csr_matrix(dense)
