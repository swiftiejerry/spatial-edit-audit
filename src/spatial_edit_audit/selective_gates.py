"""Selective multi-gate prototype (M0), calibrated dual-channel baseline, and counterfactuals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .contrast import local_molecular_contrast, robust_feature_scale
from .preprocessing import SectionBaseline
from .spatial_graph import compute_two_hop_ring


@dataclass(frozen=True)
class GateConfig:
    """Component gate configuration for selective local refinement evaluation."""

    k: int = 6
    rare_max_fraction: float = 0.15
    min_component_size: int = 5
    max_total_edit_fraction: float = 0.15
    min_control_multiplier: float = 1.0
    n_null: int = 199
    min_local_effect: float = 0.75
    min_affected_feature_fraction: float = 0.10
    max_local_null_p: float = 0.05
    n_bound_boot: int = 199
    bound_confidence: float = 0.95
    min_gain_lower_bound: float = 0.50
    max_damage_upper_bound: float = 0.55
    min_spatial_coherence: float = 0.55
    min_independent_agreement: float = 0.60
    max_boundary_leakage: float = 0.55
    disable_damage_gate: bool = False


def _boot_seed(members: np.ndarray, controls: np.ndarray, n_obs: int) -> int:
    raw = ",".join(map(str, np.asarray(members, dtype=int)))
    raw += "|" + ",".join(map(str, np.asarray(controls, dtype=int))) + f"/{int(n_obs)}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "little") % (2**32)


def _patch_effect(
    members: np.ndarray,
    controls: np.ndarray,
    features: np.ndarray,
    precomputed_scale: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    scale, valid = precomputed_scale if precomputed_scale is not None else robust_feature_scale(features)
    if not np.any(valid):
        return 0.0
    diff = np.abs(features[members][:, valid].mean(axis=0) - features[controls][:, valid].mean(axis=0)) / scale[valid]
    return float(np.mean(diff))


def _departure(ix: np.ndarray, reference: np.ndarray) -> float:
    if len(ix) == 0:
        return 1.0
    _, counts = np.unique(reference[ix], return_counts=True)
    return float(1.0 - counts.max() / len(ix))


def _boundary_leakage(ix: np.ndarray, neighbors: np.ndarray) -> float:
    chosen = set(np.asarray(ix, dtype=int).tolist())
    total = 0
    boundary = 0
    for spot in ix:
        nbrs = neighbors[int(spot)]
        total += len(nbrs)
        boundary += sum(int(j not in chosen) for j in nbrs)
    return float(boundary / total) if total else 1.0


def empirical_bounds(
    members: np.ndarray,
    controls: np.ndarray,
    features: np.ndarray,
    reference_labels: np.ndarray,
    neighbors: np.ndarray,
    *,
    n_boot: int = 199,
    confidence: float = 0.95,
    precomputed_scale: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute within-tile bootstrap bounds ``gain_LCB`` and ``damage_UCB``."""
    members = np.asarray(members, dtype=int)
    controls = np.asarray(controls, dtype=int)
    if members.size == 0 or controls.size == 0:
        return {"status": "REJECT", "gain_lower_bound": 0.0, "damage_upper_bound": 1.0}
    x = np.asarray(features, dtype=np.float64)
    scale_info = precomputed_scale if precomputed_scale is not None else robust_feature_scale(x)
    rng = np.random.default_rng(_boot_seed(members, controls, len(x)))
    gain_samples = np.empty(int(n_boot), dtype=np.float64)
    damage_samples = np.empty(int(n_boot), dtype=np.float64)

    static_leakage = _boundary_leakage(members, neighbors)
    for i in range(int(n_boot)):
        m_draw = rng.choice(members, size=len(members), replace=True)
        c_draw = rng.choice(controls, size=len(controls), replace=True)
        gain_samples[i] = _patch_effect(m_draw, c_draw, x, precomputed_scale=scale_info)
        damage_samples[i] = min(1.0, max(_departure(m_draw, reference_labels), static_leakage))

    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "status": "OK",
        "gain_lower_bound": float(np.quantile(gain_samples, alpha)),
        "damage_upper_bound": float(np.quantile(damage_samples, 1.0 - alpha)),
        "observed_gain": float(_patch_effect(members, controls, x, precomputed_scale=scale_info)),
        "observed_reference_departure": _departure(members, reference_labels),
        "observed_boundary_leakage": static_leakage,
    }


def evaluate_multigate_prototype(
    baseline: SectionBaseline,
    x_q: np.ndarray,
    i_case: np.ndarray,
    *,
    reference_labels: np.ndarray | None = None,
    config: GateConfig | None = None,
) -> dict[str, Any]:
    """Run the single-channel confirmation multi-gate prototype (M0) over candidate tiles.

    Notice that under directed 6-NN spatial coherence ``gamma(A) = 1 - lambda(A)``,
    the bound ``gamma(A) >= 0.55`` algebraically implies ``lambda(A) <= 0.45 < 0.55``.
    """
    cfg = config or GateConfig()
    ref = baseline.r_static if reference_labels is None else np.asarray(reference_labels, dtype=str)
    n_obs = baseline.n_spots
    prop_labels = np.asarray([f"T{int(v):04d}" for v in baseline.tile_of], dtype=str)
    scale_info = robust_feature_scale(x_q)

    records: list[dict[str, Any]] = []
    proposed_accepted_spots = 0
    for cid in sorted(np.unique(prop_labels)):
        members = np.flatnonzero(prop_labels == cid)
        if len(members) / n_obs > cfg.rare_max_fraction:
            continue
        mask = np.zeros(n_obs, dtype=bool)
        mask[members] = True
        sub = baseline.graph[mask][:, mask]
        n_comp, comp_labels = connected_components(sub, directed=False)
        for comp_id in range(n_comp):
            comp = members[comp_labels == comp_id]
            comp_set = set(comp.tolist())
            local_nbrs = baseline.neighbors[comp]
            internal_edges = sum(int(j in comp_set) for row in local_nbrs for j in row)
            total_edges = int(local_nbrs.size)
            coherence = internal_edges / total_edges if total_edges else 0.0
            leakage = (total_edges - internal_edges) / total_edges if total_edges else 1.0
            _, i_counts = np.unique(i_case[comp], return_counts=True)
            indep_agree = float(i_counts.max() / len(comp))
            area_frac = float(len(comp) / n_obs)

            local = local_molecular_contrast(
                comp,
                baseline.graph,
                x_q,
                min_control_multiplier=cfg.min_control_multiplier,
                n_null=cfg.n_null,
                min_effect=cfg.min_local_effect,
                min_affected_fraction=cfg.min_affected_feature_fraction,
                max_null_p=cfg.max_local_null_p,
                precomputed_scale=scale_info,
            )
            controls = compute_two_hop_ring(comp, baseline.graph, hops=2)
            bounds = empirical_bounds(
                comp,
                controls,
                x_q,
                ref,
                baseline.neighbors,
                n_boot=cfg.n_bound_boot,
                confidence=cfg.bound_confidence,
                precomputed_scale=scale_info,
            )
            checks = {
                "min_component_size": len(comp) >= cfg.min_component_size,
                "rare_area": area_frac <= cfg.rare_max_fraction,
                "spatial_coherence": coherence >= cfg.min_spatial_coherence,
                "independent_agreement": indep_agree >= cfg.min_independent_agreement,
                "boundary_leakage": leakage <= cfg.max_boundary_leakage,
                "local_molecular_contrast": local["status"] == "ACCEPT",
                "gain_lower_bound": bounds["gain_lower_bound"] >= cfg.min_gain_lower_bound,
                "damage_upper_bound": (
                    True if cfg.disable_damage_gate else bounds["damage_upper_bound"] <= cfg.max_damage_upper_bound
                ),
            }
            accepted = all(checks.values())
            if accepted:
                proposed_accepted_spots += len(comp)
            records.append(
                {
                    "candidate_id": cid,
                    "component_id": comp_id,
                    "n_spots": int(len(comp)),
                    "spatial_coherence": coherence,
                    "boundary_leakage": leakage,
                    "independent_agreement": indep_agree,
                    "effect_Q": float(local["effect"]),
                    "gain_lower_bound": float(bounds["gain_lower_bound"]),
                    "damage_upper_bound": float(bounds["damage_upper_bound"]),
                    "checks": checks,
                    "status": "ACCEPT" if accepted else "REJECT",
                    "failed": [k for k, v in checks.items() if not v],
                }
            )

    budget_exceeded = proposed_accepted_spots / n_obs > cfg.max_total_edit_fraction
    if budget_exceeded:
        for rec in records:
            if rec["status"] == "ACCEPT":
                rec["status"] = "REJECT"
                rec["failed"].append("joint_edit_budget")
    accepted_ids = [r["candidate_id"] for r in records if r["status"] == "ACCEPT"]
    return {
        "decision": "ACCEPT" if accepted_ids else "ABSTAIN",
        "accepted_ids": accepted_ids,
        "k_accepted": len(accepted_ids),
        "budget_exceeded": budget_exceeded,
        "components": records,
    }


def apply_reference_intervention(baseline: SectionBaseline, target_tile_id: str, mode: str) -> np.ndarray:
    """Apply paired counterfactual edits to static reference partition ``R_static`` inside target tile.

    Args:
        baseline: Loaded section baseline.
        target_tile_id: Target candidate ID (e.g., ``"T0012"``).
        mode: ``"homogeneous"`` (map target spots to local majority class, ``dep=0``)
              or ``"three_way_split"`` (split uniform target evenly across three global classes).
    """
    changed = baseline.r_static.copy()
    members = np.flatnonzero(baseline.tile_of == int(target_tile_id[1:]))
    values, counts = np.unique(changed[members], return_counts=True)
    majority = sorted([(int(n), str(v)) for v, n in zip(values, counts)], key=lambda z: (-z[0], z[1]))[0][1]
    if mode == "homogeneous":
        changed[members] = majority
        return changed
    if mode == "three_way_split":
        global_vals, global_cnts = np.unique(changed, return_counts=True)
        alts = sorted(
            [(int(n), str(v)) for v, n in zip(global_vals, global_cnts) if str(v) != majority],
            key=lambda z: (-z[0], z[1]),
        )
        if len(alts) < 2:
            raise RuntimeError("At least two alternative global classes are required for three_way_split")
        chunks = np.array_split(np.sort(members), 3)
        changed[chunks[0]] = majority
        changed[chunks[1]] = alts[0][1]
        changed[chunks[2]] = alts[1][1]
        return changed
    raise ValueError(f"Unknown counterfactual intervention mode: {mode!r}")


def evaluate_dual_calibrated_gate(
    scores_df: Any,
    static_tile_table: Any,
    q_min: float,
    n_spots: int,
    *,
    spot_cap_fraction: float = 0.05,
    min_size: int = 5,
    rare_max_fraction: float = 0.15,
) -> list[str]:
    """Calibrated dual-channel comparator free of static reference-partition anchoring.

    Selects scorable tiles passing structural compactness (``size >= 5``, ``area <= 0.15``)
    and calibrated dual-channel minimum contrast ``min(e_P, e_Q) >= q_{b,min}`` in
    descending score order up to the total section edit spot cap.
    """
    ok = scores_df[scores_df["scorable"].astype(bool)].copy()
    eligible = ok[
        (ok["min"] >= float(q_min))
        & (ok["n_points"] >= int(min_size))
        & (ok["n_points"] / float(n_spots) <= float(rare_max_fraction))
    ].sort_values(["min", "candidate_id"], ascending=[False, True], kind="mergesort")

    accepted: list[str] = []
    used_spots = 0
    for _, row in eligible.iterrows():
        pts = int(row["n_points"])
        if (used_spots + pts) / float(n_spots) <= float(spot_cap_fraction):
            accepted.append(str(row["candidate_id"]))
            used_spots += pts
    return accepted
