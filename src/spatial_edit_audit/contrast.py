"""Local two-hop molecular contrast scoring, aggregation rules, and fixed-budget ranking."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .spatial_graph import compute_two_hop_ring

RULES = ("P", "Q", "mean", "min")
REASONS_UNSCORABLE = {"insufficient_spatial_controls", "constant_independent_features"}


def _stable_seed(members: np.ndarray, n_obs: int) -> int:
    raw = ",".join(map(str, np.asarray(members, dtype=int))) + f"/{int(n_obs)}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "little") % (2**32)


def robust_feature_scale(x: np.ndarray, *, ddof: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature MAD x 1.4826 scale with population-SD (default ddof=0) fallback when MAD <= 1e-12."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    fallback = np.std(x, axis=0, ddof=int(ddof))
    scale = np.where((mad <= 1e-12) & (fallback > 1e-12), fallback, mad)
    valid = scale > 1e-12
    return scale, valid


def local_molecular_contrast(
    members: np.ndarray,
    graph: csr_matrix,
    features: np.ndarray,
    *,
    min_control_multiplier: float = 1.0,
    n_null: int = 199,
    min_effect: float = 0.75,
    min_affected_fraction: float = 0.10,
    max_null_p: float = 0.05,
    precomputed_scale: tuple[np.ndarray, np.ndarray] | None = None,
    disjoint_perm_ref: bool = False,
) -> dict[str, Any]:
    """Evaluate standardized local molecular contrast against a disjoint two-hop spatial ring."""
    members = np.asarray(members, dtype=int)
    x = np.asarray(features, dtype=np.float64)
    n_obs = len(x)
    if members.size == 0 or x.ndim != 2 or x.shape[0] != n_obs:
        raise ValueError("Incompatible candidate members or feature matrix shape")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix must be finite")

    controls = compute_two_hop_ring(members, graph, hops=2)
    required = max(int(math.ceil(min_control_multiplier * len(members))), 4)
    if len(controls) < required:
        return {
            "status": "REJECT",
            "reason": "insufficient_spatial_controls",
            "n_controls": int(len(controls)),
            "required_controls": required,
            "effect": 0.0,
            "affected_feature_fraction": 0.0,
            "null_p": 1.0,
            "members": members.tolist(),
        }

    scale, valid = precomputed_scale if precomputed_scale is not None else robust_feature_scale(x)
    if not np.any(valid):
        return {
            "status": "REJECT",
            "reason": "constant_independent_features",
            "n_controls": int(len(controls)),
            "required_controls": required,
            "effect": 0.0,
            "affected_feature_fraction": 0.0,
            "null_p": 1.0,
            "members": members.tolist(),
        }

    effect_vec = np.zeros(x.shape[1], dtype=np.float64)
    effect_vec[valid] = np.abs(x[members][:, valid].mean(axis=0) - x[controls][:, valid].mean(axis=0)) / scale[valid]
    effect = float(np.mean(effect_vec[valid]))
    affected = float(np.mean(effect_vec[valid] >= min_effect))

    rng = np.random.default_rng(_stable_seed(members, n_obs))
    null_effects = np.empty(int(n_null), dtype=np.float64)
    replace = len(controls) < len(members)
    control_mean_full = x[controls][:, valid].mean(axis=0)
    for i in range(int(n_null)):
        draw = rng.choice(controls, size=len(members), replace=replace)
        if disjoint_perm_ref and not replace and len(controls) > len(members):
            rem = np.setdiff1d(controls, draw, assume_unique=False)
            ref_mean = x[rem][:, valid].mean(axis=0)
        else:
            ref_mean = control_mean_full
        vals = np.abs(x[draw][:, valid].mean(axis=0) - ref_mean) / scale[valid]
        null_effects[i] = float(np.mean(vals))
    null_p = float((1 + np.sum(null_effects >= effect)) / (1 + len(null_effects)))

    checks = {
        "enough_controls": len(controls) >= required,
        "effect": effect >= min_effect,
        "affected_features": affected >= min_affected_fraction,
        "null_p": null_p <= max_null_p,
    }
    return {
        "status": "ACCEPT" if all(checks.values()) else "REJECT",
        "reason": "passed_local_molecular_contrast" if all(checks.values()) else "failed:" + ",".join(k for k, v in checks.items() if not v),
        "n_controls": int(len(controls)),
        "required_controls": required,
        "effect": effect,
        "affected_feature_fraction": affected,
        "null_p": null_p,
        "checks": checks,
        "members": members.tolist(),
    }


def build_contrast_operator(
    candidates: list[tuple[str, np.ndarray]],
    graph: csr_matrix,
    n_spots: int,
    *,
    min_control_multiplier: float = 1.0,
) -> tuple[csr_matrix, np.ndarray]:
    """Precompile a sparse difference operator (mean_tile - mean_ring) for fast batch scoring."""
    rows, cols, vals = [], [], []
    scorable = np.zeros(len(candidates), dtype=bool)
    for idx, (_cid, members) in enumerate(candidates):
        ring = compute_two_hop_ring(members, graph, hops=2)
        req = max(int(math.ceil(min_control_multiplier * len(members))), 4)
        if len(ring) >= req:
            scorable[idx] = True
            wm = 1.0 / len(members)
            wr = -1.0 / len(ring)
            rows.extend([idx] * len(members) + [idx] * len(ring))
            cols.extend(members.tolist() + ring.tolist())
            vals.extend([wm] * len(members) + [wr] * len(ring))
    op = csr_matrix((vals, (rows, cols)), shape=(len(candidates), n_spots), dtype=np.float64)
    return op, scorable


def score_candidate_pool_fast(
    op: csr_matrix,
    scorable: np.ndarray,
    x_p: np.ndarray,
    x_q: np.ndarray,
    *,
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized scoring exact to ``local_molecular_contrast`` up to IEEE-754 rounding (<1e-12)."""
    def _score_one(x: np.ndarray) -> np.ndarray:
        scale, valid = robust_feature_scale(x, ddof=ddof)
        if not np.any(valid):
            out = np.zeros(op.shape[0], dtype=np.float64)
            out[~scorable] = np.nan
            return out
        diffs = np.abs((op @ x)[:, valid]) / scale[valid]
        out = np.mean(diffs, axis=1)
        out[~scorable] = np.nan
        return out

    return _score_one(x_p), _score_one(x_q)


def aggregate_rule_scores(e_p: np.ndarray, e_q: np.ndarray) -> dict[str, np.ndarray]:
    """Compute scalar priority rules P, Q, mean, and min from channel contrasts."""
    return {
        "P": e_p,
        "Q": e_q,
        "mean": 0.5 * (e_p + e_q),
        "min": np.minimum(e_p, e_q),
    }


def score_candidate_pool(
    graph: csr_matrix,
    x_p: np.ndarray,
    x_q: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    *,
    min_control_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute scalar contrast rules (P, Q, mean, min) across a candidate list."""
    rows: list[dict[str, Any]] = []
    for cid, members in candidates:
        res_p = local_molecular_contrast(members, graph, x_p, min_control_multiplier=min_control_multiplier, n_null=0)
        res_q = local_molecular_contrast(members, graph, x_q, min_control_multiplier=min_control_multiplier, n_null=0)
        rp, rq = str(res_p["reason"]), str(res_q["reason"])
        ep, eq = float(res_p["effect"]), float(res_q["effect"])
        ok = (rp not in REASONS_UNSCORABLE) and (rq not in REASONS_UNSCORABLE)
        rows.append(
            {
                "candidate_id": str(cid),
                "n_points": int(len(members)),
                "e_P": ep,
                "e_Q": eq,
                "reason_P": rp,
                "reason_Q": rq,
                "scorable": bool(ok),
                "P": ep,
                "Q": eq,
                "mean": 0.5 * (ep + eq),
                "min": min(ep, eq),
            }
        )
    return pd.DataFrame.from_records(rows)


def budget_tile_count(fraction: float, n_candidates: int) -> int:
    """Compute rounded integer tile quota m_b = max(1, floor(f * |T_b| + 0.5))."""
    return max(1, int(math.floor(float(fraction) * int(n_candidates) + 0.5)))


def rank_scorable_candidates(scores: pd.DataFrame, rule: str) -> list[str]:
    """Rank scorable candidates by descending score with lexicographic candidate_id tie-breaking."""
    eligible = scores[scores["scorable"].astype(bool)].sort_values(
        [rule, "candidate_id"], ascending=[False, True], kind="mergesort"
    )
    return eligible["candidate_id"].astype(str).tolist()
