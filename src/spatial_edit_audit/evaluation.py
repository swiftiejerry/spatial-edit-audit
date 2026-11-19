"""Section-stratified bootstrap inference, quota-coupling summaries, and counterfactual audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contrast import RULES


def section_stratified_bootstrap_ci(
    per_section_diffs: dict[str, list[float] | np.ndarray],
    *,
    seed: int = 20261111,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """Compute equal-section weighted 95% percentile bootstrap CI sampling cases within slice."""
    rng = np.random.default_rng(seed)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in per_section_diffs.items()}
    stats = np.empty(int(n_boot), dtype=np.float64)
    for b_idx in range(int(n_boot)):
        means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for arr in arrays.values()]
        stats[b_idx] = float(np.mean(means))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def summarize_fixed_budget_contrasts(
    cases_df: pd.DataFrame,
    *,
    focal_rule: str = "min",
    comparators: tuple[str, ...] = ("Q", "mean", "P"),
    budget_suffix: str = "0.03",
    seed: int = 20261111,
) -> dict[str, Any]:
    """Summarize hit rates, equal-background differences, CIs, and discordant win:loss counts."""
    out: dict[str, Any] = {
        "n_cases": int(len(cases_df)),
        "hit_rates": {r: float(cases_df[f"hit_{r}_{budget_suffix}"].mean()) for r in RULES},
        "contrasts": {},
    }
    focal = cases_df[f"hit_{focal_rule}_{budget_suffix}"].to_numpy(dtype=float)
    for comp in comparators:
        other = cases_df[f"hit_{comp}_{budget_suffix}"].to_numpy(dtype=float)
        per_bg = {
            str(bg): (g[f"hit_{focal_rule}_{budget_suffix}"] - g[f"hit_{comp}_{budget_suffix}"]).to_numpy(dtype=float)
            for bg, g in cases_df.groupby("background")
        }
        bg_means = {bg: float(np.mean(vals)) for bg, vals in per_bg.items()}
        ci_lo, ci_hi = section_stratified_bootstrap_ci(per_bg, seed=seed)
        out["contrasts"][f"{focal_rule}_minus_{comp}"] = {
            "equal_background_mean": float(np.mean(list(bg_means.values()))),
            "by_background": bg_means,
            "ci95": [ci_lo, ci_hi],
            "discordant_wins_losses": [
                int(np.sum((focal == 1) & (other == 0))),
                int(np.sum((focal == 0) & (other == 1))),
            ],
        }
    return out


def summarize_quota_coupling(cases_df: pd.DataFrame) -> dict[str, Any]:
    """Quantify forced zero-budget ties when continuous scorers match variable M0 counts ``k_m0``."""
    k_zero = cases_df["k_m0"].astype(int) == 0
    hits_fixed = cases_df[[f"hit_{r}_0.03" for r in RULES]].astype(int)
    disagree = hits_fixed.nunique(axis=1) > 1
    any_hit = hits_fixed.max(axis=1) == 1
    return {
        "n_cases": int(len(cases_df)),
        "zero_budget_count": int(k_zero.sum()),
        "zero_budget_fraction": float(k_zero.mean()),
        "latent_any_rule_hits_in_zero": int((k_zero & any_hit).sum()),
        "latent_rule_hits_in_zero": {r: int((k_zero & (hits_fixed[f"hit_{r}_0.03"] == 1)).sum()) for r in RULES},
        "rules_disagree_in_zero": int((k_zero & disagree).sum()),
    }


def compare_counterfactual_arms(records_df: pd.DataFrame) -> dict[str, Any]:
    """Compute paired native vs disable_damage and relabel transition metrics across S6 and S8."""
    out: dict[str, Any] = {}
    for stratum in sorted(records_df["stratum"].unique().tolist()):
        sub = records_df[records_df["stratum"] == stratum]
        out[f"S{int(stratum)}"] = {
            "n": int(len(sub)),
            "native_accept_rate": float((sub["native_status"] == "ACCEPT").mean()),
            "disable_damage_accept_rate": float((sub["no_damage_status"] == "ACCEPT").mean()),
            "relabel_accept_rate": float((sub["relabel_status"] == "ACCEPT").mean()),
        }
    return out
