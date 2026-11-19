"""Split-conformal empirical null calibration and slice-clustered Beta-binomial diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import betabinom, binom


def fit_order_statistic_cutoffs(
    maxima_by_rule: dict[str, list[float] | np.ndarray],
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute conformal order-statistic thresholds at rank k = ceil((n_cal + 1) * (1 - alpha))."""
    cutoffs: dict[str, float] = {}
    for rule, vals in maxima_by_rule.items():
        arr = np.sort(np.asarray(vals, dtype=np.float64))
        if arr.size == 0:
            raise ValueError(f"Empty calibration maxima for rule {rule!r}")
        k_rank = min(len(arr), int(math.ceil((len(arr) + 1) * (1.0 - float(alpha)))))
        cutoffs[rule] = float(arr[k_rank - 1])
    return cutoffs


def mondrian_stratum_cutoffs(
    maxima: list[float] | np.ndarray,
    strata: list[int] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[int, float]:
    """Compute stratum-conditional conformal cutoffs separately within each null stratum (N1, N2, N3)."""
    max_arr = np.asarray(maxima, dtype=np.float64)
    st_arr = np.asarray(strata, dtype=int)
    cutoffs: dict[int, float] = {}
    for s in sorted(np.unique(st_arr).tolist()):
        sub = np.sort(max_arr[st_arr == s])
        k_rank = min(len(sub), int(math.ceil((len(sub) + 1) * (1.0 - float(alpha)))))
        cutoffs[int(s)] = float(sub[k_rank - 1])
    return cutoffs


def betabinomial_tail_summary(
    observed_per_section: list[int],
    *,
    n_cal: int = 199,
    k_rank: int = 190,
    n_test_per_section: int = 60,
    alpha_nominal: float = 0.05,
) -> dict[str, Any]:
    """Evaluate pooled and per-section exceedance probabilities under Beta(n+1-k, k)-binomial clustering.

    Under finite split-conformal calibration of ``n_cal`` draws selecting rank ``k_rank``,
    all ``n_test_per_section`` test cases sharing one realized threshold have conditional
    false-acceptance probability theta ~ Beta(a, b) with a = n_cal + 1 - k_rank, b = k_rank.
    """
    a = int(n_cal) + 1 - int(k_rank)
    b = int(k_rank)
    total_obs = int(sum(observed_per_section))
    total_cases = int(len(observed_per_section) * n_test_per_section)

    pmf_section = betabinom.pmf(np.arange(n_test_per_section + 1), n_test_per_section, a, b)
    pmf_pooled = np.array([1.0], dtype=np.float64)
    for _ in observed_per_section:
        pmf_pooled = np.convolve(pmf_pooled, pmf_section)

    cdf_section = np.cumsum(pmf_section)
    return {
        "beta_params": (a, b),
        "observed_per_section": [int(x) for x in observed_per_section],
        "pooled_count": total_obs,
        "pooled_denominator": total_cases,
        "pooled_rate": total_obs / float(total_cases),
        "exact_binomial_p_naive": float(binom.sf(total_obs - 1, total_cases, alpha_nominal)) if total_obs > 0 else 1.0,
        "betabinomial_p_pooled": float(pmf_pooled[total_obs:].sum()),
        "section_p_values": [float(pmf_section[x:].sum()) for x in observed_per_section],
        "section_95pct_band": (int(np.searchsorted(cdf_section, 0.025)), int(np.searchsorted(cdf_section, 0.975))),
        "expected_section_mean": float(n_test_per_section * a / (a + b)),
    }
