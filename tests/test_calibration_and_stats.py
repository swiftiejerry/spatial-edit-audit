"""Unit tests for order-statistic calibration, Mondrian cutoffs, and Beta-binomial inference."""

import numpy as np

from spatial_edit_audit.calibration import (
    betabinomial_tail_summary,
    fit_order_statistic_cutoffs,
    mondrian_stratum_cutoffs,
)


def test_order_statistic_index_matches_k190_of_199():
    arr = np.arange(1.0, 200.0)
    cutoffs = fit_order_statistic_cutoffs({"min": arr}, alpha=0.05)
    # ceil((199 + 1) * 0.95) = 190 -> 1-based rank 190 is 190.0
    assert cutoffs["min"] == 190.0


def test_mondrian_stratum_cutoffs_separate_strata():
    maxima = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    strata = np.array([1, 1, 1, 2, 2, 2])
    cuts = mondrian_stratum_cutoffs(maxima, strata, alpha=0.25)
    assert set(cuts.keys()) == {1, 2}
    assert cuts[1] <= 3.0 and cuts[2] >= 10.0


def test_betabinomial_four_section_tail_matches_exact_values():
    summary = betabinomial_tail_summary([2, 3, 4, 10], n_cal=199, k_rank=190, n_test_per_section=60)
    assert summary["beta_params"] == (10, 190)
    assert summary["pooled_count"] == 19
    assert abs(summary["exact_binomial_p_naive"] - 0.03357104) < 1e-6
    assert abs(summary["betabinomial_p_pooled"] - 0.05429884) < 1e-6
    assert abs(summary["section_p_values"][3] - 0.00357081) < 1e-6
    assert summary["section_95pct_band"] == (0, 7)
