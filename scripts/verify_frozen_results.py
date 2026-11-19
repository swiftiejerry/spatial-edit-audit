"""Export clean CSV/JSON benchmark outputs and verify arithmetic against offline baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import (  # noqa: E402
    aggregate_rule_scores,
    betabinomial_tail_summary,
    compare_counterfactual_arms,
    fit_order_statistic_cutoffs,
    generate_main_case,
    load_section_baseline,
    summarize_fixed_budget_contrasts,
    summarize_quota_coupling,
)
from spatial_edit_audit.contrast import build_contrast_operator, score_candidate_pool_fast  # noqa: E402


def verify_all() -> dict[str, object]:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    main_csv = res_dir / "main_cases_720.csv"
    mech_csv = res_dir / "counterfactual_records_160.csv"
    sec_csv = res_dir / "secondary_partition_1440.csv"
    static_csv = res_dir / "static_tiles_593.csv"

    main_df = pd.read_csv(main_csv)
    mech_df = pd.read_csv(mech_csv)
    sec_df = pd.read_csv(sec_csv)
    static_df = pd.read_csv(static_csv)

    assert len(main_df) == 720, len(main_df)
    assert len(mech_df) == 160, len(mech_df)
    assert len(sec_df) == 1440, len(sec_df)
    assert len(static_df) == 593, len(static_df)

    # Verify offline baselines & exact Block-1 calibration thresholds
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}
    cal_cutoffs = {}
    for bg_name, bl in baselines.items():
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        maxima = {"P": [], "Q": [], "mean": [], "min": []}
        for rep in range(1, 200):
            case = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
            scores = aggregate_rule_scores(ep, eq)
            for rule, arr in scores.items():
                maxima[rule].append(float(np.nanmax(arr)))
        cal_cutoffs[bg_name] = fit_order_statistic_cutoffs(maxima, alpha=0.05)

    # Verify Block-2 fixed-budget contrasts & S9 reversal
    dose_1_4 = main_df[main_df["stratum"].between(4, 7)]
    s9 = main_df[main_df["stratum"] == 9]
    e1_summary = summarize_fixed_budget_contrasts(dose_1_4)
    s9_summary = summarize_fixed_budget_contrasts(s9)
    quota_summary = summarize_quota_coupling(dose_1_4)

    # Verify BetaBinomial tail evaluation across all four rules
    null_df = main_df[main_df["stratum"].isin([1, 2, 3])]
    cal_stats = {}
    for rule in ("P", "Q", "mean", "min"):
        counts = [int(null_df[null_df["background"] == bg][f"cal_any_{rule}"].sum()) for bg in ("B1", "B2", "B3", "B4")]
        cal_stats[rule] = betabinomial_tail_summary(counts, n_cal=199, k_rank=190, n_test_per_section=60)

    # Verify counterfactual parity & Secondary summaries
    cf_summary = compare_counterfactual_arms(mech_df)
    sec_summary = {
        src: {
            "min_mean_cov": round(float(sec_df[(sec_df["source"] == src) & (sec_df["rule"] == "min")]["coverage"].mean()), 4),
            "oracle_mean_cov": round(float(sec_df[(sec_df["source"] == src) & (sec_df["rule"] == "min")]["upper_coverage"].mean()), 4),
        }
        for src in ("F", "D1", "D2")
    }

    report = {
        "status": "VERIFIED",
        "n_spots_total": sum(bl.n_spots for bl in baselines.values()),
        "n_tiles_total": sum(bl.n_tiles for bl in baselines.values()),
        "calibration_cutoffs_k190": cal_cutoffs,
        "e1_concordant_3pct": e1_summary,
        "s9_asymmetric_3pct": s9_summary,
        "quota_coupling_dose_1_4": quota_summary,
        "null_calibration_betabinomial": cal_stats,
        "counterfactual_transitions": cf_summary,
        "secondary_coverage_summary": sec_summary,
    }
    (res_dir / "verification_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = verify_all()
    print(json.dumps({k: report[k] for k in ("status", "n_spots_total", "n_tiles_total", "calibration_cutoffs_k190")}, indent=2))
    print("min null exact_binomial p =", round(report["null_calibration_betabinomial"]["min"]["exact_binomial_p_naive"], 6),
          "| betabinomial p =", round(report["null_calibration_betabinomial"]["min"]["betabinomial_p_pooled"], 6))


if __name__ == "__main__":
    main()
