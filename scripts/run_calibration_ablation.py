"""Verify split-conformal calibration sensitivity under balanced 1:1:1 sampling, Mondrian cuts, and larger calibration pools."""

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
    fit_order_statistic_cutoffs,
    generate_main_case,
    load_section_baseline,
    mondrian_stratum_cutoffs,
)
from spatial_edit_audit.contrast import RULES, build_contrast_operator, score_candidate_pool_fast  # noqa: E402
from spatial_edit_audit.simulation import stream_rng  # noqa: E402


def main() -> None:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    main_df = pd.read_csv(res_dir / "main_cases_720.csv")
    null_df = main_df[main_df["stratum"].isin([1, 2, 3])].copy()

    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    # Reconstruct exact Block-2 test null maximum scores (60 per slice) directly from deterministic streams
    b2_test_null = {}
    cal1_strata_and_max = {}
    for bg_name, bl in baselines.items():
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        test_rows = []
        sub = null_df[null_df["background"] == bg_name]
        # Regenerate exact Block-2 null cases across N1..N3 (20 each)
        for s in (1, 2, 3):
            for rep in range(1, 21):
                case = generate_main_case(bl, block=2, stratum=s, rep=rep, fit_i_case=False)
                ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
                sc = aggregate_rule_scores(ep, eq)
                test_rows.append({"stratum": s, **{r: float(np.nanmax(sc[r])) for r in RULES}})
        b2_test_null[bg_name] = pd.DataFrame(test_rows)

        # Primary Block-1 random N1-N3 stratum breakdown
        cal_rows = []
        for rep in range(1, 200):
            rng = stream_rng(1, bl.section_index, 0, rep)
            actual_s = int(rng.integers(1, 4))
            case = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
            sc = aggregate_rule_scores(ep, eq)
            cal_rows.append({"stratum": actual_s, **{r: float(np.nanmax(sc[r])) for r in RULES}})
        cal1_strata_and_max[bg_name] = pd.DataFrame(cal_rows)

    out = {"primary_block1_stratum_stats": {}, "mondrian_on_block1": {}, "balanced_ablation": {}}

    # 1. Inspect primary Block-1 stratum distributions + Mondrian conditional cutoffs
    mondrian_tot = 0
    for bg_name, cdf in cal1_strata_and_max.items():
        q_raw = fit_order_statistic_cutoffs({r: cdf[r].to_numpy() for r in RULES}, alpha=0.05)
        q_mon = mondrian_stratum_cutoffs(cdf["min"].to_numpy(), cdf["stratum"].to_numpy(), alpha=0.05)
        tdf = b2_test_null[bg_name]
        mon_hits = int(sum(row["min"] >= q_mon[int(row["stratum"])] for _, row in tdf.iterrows()))
        mondrian_tot += mon_hits
        out["primary_block1_stratum_stats"][bg_name] = {
            f"N{s}": {
                "n_cal": int((cdf["stratum"] == s).sum()),
                "cal_min_median": float(cdf[cdf["stratum"] == s]["min"].median()),
                "cal_min_p95": float(np.quantile(cdf[cdf["stratum"] == s]["min"], 0.95)),
                "cal_min_max": float(cdf[cdf["stratum"] == s]["min"].max()),
                "test_min_hits": int((tdf[tdf["stratum"] == s]["min"] >= q_raw["min"]).sum()),
            }
            for s in (1, 2, 3)
        }
        out["mondrian_on_block1"][bg_name] = {"cutoffs": q_mon, "hits_of_60": mon_hits}
    out["mondrian_pooled_hits_of_240"] = mondrian_tot

    # 2. Balanced 1:1:1 calibration sensitivity (n_cal = 198 and n_cal = 999)
    for n_per_s in (66, 333):
        n_tot = 3 * n_per_s
        key = f"balanced_{n_tot}"
        sec_res = {}
        pooled = {r: 0 for r in RULES}
        for bg_name, bl in baselines.items():
            op, scorable = build_contrast_operator(
                [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
            )
            mx = {r: [] for r in RULES}
            for s in (1, 2, 3):
                for rep in range(1, n_per_s + 1):
                    case = generate_main_case(bl, block=100, stratum=s, rep=rep, fit_i_case=False)
                    ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
                    sc = aggregate_rule_scores(ep, eq)
                    for r in RULES:
                        mx[r].append(float(np.nanmax(sc[r])))
            cuts = fit_order_statistic_cutoffs(mx, alpha=0.05)
            tdf = b2_test_null[bg_name]
            fa = {r: int((tdf[r] >= cuts[r]).sum()) for r in RULES}
            for r in RULES:
                pooled[r] += fa[r]
            sec_res[bg_name] = {"cutoffs": cuts, "false_accepts_of_60": fa}
        out["balanced_ablation"][key] = {
            "n_cal": n_tot,
            "by_section": sec_res,
            "pooled_of_240": pooled,
            "pooled_rates": {r: round(v / 240.0, 4) for r, v in pooled.items()},
        }

    (res_dir / "calibration_ablation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mondrian_240": mondrian_tot,
        "balanced_198_rates": out["balanced_ablation"]["balanced_198"]["pooled_rates"],
        "balanced_999_rates": out["balanced_ablation"]["balanced_999"]["pooled_rates"],
    }, indent=2))


if __name__ == "__main__":
    main()
