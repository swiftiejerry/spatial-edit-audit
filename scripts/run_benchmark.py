"""End-to-end deterministic simulation benchmark runner over offline reference bundles."""

from __future__ import annotations

import argparse
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
)
from spatial_edit_audit.contrast import RULES, budget_tile_count, build_contrast_operator, rank_scorable_candidates, score_candidate_pool_fast  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Block 1 calibration and Block 2 main evaluations offline.")
    parser.add_argument("--sections", nargs="+", default=["B1", "B2", "B3", "B4"])
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save per-case CSV outputs.")
    args = parser.parse_args()

    ref_dir = ROOT / "data" / "reference"
    rows = []
    for bg_name in args.sections:
        bl = load_section_baseline(ref_dir / f"{bg_name}_baseline.npz")
        cands = [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)]
        op, scorable = build_contrast_operator(cands, bl.graph, bl.n_spots)
        c_ids = bl.tile_table["tile_id"].astype(str).tolist()
        m_3pct = budget_tile_count(0.03, bl.n_tiles)

        # Block 1 calibration
        maxima = {r: [] for r in RULES}
        for rep in range(1, 200):
            case = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
            sc = aggregate_rule_scores(ep, eq)
            for r in RULES:
                maxima[r].append(float(np.nanmax(sc[r])))
        cutoffs = fit_order_statistic_cutoffs(maxima, alpha=0.05)

        # Block 2 main evaluation
        for stratum in range(1, 10):
            for rep in range(1, 21):
                case = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
                sc_df = pd.DataFrame(
                    {
                        "candidate_id": c_ids,
                        "scorable": scorable,
                        "P": ep,
                        "Q": eq,
                        "mean": 0.5 * (ep + eq),
                        "min": np.minimum(ep, eq),
                    }
                )
                t_id = case["truth"]["positions"].get("T")
                rec = {"background": bg_name, "stratum": stratum, "rep": rep}
                for r in RULES:
                    top_m = rank_scorable_candidates(sc_df, r)[:m_3pct]
                    rec[f"hit_{r}_0.03"] = int(t_id in top_m) if t_id else None
                    rec[f"cal_any_{r}"] = int(float(np.nanmax(sc_df[r])) >= cutoffs[r])
                rows.append(rec)

    df = pd.DataFrame(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8")

    dose = df[df["stratum"].between(4, 7)]
    null_df = df[df["stratum"].isin([1, 2, 3])]
    summary = {
        "n_cases": int(len(df)),
        "concordant_3pct_hit_rates": {r: round(float(dose[f"hit_{r}_0.03"].mean()), 4) for r in RULES},
        "s9_3pct_hit_rates": {r: round(float(df[df["stratum"] == 9][f"hit_{r}_0.03"].mean()), 4) for r in RULES},
        "null_false_accepts_of_240": {r: int(null_df[f"cal_any_{r}"].sum()) for r in RULES},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
