"""Evaluate Secondary grown-patch partitioners (F, D1 raw, D1 size-filtered, D2) at dose=6.0 and dose=3.0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import (  # noqa: E402
    build_grown_patch_candidates,
    evaluate_partition_coverage,
    generate_secondary_case,
    load_section_baseline,
)
from spatial_edit_audit.contrast import RULES, build_contrast_operator, score_candidate_pool_fast  # noqa: E402
import pandas as pd  # noqa: E402


def fast_score_df(op, scorable, candidates, x_p, x_q):
    ep, eq = score_candidate_pool_fast(op, scorable, x_p, x_q)
    return pd.DataFrame(
        {
            "candidate_id": [c[0] for c in candidates],
            "n_points": [len(c[1]) for c in candidates],
            "e_P": ep,
            "e_Q": eq,
            "scorable": scorable,
            "P": ep,
            "Q": eq,
            "mean": 0.5 * (ep + eq),
            "min": np.minimum(ep, eq),
        }
    )


def main() -> None:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    summary: dict[str, object] = {}
    for dose_label, dose_val in (("dose_6.0", 6.0), ("dose_3.0", 3.0)):
        dose_out: dict[str, object] = {}
        for bg_name, bl in baselines.items():
            srcs = ("F", "D1", "D1_filtered", "D2")
            bg_rows: dict[str, list[float]] = {k: [] for k in srcs}
            bg_upper: dict[str, list[float]] = {k: [] for k in srcs}
            bg_out: dict[str, list[int]] = {k: [] for k in srcs}
            bg_excl: dict[str, list[int]] = {k: [] for k in srcs}
            bg_sel_pts: dict[str, list[int]] = {k: [] for k in srcs}
            bg_max_sz: dict[str, list[int]] = {k: [] for k in srcs}
            for rep in range(1, 31):
                case = generate_secondary_case(bl, rep=rep, block=3, dose=dose_val)
                pools = build_grown_patch_candidates(bl, case["X_P"], include_filtered_ward=True)
                for src, cands in pools.items():
                    op, scorable = build_contrast_operator(cands, bl.graph, bl.n_spots)
                    sc_df = fast_score_df(op, scorable, cands, case["X_P"], case["X_Q"])
                    covs = {r["rule"]: r for r in evaluate_partition_coverage(cands, sc_df, case["truth"])}
                    covered_union = set().union(*(set(int(v) for v in m) for _, m in cands)) if cands else set()
                    bg_rows[src].append(float(covs["min"]["target_coverage"]))
                    bg_upper[src].append(float(covs["min"]["oracle_upper_coverage"]))
                    bg_out[src].append(int(covs["min"]["outside_points"]))
                    bg_excl[src].append(int(bl.n_spots - len(covered_union)))
                    bg_sel_pts[src].append(int(covs["min"]["selected_point_count"]))
                    bg_max_sz[src].append(int(max((len(m) for _, m in cands), default=0)))
            dose_out[bg_name] = {
                src: {
                    "min_mean_coverage": round(float(np.mean(bg_rows[src])), 4),
                    "oracle_upper_coverage": round(float(np.mean(bg_upper[src])), 4),
                    "mean_outside_points": round(float(np.mean(bg_out[src])), 2),
                    "mean_excluded_spots": round(float(np.mean(bg_excl[src])), 2),
                    "excluded_spots_min": int(min(bg_excl[src])),
                    "excluded_spots_max": int(max(bg_excl[src])),
                    "mean_excluded_pct": round(float(np.mean(bg_excl[src]) / bl.n_spots * 100.0), 2),
                    "mean_selected_points": round(float(np.mean(bg_sel_pts[src])), 2),
                    "mean_max_candidate_size": round(float(np.mean(bg_max_sz[src])), 2),
                }
                for src in srcs
            }
        pooled = {
            src: round(float(np.mean([dose_out[bg][src]["min_mean_coverage"] for bg in baselines])), 4)
            for src in ("F", "D1", "D1_filtered", "D2")
        }
        summary[dose_label] = {"by_section": dose_out, "pooled_mean_coverage": pooled}

    (res_dir / "partition_ablation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["pooled_mean_coverage"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
