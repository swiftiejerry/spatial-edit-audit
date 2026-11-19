"""Run calibrated dual-channel selective gates, graph smoothing, and neighbor-augmented clustering."""

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
    evaluate_dual_calibrated_gate,
    fit_order_statistic_cutoffs,
    generate_main_case,
    load_section_baseline,
)
from spatial_edit_audit.baselines import evaluate_neighbor_augmented_clustering, score_smoothed_views  # noqa: E402
from spatial_edit_audit.contrast import build_contrast_operator, score_candidate_pool_fast  # noqa: E402


def main() -> None:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    ops = {}
    cal_cutoffs = {}
    for bg_name, bl in baselines.items():
        cands = [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)]
        op, scorable = build_contrast_operator(cands, bl.graph, bl.n_spots)
        ops[bg_name] = (op, scorable)
        mx = []
        for rep in range(1, 200):
            c = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
            mx.append(float(np.nanmax(np.minimum(ep, eq))))
        cal_cutoffs[bg_name] = fit_order_statistic_cutoffs({"min": mx}, alpha=0.05)["min"]

    # 1. Calibrated dual-channel comparator across all 720 Block-2 cases (5% and 15% spot caps)
    dual_results = {}
    case_cache = []
    for bg_name, bl in baselines.items():
        op, scorable = ops[bg_name]
        c_ids = bl.tile_table["tile_id"].astype(str).tolist()
        c_pts = bl.tile_table["n_points"].astype(int).tolist()
        for stratum in range(1, 10):
            for rep in range(1, 21):
                c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
                sc_df = pd.DataFrame(
                    {
                        "candidate_id": c_ids,
                        "n_points": c_pts,
                        "e_P": ep,
                        "e_Q": eq,
                        "scorable": scorable,
                        "min": np.minimum(ep, eq),
                    }
                )
                case_cache.append((bg_name, bl.n_spots, stratum, c["truth"]["positions"].get("T"), sc_df))

    for cap in (0.05, 0.15):
        recs = []
        for bg_name, n_spots, stratum, tgt, sc_df in case_cache:
            kept = evaluate_dual_calibrated_gate(
                sc_df, None, cal_cutoffs[bg_name], n_spots, spot_cap_fraction=cap
            )
            recs.append(
                {
                    "background": bg_name,
                    "stratum": stratum,
                    "k_cal": len(kept),
                    "any_accept": int(len(kept) > 0),
                    "target_accept": int(tgt in kept) if tgt else 0,
                }
            )
        rdf = pd.DataFrame(recs)
        dose = rdf[rdf["stratum"].between(4, 7)]
        dual_results[f"cap_{int(cap * 100)}pct"] = {
            "null_false_accepts_of_240": int(rdf[rdf["stratum"].isin([1, 2, 3])]["any_accept"].sum()),
            "b2_null_of_60": int(rdf[(rdf["background"] == "B2") & (rdf["stratum"].isin([1, 2, 3]))]["any_accept"].sum()),
            "dose_1_4_targets_of_320": int(dose["target_accept"].sum()),
            "s6_s7_targets_of_160": int(rdf[rdf["stratum"].isin([6, 7])]["target_accept"].sum()),
            "s8_cross_boundary_of_80": int(rdf[rdf["stratum"] == 8]["target_accept"].sum()),
            "s9_asymmetric_of_80": int(rdf[rdf["stratum"] == 9]["target_accept"].sum()),
            "zero_budget_ties_dose_1_4": int((dose["k_cal"] == 0).sum()),
        }

    # 2. Linear spatial graph diffusion smoothing (alpha in 0.0, 0.3, 0.5, 0.7)
    smoothing_out = {}
    for alpha in (0.0, 0.3, 0.5, 0.7):
        s6_hits, s8_hits, s6_med, s8_med = [], [], [], []
        for bg_name, bl in baselines.items():
            for stratum, h_bucket, m_bucket in ((6, s6_hits, s6_med), (8, s8_hits, s8_med)):
                for rep in range(1, 21):
                    c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                    res = score_smoothed_views(bl, c["X_P"], c["X_Q"], alpha=alpha)
                    t_idx = c["positions"]["T"]
                    h_bucket.append(int(t_idx in res["selected_indices"]))
                    m_bucket.append(float(res["e_min"][t_idx]))
        smoothing_out[f"alpha_{alpha:.1f}"] = {
            "s6_hit_rate": round(float(np.mean(s6_hits)), 4),
            "s8_hit_rate": round(float(np.mean(s8_hits)), 4),
            "s6_median_min": round(float(np.nanmedian(s6_med)), 4),
            "s8_median_min": round(float(np.nanmedian(s8_med)), 4),
        }

    # 3. BANKSY-style neighbor-augmented KMeans clustering (lambda in 0.0, 0.3, 0.8)
    banksy_out = {}
    for lam in (0.0, 0.3, 0.8):
        f1s = {6: [], 8: [], 9: []}
        for bg_name, bl in baselines.items():
            for stratum in (6, 8, 9):
                for rep in range(1, 6):
                    c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                    f1 = evaluate_neighbor_augmented_clustering(
                        bl, c["X_P"], c["X_Q"], bl.members_by_tile[c["positions"]["T"]], neighbor_weight=lam
                    )
                    f1s[stratum].append(f1)
        banksy_out[f"lambda_{lam:.1f}"] = {
            f"S{s}_mean_best_f1": round(float(np.mean(vals)), 4) for s, vals in f1s.items()
        }

    summary = {
        "calibrated_dual_comparator": dual_results,
        "graph_diffusion_smoothing": smoothing_out,
        "neighbor_augmented_clustering": banksy_out,
    }
    (res_dir / "spatial_baselines_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
