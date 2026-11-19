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
        exact_matches_vs_cal_min = 0
        for bg_name, n_spots, stratum, tgt, sc_df in case_cache:
            kept = evaluate_dual_calibrated_gate(
                sc_df, None, cal_cutoffs[bg_name], n_spots, spot_cap_fraction=cap
            )
            # Compare against unconstrained cal_min (min >= q_{b,min} across scorable tiles)
            cal_min_set = set(
                sc_df[sc_df["scorable"].astype(bool) & (sc_df["min"] >= cal_cutoffs[bg_name])]["candidate_id"].astype(str)
            )
            exact_matches_vs_cal_min += int(set(kept) == cal_min_set)
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
        zero_by_stratum = {
            f"S{s}": int(((rdf["stratum"] == s) & (rdf["k_cal"] == 0)).sum())
            for s in (4, 5, 6, 7)
        }
        dual_results[f"cap_{int(cap * 100)}pct"] = {
            "null_false_accepts_of_240": int(rdf[rdf["stratum"].isin([1, 2, 3])]["any_accept"].sum()),
            "b2_null_of_60": int(rdf[(rdf["background"] == "B2") & (rdf["stratum"].isin([1, 2, 3]))]["any_accept"].sum()),
            "dose_1_4_targets_of_320": int(dose["target_accept"].sum()),
            "s6_s7_targets_of_160": int(rdf[rdf["stratum"].isin([6, 7])]["target_accept"].sum()),
            "s8_cross_boundary_of_80": int(rdf[rdf["stratum"] == 8]["target_accept"].sum()),
            "s9_asymmetric_of_80": int(rdf[rdf["stratum"] == 9]["target_accept"].sum()),
            "zero_budget_ties_dose_1_4": int((dose["k_cal"] == 0).sum()),
            "zero_budget_ties_by_stratum": zero_by_stratum,
            "identical_decisions_to_cal_min_of_720": exact_matches_vs_cal_min,
        }

    # Check mean / P / Q / min N2 decoy capture under unthresholded top-3% budget
    n2_df_rates = {}
    main_csv = res_dir / "main_cases_720.csv"
    if main_csv.exists():
        m_df = pd.read_csv(main_csv)
        n2_sub = m_df[m_df["stratum"] == 2]
        for r in ("P", "Q", "mean", "min"):
            a_hits = (n2_sub[f"A_{r}_0.03"] > 0).sum()
            b_hits = (n2_sub[f"B_{r}_0.03"] > 0).sum()
            any_hits = ((n2_sub[f"A_{r}_0.03"] > 0) | (n2_sub[f"B_{r}_0.03"] > 0)).sum()
            n2_df_rates[r] = {
                "decoy_a_of_80": int(a_hits),
                "decoy_b_of_80": int(b_hits),
                "any_decoy_of_80": int(any_hits),
                "any_decoy_rate": round(float(any_hits / len(n2_sub)), 4),
            }

    # 2. Linear spatial graph diffusion smoothing (alpha in 0.0, 0.3, 0.5, 0.7) across N=80/stratum (S6, S8, S9, N2)
    smoothing_out = {}
    for alpha in (0.0, 0.3, 0.5, 0.7):
        s6_hits, s8_hits, s9_hits, s6_med, s8_med, s9_med = [], [], [], [], [], []
        n2_a, n2_b, n2_any = [], [], []
        for bg_name, bl in baselines.items():
            for stratum, h_bucket, m_bucket in ((6, s6_hits, s6_med), (8, s8_hits, s8_med), (9, s9_hits, s9_med)):
                for rep in range(1, 21):
                    c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                    res = score_smoothed_views(bl, c["X_P"], c["X_Q"], alpha=alpha)
                    t_idx = c["positions"]["T"]
                    h_bucket.append(int(t_idx in res["selected_indices"]))
                    m_bucket.append(float(res["e_min"][t_idx]))
            for rep in range(1, 21):
                c = generate_main_case(bl, block=2, stratum=2, rep=rep, fit_i_case=False)
                res = score_smoothed_views(bl, c["X_P"], c["X_Q"], alpha=alpha)
                chosen = set(res["selected_indices"])
                da = int(c["positions"]["A"] in chosen)
                db = int(c["positions"]["B"] in chosen)
                n2_a.append(da)
                n2_b.append(db)
                n2_any.append(int(da or db))
        smoothing_out[f"alpha_{alpha:.1f}"] = {
            "n_per_stratum": 80,
            "s6_hit_rate": round(float(np.mean(s6_hits)), 4),
            "s8_hit_rate": round(float(np.mean(s8_hits)), 4),
            "s9_hit_rate": round(float(np.mean(s9_hits)), 4),
            "n2_decoy_a_rate": round(float(np.mean(n2_a)), 4),
            "n2_decoy_b_rate": round(float(np.mean(n2_b)), 4),
            "n2_any_decoy_rate": round(float(np.mean(n2_any)), 4),
            "s6_median_min": round(float(np.nanmedian(s6_med)), 4),
            "s8_median_min": round(float(np.nanmedian(s8_med)), 4),
            "s9_median_min": round(float(np.nanmedian(s9_med)), 4),
        }

    # 3. BANKSY-style neighbor-augmented KMeans clustering (lambda in 0.0, 0.3, 0.8), N=80/stratum (20 reps x 4 sections)
    # Compare coarse k=12 vs matched tile-granularity k=round(n_b/20) (=bl.n_tiles)
    banksy_out = {}
    for k_mode, k_arg in (("k_12", 12), ("k_matched_tiles", None)):
        mode_out = {}
        for lam in (0.0, 0.3, 0.8):
            f1s = {6: [], 8: [], 9: []}
            n2_f1_a, n2_f1_b = [], []
            for bg_name, bl in baselines.items():
                for stratum in (6, 8, 9):
                    for rep in range(1, 21):
                        c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                        f1 = evaluate_neighbor_augmented_clustering(
                            bl,
                            c["X_P"],
                            c["X_Q"],
                            bl.members_by_tile[c["positions"]["T"]],
                            neighbor_weight=lam,
                            n_clusters=k_arg,
                        )
                        f1s[stratum].append(f1)
                for rep in range(1, 21):
                    c = generate_main_case(bl, block=2, stratum=2, rep=rep, fit_i_case=False)
                    n2_f1_a.append(
                        evaluate_neighbor_augmented_clustering(
                            bl, c["X_P"], c["X_Q"], bl.members_by_tile[c["positions"]["A"]], neighbor_weight=lam, n_clusters=k_arg
                        )
                    )
                    n2_f1_b.append(
                        evaluate_neighbor_augmented_clustering(
                            bl, c["X_P"], c["X_Q"], bl.members_by_tile[c["positions"]["B"]], neighbor_weight=lam, n_clusters=k_arg
                        )
                    )
            mode_out[f"lambda_{lam:.1f}"] = {
                "n_per_stratum": 80,
                **{f"S{s}_mean_best_f1": round(float(np.mean(vals)), 4) for s, vals in f1s.items()},
                "N2_decoyA_mean_best_f1": round(float(np.mean(n2_f1_a)), 4),
                "N2_decoyB_mean_best_f1": round(float(np.mean(n2_f1_b)), 4),
            }
        banksy_out[k_mode] = mode_out

    summary = {
        "calibrated_dual_comparator": dual_results,
        "unthresholded_n2_decoy_rates": n2_df_rates,
        "graph_diffusion_smoothing": smoothing_out,
        "neighbor_augmented_clustering": banksy_out,
    }
    (res_dir / "spatial_baselines_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
