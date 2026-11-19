"""Deterministically regenerate all frozen CSV result tables directly from offline reference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import (  # noqa: E402
    GateConfig,
    aggregate_rule_scores,
    apply_reference_intervention,
    build_grown_patch_candidates,
    evaluate_multigate_prototype,
    evaluate_partition_coverage,
    fit_order_statistic_cutoffs,
    generate_main_case,
    generate_secondary_case,
    load_section_baseline,
)
from scipy.sparse.csgraph import connected_components  # noqa: E402
from spatial_edit_audit.contrast import (  # noqa: E402
    RULES,
    budget_tile_count,
    build_contrast_operator,
    local_molecular_contrast,
    rank_scorable_candidates,
    robust_feature_scale,
    score_candidate_pool_fast,
)
from spatial_edit_audit.selective_gates import empirical_bounds  # noqa: E402
from spatial_edit_audit.spatial_graph import compute_two_hop_ring  # noqa: E402


def _m0_any_fast(bl, x_q: np.ndarray, i_case: np.ndarray) -> int:
    """Exact M0 decision screen filtering necessary non-bootstrap gates before empirical_bounds."""
    cfg = GateConfig(disable_damage_gate=False)
    scale_info = robust_feature_scale(x_q)
    accepted_spots = 0
    n_obs = bl.n_spots
    for tid, members in enumerate(bl.members_by_tile):
        if len(members) / n_obs > cfg.rare_max_fraction:
            continue
        mask = np.zeros(n_obs, dtype=bool)
        mask[members] = True
        n_comp, comp_labels = connected_components(bl.graph[mask][:, mask], directed=False)
        for comp_id in range(n_comp):
            comp = members[comp_labels == comp_id]
            if len(comp) < cfg.min_component_size or len(comp) / n_obs > cfg.rare_max_fraction:
                continue
            comp_set = set(comp.tolist())
            nbrs = bl.neighbors[comp]
            coherence = sum(int(j in comp_set) for row in nbrs for j in row) / int(nbrs.size)
            if coherence < cfg.min_spatial_coherence or (1.0 - coherence) > cfg.max_boundary_leakage:
                continue
            _, icnt = np.unique(i_case[comp], return_counts=True)
            if float(icnt.max() / len(comp)) < cfg.min_independent_agreement:
                continue
            loc = local_molecular_contrast(comp, bl.graph, x_q, precomputed_scale=scale_info)
            if loc["status"] != "ACCEPT":
                continue
            controls = compute_two_hop_ring(comp, bl.graph, hops=2)
            bounds = empirical_bounds(comp, controls, x_q, bl.r_static, bl.neighbors, precomputed_scale=scale_info)
            if bounds["gain_lower_bound"] >= cfg.min_gain_lower_bound and bounds["damage_upper_bound"] <= cfg.max_damage_upper_bound:
                accepted_spots += len(comp)
                if accepted_spots / n_obs > cfg.max_total_edit_fraction:
                    return 0
    return int(accepted_spots > 0)


def _static_table(baselines: dict) -> pd.DataFrame:
    cfg = GateConfig()
    rows = []
    for bg_name, bl in baselines.items():
        zero_x = np.zeros((bl.n_spots, 1), dtype=np.float64)
        for tid, members in enumerate(bl.members_by_tile):
            cid = f"T{tid:04d}"
            m_set = set(members.tolist())
            nbrs = bl.neighbors[members]
            internal = sum(int(j in m_set) for row in nbrs for j in row)
            total = int(nbrs.size)
            coherence = internal / total if total else 0.0
            leakage = (total - internal) / total if total else 1.0
            _, i_cnt = np.unique(bl.i_background[members], return_counts=True)
            agree = float(i_cnt.max() / len(members))

            controls = bl.rings_by_tile[tid]
            bounds_nat = empirical_bounds(
                members, controls, zero_x, bl.r_static, bl.neighbors,
                n_boot=cfg.n_bound_boot, confidence=cfg.bound_confidence,
            )
            hom_r = apply_reference_intervention(bl, cid, "homogeneous")
            bounds_hom = empirical_bounds(
                members, controls, zero_x, hom_r, bl.neighbors,
                n_boot=cfg.n_bound_boot, confidence=cfg.bound_confidence,
            )
            dep = float(bl.tile_table.iloc[tid]["dep"])
            cat = str(bl.tile_table.iloc[tid]["category"])
            three_way = np.nan
            if cat == "LOW":
                tw_r = apply_reference_intervention(bl, cid, "three_way_split")
                b_tw = empirical_bounds(
                    members, controls, zero_x, tw_r, bl.neighbors,
                    n_boot=cfg.n_bound_boot, confidence=cfg.bound_confidence,
                )
                three_way = float(b_tw["damage_upper_bound"])

            size_ok = len(members) >= cfg.min_component_size
            area_ok = len(members) / bl.n_spots <= cfg.rare_max_fraction
            coh_ok = coherence >= cfg.min_spatial_coherence
            leak_ok = leakage <= cfg.max_boundary_leakage
            agree_ok = agree >= cfg.min_independent_agreement
            dmg_ok = bounds_nat["damage_upper_bound"] <= cfg.max_damage_upper_bound
            other_ok = size_ok and area_ok and coh_ok and leak_ok and agree_ok
            rows.append(
                {
                    "background": bg_name,
                    "tile": cid,
                    "n_points": int(len(members)),
                    "dep": dep,
                    "category": cat,
                    "spatial_coherence": coherence,
                    "boundary_leakage": leakage,
                    "independent_agreement": agree,
                    "damage_ucb_native": float(bounds_nat["damage_upper_bound"]),
                    "size_ok": bool(size_ok),
                    "area_ok": bool(area_ok),
                    "coherence_ok": bool(coh_ok),
                    "leakage_ok": bool(leak_ok),
                    "agree_ok": bool(agree_ok),
                    "damage_ok_native": bool(dmg_ok),
                    "damage_ucb_homogeneous": float(bounds_hom["damage_upper_bound"]),
                    "damage_ucb_three_way": three_way,
                    "static_other_ok": bool(other_ok),
                    "static_all_ok": bool(other_ok and dmg_ok),
                }
            )
    return pd.DataFrame(rows)


def _secondary_table(baselines: dict) -> pd.DataFrame:
    rows = []
    for bg_name, bl in baselines.items():
        for rep in range(1, 31):
            case_id = f"S{rep:04d}"
            sec = generate_secondary_case(bl, rep=rep, block=3, dose=6.0)
            pools = build_grown_patch_candidates(bl, sec["X_P"], include_filtered_ward=False)
            for src in ("F", "D1", "D2"):
                cands = pools[src]
                op, scorable = build_contrast_operator(cands, bl.graph, bl.n_spots)
                ep, eq = score_candidate_pool_fast(op, scorable, sec["X_P"], sec["X_Q"])
                sc_df = pd.DataFrame(
                    {
                        "candidate_id": [c[0] for c in cands],
                        "n_points": [len(c[1]) for c in cands],
                        "scorable": scorable,
                        "P": ep,
                        "Q": eq,
                        "mean": 0.5 * (ep + eq),
                        "min": np.minimum(ep, eq),
                    }
                )
                for cov in evaluate_partition_coverage(cands, sc_df, sec["truth"], budget_fraction=0.03):
                    rows.append(
                        {
                            "background": bg_name,
                            "case": case_id,
                            "source": src,
                            "rule": cov["rule"],
                            "m": int(cov["m_requested"]),
                            "n_candidates": int(cov["candidate_count"]),
                            "coverage": float(cov["target_coverage"]),
                            "outside_points": int(cov["outside_points"]),
                            "decoy_A_overlap": int(cov["decoy_A_overlap_points"]),
                            "decoy_B_overlap": int(cov["decoy_B_overlap_points"]),
                            "upper_coverage": float(cov["oracle_upper_coverage"]),
                        }
                    )
    return pd.DataFrame(rows)


def _counterfactual_table(baselines: dict) -> pd.DataFrame:
    cfg_nat = GateConfig(disable_damage_gate=False)
    cfg_no = GateConfig(disable_damage_gate=True)
    rows = []
    for bg_name, bl in baselines.items():
        for stratum, mode in ((6, "three_way_split"), (8, "homogeneous")):
            for rep in range(1, 21):
                c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=True)
                t_id = c["truth"]["positions"]["T"]
                r_nat = evaluate_multigate_prototype(bl, c["X_Q"], c["I_case"], config=cfg_nat)
                r_no = evaluate_multigate_prototype(bl, c["X_Q"], c["I_case"], config=cfg_no)
                r_rel = evaluate_multigate_prototype(
                    bl,
                    c["X_Q"],
                    c["I_case"],
                    reference_labels=apply_reference_intervention(bl, t_id, mode),
                    config=cfg_nat,
                )

                def _status_fail(res: dict) -> tuple[str, str]:
                    comp = next((r for r in res["components"] if r["candidate_id"] == t_id), None)
                    if comp is None:
                        return "REJECT", "missing"
                    return comp["status"], ";".join(sorted(comp["failed"]))

                ns, nf = _status_fail(r_nat)
                os, of = _status_fail(r_no)
                rs, rf = _status_fail(r_rel)
                rows.append(
                    {
                        "background": bg_name,
                        "stratum": stratum,
                        "target": t_id,
                        "native_status": ns,
                        "native_failed": nf,
                        "no_damage_status": os,
                        "no_damage_failed": of,
                        "relabel_status": rs,
                        "relabel_failed": rf,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate and verify frozen benchmark tables from offline NPZ bundles.")
    parser.add_argument("--write-clean-static", action="store_true", help="Rewrite static_tiles_593.csv without vestigial columns.")
    args = parser.parse_args()

    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    # 1. Static tiles table (593 rows)
    print("[1/4] Reproducing static tiles table (593 rows)...", flush=True)
    st_regen = _static_table(baselines)
    if args.write_clean_static:
        st_regen.to_csv(res_dir / "static_tiles_593.csv", index=False, encoding="utf-8")
    st_frozen = pd.read_csv(res_dir / "static_tiles_593.csv")
    pd.testing.assert_frame_equal(st_regen, st_frozen, atol=1e-12)

    # 2. Secondary partition table (1,440 rows)
    print("[2/4] Reproducing secondary partition table (1,440 rows)...", flush=True)
    sec_regen = _secondary_table(baselines)
    sec_frozen = pd.read_csv(res_dir / "secondary_partition_1440.csv")
    pd.testing.assert_frame_equal(sec_regen, sec_frozen, atol=1e-12)

    # 3. Counterfactual records table (160 rows, matched canonically without arbitrary tie order)
    print("[3/4] Reproducing counterfactual records (160 rows)...", flush=True)
    cf_sort_cols = [
        "background",
        "stratum",
        "target",
        "native_status",
        "native_failed",
        "no_damage_status",
        "no_damage_failed",
        "relabel_status",
        "relabel_failed",
    ]
    cf_regen = _counterfactual_table(baselines)
    cf_regen_cmp = cf_regen.fillna("").sort_values(cf_sort_cols).reset_index(drop=True)
    cf_frozen_cmp = (
        pd.read_csv(res_dir / "counterfactual_records_160.csv")
        .drop(columns=["case_id"])
        .fillna("")
        .sort_values(cf_sort_cols)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(cf_regen_cmp, cf_frozen_cmp)

    # 4. Main cases table (720 rows): regenerate 3% & calibrated metrics across all strata and check parity
    print("[4/4] Reproducing main 720 cases...", flush=True)
    main_frozen = pd.read_csv(res_dir / "main_cases_720.csv")
    regen_cal_min = []
    regen_hit3_min = []
    regen_m0_any = []
    for bg_name, bl in baselines.items():
        print(f"      -> section {bg_name}...", flush=True)
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        c_ids = bl.tile_table["tile_id"].astype(str).tolist()
        m3 = budget_tile_count(0.03, bl.n_tiles)
        mx = {"min": []}
        for rep in range(1, 200):
            c1 = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, c1["X_P"], c1["X_Q"])
            mx["min"].append(float(np.nanmax(np.minimum(ep, eq))))
        q_min = fit_order_statistic_cutoffs(mx, alpha=0.05)["min"]
        for s in range(1, 10):
            for rep in range(1, 21):
                c2 = generate_main_case(bl, block=2, stratum=s, rep=rep, fit_i_case=True)
                ep, eq = score_candidate_pool_fast(op, scorable, c2["X_P"], c2["X_Q"])
                sc_df = pd.DataFrame({"candidate_id": c_ids, "scorable": scorable, "min": np.minimum(ep, eq)})
                top3 = rank_scorable_candidates(sc_df, "min")[:m3]
                t_id = c2["truth"]["positions"].get("T")
                regen_cal_min.append(int(float(np.nanmax(sc_df["min"])) >= q_min))
                if t_id:
                    regen_hit3_min.append(int(t_id in top3))
                regen_m0_any.append(_m0_any_fast(bl, c2["X_Q"], c2["I_case"]))

    assert sum(regen_hit3_min) == int(main_frozen["hit_min_0.03"].dropna().sum())
    assert sum(regen_cal_min) == int(main_frozen["cal_any_min"].sum())
    assert sum(regen_m0_any) == int(main_frozen["m0_any"].sum())

    summary = {
        "status": "REPRODUCED_FROM_OFFLINE_NPZ",
        "static_rows_verified": int(len(st_regen)),
        "secondary_rows_verified": int(len(sec_regen)),
        "counterfactual_rows_verified": int(len(cf_regen)),
        "main_720_hit_min_3pct_sum": int(sum(regen_hit3_min)),
        "main_720_cal_any_min_sum": int(sum(regen_cal_min)),
        "main_720_m0_any_sum": int(sum(regen_m0_any)),
    }
    (res_dir / "reproduction_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
