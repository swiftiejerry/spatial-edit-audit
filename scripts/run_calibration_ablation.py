"""Verify split-conformal calibration sensitivity: B4 reweighting, Mondrian cuts, and multi-seed Monte Carlo."""

from __future__ import annotations

import json
import math
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


def _weighted_quantile_one_sided(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    sw = weights[order] / float(np.sum(weights))
    cum = np.cumsum(sw)
    idx = int(np.searchsorted(cum, q, side="left"))
    return float(sv[min(idx, len(sv) - 1)])


def main() -> None:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    ops: dict[str, tuple[object, np.ndarray]] = {}
    b2_test_null: dict[str, pd.DataFrame] = {}
    cal1_primary: dict[str, pd.DataFrame] = {}

    for bg_name, bl in baselines.items():
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        ops[bg_name] = (op, scorable)

        # Frozen Block-2 null test realizations (20 per stratum N1..N3)
        test_rows = []
        for s in (1, 2, 3):
            for rep in range(1, 21):
                case = generate_main_case(bl, block=2, stratum=s, rep=rep, fit_i_case=False)
                ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
                sc = aggregate_rule_scores(ep, eq)
                test_rows.append({"stratum": s, **{r: float(np.nanmax(sc[r])) for r in RULES}})
        b2_test_null[bg_name] = pd.DataFrame(test_rows)

        # Primary Block-1 random N1-N3 maxima and realized stratum counts
        cal_rows = []
        for rep in range(1, 200):
            rng = stream_rng(1, bl.section_index, 0, rep)
            actual_s = int(rng.integers(1, 4))
            case = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, case["X_P"], case["X_Q"])
            sc = aggregate_rule_scores(ep, eq)
            cal_rows.append({"stratum": actual_s, **{r: float(np.nanmax(sc[r])) for r in RULES}})
        cal1_primary[bg_name] = pd.DataFrame(cal_rows)

    out: dict[str, object] = {
        "primary_block1_stratum_stats": {},
        "b4_reweighting_check": {},
        "mondrian_on_block1": {},
        "single_seed_balanced_ablation": {},
        "multiseed_monte_carlo": {},
    }

    # 1. Inspect primary Block-1 stratum breakdown and Mondrian cutoffs
    mondrian_tot = 0
    for bg_name, cdf in cal1_primary.items():
        q_raw = fit_order_statistic_cutoffs({r: cdf[r].to_numpy() for r in RULES}, alpha=0.05)
        q_mon = mondrian_stratum_cutoffs(cdf["min"].to_numpy(), cdf["stratum"].to_numpy(), alpha=0.05)
        tdf = b2_test_null[bg_name]
        mon_hits = int(sum(row["min"] >= q_mon[int(row["stratum"])] for _, row in tdf.iterrows()))
        mondrian_tot += mon_hits
        out["primary_block1_stratum_stats"][bg_name] = {
            f"N{s}": {
                "n_cal": int((cdf["stratum"] == s).sum()),
                "cal_min_median": round(float(cdf[cdf["stratum"] == s]["min"].median()), 6),
                "cal_min_p95": round(float(np.quantile(cdf[cdf["stratum"] == s]["min"], 0.95)), 6),
                "cal_min_max": round(float(cdf[cdf["stratum"] == s]["min"].max()), 6),
                "test_min_hits_of_20": int((tdf[tdf["stratum"] == s]["min"] >= q_raw["min"]).sum()),
            }
            for s in (1, 2, 3)
        }
        out["mondrian_on_block1"][bg_name] = {"cutoffs": q_mon, "hits_of_60": mon_hits}
    out["mondrian_pooled_hits_of_240"] = mondrian_tot

    # Exact 1:1:1 inverse-probability reweighting check on B4 primary Block-1 draws
    b4_cal = cal1_primary["B4"]
    b4_test = b2_test_null["B4"]
    q_b4_unweighted = fit_order_statistic_cutoffs({"min": b4_cal["min"].to_numpy()}, alpha=0.05)["min"]
    counts_b4 = b4_cal["stratum"].value_counts().to_dict()
    weights_b4 = np.array([1.0 / (3.0 * counts_b4[int(s)]) for s in b4_cal["stratum"]], dtype=np.float64)
    q_b4_reweighted = _weighted_quantile_one_sided(b4_cal["min"].to_numpy(), weights_b4, 190.0 / 199.0)
    sorted_b4 = b4_cal.sort_values("min").reset_index(drop=True)
    out["b4_reweighting_check"] = {
        "block1_stratum_counts": {f"N{int(k)}": int(v) for k, v in sorted(counts_b4.items())},
        "q_b4_unweighted_k190": round(q_b4_unweighted, 6),
        "rank190_stratum": f"N{int(sorted_b4.iloc[189]['stratum'])}",
        "q_b4_reweighted_1_1_1": round(q_b4_reweighted, 6),
        "b4_test_fa_unweighted": int((b4_test["min"] >= q_b4_unweighted).sum()),
        "b4_test_fa_reweighted": int((b4_test["min"] >= q_b4_reweighted).sum()),
    }

    # 2. Multi-seed (200 seeds) independent Monte Carlo study at exact alpha = 0.05
    # Generate a fresh bank of 1,000 independent null maxima per stratum per section (Block 101 & 102 -> 500 each)
    # plus an independent held-out test bank (Block 103, 180 draws = 60 per stratum per section).
    cal_bank: dict[str, dict[int, np.ndarray]] = {b: {} for b in baselines}
    holdout_bank: dict[str, dict[int, np.ndarray]] = {b: {} for b in baselines}
    for bg_name, bl in baselines.items():
        op, scorable = ops[bg_name]
        for s in (1, 2, 3):
            cal_vals = []
            for block_id in (101, 102):
                for rep in range(1, 501):
                    c = generate_main_case(bl, block=block_id, stratum=s, rep=rep, fit_i_case=False)
                    ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
                    cal_vals.append(float(np.nanmax(np.minimum(ep, eq))))
            cal_bank[bg_name][s] = np.asarray(cal_vals, dtype=np.float64)

            h_vals = []
            for rep in range(1, 61):
                c = generate_main_case(bl, block=103, stratum=s, rep=rep, fit_i_case=False)
                ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
                h_vals.append(float(np.nanmax(np.minimum(ep, eq))))
            holdout_bank[bg_name][s] = np.asarray(h_vals, dtype=np.float64)

    n_seeds = 200
    rng = np.random.default_rng(20261119)
    protocols = ("unstrat_199_k190", "balanced_199_k190", "balanced_219_k209", "balanced_999_k950", "mondrian_199", "mondrian_999")
    mc_frozen = {p: {"pooled": [], "b4": []} for p in protocols}
    mc_holdout = {p: {"pooled": [], "b4": []} for p in protocols}

    for _ in range(n_seeds):
        tot_f = {p: 0 for p in protocols}
        tot_h = {p: 0 for p in protocols}
        for bg_name in ("B1", "B2", "B3", "B4"):
            t_df = b2_test_null[bg_name]
            t_by_s = {s: t_df[t_df["stratum"] == s]["min"].to_numpy() for s in (1, 2, 3)}
            h_by_s = holdout_bank[bg_name]
            bk = cal_bank[bg_name]

            # Draw disjoint/random index sets per seed
            c_un = rng.integers(1, 4, size=199)
            pools = {
                "unstrat_199_k190": {
                    s: rng.choice(bk[s], size=int((c_un == s).sum()), replace=False) for s in (1, 2, 3)
                },
                "balanced_199_k190": {
                    1: rng.choice(bk[1], size=67, replace=False),
                    2: rng.choice(bk[2], size=66, replace=False),
                    3: rng.choice(bk[3], size=66, replace=False),
                },
                "balanced_219_k209": {
                    s: rng.choice(bk[s], size=73, replace=False) for s in (1, 2, 3)
                },
                "balanced_999_k950": {
                    s: rng.choice(bk[s], size=333, replace=False) for s in (1, 2, 3)
                },
            }
            for p_name, grp in pools.items():
                all_vals = np.sort(np.concatenate([grp[1], grp[2], grp[3]]))
                k_rank = int(math.ceil((len(all_vals) + 1) * 0.95))
                q_cut = float(all_vals[k_rank - 1])
                fa_f = sum(int((t_by_s[s] >= q_cut).sum()) for s in (1, 2, 3))
                fa_h = sum(int((h_by_s[s] >= q_cut).sum()) for s in (1, 2, 3))
                tot_f[p_name] += fa_f
                tot_h[p_name] += fa_h
                if bg_name == "B4":
                    mc_frozen[p_name]["b4"].append(fa_f / 60.0)
                    mc_holdout[p_name]["b4"].append(fa_h / 180.0)

            for mon_name, base_key in (("mondrian_199", "balanced_199_k190"), ("mondrian_999", "balanced_999_k950")):
                grp = pools[base_key]
                fa_f = 0
                fa_h = 0
                for s in (1, 2, 3):
                    sub = np.sort(grp[s])
                    ks = min(len(sub), int(math.ceil((len(sub) + 1) * 0.95)))
                    qs = float(sub[ks - 1])
                    fa_f += int((t_by_s[s] >= qs).sum())
                    fa_h += int((h_by_s[s] >= qs).sum())
                tot_f[mon_name] += fa_f
                tot_h[mon_name] += fa_h
                if bg_name == "B4":
                    mc_frozen[mon_name]["b4"].append(fa_f / 60.0)
                    mc_holdout[mon_name]["b4"].append(fa_h / 180.0)

        for p in protocols:
            mc_frozen[p]["pooled"].append(tot_f[p] / 240.0)
            mc_holdout[p]["pooled"].append(tot_h[p] / 720.0)

    def _summ(arr: list[float]) -> dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": round(float(np.mean(a)), 4),
            "sd": round(float(np.std(a, ddof=1)), 4),
            "median": round(float(np.median(a)), 4),
            "q25": round(float(np.quantile(a, 0.25)), 4),
            "q75": round(float(np.quantile(a, 0.75)), 4),
            "p95": round(float(np.quantile(a, 0.95)), 4),
        }

    out["multiseed_monte_carlo"] = {
        "n_seeds": n_seeds,
        "evaluated_on_frozen_240": {
            p: {"pooled_rate": _summ(v["pooled"]), "b4_rate": _summ(v["b4"])} for p, v in mc_frozen.items()
        },
        "evaluated_on_independent_holdout_720": {
            p: {"pooled_rate": _summ(v["pooled"]), "b4_rate": _summ(v["b4"])} for p, v in mc_holdout.items()
        },
    }

    # 3. Deterministic block=100 single-seed 1:1:1 balanced calibration sensitivity (n_cal = 198 and n_cal = 999)
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
        out["single_seed_balanced_ablation"][key] = {
            "n_cal": n_tot,
            "by_section": sec_res,
            "pooled_of_240": pooled,
            "pooled_rates": {r: round(v / 240.0, 4) for r, v in pooled.items()},
        }

    (res_dir / "calibration_ablation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mondrian_primary_240": mondrian_tot,
        "b4_reweight": out["b4_reweighting_check"],
        "single_seed_198_rates": out["single_seed_balanced_ablation"]["balanced_198"]["pooled_rates"],
        "single_seed_999_rates": out["single_seed_balanced_ablation"]["balanced_999"]["pooled_rates"],
        "mc_holdout_720_means": {
            p: out["multiseed_monte_carlo"]["evaluated_on_independent_holdout_720"][p]["pooled_rate"]["mean"]
            for p in protocols
        },
        "mc_holdout_720_b4_means": {
            p: out["multiseed_monte_carlo"]["evaluated_on_independent_holdout_720"][p]["b4_rate"]["mean"]
            for p in protocols
        },
    }, indent=2))


if __name__ == "__main__":
    main()
