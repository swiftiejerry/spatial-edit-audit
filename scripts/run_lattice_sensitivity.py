"""Evaluate coordinate-tie sensitivity (Visium sqrt(3) scaling, MISAR order permutation, k=4/8) and Gate-6/ddof sensitivity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import (  # noqa: E402
    GateConfig,
    apply_reference_intervention,
    evaluate_multigate_prototype,
    fit_order_statistic_cutoffs,
    generate_main_case,
    load_section_baseline,
)
from spatial_edit_audit.contrast import (  # noqa: E402
    budget_tile_count,
    build_contrast_operator,
    local_molecular_contrast,
    score_candidate_pool_fast,
)
from spatial_edit_audit.preprocessing import _assemble_section_baseline  # noqa: E402
from spatial_edit_audit.spatial_graph import build_knn_graph, partition_spatial_tiles  # noqa: E402


def _rebuild_with_coords(bl, new_coords: np.ndarray, *, k_nn: int = 6, retile: bool = True):
    graph, _ = build_knn_graph(new_coords, k=k_nn)
    if retile:
        tile_of, tile_table = partition_spatial_tiles(new_coords, bl.r_static, graph, seed=20261111)
    else:
        tile_of, tile_table = bl.tile_of.copy(), bl.tile_table.copy()
    bl_new = _assemble_section_baseline(
        bl.name,
        bl.section_index,
        bl.p0,
        bl.q0,
        new_coords,
        bl.ids,
        bl.r_static,
        bl.i_background,
        tile_of,
        tile_table,
    )
    bl_new.graph, bl_new.neighbors = graph, build_knn_graph(new_coords, k=k_nn)[1]
    return bl_new


def _quick_main_eval(bl) -> dict[str, float]:
    op, scorable = build_contrast_operator(
        [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
    )
    m_cap = budget_tile_count(0.03, bl.n_tiles)
    idx = np.flatnonzero(scorable)

    cal_max = []
    for rep in range(1, 200):
        c = generate_main_case(bl, block=1, stratum=0, rep=rep, calibration_draw=True, fit_i_case=False)
        ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
        cal_max.append(float(np.nanmax(np.minimum(ep, eq))))
    q_min = fit_order_statistic_cutoffs({"min": cal_max}, alpha=0.05)["min"]

    s6_min, s6_q, s8_min, s8_m0 = [], [], [], []
    n2_min = []
    for stratum in (2, 6, 8):
        for rep in range(1, 21):
            c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=(stratum == 8))
            ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
            s_min = np.minimum(ep, eq)
            chosen_min = set(idx[np.lexsort((idx, -s_min[idx]))][:m_cap].tolist())
            chosen_q = set(idx[np.lexsort((idx, -eq[idx]))][:m_cap].tolist())
            pos = c["positions"]
            if stratum == 6:
                s6_min.append(int(pos["T"] in chosen_min))
                s6_q.append(int(pos["T"] in chosen_q))
            elif stratum == 8:
                s8_min.append(int(pos["T"] in chosen_min))
                r_m0 = evaluate_multigate_prototype(bl, c["X_Q"], c["I_case"], config=GateConfig(disable_damage_gate=False))
                s8_m0.append(int(f"T{pos['T']:04d}" in r_m0["accepted_ids"]))
            elif stratum == 2:
                n2_min.append(int((pos["A"] in chosen_min) or (pos["B"] in chosen_min)))

    null_cal_hits = 0
    for s in (1, 2, 3):
        for rep in range(1, 21):
            c = generate_main_case(bl, block=2, stratum=s, rep=rep, fit_i_case=False)
            ep, eq = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"])
            null_cal_hits += int(float(np.nanmax(np.minimum(ep, eq))) >= q_min)

    return {
        "n_tiles": int(bl.n_tiles),
        "n_scorable": int(scorable.sum()),
        "q_min_k190": round(q_min, 4),
        "null_cal_fa_of_60": int(null_cal_hits),
        "n2_min_any_decoy": round(float(np.mean(n2_min)), 4),
        "s6_min_3pct": round(float(np.mean(s6_min)), 4),
        "s6_q_3pct": round(float(np.mean(s6_q)), 4),
        "s8_min_3pct": round(float(np.mean(s8_min)), 4),
        "s8_m0_native": round(float(np.mean(s8_m0)), 4),
    }


def main() -> None:
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    # 1. Visium staggered grid equilateral aspect-ratio scaling (row *= sqrt(3)) on B1/B2
    visium_aspect: dict[str, object] = {}
    for b in ("B1", "B2"):
        bl = baselines[b]
        scaled_coords = bl.coords.copy()
        scaled_coords[:, 0] *= float(np.sqrt(3.0))
        bl_hex = _rebuild_with_coords(bl, scaled_coords, k_nn=6)
        edge_jaccard = float(
            bl.graph.multiply(bl_hex.graph).nnz
            / (bl.graph.nnz + bl_hex.graph.nnz - bl.graph.multiply(bl_hex.graph).nnz)
        )
        visium_aspect[b] = {
            "undirected_edge_jaccard": round(edge_jaccard, 4),
            "native": _quick_main_eval(bl),
            "hex_sqrt3_row_scaled": _quick_main_eval(bl_hex),
        }

    # 2. MISAR-seq square grid coordinate permutation tie test and k=4/8 connectivity on B3/B4
    rng = np.random.default_rng(20261119)
    misar_sensitivity: dict[str, object] = {}
    for b in ("B3", "B4"):
        bl = baselines[b]
        perm = rng.permutation(bl.n_spots)
        inv = np.argsort(perm)
        graph_perm, _ = build_knn_graph(bl.coords[perm], k=6)
        graph_perm_un = graph_perm[inv][:, inv]
        inter = bl.graph.multiply(graph_perm_un).nnz
        perm_jaccard = float(inter / (bl.graph.nnz + graph_perm_un.nnz - inter))

        # Evaluate k=4 (rook) and k=8 (queen) neighbor graphs holding tile partition fixed
        bl_k4 = _rebuild_with_coords(bl, bl.coords, k_nn=4, retile=False)
        bl_k8 = _rebuild_with_coords(bl, bl.coords, k_nn=8, retile=False)
        misar_sensitivity[b] = {
            "spot_order_perm_6nn_edge_jaccard": round(perm_jaccard, 4),
            "native_k6": _quick_main_eval(bl),
            "k4_rook_fixed_tiles": _quick_main_eval(bl_k4),
            "k8_queen_fixed_tiles": _quick_main_eval(bl_k8),
        }

    # 3. Gate-6 strict leave-out permutation (C(A) \ D_r) and ddof=1 fallback parity check across all S6/S8 cases
    gate6_disjoint_flips = 0
    ddof1_max_diff = 0.0
    total_cf = 0
    for bg_name, bl in baselines.items():
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        for stratum in (6, 8):
            for rep in range(1, 21):
                c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                t_idx = c["positions"]["T"]
                members = bl.members_by_tile[t_idx]
                res_full = local_molecular_contrast(members, bl.graph, c["X_Q"], disjoint_perm_ref=False)
                res_disj = local_molecular_contrast(members, bl.graph, c["X_Q"], disjoint_perm_ref=True)
                gate6_disjoint_flips += int(res_full["status"] != res_disj["status"])
                total_cf += 1

                ep0, eq0 = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"], ddof=0)
                ep1, eq1 = score_candidate_pool_fast(op, scorable, c["X_P"], c["X_Q"], ddof=1)
                ddof1_max_diff = max(
                    ddof1_max_diff,
                    float(np.nanmax(np.abs(ep0 - ep1))),
                    float(np.nanmax(np.abs(eq0 - eq1))),
                )

    summary = {
        "visium_sqrt3_aspect_ratio": visium_aspect,
        "misar_grid_ties_and_k_sensitivity": misar_sensitivity,
        "gate6_disjoint_perm_ref": {
            "total_target_cases_s6_s8": total_cf,
            "gate6_status_flips_full_vs_disjoint": int(gate6_disjoint_flips),
        },
        "fallback_sd_ddof_sensitivity": {
            "max_abs_score_diff_ddof0_vs_ddof1": round(ddof1_max_diff, 12),
        },
    }
    (res_dir / "lattice_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "b1_jacc": visium_aspect["B1"]["undirected_edge_jaccard"],
        "b2_jacc": visium_aspect["B2"]["undirected_edge_jaccard"],
        "b3_perm_jacc": misar_sensitivity["B3"]["spot_order_perm_6nn_edge_jaccard"],
        "b4_perm_jacc": misar_sensitivity["B4"]["spot_order_perm_6nn_edge_jaccard"],
        "gate6_flips": gate6_disjoint_flips,
        "ddof_diff": ddof1_max_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
