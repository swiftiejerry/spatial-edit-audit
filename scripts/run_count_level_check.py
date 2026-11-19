"""Verify ranking reversal and decoy resilience using Poisson raw-count perturbations and full H5AD preprocessing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import load_section_baseline, project_count_matrix_svd  # noqa: E402
from spatial_edit_audit.contrast import budget_tile_count, build_contrast_operator, score_candidate_pool_fast  # noqa: E402
from spatial_edit_audit.preprocessing import read_h5ad_raw_and_coords  # noqa: E402
from spatial_edit_audit.simulation import inject_poisson_count_shift  # noqa: E402


def _top_variable_indices(counts, top_n: int | None) -> np.ndarray:
    pos = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(pos >= math.ceil(0.01 * counts.shape[0]))
    sub = counts[:, keep].copy()
    if top_n is not None and sub.shape[1] > top_n:
        tot = np.asarray(sub.sum(axis=1)).ravel()
        fac = np.where(tot > 0, 1e4 / tot, 0.0)
        sub = (diags(fac) @ sub).tocsr()
        sub.data = np.log1p(sub.data)
        m = np.asarray(sub.mean(axis=0)).ravel()
        m2 = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
        ord_idx = np.lexsort((keep, -np.maximum(m2 - m * m, 0.0)))[:top_n]
        return keep[ord_idx]
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run count-level raw-matrix perturbation sensitivity check.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--b1-dir", type=Path, default=None, help="Optional explicit directory for B1 raw H5AD files.")
    parser.add_argument("--b2-dir", type=Path, default=None, help="Optional explicit directory for B2 raw H5AD files.")
    parser.add_argument("--b3-dir", type=Path, default=None, help="Optional explicit directory for B3 raw H5AD files.")
    parser.add_argument("--b4-dir", type=Path, default=None, help="Optional explicit directory for B4 raw H5AD files.")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=ROOT / "results" / "count_level_verification.json")
    args = parser.parse_args()

    ref_dir = ROOT / "data" / "reference"
    out = {}
    specs = [
        ("B1", args.b1_dir or (args.raw_dir / "B1"), "adata_RNA.h5ad", "adata_ADT.h5ad", 2000, None),
        ("B2", args.b2_dir or (args.raw_dir / "B2"), "adata_RNA.h5ad", "adata_ADT.h5ad", 2000, None),
        ("B3", args.b3_dir or (args.raw_dir / "B3"), "adata_RNA.h5ad", "adata_ATAC.h5ad", 2000, 5000),
        ("B4", args.b4_dir or (args.raw_dir / "B4"), "adata_RNA.h5ad", "adata_ATAC.h5ad", 2000, 5000),
    ]

    for bg_name, sec_dir, rna_fn, pair_fn, p_top, q_top in specs:
        rna_path = sec_dir / rna_fn
        pair_path = sec_dir / pair_fn
        if not (rna_path.exists() and pair_path.exists()):
            print(f"Skipping {bg_name}: raw H5AD files not found under {sec_dir}")
            continue
        bl = load_section_baseline(ref_dir / f"{bg_name}_baseline.npz")
        _, rna_counts, _ = read_h5ad_raw_and_coords(rna_path)
        pair_regex = r"(?i)(?:mouse|rat).{0,10}igg|isotype" if "ADT" in pair_fn else None
        _, pair_counts, _ = read_h5ad_raw_and_coords(pair_path, exclude_var_regex=pair_regex)
        r_feats = _top_variable_indices(rna_counts, 200)
        q_feats = _top_variable_indices(pair_counts, min(200, pair_counts.shape[1]))

        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        low_tiles = np.flatnonzero(bl.tile_table["category"].to_numpy() == "LOW")
        m_cap = budget_tile_count(0.03, bl.n_tiles)

        out[bg_name] = {}
        for regime, fc_p, fc_q in (("concordant_2.5x", 2.5, 2.5), ("asymmetric_4.0x_1.2x", 4.0, 1.2)):
            hits = {r: 0 for r in ("P", "Q", "mean", "min")}
            for rep in range(1, int(args.replicates) + 1):
                rng = np.random.default_rng(20261119 + rep + bl.section_index * 1000 + (0 if fc_p == fc_q else 100))
                t_id = int(rng.choice(low_tiles))
                pool_a = [t for t in range(bl.n_tiles) if t != t_id and bl.isolated_matrix[t, t_id]]
                a_id = int(rng.choice(pool_a))
                pool_b = [t for t in pool_a if t != a_id and bl.isolated_matrix[t, a_id]]
                b_id = int(rng.choice(pool_b))

                rc = inject_poisson_count_shift(rna_counts, bl.members_by_tile[t_id], r_feats, rng, fc_p)
                rc = inject_poisson_count_shift(rc, bl.members_by_tile[a_id], r_feats, rng, 3.5)
                pc = inject_poisson_count_shift(pair_counts, bl.members_by_tile[t_id], q_feats, rng, fc_q)
                pc = inject_poisson_count_shift(pc, bl.members_by_tile[b_id], q_feats, rng, 3.5)

                p_view, _ = project_count_matrix_svd(rc, n_top_features=p_top, n_components=16)
                q_view, _ = project_count_matrix_svd(pc, n_top_features=q_top, n_components=16)
                ep, eq = score_candidate_pool_fast(op, scorable, p_view, q_view)
                scores = {"P": ep, "Q": eq, "mean": 0.5 * (ep + eq), "min": np.minimum(ep, eq)}
                idx = np.flatnonzero(scorable)
                for r in hits:
                    chosen = idx[np.lexsort((idx, -scores[r][idx]))][:m_cap]
                    if t_id in chosen:
                        hits[r] += 1
            out[bg_name][regime] = hits

    if out:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
