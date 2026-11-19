"""Optional baseline comparison against SpatialGlue (requires PyTorch + PyG + upstream SpatialGlue)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import generate_main_case, load_section_baseline  # noqa: E402
from spatial_edit_audit.baselines import score_dual_latent_views  # noqa: E402
from spatial_edit_audit.contrast import budget_tile_count, build_contrast_operator, score_candidate_pool_fast  # noqa: E402


def _best_cluster_f1(emb: np.ndarray, t_mask: np.ndarray, n_clusters: int, seed: int = 2026) -> float:
    preds = KMeans(n_clusters=int(n_clusters), n_init=3, random_state=seed).fit_predict(emb)
    n_t = int(t_mask.sum())
    best = 0.0
    for cid in np.unique(preds):
        c_mask = preds == cid
        tp = int((c_mask & t_mask).sum())
        if tp > 0:
            f1 = 2.0 * tp / float(int(c_mask.sum()) + n_t)
            if f1 > best:
                best = f1
    return float(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark published SpatialGlue GNN integration on B1-B4 cases.")
    parser.add_argument("--spatialglue-parent", type=Path, required=True, help="Directory containing the SpatialGlue package.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.spatialglue_parent))
    import torch  # noqa: E402
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue  # noqa: E402
    from SpatialGlue.preprocess import construct_neighbor_graph  # noqa: E402

    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_dir = ROOT / "data" / "reference"
    res_dir = ROOT / "results"

    specs = [
        ("B1", "Stereo-CITE-seq"),
        ("B2", "Stereo-CITE-seq"),
        ("B3", "Spatial-epigenome-transcriptome"),
        ("B4", "Spatial-epigenome-transcriptome"),
    ]
    out: dict[str, object] = {}
    for bg_name, dtype in specs:
        bl = load_section_baseline(ref_dir / f"{bg_name}_baseline.npz")
        op, scorable = build_contrast_operator(
            [(f"T{i:04d}", m) for i, m in enumerate(bl.members_by_tile)], bl.graph, bl.n_spots
        )
        m_cap = budget_tile_count(0.03, bl.n_tiles)
        out[bg_name] = {}
        for s_label, stratum in (("S6_concordant", 6), ("S8_boundary", 8), ("S9_asymmetric", 9), ("N2_decoys", 2)):
            hits_c, hits_dmin, hits_dmean = [], [], []
            f1_km12, f1_kmtiles = [], []
            decoy_a, decoy_b, any_decoy = [], [], []
            dmin_a, dmin_b, dmin_any = [], [], []
            for rep in range(1, int(args.replicates) + 1):
                c = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=False)
                a1 = ad.AnnData(X=np.zeros((bl.n_spots, 1), dtype=np.float32))
                a2 = ad.AnnData(X=np.zeros((bl.n_spots, 1), dtype=np.float32))
                a1.obsm["spatial"] = bl.coords.astype(np.float32)
                a2.obsm["spatial"] = bl.coords.astype(np.float32)
                a1.obsm["feat"] = c["X_P"].astype(np.float32)
                a2.obsm["feat"] = c["X_Q"].astype(np.float32)
                data = construct_neighbor_graph(a1, a2, datatype=dtype, n_neighbors=6)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model = Train_SpatialGlue(
                        data, datatype=dtype, device=device, epochs=args.epochs, random_seed=2026
                    )
                    res = model.train()
                emb = res["SpatialGlue"]
                emb_1 = res["emb_latent_omics1"]
                emb_2 = res["emb_latent_omics2"]

                ep_sg, _ = score_candidate_pool_fast(op, scorable, emb, emb)
                idx = np.flatnonzero(scorable)
                chosen = idx[np.lexsort((idx, -ep_sg[idx]))][:m_cap]

                dual_res = score_dual_latent_views(bl, emb_1, emb_2, budget_fraction=0.03)
                chosen_dmin = dual_res["selected"]["min"]
                chosen_dmean = dual_res["selected"]["mean"]

                pos = c["positions"]
                if "T" in pos:
                    t_idx = pos["T"]
                    hits_c.append(int(t_idx in chosen))
                    hits_dmin.append(int(t_idx in chosen_dmin))
                    hits_dmean.append(int(t_idx in chosen_dmean))
                    t_mask = np.zeros(bl.n_spots, dtype=bool)
                    t_mask[bl.members_by_tile[t_idx]] = True
                    f1_km12.append(_best_cluster_f1(emb, t_mask, n_clusters=12))
                    f1_kmtiles.append(_best_cluster_f1(emb, t_mask, n_clusters=bl.n_tiles))
                da = int(pos["A"] in chosen) if "A" in pos else 0
                db = int(pos["B"] in chosen) if "B" in pos else 0
                decoy_a.append(da)
                decoy_b.append(db)
                any_decoy.append(int(da or db))
                dm_a = int(pos["A"] in chosen_dmin) if "A" in pos else 0
                dm_b = int(pos["B"] in chosen_dmin) if "B" in pos else 0
                dmin_a.append(dm_a)
                dmin_b.append(dm_b)
                dmin_any.append(int(dm_a or dm_b))
            out[bg_name][s_label] = {
                "sg_contrast_3pct_hit": None if not hits_c else round(float(np.mean(hits_c)), 4),
                "sg_dual_latent_min_3pct_hit": None if not hits_dmin else round(float(np.mean(hits_dmin)), 4),
                "sg_dual_latent_mean_3pct_hit": None if not hits_dmean else round(float(np.mean(hits_dmean)), 4),
                "sg_kmeans12_best_f1": None if not f1_km12 else round(float(np.mean(f1_km12)), 4),
                "sg_kmeans_matched_best_f1": None if not f1_kmtiles else round(float(np.mean(f1_kmtiles)), 4),
                "decoy_a_3pct_rate": round(float(np.mean(decoy_a)), 4),
                "decoy_b_3pct_rate": round(float(np.mean(decoy_b)), 4),
                "any_decoy_3pct_rate": round(float(np.mean(any_decoy)), 4),
                "dual_latent_min_decoy_a_rate": round(float(np.mean(dmin_a)), 4),
                "dual_latent_min_decoy_b_rate": round(float(np.mean(dmin_b)), 4),
                "dual_latent_min_any_decoy_rate": round(float(np.mean(dmin_any)), 4),
            }

    (res_dir / "spatialglue_baseline_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
