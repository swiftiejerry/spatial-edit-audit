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
from spatial_edit_audit.contrast import budget_tile_count, build_contrast_operator, score_candidate_pool_fast  # noqa: E402


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
            hits_c, f1_km, decoy_a, decoy_b, any_decoy = [], [], [], [], []
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
                ep_sg, _ = score_candidate_pool_fast(op, scorable, emb, emb)
                idx = np.flatnonzero(scorable)
                chosen = idx[np.lexsort((idx, -ep_sg[idx]))][:m_cap]
                pos = c["positions"]
                if "T" in pos:
                    hits_c.append(int(pos["T"] in chosen))
                    preds = KMeans(n_clusters=12, n_init=3, random_state=2026).fit_predict(emb)
                    t_mask = np.zeros(bl.n_spots, dtype=bool)
                    t_mask[bl.members_by_tile[pos["T"]]] = True
                    best = max(
                        2.0 * int(((preds == cid) & t_mask).sum()) / float(int((preds == cid).sum()) + int(t_mask.sum()))
                        for cid in np.unique(preds)
                        if ((preds == cid) & t_mask).any()
                    )
                    f1_km.append(best)
                da = int(pos["A"] in chosen) if "A" in pos else 0
                db = int(pos["B"] in chosen) if "B" in pos else 0
                decoy_a.append(da)
                decoy_b.append(db)
                any_decoy.append(int(da or db))
            out[bg_name][s_label] = {
                "sg_contrast_3pct_hit": None if not hits_c else round(float(np.mean(hits_c)), 4),
                "sg_kmeans_best_f1": None if not f1_km else round(float(np.mean(f1_km)), 4),
                "decoy_a_3pct_rate": round(float(np.mean(decoy_a)), 4),
                "decoy_b_3pct_rate": round(float(np.mean(decoy_b)), 4),
                "any_decoy_3pct_rate": round(float(np.mean(any_decoy)), 4),
            }

    (res_dir / "spatialglue_baseline_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
