"""Build compressed offline baseline bundles from raw H5AD matrices (annotation-blind)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit.preprocessing import build_section_baseline  # noqa: E402

SECTION_SPECS = {
    "B1": (1, "B1/adata_RNA.h5ad", "B1/adata_ADT.h5ad", 2000, None),
    "B2": (2, "B2/adata_RNA.h5ad", "B2/adata_ADT.h5ad", 2000, None),
    "B3": (3, "B3/adata_RNA.h5ad", "B3/adata_ATAC.h5ad", 2000, 5000),
    "B4": (4, "B4/adata_RNA.h5ad", "B4/adata_ATAC.h5ad", 2000, 5000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline SVD baseline bundles from raw H5AD files.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "reference")
    parser.add_argument(
        "--exclude-adt-regex",
        type=str,
        default=r"(?i)(?:mouse|rat).{0,10}igg|isotype",
        help="Regex filtering mouse/rat IgG isotype control antibodies from raw ADT matrices.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (idx, rna_rel, pair_rel, p_top, q_top) in SECTION_SPECS.items():
        rna_path = args.raw_dir / rna_rel
        pair_path = args.raw_dir / pair_rel
        if not (rna_path.exists() and pair_path.exists()):
            print(f"Skipping {name}: raw inputs not present at {args.raw_dir}")
            continue
        q_regex = args.exclude_adt_regex if "ADT" in pair_rel else None
        bl = build_section_baseline(
            name,
            idx,
            rna_path,
            pair_path,
            p_top_features=p_top,
            q_top_features=q_top,
            exclude_q_regex=q_regex,
        )
        out = args.out_dir / f"{name}_baseline.npz"
        np.savez_compressed(
            out,
            P0=bl.p0,
            Q0=bl.q0,
            coords=bl.coords,
            ids=bl.ids,
            R_static=bl.r_static,
            I_bg=bl.i_background,
            tiles=bl.tile_of,
            tile_id=bl.tile_table["tile_id"].to_numpy(dtype=str),
            tile_n_points=bl.tile_table["n_points"].to_numpy(dtype=np.int32),
            tile_dep=bl.tile_table["dep"].to_numpy(dtype=np.float64),
            tile_category=bl.tile_table["category"].to_numpy(dtype=str),
            tile_adjacent=bl.tile_table["adjacent_tiles"].to_numpy(dtype=str),
            tile_scorable=bl.tile_table["scorable"].to_numpy(dtype=bool),
        )
        print(f"Wrote {out} ({bl.n_spots} spots, {bl.n_tiles} tiles)")


if __name__ == "__main__":
    main()
