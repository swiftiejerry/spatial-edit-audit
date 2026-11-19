"""Annotation-blind preprocessing, SVD standardization, and baseline bundle loading."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, vstack
from sklearn.cluster import KMeans

from .spatial_graph import build_knn_graph, compute_two_hop_ring, partition_spatial_tiles


@dataclass
class SectionBaseline:
    """Container holding baseline views, spatial graph, candidate tiles, and static partitions."""

    name: str
    section_index: int
    p0: np.ndarray
    q0: np.ndarray
    coords: np.ndarray
    ids: np.ndarray
    r_static: np.ndarray
    i_background: np.ndarray
    tile_of: np.ndarray
    tile_table: pd.DataFrame
    members_by_tile: list[np.ndarray]
    rings_by_tile: list[np.ndarray]
    graph: csr_matrix
    neighbors: np.ndarray
    adjacent_tiles: list[set[int]]
    isolated_matrix: np.ndarray

    @property
    def n_spots(self) -> int:
        return len(self.ids)

    @property
    def n_tiles(self) -> int:
        return len(self.members_by_tile)


def _decode_attr(val: Any) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return bytes(val).decode("utf-8")
    return str(val)


def read_h5ad_raw_and_coords(
    path: Path | str,
    *,
    exclude_var_regex: str | None = None,
) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Read strictly obs IDs, raw count matrix ``X``, and spatial coordinates without annotations."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        idx_key = _decode_attr(obs.attrs.get("_index", "_index"))
        idx_ds = obs[idx_key]
        ids = np.asarray(
            idx_ds.asstr()[()] if h5py.check_string_dtype(idx_ds.dtype) is not None else idx_ds[()],
            dtype=str,
        )
        if ids.ndim != 1 or len(ids) == 0 or np.unique(ids).size != len(ids):
            raise ValueError(f"Non-unique or empty spot IDs in {path}")
        var = handle["var"]
        var_key = _decode_attr(var.attrs.get("_index", "_index"))
        var_ds = var[var_key]
        var_names = np.asarray(
            var_ds.asstr()[()] if h5py.check_string_dtype(var_ds.dtype) is not None else var_ds[()],
            dtype=str,
        )
        n_vars = int(len(var_names))

        x_node = handle["X"]
        if isinstance(x_node, h5py.Group):
            encoding = _decode_attr(x_node.attrs.get("encoding-type", ""))
            if encoding != "csr_matrix":
                raise ValueError(f"Expected CSR sparse matrix in {path}, got {encoding!r}")
            shape = tuple(int(v) for v in x_node.attrs["shape"])
            counts = csr_matrix(
                (
                    np.asarray(x_node["data"], dtype=np.float64),
                    np.asarray(x_node["indices"], dtype=np.int64),
                    np.asarray(x_node["indptr"], dtype=np.int64),
                ),
                shape=shape,
                dtype=np.float64,
            )
        elif isinstance(x_node, h5py.Dataset):
            blocks = [
                csr_matrix(np.asarray(x_node[s : s + 128, :], dtype=np.float64))
                for s in range(0, x_node.shape[0], 128)
            ]
            counts = vstack(blocks, format="csr", dtype=np.float64)
        else:
            raise ValueError(f"Unsupported matrix layout in {path}")
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if counts.shape != (len(ids), n_vars):
            raise ValueError(f"Shape mismatch in {path}: {counts.shape} vs ({len(ids)}, {n_vars})")
        if exclude_var_regex:
            pat = re.compile(exclude_var_regex)
            keep_cols = np.asarray([not bool(pat.search(str(vn))) for vn in var_names], dtype=bool)
            if not np.any(keep_cols):
                raise ValueError(f"Regex {exclude_var_regex!r} excluded all features in {path}")
            counts = counts[:, keep_cols].tocsr()
        if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
            raise ValueError(f"Non-finite or negative count entries in {path}")
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float64)
        if coords.shape != (len(ids), 2) or not np.isfinite(coords).all():
            raise ValueError(f"Invalid spatial coordinates in {path}")
    return ids, counts, coords


def project_count_matrix_svd(
    counts: csr_matrix,
    *,
    n_top_features: int | None,
    n_components: int = 16,
) -> tuple[np.ndarray, int]:
    """Filter >=1% prevalent features, normalize to 1e4, log1p, select top HVF, project to MAD-scaled SVD."""
    mat = csr_matrix(counts, dtype=np.float64)
    mat.sum_duplicates()
    mat.eliminate_zeros()
    n_spots = mat.shape[0]
    min_pos = int(math.ceil(0.01 * n_spots))
    pos_counts = np.asarray((mat > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(pos_counts >= min_pos)
    if keep.size == 0:
        raise ValueError("No feature passed the 1% prevalence threshold")
    mat = mat[:, keep].tocsr()

    totals = np.asarray(mat.sum(axis=1)).ravel()
    factors = np.zeros_like(totals, dtype=np.float64)
    np.divide(1e4, totals, out=factors, where=totals > 0)
    mat = (diags(factors, format="csr") @ mat).tocsr()
    mat.data = np.log1p(mat.data)

    if n_top_features is not None and mat.shape[1] > int(n_top_features):
        mean = np.asarray(mat.mean(axis=0)).ravel()
        mean_sq = np.asarray(mat.multiply(mat).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean * mean, 0.0)
        order = np.lexsort((keep, -var))
        sel = np.sort(order[: int(n_top_features)])
        mat = mat[:, sel].tocsr()

    retained = int(mat.shape[1])
    if min(n_spots, retained) < n_components:
        raise ValueError(f"Insufficient rank for {n_components} SVD axes: ({n_spots}, {retained})")
    dense = mat.toarray()
    dense -= dense.mean(axis=0, dtype=np.float64)
    u, svals, vt = np.linalg.svd(dense, full_matrices=False)
    emb = u[:, :n_components] * svals[:n_components]
    for col in range(n_components):
        pivot = int(np.argmax(np.abs(vt[col])))
        if vt[col, pivot] < 0:
            emb[:, col] *= -1.0
    med = np.median(emb, axis=0)
    mad = np.median(np.abs(emb - med), axis=0) * 1.4826
    if np.any(mad <= 1e-12):
        raise ValueError("Degenerate zero-MAD SVD component encountered")
    emb = (emb - med) / mad
    return emb.astype(np.float64, copy=False), retained


def _assemble_section_baseline(
    name: str,
    section_index: int,
    p0: np.ndarray,
    q0: np.ndarray,
    coords: np.ndarray,
    ids: np.ndarray,
    r_static: np.ndarray,
    i_background: np.ndarray,
    tile_of: np.ndarray,
    tile_table: pd.DataFrame,
) -> SectionBaseline:
    graph, neighbors = build_knn_graph(coords, k=6)
    n_tiles = len(tile_table)
    members_by_tile = [np.flatnonzero(tile_of == tid) for tid in range(n_tiles)]
    rings_by_tile = [compute_two_hop_ring(m, graph, hops=2) for m in members_by_tile]

    adjacent: list[set[int]] = [set() for _ in range(n_tiles)]
    coo = graph.tocoo()
    for u, v in zip(coo.row, coo.col):
        tu, tv = int(tile_of[int(u)]), int(tile_of[int(v)])
        if tu != tv:
            adjacent[tu].add(tv)

    isolated = np.ones((n_tiles, n_tiles), dtype=bool)
    for tid in range(n_tiles):
        isolated[tid, tid] = False
        for ring_spot in rings_by_tile[tid]:
            other = int(tile_of[int(ring_spot)])
            isolated[tid, other] = False
            isolated[other, tid] = False

    return SectionBaseline(
        name=name,
        section_index=section_index,
        p0=np.asarray(p0, dtype=np.float64),
        q0=np.asarray(q0, dtype=np.float64),
        coords=np.asarray(coords, dtype=np.float64),
        ids=np.asarray(ids, dtype=str),
        r_static=np.asarray(r_static, dtype=str),
        i_background=np.asarray(i_background, dtype=str),
        tile_of=np.asarray(tile_of, dtype=int),
        tile_table=tile_table,
        members_by_tile=members_by_tile,
        rings_by_tile=rings_by_tile,
        graph=graph,
        neighbors=neighbors,
        adjacent_tiles=adjacent,
        isolated_matrix=isolated,
    )


def build_section_baseline(
    name: str,
    section_index: int,
    rna_h5ad: Path | str,
    paired_h5ad: Path | str,
    *,
    p_top_features: int = 2000,
    q_top_features: int | None = None,
    exclude_p_regex: str | None = None,
    exclude_q_regex: str | None = None,
    seed: int = 20261111,
) -> SectionBaseline:
    """Build a complete baseline section directly from paired raw H5AD files."""
    ids_p, counts_p, coords_p = read_h5ad_raw_and_coords(rna_h5ad, exclude_var_regex=exclude_p_regex)
    ids_q, counts_q, coords_q = read_h5ad_raw_and_coords(paired_h5ad, exclude_var_regex=exclude_q_regex)
    if not np.array_equal(ids_p, ids_q):
        raise ValueError(f"{name}: RNA and paired spot IDs disagree")
    if not np.array_equal(coords_p, coords_q):
        raise ValueError(f"{name}: RNA and paired spatial coordinates disagree")

    p0, _ = project_count_matrix_svd(counts_p, n_top_features=p_top_features, n_components=16)
    q0, _ = project_count_matrix_svd(counts_q, n_top_features=q_top_features, n_components=16)

    graph, _ = build_knn_graph(coords_p, k=6)
    r_idx = KMeans(n_clusters=8, n_init=10, random_state=seed).fit_predict(p0)
    i_idx = KMeans(n_clusters=8, n_init=10, random_state=seed).fit_predict(q0)
    r_static = np.asarray([f"R{int(v)}" for v in r_idx], dtype=str)
    i_bg = np.asarray([f"I{int(v)}" for v in i_idx], dtype=str)
    tile_of, tile_table = partition_spatial_tiles(coords_p, r_static, graph, seed=seed)
    return _assemble_section_baseline(
        name, section_index, p0, q0, coords_p, ids_p, r_static, i_bg, tile_of, tile_table
    )


def load_section_baseline(
    npz_path: Path | str,
    name: str | None = None,
    section_index: int | None = None,
) -> SectionBaseline:
    """Load an offline compressed baseline bundle produced under ``data/reference/``."""
    npz_path = Path(npz_path)
    resolved_name = name or npz_path.stem.split("_")[0]
    resolved_idx = section_index if section_index is not None else int(resolved_name[1:])
    with np.load(npz_path, allow_pickle=False) as packed:
        tile_table = pd.DataFrame(
            {
                "tile_id": packed["tile_id"].astype(str),
                "n_points": packed["tile_n_points"].astype(int),
                "dep": packed["tile_dep"].astype(float),
                "category": packed["tile_category"].astype(str),
                "adjacent_tiles": packed["tile_adjacent"].astype(str),
                "scorable": packed["tile_scorable"].astype(bool),
            }
        )
        return _assemble_section_baseline(
            name=resolved_name,
            section_index=resolved_idx,
            p0=packed["P0"],
            q0=packed["Q0"],
            coords=packed["coords"],
            ids=packed["ids"],
            r_static=packed["R_static"],
            i_background=packed["I_bg"],
            tile_of=packed["tiles"],
            tile_table=tile_table,
        )
