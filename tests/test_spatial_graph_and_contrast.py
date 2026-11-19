"""Unit tests for spatial neighbor graph construction, two-hop rings, and local contrast scoring."""

import numpy as np

from spatial_edit_audit.contrast import (
    aggregate_rule_scores,
    budget_tile_count,
    build_contrast_operator,
    local_molecular_contrast,
    score_candidate_pool,
    score_candidate_pool_fast,
)
from spatial_edit_audit.spatial_graph import build_knn_graph, compute_two_hop_ring


def test_knn_excludes_self_and_handles_duplicates():
    coords = np.zeros((6, 2), dtype=np.float64)
    graph, nbrs = build_knn_graph(coords, k=3)
    assert nbrs.shape == (6, 3)
    assert graph.shape == (6, 6)
    for i, row in enumerate(nbrs):
        assert i not in row
        assert len(set(row)) == 3


def test_two_hop_ring_is_disjoint_from_members():
    coords = np.column_stack([np.arange(15, dtype=np.float64), np.zeros(15, dtype=np.float64)])
    graph, _ = build_knn_graph(coords, k=2)
    members = np.array([5, 6, 7], dtype=int)
    ring = compute_two_hop_ring(members, graph, hops=2)
    assert set(ring.tolist()).isdisjoint(set(members.tolist()))
    assert len(ring) >= 4


def test_fast_contrast_operator_matches_reference_scoring():
    rng = np.random.default_rng(42)
    coords = np.column_stack([np.repeat(np.arange(10), 10), np.tile(np.arange(10), 10)]).astype(np.float64)
    graph, _ = build_knn_graph(coords, k=6)
    x_p = rng.standard_normal((100, 16))
    x_q = rng.standard_normal((100, 16))
    cands = [("T0000", np.arange(0, 15)), ("T0001", np.arange(45, 60))]
    x_p[0:15, :4] += 2.5
    x_q[0:15, :4] += 2.5

    ref_df = score_candidate_pool(graph, x_p, x_q, cands)
    op, scorable = build_contrast_operator(cands, graph, 100)
    ep_fast, eq_fast = score_candidate_pool_fast(op, scorable, x_p, x_q)
    np.testing.assert_allclose(ep_fast, ref_df["e_P"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(eq_fast, ref_df["e_Q"].to_numpy(), atol=1e-12)
    agg = aggregate_rule_scores(ep_fast, eq_fast)
    assert agg["min"][0] == min(ep_fast[0], eq_fast[0])
    assert agg["mean"][0] == 0.5 * (ep_fast[0] + eq_fast[0])
    assert budget_tile_count(0.03, 174) == 5
    assert budget_tile_count(0.03, 97) == 3


def test_constant_features_fail_closed():
    coords = np.column_stack([np.arange(30, dtype=np.float64), np.zeros(30, dtype=np.float64)])
    graph, _ = build_knn_graph(coords, k=6)
    res = local_molecular_contrast(np.arange(6), graph, np.zeros((30, 4)), n_null=19)
    assert res["status"] == "REJECT"
    assert res["reason"] == "constant_independent_features"


def test_disjoint_perm_ref_and_str_loader(tmp_path):
    from pathlib import Path
    from spatial_edit_audit.baselines import evaluate_neighbor_augmented_clustering, score_dual_latent_views
    from spatial_edit_audit.preprocessing import load_section_baseline

    ref_npz = Path(__file__).resolve().parents[1] / "data" / "reference" / "B3_baseline.npz"
    bl = load_section_baseline(str(ref_npz))
    assert bl.name == "B3"
    res = local_molecular_contrast(bl.members_by_tile[0], bl.graph, bl.p0, n_null=19, disjoint_perm_ref=True)
    assert "null_p" in res
    f1 = evaluate_neighbor_augmented_clustering(bl, bl.p0, bl.q0, bl.members_by_tile[0], n_clusters=None)
    assert 0.0 <= f1 <= 1.0
    dual = score_dual_latent_views(bl, bl.p0, bl.q0)
    assert len(dual["selected"]["min"]) == budget_tile_count(0.03, bl.n_tiles)
