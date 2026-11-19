"""Unit tests for M0 topological nesting, counterfactual invariants, and dual calibrated gates."""

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_edit_audit import (
    GateConfig,
    apply_reference_intervention,
    evaluate_dual_calibrated_gate,
    evaluate_multigate_prototype,
    load_section_baseline,
)

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_gamma_coherence_subsumes_leakage_gate_across_all_tiles():
    """Since lambda(A) = 1 - gamma(A), gamma >= 0.55 implies lambda <= 0.45 < 0.55."""
    for bg_name in ("B1", "B2", "B3", "B4"):
        bl = load_section_baseline(REF_DIR / f"{bg_name}_baseline.npz")
        for members in bl.members_by_tile:
            m_set = set(members.tolist())
            nbrs = bl.neighbors[members]
            internal = sum(int(j in m_set) for row in nbrs for j in row)
            total = int(nbrs.size)
            gamma = internal / total
            lam = 1.0 - gamma
            assert gamma >= 0.55
            assert lam <= 0.45 < 0.55


def test_relabel_homogeneous_forces_damage_gate_pass_on_cross_boundary_tile():
    bl = load_section_baseline(REF_DIR / "B1_baseline.npz")
    high_tile = str(bl.tile_table[bl.tile_table["category"] == "HIGH"]["tile_id"].iloc[0])
    relabeled = apply_reference_intervention(bl, high_tile, "homogeneous")
    members = bl.members_by_tile[int(high_tile[1:])]
    assert len(set(relabeled[members].tolist())) == 1

    # Boost Q in that high-departure boundary tile so it clears contrast/agreement gates
    rng = np.random.default_rng(7)
    x_q = bl.q0.copy() + 0.2 * rng.standard_normal(bl.q0.shape)
    x_q[members, :4] += 4.0
    i_case = bl.i_background.copy()
    i_case[members] = "I0"

    r_native = evaluate_multigate_prototype(bl, x_q, i_case, config=GateConfig(n_null=39, n_bound_boot=39))
    r_no_dmg = evaluate_multigate_prototype(
        bl, x_q, i_case, config=GateConfig(n_null=39, n_bound_boot=39, disable_damage_gate=True)
    )
    r_relabel = evaluate_multigate_prototype(
        bl, x_q, i_case, reference_labels=relabeled, config=GateConfig(n_null=39, n_bound_boot=39)
    )
    assert high_tile not in r_native["accepted_ids"]
    assert high_tile in r_no_dmg["accepted_ids"]
    assert high_tile in r_relabel["accepted_ids"]


def test_calibrated_dual_gate_respects_spot_cap():
    df = pd.DataFrame(
        {
            "candidate_id": ["T0000", "T0001", "T0002"],
            "n_points": [20, 20, 20],
            "scorable": [True, True, True],
            "min": [1.5, 1.2, 0.9],
        }
    )
    kept = evaluate_dual_calibrated_gate(df, None, q_min=0.8, n_spots=600, spot_cap_fraction=0.05)
    # 5% of 600 = 30 spots -> only top 20-spot tile fits
    assert kept == ["T0000"]
