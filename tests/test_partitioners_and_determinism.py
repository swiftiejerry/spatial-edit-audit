"""Unit tests for secondary candidate partitioners and PCG64 deterministic simulation streams."""

from pathlib import Path

import numpy as np

from spatial_edit_audit import (
    build_grown_patch_candidates,
    generate_main_case,
    generate_secondary_case,
    load_section_baseline,
)

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"


def test_pcg64_simulation_is_strictly_deterministic():
    bl = load_section_baseline(REF_DIR / "B3_baseline.npz")
    c1 = generate_main_case(bl, block=2, stratum=5, rep=3, fit_i_case=False)
    c2 = generate_main_case(bl, block=2, stratum=5, rep=3, fit_i_case=False)
    np.testing.assert_array_equal(c1["X_P"], c2["X_P"])
    np.testing.assert_array_equal(c1["X_Q"], c2["X_Q"])
    assert c1["positions"] == c2["positions"]


def test_filtered_ward_removes_sub5_fragments_on_sparse_b3():
    bl = load_section_baseline(REF_DIR / "B3_baseline.npz")
    sec = generate_secondary_case(bl, rep=1, block=3, dose=6.0)
    pools = build_grown_patch_candidates(bl, sec["X_P"], include_filtered_ward=True)
    assert any(len(m) < 5 for _, m in pools["D1"])
    assert all(len(m) >= 5 for _, m in pools["D1_filtered"])
    assert len(pools["D1_filtered"]) < len(pools["D1"])
    assert all(len(m) >= 5 for _, m in pools["D2"])
