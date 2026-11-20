# v1.0.0 — first archived release

Controlled simulation and counterfactual audit toolkit for evaluating selective local spatial multi-omics edits.

## What this release contains

- **Simulation engine** over four public spatial multi-omics sections (11,888 spots; 593 candidate tiles, 591 scorable), three disjoint blocks, 1,636 realizations.
- **Contrast and calibration scorers**: two-hop control-ring contrast, four aggregation rules (`P`, `Q`, `mean`, `min`), fixed-budget selection at 1%/3%/5%, and split-conformal thresholds at rank `k=190` of 199 null maxima.
- **Abstaining prototype** `M0` with its eight component gates, plus the label-free variant `M_dual_cal`.
- **Counterfactual arms** (`no_damage`, `relabel`) for the static-reference circularity audit.
- **External baseline wrappers**: SpatialGlue (optional, upstream package required), BANKSY-style neighbourhood-augmented $k$-means, linear graph diffusion.
- **Sensitivity suites**: raw-count Poisson perturbations, lattice geometry and tie-breaking, disjoint permutation rings, standard-deviation estimator choice, 200-seed calibration Monte Carlo.
- **Reproduction scripts** that rebuild the frozen tables offline from compressed reference baselines.

## Verification at release

- `verify_frozen_results.py` → `VERIFIED`
- `reproduce_frozen_tables.py` → `REPRODUCED_FROM_OFFLINE_NPZ`
- `pytest -v` → 13 passed

## Boundaries

Reported rates are conditional on the four fixed scaffolds and on the shift model; the bootstrap intervals measure Monte Carlo variability within these sections, not cross-donor biological generalization.

## Citation

Archived on Zenodo; see `CITATION.cff` and the DOI badge in the README once the deposit completes.
