# spatial-edit-audit

Controlled simulation, calibration, and counterfactual auditing suite for evaluating selective local spatial multi-omics edits.

## Overview

`spatial-edit-audit` provides a deterministic, annotation-blind testbed built on four public spatial multi-omics tissue sections (`B1` Human Lymph Node RNA+ADT, `B2` Human Tonsil RNA+ADT, `B3` Mouse Embryonic Brain E15.5 RNA+ATAC, and `B4` Mouse Embryonic Brain E18.5 RNA+ATAC; `11,888` spots partitioned into `593` compact contiguous candidate tiles, of which `591` are scorable outside two two-hop-isolated `B3` tiles). It systematically isolates four recurring failure modes when local spatial edits transition from single-modality heuristics to selective multimodal deployment:

1. **Channel aggregation vs. unilateral decoys (`E1`)**: On concordant targets across doses `1.0–4.0` (`N = 320` cases, fixed `3%` top-$k$ tile budget), conjunctive minimum scoring (`min = min(e_P, e_Q)`) improves target hit rate to `0.7344` (`+0.1688` over `Q` alone, `95% CI [0.1281, 0.2094]`; `+0.0594` over `mean`, `95% CI [0.0344, 0.0845]`; `+0.0281` over `P`, `95% CI [0.0062, 0.0531]`) by rejecting single-channel decoys (`Decoy A: 479/480` under `P`, `47/480` under `min`; `Decoy B: 461/480` under `Q`, `16/480` under `min`). Under asymmetric perturbations (`S9: 3.0 vs 1.0`, `N = 80`), `min` drops to `0.2000` (`-0.4375` vs. `mean`, `95% CI [-0.5500, -0.3375]`).
2. **Batch quota coupling (`PH-1`)**: Matching continuous scoring budgets to variable per-case acceptance counts (`k_m0`) forces `81/320` (`25.3%`) zero-budget outcomes across doses `1.0–4.0`, erasing `44/81` (`0.5432`) latent rule hits (`42/81` under `min`, `38/81` under `P`). An uncircular dual-channel calibrated gate (`M_dual_cal`, `min >= q_{b,min}`, circularity-free) lifts `S8` cross-boundary detection (`70/80 = 87.5%` vs. `6/80 = 7.5%`) and holds pooled null acceptance to `19/240` (`7.9%`), yet still forces `145/320` zero-budget ties when continuous rules match `k_cal` post hoc.
3. **Cross-section split-conformal calibration (`E2`)**: Order-statistic cutoffs (`k = 190` of `n_cal = 199` per section, nominal `alpha = 0.05`) yield `19/240` (`7.92%`) pooled null acceptances (`exact binomial p = 0.0336`; finite-sample Beta(`10, 190`)-binomial `p = 0.0543` over the four fixed tissue sections), driven by `B4` (`10/60`, section-level tail `p = 0.0036`) where `N1` baseline maxima compressed `q_{B4,min} = 0.6570` against the `N1` upper shoulder (`p95 = 0.6532`). Balanced `1:1:1` null calibration at `n_cal = 198` (`7/240 = 2.92%`, `B4 = 1/60`) and `n_cal = 999` (`11/240 = 4.58%`, `B4 = 2/60`) restores nominal error control across sections.
4. **Static reference circularity (`M-a..M-d`) & partition geometry**: Paired counterfactual interventions on `S6` and `S8` (`N = 160` records) confirm that conditioning a static upper-damage gate (`damage_UCB <= 0.55`) on fixed reference partition `R` rejects `74/80` cross-boundary `S8` targets natively (`71/74 = 95.9%` failing `damage_upper_bound`), reversible by either removing `R` (`S8: 0.075 -> 0.600`) or replacing `R` with post-edit homogeneous labels (`0.075 -> 0.600`) while collapsing interior `S6` (`0.725 -> 0.000`). In grown-patch localization (`N = 120` Secondary cases, dose `6.0`), unfiltered Ward clustering (`D1`) drops to `0.013` coverage on sparse `B3` (`0.662` pooled vs. `0.380` static tiling `F` and `0.929` `KMeans D2`); filtering sub-5 spot fragments (`D1_filtered`) recovers `0.753` on `B3` and `0.9125` pooled.

## Baseline & Extension Results

| Benchmark / Ablation | Key Verified Output | Script |
| --- | --- | --- |
| Offline reproduction (`1,636` cases) | `VERIFIED` byte-exact parity across Blocks `1–3` | `scripts/verify_frozen_results.py` |
| Balanced null calibration | `n_cal=198 -> 7/240 (2.92%)`; `n_cal=999 -> 11/240 (4.58%)` | `scripts/run_calibration_ablation.py` |
| Secondary partition ablation | Dose `6.0`: `D1_filtered=0.9125` vs `D1=0.6621`, `D2=0.9292`, `F=0.3800`; Dose `3.0`: `D1_filtered=0.2662` vs `D2=0.1575` | `scripts/run_partition_ablation.py` |
| Calibrated dual gate (`M_dual_cal`) | `19/240` null (`B2=3/60`), `70/80` `S8`, `145/320` zero-budget ties | `scripts/run_spatial_baselines.py` |
| Graph diffusion (`alpha = 0.0 -> 0.7`) | `S6 = 1.000`, `S8 = 0.975 -> 0.9625` (no boundary collapse) | `scripts/run_spatial_baselines.py` |
| Neighbor-augmented `KMeans` (`lambda = 0 -> 0.8`) | `S6` F1 `0.2559 -> 0.4029`; `S8` F1 `0.1891 -> 0.2328`; `S9` F1 `0.1742 -> 0.1580` | `scripts/run_spatial_baselines.py` |
| `SpatialGlue` GPU (`150` epochs, `5` reps/cell) | Contrast `3%` hit: `S6=0.85, S8=0.90, S9=0.80`; `SG-KMeans` F1: `0.14–0.23`; `N2` any-decoy: `0.80` | `scripts/run_spatialglue_baseline.py` |
| Raw-count Poisson pipeline (`B1`, `B3`) | `B1`: concordant `min=4/5`, asym. `min=1/5, P=4/5`; `B3` sparse ATAC: concordant `min=1/5, mean=0/5` | `scripts/run_count_level_check.py` |

## Repository Layout

```text
spatial-edit-audit/
├── configs/
│   ├── benchmark_protocol.yaml   # Seed root, latent dimension, preprocessing & block protocols
│   ├── datasets.yaml             # Section specs and bundled NPZ paths
│   └── gate_thresholds.yaml      # Component gates for M0 and calibrated dual comparators
├── data/
│   └── reference/                # Compressed annotation-blind reference bundles (B1–B4, 2.95 MB total)
├── docs/
│   ├── DATA_PROVENANCE.md        # Coordinate units, upstream accessions, and archive provenance gaps
│   ├── FORMULATION_AND_GATES.md  # Full equations (SVD sign rule, MAD/SD fallback, C(A), gamma/lambda, UCB/LCB)
│   └── STATISTICAL_DESIGN.md     # Hierarchical unit taxonomy and Beta-binomial calibration model
├── results/
│   ├── *.csv                     # Self-contained case/tile/counterfactual/partition tables
│   └── *.json                    # Verified JSON summaries for all CLI benchmarks and ablations
├── scripts/
│   ├── build_reference_views.py
│   ├── run_benchmark.py
│   ├── run_calibration_ablation.py
│   ├── run_count_level_check.py
│   ├── run_counterfactual_audit.py
│   ├── run_partition_ablation.py
│   ├── run_spatial_baselines.py
│   ├── run_spatialglue_baseline.py
│   └── verify_frozen_results.py
├── src/spatial_edit_audit/       # Modular Python package (preprocessing, contrast, gates, simulation, baselines)
└── tests/                        # Unit and property tests (pytest)
```

## Installation

Python `>= 3.10` is required. From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional neural graph baselines (`scripts/run_spatialglue_baseline.py`) additionally require PyTorch, PyTorch Geometric, `scanpy`, and the upstream `SpatialGlue` package (`pip install -e .[baselines]`).

## Reproducing Audits (CPU, Offline Reference Bundles)

All core statistical checks, simulation replays, calibration ablations, counterfactual transitions, and CPU baseline comparisons run directly from the bundled compressed reference archives under `data/reference/`:

```bash
# 1. Run unit and contract test suite (12 tests)
pytest -v

# 2. Verify exact reproduction of benchmark metrics, Beta-binomial tails, and S6/S8 flips
python scripts/verify_frozen_results.py

# 3. Re-run calibration ablations (balanced 1:1:1 vs Mondrian)
python scripts/run_calibration_ablation.py

# 4. Re-run Grown-Patch secondary partition ablations (D1 raw vs D1_filtered vs D2 vs F)
python scripts/run_partition_ablation.py

# 5. Re-run dual-calibrated, graph-diffusion, and neighbor-augmented clustering comparators
python scripts/run_spatial_baselines.py

# 6. Full deterministic replay across Blocks 1-3 and counterfactual S6/S8 interventions
python scripts/run_benchmark.py --output-dir results/runs/replay
python scripts/run_counterfactual_audit.py
```

## Optional Raw-Count & Deep-Baseline Extensions

1. Place raw `.h5ad` inputs under `data/raw/{B1,B2,B3,B4}/` (`adata_RNA.h5ad`, `adata_ADT.h5ad`, `adata_ATAC.h5ad`) — see [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for upstream Zenodo/GEO/NGDC accessions — and run:
   ```bash
   python scripts/build_reference_views.py --raw-dir data/raw --out-dir data/reference
   python scripts/run_count_level_check.py --raw-dir data/raw --replicates 5
   ```
2. Run the PyTorch/PyG `SpatialGlue` comparison (`epochs = 150`, CUDA if available):
   ```bash
   python scripts/run_spatialglue_baseline.py --spatialglue-parent /path/to/SpatialGlue_parent --replicates 5
   ```

## Scope and Boundaries

- This toolkit evaluates annotation-blind local statistical detection, split-conformal null calibration, and gate interactions under controlled PCG64 additive and Poisson raw-count perturbations across four fixed spatial multi-omics sections.
- Reported bootstrap intervals measure Monte Carlo resampling variability conditional on these four sections rather than cross-patient biological generalization across external donor cohorts.

## License

Released under the [MIT License](LICENSE).
