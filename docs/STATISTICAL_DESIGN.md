# Statistical Hierarchy and Conformal Calibration Analysis

## 1. Nested Four-Level Unit Taxonomy

Evaluation outputs strictly separate four nested statistical denominators:

| Level | Unit | Count | Inference Boundary |
|---|---|---:|---|
| Level 1 | Physical tissue section (`B1`–`B4`) | `4` | Fixed spatial carriers; equal section weighting; bootstrap resamples cases within slice, never across slices as independent donors. |
| Level 2 | Candidate spatial tile | `593` (`591` scorable) | Static structural profiling (`11–30` spots, median `20`); `2` sparse `B3` edge tiles lack a full two-hop ring and stay in denominators. |
| Level 3 | Simulated realization (`PCG64` seeded) | `1,636` | `796` calibration (`Block 1`) + `720` main (`Block 2`) + `120` secondary (`Block 3`). Null realizations within a section share one cutoff $q_{b,r}$. |
| Level 4 | Derived evaluation record | `160` (`S6`/`S8`) / `1,440` (`Block 3`) | Counterfactual records and `(case x source x rule)` evaluations sharing fixed molecular perturbations per case. |

## 2. Shared-Threshold Conformal Clustering (`BetaBinomial` Model)

Under finite split-conformal calibration over $n_{\mathrm{cal}} = 199$ Block-1 null maxima per section, choosing the order statistic $q_{b,r} = M_{(k)}$ at $k = \lceil (n_{\mathrm{cal}} + 1)(1 - \alpha) \rceil = 190$ controls marginal exceedance at $\le \alpha = 0.05$ when test cases are exchangeable with calibration draws. Conditioned on the calibration threshold $q_{b,r}$, the per-section exceedance probability is
$$\theta_{b,r} \sim \mathrm{Beta}(n_{\mathrm{cal}} + 1 - k,\ k) = \mathrm{Beta}(10, 190).$$
Because all $m = 60$ test null realizations in section $b$ share the same realized cutoff $q_{b,r}$, section exceedance counts follow a correlated $\mathrm{BetaBinomial}(60, 10, 190)$ distribution (mean $3.0$, 95% predictive interval $0\text{--}7$ of $60$). Evaluating total pooled exceedances across four independent sections by exact binomial (`240` independent Bernoulli trials) understates variance (`p = 0.0336` for `19/240`), whereas exact four-slice `BetaBinomial` convolution yields `p = 0.0543` (`> 0.05`), consistent with calibrated nominal 5% control accompanied by one upper-tail section fluctuation (`B4`, `10/60`, section tail `p = 0.0036`; across 16 rule-by-section cells, $\ge 1$ tail hit occurs with probability `0.0556`).

## 3. Calibration Balance, Reweighting, and Multi-Seed Monte Carlo Sensitivity

In the primary Block-1 protocol on `B4`, uniform random draws across `{N1, N2, N3}` produced `40.2% N1` (`80/199`), `33.7% N2` (`67/199`), and `26.1% N3` (`52/199`), yielding `q_{B4,min} = 0.657046` (rank `190` drawn from `N3`). Because `P`, `Q`, and `mean` thresholds are dominated by `delta=3.0` unilateral decoys in `N2`/`N3`, pure-noise `N1` test draws produce `0/80` false accepts under `cal_P/Q/mean`, whereas conjunctive `min` suppresses decoys so `N1`, `N2`, and `N3` maxima share a narrow range (`p95 = 0.6532` vs `0.6460`).

Inverse-probability `1:1:1` reweighting of the primary `B4` calibration draws shifts `q_{B4,min}` minimally from `0.657046` to `0.657233`, leaving the frozen test false-acceptance rate unchanged at `10/60`. Running `scripts/run_calibration_ablation.py` over `200` Monte Carlo seeds evaluated on a disjoint `720`-case independent holdout test bank (`60` per stratum per section) confirms that unstratified `n_cal=199` (`0.0371`, `sd = 0.0057`), balanced `n_cal=199` (`67/66/66`, `0.0369`, `sd = 0.0060`), and balanced `n_cal=219` (`73x3`, `0.0368`, `sd = 0.0066`) agree in mean error rate (`B4` `0.0444–0.0454`), demonstrating that the realized `10/60` `B4` fluctuation reflects finite-sample `Beta(10, 190)` order-statistic variance combined with Block-2 test draws rather than systematic `N1` share bias. Scaling to `n_cal=999` (`333x3`, `k=950`, exact `alpha=0.05`) cuts Monte Carlo standard deviation by more than half (`0.0029` pooled on `720` holdout, `B4 p95 = 0.0500`), while `Mondrian` conditional cuts yield `0.0292` (`199`) and `0.0306` (`999`) pooled (`17/240` on the primary test set).
