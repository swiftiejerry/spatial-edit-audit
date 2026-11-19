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

## 3. Calibration Balance and Pool-Size Sensitivity

In the primary Block-1 protocol, uniform random draws across `{N1, N2, N3}` produced `40.2% N1` (`80/199`), `33.7% N2` (`67/199`), and `26.1% N3` (`52/199`) on `B4`, slightly depressing `q_{B4,min}` (`0.6570`) near `B4`'s native `N1` baseline upper shoulder (`p95 = 0.6532`). Unlike `P`, `Q`, and `mean`—whose Block-1 thresholds are elevated by `delta=3.0` unilateral decoys in `N2`/`N3` so `N1` draws produce `0/80` false accepts—conjunctive `min` suppresses unilateral decoys, leaving `N1` and `N2`/`N3` null maxima closely aligned. Running `scripts/run_calibration_ablation.py` verifies that restoring exact `1:1:1` calibration stratification (`n_cal = 198`) and scaling calibration size (`n_cal = 999`, `k = 950`) reduce pooled `cal_min` false acceptance to `7/240` (`2.9%`, `B4 = 1/60`) and `11/240` (`4.6%`, `B4 = 2/60`), respectively.
