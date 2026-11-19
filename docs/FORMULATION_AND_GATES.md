# Mathematical Formulations and Gate Specifications

This document provides complete formal definitions for all scoring rules,
prototype gates, and candidate partitioners implemented in `spatial_edit_audit`.

## 1. Baseline Preprocessing and SVD Sign Alignment

For each raw count matrix $C \in \mathbb{R}^{n_b \times p_{\text{raw}}}$ on section $b$:

1. Retain features with positive counts in at least $\lceil 0.01\,n_b \rceil$ spots.
2. Library-size scale each spot to total sum $10^4$ and apply $\log(1 + x)$.
3. Select features by descending population variance over the log-normalized matrix
   (top $2{,}000$ genes for RNA $P$; all $31$ non-isotype antibodies for ADT $Q$; top
   $5{,}000$ peaks for ATAC $Q$), with ties broken deterministically by original feature index.
4. Center columns and compute rank-16 SVD $Z_{\text{raw}} = U_{:,1:16}\Sigma_{1:16}$.
   For **sign alignment**, locate the maximum-magnitude entry $j^* = \arg\max_j |V^T_{c,j}|$
   in each right singular vector $c \in \{1,\dots,16\}$ and negate column $Z_{:,c}$ if $V^T_{c,j^*} < 0$.
5. Standardize each axis $c$ by its global median and robust scale
   $s_c = 1.4826 \times \operatorname{median}_i |z_{i,c} - \tilde{z}_c|$ into baseline views $P_0, Q_0 \in \mathbb{R}^{n_b \times 16}$.

## 2. Spatial Neighbor Graph and Disjoint Two-Hop Control Ring

1. For Euclidean array coordinates $\mathbf{y}_i \in \mathbb{R}^2$, each spot selects its
   $k=6$ non-self nearest neighbors $N_6(i)$, breaking distance ties lexicographically by
   target spot index. Let $G = A \vee A^T$ be the symmetrized undirected adjacency graph.
2. Candidate tiles $\mathcal{T}_b$ are extracted via `KMeans(n_clusters=round(n_b / 20), n_init=1, random_state=20261111)`
   followed by splitting each spatial cluster into undirected $G$-connected components.
3. For any candidate spot subset $A \subset \{1,\dots,n_b\}$, the **disjoint two-hop control ring** is
   $$C(A) = \bigl(\mathcal{N}_G^{(1)}(A) \cup \mathcal{N}_G^{(2)}(A)\bigr) \setminus A,$$
   where $\mathcal{N}_G^{(1)}(A) = \{v : \exists u \in A,\ (u,v) \in E(G)\}$ and
   $\mathcal{N}_G^{(2)}(A) = \{w : \exists v \in \mathcal{N}_G^{(1)}(A),\ (v,w) \in E(G)\}$.
4. Scorability requires $|C(A)| \ge \max(\lceil \rho_{\min}|A| \rceil, 4)$ with
   `min_control_multiplier` $\rho_{\min} = 1.0$.

## 3. Local Molecular Contrast and Fallback Feature Scale

Given a view matrix $X^{(M)} \in \mathbb{R}^{n_b \times d}$ ($M \in \{P, Q\}$), define the
per-feature scale
$$s^{(M)}_j = \begin{cases}
1.4826\,\mathrm{MAD}_j(X^{(M)}), & \text{if } 1.4826\,\mathrm{MAD}_j > 10^{-12},\\
\mathrm{SD}_j(X^{(M)}), & \text{if } 1.4826\,\mathrm{MAD}_j \le 10^{-12} \text{ and } \mathrm{SD}_j > 10^{-12},
\end{cases}$$
and let $V = \{j \in \{1,\dots,d\} : s^{(M)}_j > 10^{-12}\}$ denote the **valid feature set**.
The standardized two-hop contrast is
$$e_M(A) = \frac{1}{|V|} \sum_{j \in V} \frac{\left|\bar{x}^{(M)}_{A,j} - \bar{x}^{(M)}_{C(A),j}\right|}{s^{(M)}_j}.$$

Four scalar priority aggregations are defined: $P = e_P$, $Q = e_Q$,
$\mathrm{mean} = \frac{1}{2}(e_P + e_Q)$, and $\min = \min(e_P, e_Q)$.

## 4. Eight-Gate Confirmation Prototype ($M_0$)

Prototype $M_0$ evaluates every $G$-connected candidate component $A$ strictly on
confirmation view $X_Q$, per-case clustering $I_{\mathrm{case}}$ (8-cluster `KMeans(n_init=10, random_state=20261111)` fitted on $X_Q$), and static baseline RNA clustering $R_b$ (8-cluster `KMeans` fitted on $P_0$):

1. **Minimum size**: $|A| \ge 5$.
2. **Rare area ceiling**: $|A| / n_b \le 0.15$.
3. **Directed 6-NN spatial coherence**:
   $$\gamma(A) = \frac{\sum_{i \in A} |N_6(i) \cap A|}{6\,|A|} \ge 0.55.$$
4. **Boundary leakage bound**:
   $$\lambda(A) = \frac{\sum_{i \in A} |N_6(i) \setminus A|}{6\,|A|} = 1 - \gamma(A) \le 0.55.$$
   Note: because $\lambda(A) = 1 - \gamma(A)$, enforcing $\gamma(A) \ge 0.55$ algebraically
   implies $\lambda(A) \le 0.45 < 0.55$, making Gate 4 non-binding whenever Gate 3 is checked.
5. **Within-patch cluster agreement**:
   $$\mathrm{agree}(A; I_{\mathrm{case}}) = \frac{\max_{k \in \{0,\dots,7\}} |\{i \in A : I_{\mathrm{case},i} = k\}|}{|A|} \ge 0.60.$$
6. **Local molecular contrast + permutation test on $Q$**:
   - $e_Q(A) \ge 0.75$,
   - fraction of valid features with standardized difference $\ge 0.75$ is $\ge 0.10$,
   - one-sided empirical permutation $p = \frac{1 + \sum_{r=1}^{199} \mathbf{1}[e_Q(D_r) \ge e_Q(A)]}{200} \le 0.05$,
     where each null draw $D_r$ samples $|A|$ spots from control ring $C(A)$ (without replacement when $|C(A)| \ge |A|$).
7. **Bootstrap gain lower confidence bound ($\mathrm{gain}_{\mathrm{LCB}}$)**:
   Draw $B=199$ IID bootstrap resamples $(M_b^*, C_b^*)$ with replacement from $A$ and $C(A)$,
   recompute standardized contrast $e_Q(M_b^*; C_b^*)$ against full-section feature scales $s^{(Q)}$,
   and require quantile $0.025$ to satisfy $\mathrm{gain}_{\mathrm{LCB}}(A) \ge 0.50$.
8. **Bootstrap damage upper confidence bound ($\mathrm{damage}_{\mathrm{UCB}}$)**:
   On the identical $199$ bootstrap spot draws $M_b^*$ inside $A$, evaluate
   $$d_b^* = \min\!\bigl(1,\ \max\{\mathrm{dep}(M_b^*; R_b),\ \lambda(A)\}\bigr),$$
   where $\mathrm{dep}(S; R_b) = 1 - \max_c |\{i \in S : R_{b,i} = c\}| / |S|$ measures
   label heterogeneity against static partition $R_b$ while static boundary leakage $\lambda(A)$
   remains fixed across draws. Gate 8 requires the $0.975$ quantile $\mathrm{damage}_{\mathrm{UCB}}(A) \le 0.55$.
9. **Batch spot cap**: All separately accepted components must together span $\le 0.15\,n_b$ spots; otherwise the entire batch abstains.

## 5. Partitioners in Secondary Grown-Patch Evaluation

For contiguous 20-spot grown patches (`Block 3`, doses $\delta=6.0$ primary and $\delta=3.0$ moderate sensitivity):

- **Fixed tiles (`F`)**: Static coordinate tile pool $\mathcal{T}_b$.
- **Graph Ward clustering (`D1` and `D1_filtered`)**: `AgglomerativeClustering(n_clusters=|\mathcal{T}_b|, linkage="ward", connectivity=G)` on $X_P$. Raw `D1` retains all clusters (`|A| >= 1`); `D1_filtered` filters to `|A| >= 5` matching standard minimum component sizing.
- **KMeans connected components (`D2`)**: 12-cluster `KMeans(n_init=1, random_state=20261111)` on $X_P$, split into undirected $G$-connected components and filtered to $5 \le |A| \le \lfloor 0.05\,n_b \rfloor$.
