# LightGBM (NeurIPS 2017): Paper Study and Critical Analysis

> G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu.
> *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017, pp. 3146–3154.

This is the written study part of the project. The numerical reproduction is in the
[README](../README.md) and [results/RESULTS.md](../results/RESULTS.md).

---

## 1. Problem formulation and motivation

**Setting.** Gradient Boosting Decision Trees (GBDT; Friedman 2001) fit an additive model
F_T(x) = Σ_t f_t(x). Tree f_t is fit to the negative gradients
g_i = −∂ℓ(y_i, F(x_i))/∂F(x_i) of the loss at the current model. When the loss is
second-order (Newton) boosting, as in XGBoost and LightGBM, the tree also uses the hessians h_i.
Each node is split on the feature j and threshold d that maximise the *variance gain*
(Def. 3.1):

V_{j|O}(d) = (1/n_O) · [ (Σ_{x_ij ≤ d} g_i)² / n_l(d) + (Σ_{x_ij > d} g_i)² / n_r(d) ].

**Bottleneck.** With the histogram algorithm (Alg. 1), each feature is bucketed into ≤ 255
bins. Split finding then costs O(#bin × #feature), but *building the histograms* costs
**O(#data × #feature)** per tree level, and this term dominates. For data sets with 10⁷–10⁸ rows
and up to 10⁷ (sparse) features, existing systems such as XGBoost and pGBRT were slow, or ran
out of memory on 256 GB.

**Goal.** Reduce **#data** and **#feature** in the histogram construction *without* changing
the learned trees much. The proposed means are:

| Problem | Proposed fix | Idea |
|---|---|---|
| too many instances | **GOSS** – Gradient-based One-Side Sampling | an instance with a small |g_i| is already well fitted, so keep all large-gradient instances and sub-sample the rest |
| too many features | **EFB** – Exclusive Feature Bundling | in sparse data many features are (almost) never non-zero together, so merge them into one "bundle" feature |

## 2. Methods

### 2.1 GOSS (Sec. 3, Alg. 2)
1. Sort the instances by |g_i|. Keep the top a·100 % (set A).
2. From the remaining instances, draw b·100 % uniformly at random (set B).
3. Multiply the gradients (and hessians) of B by the constant **(1−a)/b**. This makes the sum over
   B an unbiased estimate of the sum over Aᶜ.
4. Learn the tree on A ∪ B with the estimated gain (Eq. 1):

   Ṽ_j(d) = (1/n) [ (Σ_{A_l} g_i + (1−a)/b · Σ_{B_l} g_i)² / n_l(d) + (same for the right side) ].

**Theorem 3.2.** With probability ≥ 1−δ,

|Ṽ_j(d) − V_j(d)| ≤ C²_{a,b} ln(1/δ) · max{1/n_l, 1/n_r} + 2·D·C_{a,b} √(ln(1/δ)/n),

where C_{a,b} = (1−a)/√b · max_{Aᶜ}|g_i|. The error is therefore O(1/n_l + 1/n_r + 1/√n), which
tends to 0 for balanced splits. Random sampling is the special case a = 0. GOSS beats it when
C_{0,β} > C_{a,β−a}, i.e. when the largest gradients are much larger than those in Aᶜ.

### 2.2 EFB (Sec. 4, Alg. 3–4)
* **Which features to bundle?** Build a conflict graph: vertices are features, and an edge weight is
  the number of rows on which both features are non-zero. Partitioning the features into the fewest
  exclusive bundles is graph colouring, which is NP-hard (Thm 4.1). The paper uses a **greedy**
  heuristic. It orders features by degree, or, more cheaply, by non-zero count. It then assigns each
  feature to the first bundle whose total conflict stays ≤ K = γ·n, and opens a new bundle
  otherwise.
* **How to merge?** Shift the bins of each feature by an offset so that the features occupy disjoint
  bin ranges of the bundle (Alg. 4). Histograms of the bundle contain the histograms of every
  member feature, so split finding over the *original* features is unchanged when γ = 0.
* Cost: histogram construction drops from O(#data × #feature) to O(#data × #bundle).

### 2.3 Remaining system
LightGBM = histogram algorithm + leaf-wise (best-first) growth with a leaf budget + histogram
subtraction (sibling = parent − smaller child) + GOSS + EFB. The paper's "lgb_baseline" already
contains a sparse-feature optimisation: per-feature tables of non-zero rows.

## 3. Main contributions (as claimed)
1. GOSS, a gradient-aware sampling scheme for GBDT with an approximation guarantee (Thm 3.2).
2. EFB, a near-lossless dimensionality reduction for sparse features, with a hardness result
   (Thm 4.1) and a greedy algorithm.
3. An open-source system. The paper reports it is up to 20× faster than the baseline and 9×
   faster than xgb_his at the same accuracy, on 5 public datasets.

## 4. Connection to course concepts (COMP3314)

| Course concept | Where it appears in the paper |
|---|---|
| **Decision trees, impurity, information gain** | The variance gain (Def. 3.1) is the regression-tree analogue of information gain or Gini decrease. Leaf-wise growth is greedy best-first splitting. Histograms are a discretisation of continuous features. |
| **Ensemble learning: boosting vs. bagging** | GBDT is boosting (sequential, bias-reducing). Friedman's SGB and LightGBM's `bagging_fraction` borrow the random row sub-sampling of bagging and random forests. |
| **AdaBoost sample weights** | GOSS is motivated by AdaBoost. AdaBoost has explicit instance weights, which GBDT lacks, so GOSS uses |g_i| as an implicit weight. The (1−a)/b factor is importance weighting: an unbiased estimator of a sum. |
| **Gradient descent / Newton's method** | Each tree is a functional gradient step. Leaf values are Newton steps −G/(H+λ). |
| **Bias–variance trade-off** | Sampling (GOSS, SGB) adds variance to the split estimates but also diversifies the trees, which can improve generalisation (paper Sec. 3.2). |
| **Dimensionality reduction / feature selection** | The paper contrasts EFB with PCA and feature filtering. EFB loses no information when γ = 0, because it merges rather than removes. |
| **Concentration inequalities / generalisation** | Thm 3.2 is a Bernstein / Hoeffding-type bound. The generalisation-error decomposition E_gen ≤ E_GOSS + E_gen(full data) follows the usual triangle-inequality argument. |
| **Complexity, NP-hardness, greedy approximation** | Thm 4.1 reduces graph colouring to bundling. |
| **Evaluation metrics** | ROC-AUC for imbalanced binary classification (Flight Delay is 19 % positive). NDCG@10 for ranking. |

## 5. Experimental evidence in the paper
* **Datasets:** Allstate (12 M × 4228, sparse), Flight Delay (10 M × 700, sparse), LETOR
  (MSLR-WEB30K, 2 M × 136, dense), KDD10 (19 M × 29 M) and KDD12 (119 M × 54 M).
* **Table 2 (time per iteration):** LightGBM is 21×, 6×, 1.6×, 14× and 13× faster than
  lgb_baseline. EFB accounts for most of the gain on sparse data, and GOSS adds about 1.5–2× more.
  The gain is smallest on dense LETOR, where EFB cannot bundle anything.
* **Table 3 (accuracy):** LightGBM is within ±0.0003 of lgb_baseline everywhere, while SGB at the
  same sampling ratio loses 0.003–0.008 AUC/NDCG.
* **Table 4:** at every sampling ratio from 0.1 to 0.4, GOSS beats SGB on LETOR by about
  0.003–0.004 NDCG@10.
* **Fig. 1–2:** time–accuracy curves. LightGBM reaches every accuracy level first.

The evidence for "speed-up at no accuracy cost" is direct (Tables 2+3). The evidence for
"GOSS > random sampling" rests on one dataset (Table 4), plus SGB's columns in Table 3.

## 6. Critical analysis

### 6.1 Strengths
* **Right bottleneck.** The paper identifies the actual cost driver, histogram construction at
  O(#data × #feature), and attacks *both* of its factors with simple, orthogonal ideas.
* **EFB is exact when γ = 0** and needs no tuning on one-hot data. It is a principled way to
  exploit sparsity in the histogram algorithm, which previously had no good sparse optimisation.
* **GOSS is a one-line change** to the training loop, works with any differentiable loss, and has a
  theoretical justification. It is now a standard option (`data_sample_strategy=goss`).
* **Strong empirical scale:** datasets up to 119 M rows and 54 M features, with the code released.

### 6.2 Weaknesses, gaps and inconsistencies found while re-implementing
1. **Inconsistent definition of b.** Sec. 3.2 says B has size b × |Aᶜ| = b(1−a)n, but Alg. 2 uses
   randN = b × len(I) = bn. The normaliser (1−a)/b is unbiased only under Alg. 2's definition.
   Under the Sec. 3.2 definition the correct factor is 1/b. The official code follows Alg. 2, and
   so does ours.
2. **"Constant approximation ratio" for greedy colouring is not true in general.** Graph colouring
   cannot be approximated within n^{1−ε} unless P = NP (Zuckerman 2007). Greedy colouring can use
   Θ(n) colours on 2-colourable graphs if the order is unlucky. What *is* true is the bound
   ≤ Δ+1 colours (max degree + 1). The paper gives no quality guarantee for the bundling. Our
   synthetic experiment shows the greedy still works well on one-hot structure.
3. **Typo in Alg. 3:** `bundlesConflict[i]` should be `bundlesConflict[j]`.
4. **Thm 3.2 bounds the error of one fixed split (j, d).** The tree picks the arg-max over all
   features and thresholds, which is the *same sample* reused for many comparisons. A uniform
   bound would need a union bound that adds a log(#feature × #bin) factor, and this is not
   discussed. Eq. (1) also divides by n_l(d) and n_r(d), the *full-data* counts, which the
   sampled algorithm does not have. Implementations, including ours and the official one, use the
   re-weighted sample counts or hessians instead, which adds further estimation error.
5. **GOSS details that matter in practice are not in the paper.** The official implementation:
   * ranks instances by |g·h| rather than |g|;
   * skips GOSS for the first 1/learning_rate iterations;
   * uses the sampled data for the leaf values too.

   The warm-up matters. Early in training every gradient is large: for logloss at the prior
   probability p, |g| takes only the two values p and 1−p. The "top-a" set is then simply a
   random subset of one class. We observed this as a slower start of our no-warm-up GOSS (see
   README).
6. **Evaluation protocol.** Table 3 reports "the iteration with the best score" *on the test set*.
   That is model selection on test data, and it biases every method upward. It probably does not
   change the ranking between methods, because the bias is similar for all of them, but a
   validation split would be cleaner. Our reproduction reports both protocols for LETOR.
7. **Variance and statistical significance.** Only LightGBM and SGB report ± standard deviations.
   The differences claimed in Table 4 (about 0.003 NDCG) come from single runs per ratio, and the
   (a, b) settings are hidden in a supplement.
8. **Fairness of the speed baselines.** xgb_exa is restricted to depth-wise growth with
   "tuned parameters". lgb_baseline is the authors' own code without the new techniques. The
   speed-ups therefore mix algorithmic gains with engineering such as cache locality and memory
   layout. The paper itself acknowledges that the EFB speed-up partly comes from better cache
   hits (Sec. 5.3). Our from-scratch implementation separates the two effects (README §5).
9. **Limited scope of EFB.** On dense data (LETOR) EFB gives almost nothing (0.49 → 0.46 s in
   Table 2), and with GOSS the gain is only 1.6×. The paper lists improving EFB for non-sparse
   features as future work.

### 6.3 Possible improvements and extensions
* **Adaptive (a, b)** chosen from the gradient distribution, e.g. from the ratio α_a in the
  condition below Thm 3.2, or by keeping a fixed fraction of the total |g| mass instead of a fixed
  fraction of instances. The authors list the "optimal selection of a and b" as future work.
* **Variance-reduced GOSS:** importance sampling with probability ∝ |g_i| (or |g_i|·h_i) over
  *all* instances, as in later work such as MVS (Ibragimov & Gusev, NeurIPS 2019). This replaces
  the hard top-a cut-off and is provably minimum-variance.
* **Bundling with better colouring heuristics** such as DSATUR or Welsh–Powell with conflict
  weights, plus feature-value-aware bundling for dense features with a dominant value. The latter
  is what LightGBM later did with "most-frequent-bin" bundling.
* **Report results with seeds and confidence intervals**, and select the iteration on a validation
  set.

## 7. Summary of what our reproduction confirms or questions
This section summarises the reproduction at reduced scale. The tables and figures are in README §5.

**Confirmed**
* **EFB.**
  * It is lossless at γ = 0: our implementation learns bit-identical trees with and without it.
  * It removes most of the cost of sparse features:
    * Flight: 657 features become 27 bundles, 2.7× faster (ours) and 9.6× faster (official);
    * synthetic KDD-like data: about 68k features become 345 bundles.
  * It does nothing on dense LETOR, which agrees with the paper's Table 2.
  * As predicted, γ trades speed for accuracy, and γ ≤ 10⁻³ is almost free.
* **GOSS vs SGB.** At every sampling ratio from 0.1 to 0.4, GOSS gives higher NDCG@10 than SGB
  (Table 4), in our implementation and in the official one. The ablation shows that the (1−a)/b
  amplification is necessary: without it, GOSS is no better than SGB.
* **Theorem 3.2.** The estimation error of the split gain decays as 1/√n.

**Questioned**
* **"GOSS does not hurt accuracy."** At 1M rows, both implementations lose about 0.01 AUC on
  Flight and about 0.004 NDCG on LETOR. The guarantee is asymptotic in n, so this claim depends
  on the data being large, and the paper does not say so. The official library's undocumented
  warm-up phase is a practical patch for exactly this problem.
* **The theoretical advantage of GOSS over random sampling.** On real gradients, the
  single-split estimation error of GOSS at an equal ratio is no smaller than that of uniform
  sampling. Late in training, the relative error is O(1) for both. The empirical benefit of
  GOSS therefore seems to come from where it puts its sample over the whole run, more than from
  the error bound the paper proves.
* **Speed-ups depend on the implementation.**
  * A modern C++ baseline without GOSS and EFB is already much faster than the paper's
    lgb_baseline.
  * Our Python/Numba version obtains smaller GOSS gains, because the costs of sampling and
    partitioning, which the paper does not discuss, become a large share of the time per
    iteration.
  * The size of the reported speed-ups (Table 2) therefore depends strongly on engineering
    details that the paper leaves out.
