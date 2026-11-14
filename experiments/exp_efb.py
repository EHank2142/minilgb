"""EFB analysis on a KDD-Cup-like synthetic high-dimensional sparse dataset.

The KDD10/KDD12 datasets of the paper (29M / 54M features, 19M / 119M rows) do not fit in
16 GB, so we generate data with the same structure: many high-cardinality categorical
fields (one-hot, mutually exclusive inside a field, Zipf popularity) plus many rare sparse
numeric "count" features that are *almost* exclusive (they motivate gamma > 0).

  (a) baseline vs EFB (gamma = 0): #groups, time/iteration, test AUC  [ours + official]
  (b) bundling order 'degree' (Alg. 3 as written, needs the conflict graph) vs 'count'
  (c) gamma sweep: #bundles, conflicts, time/iteration, AUC  (supplementary study of gamma)

  python experiments/exp_efb.py
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from common import COMMON, GBDT, RES, THREADS, BinnedDataset, auc

N_TRAIN, N_TEST = 300_000, 100_000
CARDS = [10, 50, 200, 1000, 5000, 10000, 20000, 30000]
N_RARE, P_RARE = 2000, 0.002
ITERS, LEAVES = 100, 63


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(12345)  # same ground-truth model for train and test
    logit = np.zeros(n)
    blocks = []
    codes_all = []
    for card in CARDS:
        pop = 1.0 / np.arange(1, card + 1) ** 1.1
        pop /= pop.sum()
        codes = rng.choice(card, n, p=pop)
        codes_all.append(codes)
        eff = wrng.normal(0, 0.5, card)
        logit += eff[codes]
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, card)))
    # an interaction between the two smallest fields (so that depth matters)
    inter = wrng.normal(0, 0.7, (CARDS[0], CARDS[1]))
    logit += inter[codes_all[0], codes_all[1]]
    # rare count features: non-zero with prob P_RARE, independent -> occasional conflicts
    mask = rng.random((n, N_RARE), dtype=np.float32) < P_RARE
    r, c = np.nonzero(mask)
    vals = rng.poisson(3, len(r)) + 1.0
    rare = sp.csr_matrix((vals, (r, c)), shape=(n, N_RARE))
    wr = wrng.normal(0, 0.3, N_RARE)
    logit += rare @ wr / 3
    dense = rng.normal(size=(n, 4))
    logit += dense[:, 0] - 0.5 * dense[:, 1] ** 2
    X = sp.hstack([sp.csr_matrix(dense)] + blocks + [rare], format="csr")
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit - logit.mean())))).astype(np.float64)
    return X, y


def run_ours(ds, Xte, yte, iters=ITERS):
    m = GBDT(**{**COMMON, "num_iterations": iters, "num_leaves": LEAVES}).fit(
        None, ytr, dataset=ds, verbose=0)
    t = np.array(m.history["iter_time"])
    return float(np.mean(t[1:])), auc(yte, m.predict_raw(Xte))


def run_official(Xtr, ytr, Xte, yte, enable_bundle, gamma=0.0, iters=ITERS):
    import lightgbm as lgb
    p = dict(objective="binary", learning_rate=0.1, num_leaves=LEAVES, min_data_in_leaf=0,
             min_sum_hessian_in_leaf=100, num_threads=THREADS, verbose=-1,
             enable_bundle=enable_bundle, max_conflict_rate=gamma, feature_pre_filter=False,
             force_col_wise=True)
    d = lgb.Dataset(Xtr, ytr, params=p)
    t0 = time.perf_counter(); d.construct(); ct = time.perf_counter() - t0
    st = []
    t0 = time.perf_counter()
    b = lgb.train(p, d, iters, callbacks=[lambda env: st.append(time.perf_counter())])
    it = np.diff(np.array([t0] + st))
    return float(np.mean(it[1:])), auc(yte, b.predict(Xte, raw_score=True)), ct


t0 = time.time()
Xtr, ytr = make_data(N_TRAIN, 1)
Xte, yte = make_data(N_TEST, 2)
print(f"data {Xtr.shape}, nnz/row={Xtr.nnz / Xtr.shape[0]:.1f}, pos={ytr.mean():.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)
res = {"shape": list(Xtr.shape), "nnz_per_row": Xtr.nnz / Xtr.shape[0]}

# (a) baseline vs EFB
ds_b = BinnedDataset(Xtr, efb=False)
tb, ab = run_ours(ds_b, Xte, yte)
print(f"ours baseline : groups={len(ds_b.groups)}  {tb:.3f}s/iter  AUC={ab:.5f}", flush=True)
ds_e = BinnedDataset(Xtr, efb=True, gamma=0.0, bundle_order="count")
te, ae = run_ours(ds_e, Xte, yte)
print(f"ours EFB      : groups={len(ds_e.groups)}  {te:.3f}s/iter  AUC={ae:.5f}", flush=True)
res["ours"] = {"baseline": {"groups": len(ds_b.groups), "iter_time": tb, "auc": ab},
               "efb": {"groups": len(ds_e.groups), "iter_time": te, "auc": ae,
                       "bundle_time": ds_e.bundle_time, "conflicts": ds_e.total_conflicts}}
del ds_b
ob = run_official(Xtr, ytr, Xte, yte, False)
oe = run_official(Xtr, ytr, Xte, yte, True)
print(f"official baseline {ob[0]:.3f}s/iter AUC={ob[1]:.5f} | EFB {oe[0]:.3f}s/iter AUC={oe[1]:.5f}",
      flush=True)
res["official"] = {"baseline": {"iter_time": ob[0], "auc": ob[1], "construct": ob[2]},
                   "efb": {"iter_time": oe[0], "auc": oe[1], "construct": oe[2]}}

# (b) ordering strategies
res["order"] = {}
for order in ("degree", "count"):
    ds = BinnedDataset(Xtr, efb=True, gamma=0.0, bundle_order=order, verbose=False)
    res["order"][order] = {"groups": len(ds.groups), "bundle_time": ds.bundle_time}
    print(f"order={order:6s}: groups={len(ds.groups)} bundling time {ds.bundle_time:.2f}s",
          flush=True)

# (c) gamma sweep
res["gamma"] = []
for gamma in (0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2):
    ds = BinnedDataset(Xtr, efb=True, gamma=gamma, verbose=False)
    t, a = run_ours(ds, Xte, yte)
    o = run_official(Xtr, ytr, Xte, yte, True, gamma)
    row = {"gamma": gamma, "groups": len(ds.groups), "conflict_rows": ds.total_conflicts,
           "iter_time": t, "auc": a, "official_iter_time": o[0], "official_auc": o[1]}
    res["gamma"].append(row)
    print(f"gamma={gamma:<6} groups={len(ds.groups):5d} conflicts={ds.total_conflicts:7d} "
          f"ours {t:.3f}s/iter AUC={a:.5f} | official {o[0]:.3f}s/iter AUC={o[1]:.5f}", flush=True)

with open(os.path.join(RES, "efb_synthetic.json"), "w") as f:
    json.dump(res, f, indent=1)
