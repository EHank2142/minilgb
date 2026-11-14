"""Empirical check of Theorem 3.2 (GOSS approximation error of the variance gain).

We take *real* GBDT gradients (Flight Delay, after k boosting iterations of a model trained
with our implementation), and for every candidate split point d of a set of features compute
  V_j(d)        exact variance gain (Def. 3.1) on all n rows
  ~V_j(d)       GOSS estimate (Eq. 1) with top-a and b-sampling, weight (1-a)/b
  V_rand(d)     random-sampling estimate with the same sampling ratio a+b (= SGB, a=0)
and report  (i) mean |~V - V| / max_d V  and (ii) how often the argmax split is unchanged,
over many random draws, for several (a, b) and several n (to see the O(1/sqrt n) decay).

  python experiments/exp_goss_theory.py
"""
import json
import os

import numpy as np

from common import COMMON, GBDT, BinnedDataset, RES, load
from minilgb.dataset import _Columns

rng = np.random.default_rng(0)
D = load("flight")
X, y = D["Xtr"], D["ytr"].astype(np.float64)
ds = BinnedDataset(X, efb=False, verbose=False)

# features to evaluate: the numeric ones (DepTime, Distance) + the 20 densest one-hot columns
nnz = np.diff(X.tocsc().indptr)
feats = list(range(2)) + list(np.argsort(-nnz[2:])[:20] + 2)
cols = _Columns(X)
bins = {j: BinnedDataset._bin_column(cols, j, ds.bounds[j]) for j in feats}


def gradients(k):
    """negative gradients after k iterations of plain GBDT."""
    m = GBDT(**{**COMMON, "num_iterations": k}).fit(None, y, dataset=ds, verbose=0) if k else None
    s = m.train_score if m else np.full(len(y), np.log(y.mean() / (1 - y.mean())))
    p = 1 / (1 + np.exp(-s))
    return -(p - y)


def gains(g, w, rows, b, nb):
    """Variance-gain *improvement* V(d) - V(parent) of Def. 3.1 / Eq. (1) for every threshold
    of one feature, from weighted sums of g over the (sub)sample. Counts n_l, n_r are estimated
    by the weighted counts (this is what an implementation can compute; LightGBM does the same).
    Returns (delta, exact-free count of left side)."""
    sg = np.bincount(b[rows], weights=g[rows] * w, minlength=nb)
    sc = np.bincount(b[rows], weights=w, minlength=nb)
    lg, lc = np.cumsum(sg)[:-1], np.cumsum(sc)[:-1]
    G, C = sg.sum(), sc.sum()
    rg, rc = G - lg, C - lc
    v = np.full(nb - 1, -np.inf)
    ok = (lc > 0) & (rc > 0)
    v[ok] = (lg[ok] ** 2 / lc[ok] + rg[ok] ** 2 / rc[ok] - G * G / C) / C
    return v, lc


def experiment(g, n, settings, trials=20, min_frac=0.01):
    idx_all = rng.choice(len(g), n, replace=False) if n < len(g) else np.arange(len(g))
    gs = g[idx_all]
    fb = {j: bins[j][idx_all] for j in feats}
    exact, valid = {}, {}
    for j in feats:
        v, lc = gains(gs, np.ones(n), np.arange(n), fb[j], ds.num_bins[j])
        # thresholds with both children >= 1% of the node (Theorem 3.2 needs balanced splits)
        valid[j] = (lc >= min_frac * n) & (n - lc >= min_frac * n)
        exact[j] = np.where(valid[j], v, -np.inf)
    vbest = max(exact[j].max() for j in feats)
    order = np.argsort(-np.abs(gs), kind="stable")
    out = {}
    for name, a, b in settings:
        errs, regrets = [], []
        top_n, rand_n = int(a * n), int(b * n)
        top, rest = order[:top_n], order[top_n:]
        for _ in range(trials):
            rnd = rng.choice(rest, rand_n, replace=False)
            rows = np.concatenate([top, rnd])
            # a = 0 is plain random sampling (weight 1/b, which cancels in the argmax)
            w = np.concatenate([np.ones(top_n), np.full(rand_n, (1 - a) / b)])
            e, best_est, choice = [], -np.inf, None
            for j in feats:
                v, _ = gains(gs, w, rows, fb[j], ds.num_bins[j])
                m = valid[j]
                if not m.any():
                    continue
                e.append(np.abs(v[m] - exact[j][m]).mean() / vbest)
                vm = np.where(m, v, -np.inf)
                d = int(np.argmax(vm))
                if vm[d] > best_est:
                    best_est, choice = vm[d], (j, d)
            errs.append(np.mean(e))
            regrets.append((vbest - exact[choice[0]][choice[1]]) / vbest)
        out[name] = {"rel_err": float(np.mean(errs)), "rel_err_std": float(np.std(errs)),
                     "regret": float(np.mean(regrets)), "regret_std": float(np.std(regrets))}
    return out


results = {}
for k in (0, 10, 50):
    g = gradients(k)
    for n in (10_000, 30_000, 100_000, 300_000, 1_000_000):
        # same total sampling ratio 0.2:  GOSS (a=b=0.1), GOSS(a=0.05,b=0.15), random (a=0,b=0.2)
        r = experiment(g, n, [("GOSS a=0.1 b=0.1", 0.1, 0.1),
                              ("GOSS a=0.05 b=0.15", 0.05, 0.15),
                              ("Random b=0.2", 0.0, 0.2)], trials=20 if n < 1_000_000 else 10)
        results[f"iter{k}_n{n}"] = r
        print(f"iter={k:3d} n={n:8d} " + "  ".join(f"{m}: err={v['rel_err']:.4f} regret={v['regret']:.4f}"
                                                for m, v in r.items()), flush=True)

with open(os.path.join(RES, "goss_theory.json"), "w") as f:
    json.dump(results, f, indent=1)
