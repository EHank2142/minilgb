"""Shared helpers: data loading, running our implementation and the official libraries
with identical per-iteration bookkeeping (training time excludes evaluation)."""
import json
import os
import sys
import time

THREADS = int(os.environ.get("REPRO_THREADS", "6"))
os.environ.setdefault("NUMBA_NUM_THREADS", str(THREADS))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minilgb import GBDT, BinnedDataset, auc, ndcg  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# Common hyper-parameters (the values LightGBM's own benchmark page uses for these
# comparisons; the paper defers its settings to the supplement).
COMMON = dict(learning_rate=0.1, num_leaves=255, min_data_in_leaf=0,
              min_sum_hessian_in_leaf=100.0, lambda_l2=0.0)
# GOSS rates of the paper (Sec. 5.1): a=b=0.1 for Flight Delay and LETOR
GOSS = dict(top_rate=0.1, other_rate=0.1)


def load(name):
    if name == "flight":
        Xtr = sp.load_npz(os.path.join(DATA, "flight_Xtr.npz"))
        Xte = sp.load_npz(os.path.join(DATA, "flight_Xte.npz"))
        d = np.load(os.path.join(DATA, "flight_y.npz"))
        return dict(task="binary", Xtr=Xtr, ytr=d["ytr"], qtr=None,
                    evals=[("test", Xte, d["yte"], None)])
    if name == "mslr":
        d = np.load(os.path.join(DATA, "mslr10k_fold1.npz"))
        return dict(task="lambdarank", Xtr=d["X_train"], ytr=d["y_train"], qtr=d["qptr_train"],
                    evals=[("vali", d["X_vali"], d["y_vali"], d["qptr_vali"]),
                           ("test", d["X_test"], d["y_test"], d["qptr_test"])])
    raise ValueError(name)


def metric(task, y, s, q):
    return auc(y, s) if task == "binary" else ndcg(y, s, q, 10)


def save(tag, hist, extra=None):
    out = {"history": hist, **(extra or {})}
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f)


# ----------------------------------------------------------------------------- ours
def run_ours(D, iters, ds, sampling="none", verbose=50, **kw):
    params = {**COMMON, "objective": D["task"], "num_iterations": iters, "sampling": sampling}
    if sampling == "goss":
        params.update(GOSS)
    params.update(kw)
    m = GBDT(**params).fit(None, D["ytr"], qptr=D["qtr"], dataset=ds, eval_sets=D["evals"],
                           verbose=verbose)
    return m.history


# ----------------------------------------------------------------------------- official
def _group_sizes(q):
    return np.diff(q)


def run_lightgbm(D, iters, variant):
    import lightgbm as lgb
    p = dict(objective=D["task"], learning_rate=COMMON["learning_rate"],
             num_leaves=COMMON["num_leaves"], min_data_in_leaf=COMMON["min_data_in_leaf"],
             min_sum_hessian_in_leaf=COMMON["min_sum_hessian_in_leaf"], lambda_l2=0.0,
             max_bin=255, num_threads=THREADS, verbose=-1, seed=0, force_col_wise=True)
    if variant == "lgb_baseline":
        p["enable_bundle"] = False
    elif variant == "EFB_only":
        pass
    elif variant == "LightGBM":
        p.update(data_sample_strategy="goss", **GOSS)
    elif variant == "SGB":
        p.update(bagging_fraction=GOSS["top_rate"] + GOSS["other_rate"], bagging_freq=1)
    dtr = lgb.Dataset(D["Xtr"], D["ytr"], group=None if D["qtr"] is None else _group_sizes(D["qtr"]),
                      params={**p, "feature_pre_filter": False})
    t0 = time.perf_counter()
    dtr.construct()
    construct = time.perf_counter() - t0
    stamps = []
    cb = lambda env: stamps.append(time.perf_counter())  # noqa: E731
    t0 = time.perf_counter()
    bst = lgb.train(p, dtr, iters, callbacks=[cb])
    times = np.array(stamps) - t0
    hist = {"iter": list(range(1, iters + 1)), "time": times.tolist(),
            "iter_time": np.diff(np.concatenate([[0], times])).tolist()}
    for name, Xe, ye, qe in D["evals"]:
        s = np.zeros(Xe.shape[0])
        vals = []
        for i in range(iters):  # incremental scores, one tree at a time
            s += bst.predict(Xe, start_iteration=i, num_iteration=1, raw_score=True)
            vals.append(metric(D["task"], ye, s, qe))
        hist[name] = vals
    return hist, {"construct_time": construct}


def run_xgboost(D, iters, variant):
    import xgboost as xgb
    p = dict(eta=COMMON["learning_rate"], min_child_weight=COMMON["min_sum_hessian_in_leaf"],
             reg_lambda=0.0, nthread=THREADS, seed=0)
    p["objective"] = "binary:logistic" if D["task"] == "binary" else "rank:ndcg"
    if variant == "xgb_exa":  # pre-sorted algorithm, depth-wise only
        p.update(tree_method="exact", max_depth=8)
    else:  # histogram algorithm with leaf-wise (loss-guided) growth
        p.update(tree_method="hist", grow_policy="lossguide", max_leaves=COMMON["num_leaves"],
                 max_depth=0, max_bin=255)
    kw = {} if D["qtr"] is None else {"qid": np.repeat(np.arange(len(D["qtr"]) - 1),
                                                       _group_sizes(D["qtr"]))}
    dtr = xgb.DMatrix(D["Xtr"], D["ytr"], **kw)

    class CB(xgb.callback.TrainingCallback):
        def __init__(self):
            self.stamps = []

        def after_iteration(self, model, epoch, evals_log):
            self.stamps.append(time.perf_counter())
            return False

    cb = CB()
    t0 = time.perf_counter()
    bst = xgb.train(p, dtr, iters, callbacks=[cb])
    times = np.array(cb.stamps) - t0
    hist = {"iter": list(range(1, len(times) + 1)), "time": times.tolist(),
            "iter_time": np.diff(np.concatenate([[0], times])).tolist()}
    for name, Xe, ye, qe in D["evals"]:
        de = xgb.DMatrix(Xe)
        vals = []
        for i in range(1, len(times) + 1):  # evaluated every 10 iterations (NaN otherwise)
            if i % 10 == 0 or i == 1 or i == len(times):
                s = bst.predict(de, iteration_range=(0, i), output_margin=True)
                vals.append(metric(D["task"], ye, s, qe))
            else:
                vals.append(float("nan"))
        hist[name] = vals
    return hist, {}
