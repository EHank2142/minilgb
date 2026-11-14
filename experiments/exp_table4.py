"""Table 4: GOSS vs SGB at equal sampling ratio on LETOR (here MSLR-WEB10K Fold1).

Early stopping on the validation set (patience 30); NDCG@10 reported on the test set at the
best validation iteration.  The paper's (a, b) per ratio are in its supplement (not
available to us); we use a = b = ratio / 2.  Also: an ablation of GOSS *without* the
(1-a)/b re-weighting, and the official LightGBM library as a reference.

  python experiments/exp_table4.py
"""
import json
import os
import time

import numpy as np

from common import COMMON, GBDT, RES, THREADS, BinnedDataset, load, ndcg

MAX_IT, PATIENCE = 600, 30
RATIOS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
D = load("mslr")
ds = BinnedDataset(D["Xtr"], efb=True)
vali, test = D["evals"]


def ours(**kw):
    m = GBDT(**{**COMMON, "objective": "lambdarank", "num_iterations": MAX_IT, **kw}).fit(
        None, D["ytr"], qptr=D["qtr"], dataset=ds, eval_sets=[vali, test], verbose=0,
        early_stopping_rounds=PATIENCE)
    b = m.best_iteration
    return {"best_iter": b, "vali": m.history["vali"][b - 1], "test": m.history["test"][b - 1],
            "time": m.history["time"][-1]}


def official(**kw):
    import lightgbm as lgb
    p = dict(objective="lambdarank", learning_rate=0.1, num_leaves=255, min_data_in_leaf=0,
             min_sum_hessian_in_leaf=100, num_threads=THREADS, verbose=-1, seed=0,
             feature_pre_filter=False, eval_at=[10], metric="ndcg", **kw)
    dtr = lgb.Dataset(D["Xtr"], D["ytr"], group=np.diff(D["qtr"]), params=p)
    dva = lgb.Dataset(vali[1], vali[2], group=np.diff(vali[3]), reference=dtr)
    b = lgb.train(p, dtr, MAX_IT, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    s = b.predict(test[1], num_iteration=b.best_iteration, raw_score=True)
    return {"best_iter": b.best_iteration, "test": ndcg(test[2], s, test[3], 10)}


res = {"ours": {"SGB": {}, "GOSS": {}}, "official": {"SGB": {}, "GOSS": {}}, "ablation": {}}
for r in RATIOS:
    t = time.time()
    res["ours"]["SGB"][r] = ours(sampling="sgb", bagging_fraction=r)
    res["ours"]["GOSS"][r] = ours(sampling="goss", top_rate=r / 2, other_rate=r / 2)
    res["official"]["SGB"][r] = official(bagging_fraction=r, bagging_freq=1)
    res["official"]["GOSS"][r] = official(data_sample_strategy="goss", top_rate=r / 2,
                                          other_rate=r / 2)
    print(f"ratio {r:.2f}: ours SGB {res['ours']['SGB'][r]['test']:.4f}  GOSS "
          f"{res['ours']['GOSS'][r]['test']:.4f} | official SGB {res['official']['SGB'][r]['test']:.4f}"
          f"  GOSS {res['official']['GOSS'][r]['test']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    with open(os.path.join(RES, "table4.json"), "w") as f:
        json.dump(res, f, indent=1)

# ablations at ratio 0.2
res["ablation"]["no_sampling"] = ours()
res["ablation"]["GOSS_no_reweight"] = ours(sampling="goss", top_rate=0.1, other_rate=0.1,
                                           goss_reweight=False)
res["ablation"]["GOSS_a0.15_b0.05"] = ours(sampling="goss", top_rate=0.15, other_rate=0.05)
res["ablation"]["GOSS_a0.05_b0.15"] = ours(sampling="goss", top_rate=0.05, other_rate=0.15)
for k, v in res["ablation"].items():
    print(f"ablation {k}: test {v['test']:.4f} (best iter {v['best_iter']})", flush=True)
with open(os.path.join(RES, "table4.json"), "w") as f:
    json.dump(res, f, indent=1)
