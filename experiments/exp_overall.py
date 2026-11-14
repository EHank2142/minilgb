"""Reproduces Table 2 (time / iteration), Table 3 (accuracy) and Fig. 1/2 (time-metric curves).

  python experiments/exp_overall.py flight [--iters 500] [--only ours|official]
  python experiments/exp_overall.py mslr

Methods
  ours_*     : our from-scratch implementation (minilgb)
      ours_lgb_baseline : histogram + leaf-wise, sparse-feature optimisation, no GOSS/EFB
      ours_EFB_only     : + Exclusive Feature Bundling (gamma = 0)
      ours_LightGBM     : + EFB + GOSS (a = b = 0.1)
      ours_SGB          : + EFB + Stochastic Gradient Boosting with ratio a+b
                          (EFB with gamma=0 is exact, so this equals lgb_baseline+SGB)
  official_* : the official LightGBM 4.x library with the same switches
  xgb_exa / xgb_his : XGBoost 3.x exact (pre-sorted, depth-wise) and hist (leaf-wise)
"""
import argparse
import gc
import os
import time

import numpy as np

from common import (RES, BinnedDataset, load, run_lightgbm, run_ours, run_xgboost, save)

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["flight", "mslr"])
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--only", choices=["ours", "official", "xgb", "all"], default="all")
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--skip-done", action="store_true", help="skip methods whose json exists")
args = ap.parse_args()

t = time.time()
D = load(args.dataset)
print(f"loaded {args.dataset} in {time.time()-t:.1f}s; train shape {D['Xtr'].shape}")
it = args.iters


def want(m):
    if args.skip_done and os.path.exists(os.path.join(RES, f"{args.dataset}_{m}.json")):
        return False
    return args.methods is None or m in args.methods


if args.only in ("ours", "all"):
    ds_base = ds_efb = None
    for method, efb, samp in [("ours_lgb_baseline", False, "none"), ("ours_EFB_only", True, "none"),
                              ("ours_LightGBM", True, "goss"), ("ours_SGB", True, "sgb")]:
        if not want(method):
            continue
        if efb and ds_base is not None:  # free memory: the baseline dataset is no longer needed
            ds_base = None
            gc.collect()
        if efb and ds_efb is None:
            ds_efb = BinnedDataset(D["Xtr"], efb=True, gamma=0.0)
        if not efb and ds_base is None:
            ds_base = BinnedDataset(D["Xtr"], efb=False)
        ds = ds_efb if efb else ds_base
        print(f"== {method}")
        hist = run_ours(D, it, ds, samp, bagging_fraction=0.2)
        save(f"{args.dataset}_{method}", hist,
             {"groups": len(ds.groups), "features": int(ds.num_feature),
              "construct_time": ds.construct_time, "bundle_time": ds.bundle_time})
        del ds
        gc.collect()

if args.only in ("official", "all"):
    for v in ["lgb_baseline", "EFB_only", "LightGBM", "SGB"]:
        if not want(f"official_{v}"):
            continue
        print(f"== official_{v}")
        hist, extra = run_lightgbm(D, it, v)
        print(f"   {np.mean(hist['iter_time']):.3f}s/iter, best test "
              f"{np.nanmax(hist['test']):.5f}")
        save(f"{args.dataset}_official_{v}", hist, extra)
        gc.collect()

if args.only in ("xgb", "all"):
    for v in ["xgb_his", "xgb_exa"]:
        if not want(v):
            continue
        print(f"== {v}")
        hist, extra = run_xgboost(D, it, v)
        print(f"   {np.mean(hist['iter_time']):.3f}s/iter, best test "
              f"{np.nanmax(hist['test']):.5f}")
        save(f"{args.dataset}_{v}", hist, extra)
