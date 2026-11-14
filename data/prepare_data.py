"""Prepare the two public datasets used in this reproduction.

Flight Delay  : szilard/benchm-ml airline data (train-1m.csv + test.csv), the same
                source as the "Flight Delay" dataset of the paper (which used 10M rows).
                Categorical columns are one-hot encoded -> ~650 sparse 0/1 features,
                as in the paper ("contain a lot of one-hot coding features").
MSLR-WEB10K   : Microsoft Learning-to-Rank, Fold1 (train / vali / test), 136 dense
                features. The paper used the larger MSLR-WEB30K ("LETOR", 2M rows).

Outputs go to data/processed/*.npz
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "processed")
os.makedirs(OUT, exist_ok=True)

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "Distance"]


def prepare_flight():
    t = time.time()
    tr = pd.read_csv(os.path.join(RAW, "flight_train1m.csv"))
    te = pd.read_csv(os.path.join(RAW, "flight_test.csv"))
    blocks_tr, blocks_te, names = [], [], []
    for c in NUM_COLS:
        blocks_tr.append(sp.csr_matrix(tr[c].to_numpy(np.float32)[:, None]))
        blocks_te.append(sp.csr_matrix(te[c].to_numpy(np.float32)[:, None]))
        names.append(c)
    for c in CAT_COLS:
        cats = np.sort(tr[c].astype(str).unique())
        lut = {v: i for i, v in enumerate(cats)}
        for df, blocks in ((tr, blocks_tr), (te, blocks_te)):
            codes = df[c].astype(str).map(lut).fillna(-1).to_numpy(np.int64)
            ok = codes >= 0  # unseen test categories -> all-zero row in this block
            rows = np.nonzero(ok)[0]
            m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, codes[ok])),
                              shape=(len(df), len(cats)))
            blocks.append(m)
        names += [f"{c}={v}" for v in cats]
    Xtr = sp.hstack(blocks_tr, format="csr")
    Xte = sp.hstack(blocks_te, format="csr")
    ytr = (tr["dep_delayed_15min"] == "Y").to_numpy(np.float32)
    yte = (te["dep_delayed_15min"] == "Y").to_numpy(np.float32)
    sp.save_npz(os.path.join(OUT, "flight_Xtr.npz"), Xtr)
    sp.save_npz(os.path.join(OUT, "flight_Xte.npz"), Xte)
    np.savez(os.path.join(OUT, "flight_y.npz"), ytr=ytr, yte=yte, names=np.array(names))
    print(f"[flight] train {Xtr.shape} nnz={Xtr.nnz}  test {Xte.shape}  "
          f"pos-rate={ytr.mean():.3f}  ({time.time()-t:.1f}s)")


def prepare_mslr(fold="Fold1"):
    t = time.time()
    d = os.path.join(RAW, "MSLR-WEB10K", fold)
    out = {}
    for split in ("train", "vali", "test"):
        X, y, q = load_svmlight_file(os.path.join(d, f"{split}.txt"), n_features=136,
                                     query_id=True, dtype=np.float32)
        X = np.asarray(X.todense(), dtype=np.float32)
        # documents of one query are contiguous in the file; store boundaries
        change = np.nonzero(np.diff(q))[0] + 1
        qptr = np.concatenate([[0], change, [len(q)]]).astype(np.int64)
        out[f"X_{split}"], out[f"y_{split}"], out[f"qptr_{split}"] = X, y.astype(np.float32), qptr
        print(f"[mslr] {split}: {X.shape}, {len(qptr)-1} queries, min feature value {X.min():.3g}")
    np.savez(os.path.join(OUT, "mslr10k_fold1.npz"), **out)
    print(f"[mslr] done ({time.time()-t:.1f}s)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["flight", "mslr"]
    if "flight" in which:
        prepare_flight()
    if "mslr" in which:
        prepare_mslr()
