import numpy as np
from numba import njit, prange
from scipy.stats import rankdata


def auc(y, s):
    """ROC-AUC via the Mann-Whitney statistic (ties get average rank)."""
    y = np.asarray(y) > 0.5
    r = rankdata(s)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


@njit(parallel=True, cache=True)
def _ndcg(y, s, qptr, k):
    nq = qptr.shape[0] - 1
    out = np.zeros(nq)
    for q in prange(nq):
        b = qptr[q]
        e = qptr[q + 1]
        lab = y[b:e]
        ideal = np.sort(lab)[::-1]
        idcg = 0.0
        for i in range(min(k, e - b)):
            idcg += (2.0 ** ideal[i] - 1.0) / np.log2(i + 2.0)
        if idcg == 0.0:
            out[q] = 1.0  # LightGBM convention for queries without relevant documents
            continue
        order = np.argsort(-s[b:e], kind="mergesort")
        dcg = 0.0
        for i in range(min(k, e - b)):
            dcg += (2.0 ** lab[order[i]] - 1.0) / np.log2(i + 2.0)
        out[q] = dcg / idcg
    return out


def ndcg(y, s, qptr, k=10):
    return float(_ndcg(np.asarray(y, np.float64), np.asarray(s, np.float64),
                       np.asarray(qptr, np.int64), k).mean())
