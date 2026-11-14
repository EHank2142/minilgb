"""Loss functions: first/second order gradients (written from scratch)."""
import numpy as np
from numba import njit, prange


class BinaryLogloss:
    name = "binary"

    def init_score(self, y):
        p = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        return float(np.log(p / (1 - p)))

    def grad_hess(self, score, y, g, h):
        _logloss_grad(score, y, g, h)


@njit(parallel=True, cache=True)
def _logloss_grad(score, y, g, h):
    for i in prange(score.shape[0]):
        p = 1.0 / (1.0 + np.exp(-score[i]))
        g[i] = p - y[i]
        h[i] = max(p * (1.0 - p), 1e-16)


class LambdaRank:
    """LambdaMART gradients for NDCG (Burges 2010), in the form used by LightGBM:
    pairs (i, j) with i in the top `truncation` positions of the current ranking,
    |delta NDCG| weighting, sigmoid parameter sigma, optional lambda normalisation."""

    name = "lambdarank"

    def __init__(self, qptr, y, truncation=30, sigma=1.0, norm=True):
        self.qptr = np.asarray(qptr, np.int64)
        self.trunc, self.sigma, self.norm = truncation, sigma, norm
        self.inv_maxdcg = _inv_max_dcg(self.qptr, y.astype(np.float64), truncation)

    def init_score(self, y):
        return 0.0

    def grad_hess(self, score, y, g, h):
        _lambdarank_grad(score, y, self.qptr, self.inv_maxdcg, self.trunc, self.sigma,
                         self.norm, g, h)


@njit(cache=True)
def _inv_max_dcg(qptr, y, k):
    nq = qptr.shape[0] - 1
    out = np.zeros(nq)
    for q in range(nq):
        lab = np.sort(y[qptr[q]:qptr[q + 1]])[::-1]
        d = 0.0
        for i in range(min(k, lab.shape[0])):
            d += (2.0 ** lab[i] - 1.0) / np.log2(i + 2.0)
        out[q] = 1.0 / d if d > 0 else 0.0
    return out


@njit(parallel=True, cache=True)
def _lambdarank_grad(score, y, qptr, inv_maxdcg, trunc, sigma, norm, g, h):
    nq = qptr.shape[0] - 1
    for q in prange(nq):
        b = qptr[q]
        e = qptr[q + 1]
        cnt = e - b
        for i in range(b, e):
            g[i] = 0.0
            h[i] = 0.0
        if inv_maxdcg[q] == 0.0 or cnt < 2:
            continue
        s = score[b:e]
        lab = y[b:e]
        idx = np.argsort(-s, kind="mergesort")
        best_s = s[idx[0]]
        worst_s = s[idx[cnt - 1]]
        sum_l = 0.0
        for i in range(min(cnt - 1, trunc)):
            a = idx[i]
            for j in range(i + 1, cnt):
                c = idx[j]
                if lab[a] == lab[c]:
                    continue
                if lab[a] > lab[c]:
                    hi, lo, phi, plo = a, c, i, j
                else:
                    hi, lo, phi, plo = c, a, j, i
                ds = s[hi] - s[lo]
                dgain = (2.0 ** lab[hi]) - (2.0 ** lab[lo])
                ddisc = abs(1.0 / np.log2(phi + 2.0) - 1.0 / np.log2(plo + 2.0))
                delta = dgain * ddisc * inv_maxdcg[q]
                if norm and best_s != worst_s:  # regularise by score distance (as LightGBM)
                    delta /= 0.01 + abs(ds)
                rho = 1.0 / (1.0 + np.exp(sigma * ds))
                lam = -sigma * rho * delta
                hes = sigma * sigma * rho * (1.0 - rho) * delta
                g[b + hi] += lam
                g[b + lo] -= lam
                h[b + hi] += hes
                h[b + lo] += hes
                sum_l -= 2.0 * lam
        if norm and sum_l > 0:
            f = np.log2(1.0 + sum_l) / sum_l
            for i in range(b, e):
                g[i] *= f
                h[i] *= f
        for i in range(b, e):
            if h[i] < 1e-16:
                h[i] = 1e-16
