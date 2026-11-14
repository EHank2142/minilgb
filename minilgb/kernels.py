"""Numba kernels for histogram-based tree learning (written from scratch).

Data layout (see dataset.py):
  * Every original feature f lives in a *group* (a single feature, or an EFB bundle).
    Inside its group, the non-zero bins 1..nb_f-1 of f are stored as group-bins
    start_f .. start_f+nb_f-2; group-bin 0 means "every feature of the group is in bin 0".
    For a singleton group start_f = 1, so group-bin == feature-bin.
  * Dense groups : Xd[k, row] (uint16).
  * Sparse groups: per group a list of (row, group-bin) entries for group-bin != 0.
    The entry list of each group is kept *ordered by leaf* during tree growth
    (the "per-feature table of non-zero data" of the paper, Sec. 4).
  * A histogram is an array (total_bins, 3) with columns (sum_grad, sum_hess, count).
    Bin 0 of every feature is never read: it is recovered as leaf_total - sum(other bins),
    which is what makes both sparse storage and EFB bundles exact.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def decode(v, start, nb):
    """group-bin -> feature-bin."""
    if v >= start and v <= start + nb - 2:
        return v - start + 1
    return 0


# --------------------------------------------------------------------------- histograms
@njit(parallel=True, cache=True)
def hist_dense(Xd, dgoff, order, b, e, g, h, c, hist):
    n = e - b
    gg = np.empty(n)
    hh = np.empty(n)
    cc = np.empty(n)
    rows = order[b:e]
    for i in range(n):  # gather gradients once ("ordered gradients")
        r = rows[i]
        gg[i] = g[r]
        hh[i] = h[r]
        cc[i] = c[r]
    for k in prange(Xd.shape[0]):  # one thread per group -> no write conflicts
        off = dgoff[k]
        col = Xd[k]
        for i in range(n):
            v = off + col[rows[i]]
            hist[v, 0] += gg[i]
            hist[v, 1] += hh[i]
            hist[v, 2] += cc[i]


@njit(parallel=True, cache=True)
def hist_sparse(sp_rows, sp_bins, sbeg, scnt, leaf, sgoff, g, h, c, hist):
    for k in prange(sgoff.shape[0]):
        off = sgoff[k]
        bb = sbeg[k, leaf]
        ee = bb + scnt[k, leaf]
        for i in range(bb, ee):
            r = sp_rows[i]
            v = off + sp_bins[i]
            hist[v, 0] += g[r]
            hist[v, 1] += h[r]
            hist[v, 2] += c[r]


@njit(parallel=True, cache=True)
def hist_subtract(parent, child):
    """parent <- parent - child (in place)."""
    for i in prange(parent.shape[0]):
        parent[i, 0] -= child[i, 0]
        parent[i, 1] -= child[i, 1]
        parent[i, 2] -= child[i, 2]


# --------------------------------------------------------------------------- split finding
@njit(parallel=True, cache=True)
def find_best_splits(hist, feat_base, feat_nb, tg, th, tc, min_data, min_hess, lam,
                     out_gain, out_thr, out_lg, out_lh, out_lc):
    """Best threshold for every feature. Left child = feature-bins 0..thr."""
    parent = tg * tg / (th + lam)
    for f in prange(feat_base.shape[0]):
        base = feat_base[f]
        nb = feat_nb[f]
        sg = 0.0
        sh = 0.0
        sc = 0.0
        for b in range(1, nb):
            sg += hist[base + b, 0]
            sh += hist[base + b, 1]
            sc += hist[base + b, 2]
        lg = tg - sg  # bin 0 by subtraction
        lh = th - sh
        lc = tc - sc
        best = -1.0
        bthr = -1
        blg = 0.0
        blh = 0.0
        blc = 0.0
        for t in range(nb - 1):
            if t > 0:
                lg += hist[base + t, 0]
                lh += hist[base + t, 1]
                lc += hist[base + t, 2]
            rc = tc - lc
            if rc < min_data:
                break
            if lc < min_data:
                continue
            rh = th - lh
            if lh < min_hess or rh < min_hess:
                continue
            rg = tg - lg
            gain = lg * lg / (lh + lam) + rg * rg / (rh + lam) - parent
            if gain > best:
                best = gain
                bthr = t
                blg = lg
                blh = lh
                blc = lc
        out_gain[f] = best
        out_thr[f] = bthr
        out_lg[f] = blg
        out_lh[f] = blh
        out_lc[f] = blc


# --------------------------------------------------------------------------- partition
@njit(cache=True)
def partition_dense(order, b, e, col, start, nb, thr, tmp, leaf_of, new_leaf, mark, set_mark):
    """Stable partition of order[b:e] on a dense group column. Right rows get
    leaf_of = new_leaf (and mark = 1 if set_mark, needed by partition_sparse). Returns #left."""
    nl = 0
    nr = 0
    for i in range(b, e):
        r = order[i]
        if decode(col[r], start, nb) > thr:
            tmp[nr] = r
            nr += 1
        else:
            order[b + nl] = r
            nl += 1
    for j in range(nr):
        r = tmp[j]
        order[b + nl + j] = r
        leaf_of[r] = new_leaf
        if set_mark:
            mark[r] = 1
    return nl


@njit(cache=True)
def mark_sparse(sp_rows, sp_bins, bb, ee, start, nb, thr, mark):
    for i in range(bb, ee):
        if decode(sp_bins[i], start, nb) > thr:
            mark[sp_rows[i]] = 1


@njit(cache=True)
def partition_rows(order, b, e, mark, tmp, leaf_of, new_leaf):
    """Stable partition of order[b:e] by mark (0 left / 1 right). Returns #left."""
    nl = 0
    nr = 0
    for i in range(b, e):
        r = order[i]
        if mark[r] == 0:
            order[b + nl] = r
            nl += 1
        else:
            tmp[nr] = r
            nr += 1
    for j in range(nr):
        r = tmp[j]
        order[b + nl + j] = r
        leaf_of[r] = new_leaf
    return nl


@njit(parallel=True, cache=True)
def partition_sparse(sp_rows, sp_bins, tmp_rows, tmp_bins, sbeg, scnt, leaf, new_leaf, mark):
    """Keep every sparse group's (row, bin) list ordered by leaf (the per-feature
    non-zero tables that the paper says baseline LightGBM must maintain)."""
    for k in prange(sbeg.shape[0]):
        b = sbeg[k, leaf]
        e = b + scnt[k, leaf]
        nl = 0
        nr = 0
        for i in range(b, e):
            r = sp_rows[i]
            if mark[r] == 0:
                sp_rows[b + nl] = r
                sp_bins[b + nl] = sp_bins[i]
                nl += 1
            else:
                tmp_rows[b + nr] = r
                tmp_bins[b + nr] = sp_bins[i]
                nr += 1
        for j in range(nr):
            sp_rows[b + nl + j] = tmp_rows[b + j]
            sp_bins[b + nl + j] = tmp_bins[b + j]
        scnt[k, leaf] = nl
        sbeg[k, new_leaf] = b + nl
        scnt[k, new_leaf] = nr


@njit(cache=True)
def clear_mark(order, b, e, mark):
    for i in range(b, e):
        mark[order[i]] = 0


@njit(parallel=True, cache=True)
def add_leaf_values(score, leaf_of, leaf_value):
    for i in prange(score.shape[0]):
        score[i] += leaf_value[leaf_of[i]]


@njit(parallel=True, cache=True)
def add_leaf_values_rows(score, rows, leaf_of, leaf_value):
    for i in prange(rows.shape[0]):
        r = rows[i]
        score[r] += leaf_value[leaf_of[r]]


@njit(parallel=True, cache=True)
def filter_sparse(sp_rows, sp_bins, sp_ptr, keep, out_rows, out_bins, out_beg, out_cnt):
    """Per-iteration sparse tables restricted to the sampled rows (GOSS / SGB)."""
    for k in prange(sp_ptr.shape[0] - 1):
        p = sp_ptr[k]
        for i in range(sp_ptr[k], sp_ptr[k + 1]):
            r = sp_rows[i]
            if keep[r]:
                out_rows[p] = r
                out_bins[p] = sp_bins[i]
                p += 1
        out_beg[k] = sp_ptr[k]
        out_cnt[k] = p - sp_ptr[k]


@njit(parallel=True, cache=True)
def predict_binned_rows(rows, Xd, sp_rows, sp_bins, sp_ptr, f_kind, f_gidx, f_start, f_nb,
                        split_fi, split_bin, left, right, leaf_value, score):
    """Add a tree's output to score[rows] using the *binned* training data (used for the
    rows that were not sampled by GOSS/SGB). sp_rows must be sorted within each group."""
    for i in prange(rows.shape[0]):
        r = rows[i]
        node = 0
        if split_fi.shape[0] == 0:
            score[r] += leaf_value[0]
            continue
        while node >= 0:
            f = split_fi[node]
            gi = f_gidx[f]
            if f_kind[f] == 0:
                v = Xd[gi, r]
            else:
                lo = sp_ptr[gi]
                hi = sp_ptr[gi + 1]
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if sp_rows[mid] < r:
                        lo = mid + 1
                    else:
                        hi = mid
                v = sp_bins[lo] if (lo < sp_ptr[gi + 1] and sp_rows[lo] == r) else 0
            if decode(v, f_start[f], f_nb[f]) <= split_bin[node]:
                node = left[node]
            else:
                node = right[node]
        score[r] += leaf_value[~node]


# --------------------------------------------------------------------------- prediction
@njit(parallel=True, cache=True)
def predict_csr(indptr, indices, data, split_feat, split_val, left, right, leaf_value, out):
    """Add one tree's output for raw feature values in CSR format (implicit zeros)."""
    for r in prange(indptr.shape[0] - 1):
        a = indptr[r]
        z = indptr[r + 1]
        node = 0
        if split_feat.shape[0] == 0:
            out[r] += leaf_value[0]
            continue
        while node >= 0:
            f = split_feat[node]
            lo = a
            hi = z
            while lo < hi:  # binary search feature f in the row
                mid = (lo + hi) >> 1
                if indices[mid] < f:
                    lo = mid + 1
                else:
                    hi = mid
            x = data[lo] if (lo < z and indices[lo] == f) else 0.0
            node = left[node] if x <= split_val[node] else right[node]
        out[r] += leaf_value[~node]
