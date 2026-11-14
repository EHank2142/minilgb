"""Exclusive Feature Bundling (paper Sec. 4, Algorithms 3 and 4), written from scratch."""
import numpy as np
import scipy.sparse as sp


def conflict_degrees(nz_rows, n_sample):
    """Degree of every feature in the conflict graph (edge <=> the two features are
    non-zero on at least one common sampled row). O(#feature^2) in the worst case."""
    F = len(nz_rows)
    lens = np.array([len(r) for r in nz_rows])
    indptr = np.concatenate([[0], np.cumsum(lens)])
    indices = np.concatenate(nz_rows) if F else np.zeros(0, np.int64)
    M = sp.csr_matrix((np.ones(len(indices), np.float32), indices, indptr), shape=(F, n_sample))
    C = (M @ M.T).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return np.diff(C.indptr)


def greedy_bundling(nz_rows, n_sample, num_bins, gamma=0.0, order="count",
                    max_bundle_bins=65535):
    """Algorithm 3 (Greedy Bundling).

    nz_rows   : list, for each feature the sampled row ids where its bin != 0
    num_bins  : list of #bins per feature
    gamma     : max conflict rate per bundle; K = gamma * n_sample
    order     : 'degree' (graph degree, as in Alg. 3) or 'count' (#non-zeros, the
                cheaper ordering proposed at the end of Sec. 4)
    returns   : list of bundles (lists of feature ids)
    """
    F = len(nz_rows)
    K = int(gamma * n_sample)
    if order == "degree":
        key = conflict_degrees(nz_rows, n_sample)
    elif order == "count":
        key = np.array([len(r) for r in nz_rows])
    else:
        raise ValueError(order)
    search = np.argsort(-key, kind="stable")

    cap = 16
    masks = np.zeros((cap, n_sample), dtype=bool)  # union of non-zero rows per bundle
    bundles, conflict, bins = [], [], []
    for f in search:
        rows = nz_rows[f]
        nb = num_bins[f] - 1  # non-zero bins this feature adds to a bundle
        placed = False
        if bundles:
            B = len(bundles)
            cnt = masks[:B][:, rows].sum(axis=1) if len(rows) else np.zeros(B, np.int64)
            for j in range(B):  # first fit, as in Alg. 3
                # NOTE: Alg. 3 in the paper writes bundlesConflict[i]; the bundle index j
                # is clearly what is meant.
                if cnt[j] + conflict[j] <= K and bins[j] + nb <= max_bundle_bins:
                    bundles[j].append(int(f))
                    conflict[j] += int(cnt[j])
                    bins[j] += nb
                    masks[j, rows] = True
                    placed = True
                    break
        if not placed:
            if len(bundles) == cap:
                cap *= 2
                masks = np.vstack([masks, np.zeros_like(masks)])
            masks[len(bundles), rows] = True
            bundles.append([int(f)])
            conflict.append(0)
            bins.append(1 + nb)
    return bundles


def merge_bundle(bin_columns, num_bins, n):
    """Algorithm 4 (Merge Exclusive Features): add bin offsets so that the features of
    one bundle reside in disjoint bin ranges. Returns (merged column, starts, #conflicts).

    Feature j's non-zero bins 1..nb_j-1 become start_j .. start_j+nb_j-2, bin 0 of the
    bundle means 'all features of the bundle are zero'. On conflict the later feature
    overwrites (the "random pollution" analysed in Prop. 2.1 of the supplement).
    """
    out = np.zeros(n, dtype=np.uint16)
    starts = []
    start = 1
    n_conflict = 0
    for col, nb in zip(bin_columns, num_bins):
        m = col != 0
        n_conflict += int(np.count_nonzero(out[m]))
        out[m] = (col[m].astype(np.int64) + (start - 1)).astype(np.uint16)
        starts.append(start)
        start += nb - 1
    return out, starts, n_conflict
