"""Feature binning + storage layout (dense / sparse / EFB bundles)."""
import time

import numpy as np
import scipy.sparse as sp

from .efb import greedy_bundling, merge_bundle


def find_bin_bounds(nz_vals, n_zero, max_bin):
    """Upper bounds of the bins of one feature (bin i = (bounds[i-1], bounds[i]]).
    Equal-frequency over distinct values; a feature that is >= 0 gets value 0 isolated
    in bin 0, so that bin 0 == "zero" (what EFB's Alg. 4 relies on)."""
    vals, cnts = np.unique(nz_vals, return_counts=True)
    if n_zero > 0:
        pos = np.searchsorted(vals, 0.0)
        vals = np.insert(vals, pos, 0.0)
        cnts = np.insert(cnts, pos, n_zero)
    if len(vals) <= 1:
        return np.array([np.inf])
    mids = (vals[:-1] + vals[1:]) / 2.0
    if len(vals) <= max_bin:
        cuts = mids
    else:
        cum = np.cumsum(cnts)[:-1]
        total = cnts.sum()
        targets = np.arange(1, max_bin) * (total / max_bin)
        idx = np.unique(np.searchsorted(cum, targets))
        idx = idx[idx < len(mids)]
        cuts = mids[idx]
        if n_zero > 0 and vals[0] == 0.0:
            cuts = np.union1d(cuts, mids[:1])  # isolate zero
    return np.append(cuts, np.inf)


class _Columns:
    """Uniform column access for dense ndarray / scipy sparse input."""

    def __init__(self, X):
        if sp.issparse(X):
            self.csc = X.tocsc().astype(np.float64)
            self.csc.eliminate_zeros()
            self.csc.sort_indices()
            self.dense = None
        else:
            self.csc = None
            self.dense = np.asarray(X)
        self.n, self.F = X.shape

    def nonzero(self, j):
        if self.csc is not None:
            a, b = self.csc.indptr[j], self.csc.indptr[j + 1]
            return self.csc.indices[a:b], self.csc.data[a:b]
        col = self.dense[:, j]
        rows = np.flatnonzero(col)
        return rows, col[rows].astype(np.float64)


class BinnedDataset:
    def __init__(self, X, max_bin=255, efb=False, gamma=0.0, bundle_order="count",
                 sparse_threshold=0.8, bin_sample=200_000, conflict_sample=1_000_000,
                 max_bundle_bins=65535, bundle_max_nnz_rate=0.5, seed=0, verbose=True):
        """conflict_sample: #rows on which feature conflicts are counted (all rows when
        n <= conflict_sample, so gamma=0 is exactly conflict-free).
        bundle_max_nnz_rate: features non-zero on more rows than this are never bundled
        (they would conflict with everything; skipping them keeps Alg. 3 cheap)."""
        t0 = time.time()
        cols = _Columns(X)
        n, F = cols.n, cols.F
        self.n, self.num_feature = n, F
        rng = np.random.default_rng(seed)
        bin_mask = np.zeros(n, bool)
        bin_mask[rng.choice(n, min(n, bin_sample), replace=False)] = True
        n_bs = int(bin_mask.sum())
        csample = np.sort(rng.choice(n, min(n, conflict_sample), replace=False))
        cpos = np.full(n, -1, np.int64)
        cpos[csample] = np.arange(len(csample))

        # ---- pass 1: bin boundaries, non-zero pattern on the conflict sample
        self.bounds, nbins, nz_sample = [], [], []
        for j in range(F):
            rows, vals = cols.nonzero(j)
            sel = bin_mask[rows]
            b = find_bin_bounds(vals[sel], n_bs - int(sel.sum()), max_bin)
            self.bounds.append(b)
            nbins.append(len(b))
            bins = self._bin_column(cols, j, b)
            nzr = np.flatnonzero(bins[csample] != 0).astype(np.int32)
            nz_sample.append(nzr if len(nzr) <= bundle_max_nnz_rate * len(csample) else None)
        self.num_bins = np.array(nbins, np.int64)
        usable = [j for j in range(F) if nbins[j] > 1]
        bundlable = [j for j in usable if nz_sample[j] is not None]

        # ---- grouping: singletons (baseline) or EFB bundles
        if efb:
            tb = time.time()
            sub = greedy_bundling([nz_sample[j] for j in bundlable], len(csample),
                                  [nbins[j] for j in bundlable], gamma, bundle_order,
                                  max_bundle_bins)
            groups = [[bundlable[i] for i in g] for g in sub]
            groups += [[j] for j in usable if nz_sample[j] is None]
            self.bundle_time = time.time() - tb
        else:
            groups = [[j] for j in usable]
            self.bundle_time = 0.0

        # ---- pass 2: build group columns, choose dense / sparse storage
        dense_cols, dense_gb, sp_rows, sp_bins, sp_ptr, sparse_gb = [], [], [], [], [0], []
        feat_group_kind, feat_group_idx, feat_start = {}, {}, {}
        self.total_conflicts = 0
        for grp in groups:
            colsb = [self._bin_column(cols, j, self.bounds[j]) for j in grp]
            merged, starts, ncf = merge_bundle(colsb, [nbins[j] for j in grp], n)
            self.total_conflicts += ncf
            gbins = 1 + sum(nbins[j] - 1 for j in grp)
            nz = np.flatnonzero(merged)
            if len(nz) / n < 1.0 - sparse_threshold:
                kind, idx = 1, len(sparse_gb)
                sp_rows.append(nz.astype(np.int32))
                sp_bins.append(merged[nz])
                sp_ptr.append(sp_ptr[-1] + len(nz))
                sparse_gb.append(gbins)
            else:
                kind, idx = 0, len(dense_gb)
                dense_cols.append(merged)
                dense_gb.append(gbins)
            for j, s in zip(grp, starts):
                feat_group_kind[j], feat_group_idx[j], feat_start[j] = kind, idx, s

        self.groups = groups
        self.Xd = np.vstack(dense_cols) if dense_cols else np.zeros((0, n), np.uint16)
        self.sp_rows = np.concatenate(sp_rows) if sp_rows else np.zeros(0, np.int32)
        self.sp_bins = np.concatenate(sp_bins) if sp_bins else np.zeros(0, np.uint16)
        self.sp_ptr = np.array(sp_ptr, np.int64)
        dgoff = np.concatenate([[0], np.cumsum(dense_gb)]).astype(np.int64)
        sgoff = (dgoff[-1] + np.concatenate([[0], np.cumsum(sparse_gb)])).astype(np.int64)
        self.dgoff, self.sgoff = dgoff[:-1], sgoff[:-1]
        self.total_bins = int(sgoff[-1])

        # per-feature metadata used by split finding / partition (features with 1 bin
        # are dropped: they can never be split on)
        self.feat_ids = np.array(usable, np.int64)
        self.feat_nb = self.num_bins[self.feat_ids]
        self.feat_kind = np.array([feat_group_kind[j] for j in usable], np.int64)
        self.feat_gidx = np.array([feat_group_idx[j] for j in usable], np.int64)
        self.feat_start = np.array([feat_start[j] for j in usable], np.int64)
        goff = np.array([self.dgoff[i] if k == 0 else self.sgoff[i]
                         for k, i in zip(self.feat_kind, self.feat_gidx)], np.int64)
        self.feat_base = (goff + self.feat_start - 1).astype(np.int64)
        self.num_dense_groups, self.num_sparse_groups = len(dense_gb), len(sparse_gb)
        self.construct_time = time.time() - t0
        if verbose:
            print(f"  [dataset] n={n} features={F} (usable {len(usable)}) -> groups={len(groups)} "
                  f"(dense {len(dense_gb)}, sparse {len(sparse_gb)}), total_bins={self.total_bins}, "
                  f"sparse entries={len(self.sp_rows)}, conflicts={self.total_conflicts}, "
                  f"build {self.construct_time:.1f}s (bundling {self.bundle_time:.2f}s)")

    @staticmethod
    def _bin_column(cols, j, bounds):
        rows, vals = cols.nonzero(j)
        zero_bin = np.searchsorted(bounds, 0.0, side="left")
        out = np.full(cols.n, zero_bin, dtype=np.uint16)
        out[rows] = np.searchsorted(bounds, vals, side="left")
        return out

    def split_value(self, f, thr):
        """raw-value threshold for feature f, bin threshold thr (left: x <= value)."""
        return self.bounds[f][thr]
