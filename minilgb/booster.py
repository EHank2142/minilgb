"""Gradient boosting with leaf-wise histogram trees, GOSS and SGB (written from scratch)."""
import time

import numpy as np
import scipy.sparse as sp

from . import kernels as K
from .dataset import BinnedDataset
from .metrics import auc, ndcg
from .objectives import BinaryLogloss, LambdaRank

DEFAULTS = dict(
    objective="binary",       # 'binary' | 'lambdarank'
    num_iterations=100,
    learning_rate=0.1,
    num_leaves=255,
    max_depth=-1,
    min_data_in_leaf=20,
    min_sum_hessian_in_leaf=1e-3,
    lambda_l2=0.0,
    sampling="none",          # 'none' | 'goss' | 'sgb'
    top_rate=0.1,             # GOSS a
    other_rate=0.1,           # GOSS b (fraction of the *whole* data, as in Alg. 2)
    goss_reweight=True,       # multiply small-gradient samples by (1-a)/b
    goss_warmup=0,            # iterations without sampling before GOSS starts
    goss_rank="grad",         # rank by |g| (paper) or |g*h| ('gradhess', LightGBM impl.)
    bagging_fraction=0.2,     # SGB sampling ratio
    truncation=30,            # lambdarank
    eval_at=10,
    seed=0,
)


class Tree:
    __slots__ = ("split_feat", "split_fi", "split_bin", "split_val", "left", "right",
                 "leaf_value", "gain")

    def predict_csr(self, X, out):
        K.predict_csr(X.indptr, X.indices, X.data, self.split_feat, self.split_val,
                      self.left, self.right, self.leaf_value, out)


class _Workspace:
    def __init__(self, ds, L):
        n = ds.n
        self.arange = np.arange(n, dtype=np.int64)
        self.order = np.empty(n, np.int64)
        self.tmp = np.empty(n, np.int64)
        self.leaf_of = np.zeros(n, np.int64)
        self.mark = np.zeros(n, np.uint8)
        ns = ds.num_sparse_groups
        self.sbeg = np.zeros((ns, L), np.int64)
        self.scnt = np.zeros((ns, L), np.int64)
        # working copies of the sparse tables (ds keeps a pristine, row-sorted copy)
        self.sp_rows = ds.sp_rows.copy()
        self.sp_bins = ds.sp_bins.copy()
        self.tmp_rows = np.empty_like(ds.sp_rows)
        self.tmp_bins = np.empty_like(ds.sp_bins)
        self.filtered = False
        self.hist = np.zeros((L, ds.total_bins, 3))
        F = len(ds.feat_ids)
        self.og = np.empty(F)
        self.ot = np.empty(F, np.int64)
        self.olg = np.empty(F)
        self.olh = np.empty(F)
        self.olc = np.empty(F)


class GBDT:
    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown params {unknown}")
        self.p = {**DEFAULTS, **params}
        self.trees = []

    # ------------------------------------------------------------------ tree learner
    def _build_hist(self, ds, ws, leaf, b, e, g, h, c):
        H = ws.hist[leaf]
        H[:] = 0.0
        if ds.num_dense_groups:
            K.hist_dense(ds.Xd, ds.dgoff, ws.order, b, e, g, h, c, H)
        if ds.num_sparse_groups:
            K.hist_sparse(ws.sp_rows, ws.sp_bins, ws.sbeg, ws.scnt, leaf, ds.sgoff, g, h, c, H)

    def _best_split(self, ds, ws, leaf, tot, best_gain, best):
        p = self.p
        K.find_best_splits(ws.hist[leaf], ds.feat_base, ds.feat_nb, tot[0], tot[1], tot[2],
                           p["min_data_in_leaf"], p["min_sum_hessian_in_leaf"], p["lambda_l2"],
                           ws.og, ws.ot, ws.olg, ws.olh, ws.olc)
        f = int(np.argmax(ws.og))
        best_gain[leaf] = ws.og[f]
        best[leaf] = (f, ws.ot[f], ws.olg[f], ws.olh[f], ws.olc[f])

    def _grow_tree(self, ds, ws, g, h, c, used):
        """Leaf-wise growth on the rows in `used` (all rows if None). Only those rows are
        partitioned; afterwards leaf_of[r] is valid for every r in `used`."""
        p = self.p
        L = p["num_leaves"]
        if used is None:
            m = ds.n
            ws.order[:] = ws.arange
            ws.leaf_of[:] = 0
            if ws.filtered:
                ws.sp_rows[:] = ds.sp_rows
                ws.sp_bins[:] = ds.sp_bins
                ws.filtered = False
            if ds.num_sparse_groups:
                ws.sbeg[:, 0] = ds.sp_ptr[:-1]
                ws.scnt[:, 0] = np.diff(ds.sp_ptr)
        else:
            m = len(used)
            ws.order[:m] = used
            ws.leaf_of[used] = 0
            ws.filtered = True
            if ds.num_sparse_groups:
                b0 = np.empty(ds.num_sparse_groups, np.int64)
                c0 = np.empty(ds.num_sparse_groups, np.int64)
                K.filter_sparse(ds.sp_rows, ds.sp_bins, ds.sp_ptr, c > 0, ws.sp_rows,
                                ws.sp_bins, b0, c0)
                ws.sbeg[:, 0] = b0
                ws.scnt[:, 0] = c0
        has_sparse = ds.num_sparse_groups > 0
        lb = np.zeros(L, np.int64)
        le = np.zeros(L, np.int64)
        le[0] = m
        tot = np.zeros((L, 3))
        tot[0] = (g.sum(), h.sum(), c.sum())
        depth = np.zeros(L, np.int64)
        parent_node = np.full(L, -1, np.int64)
        parent_is_left = np.zeros(L, bool)
        best_gain = np.full(L, -np.inf)
        best = {}
        sf, sfi, sb, sv, lft, rgt, gains = [], [], [], [], [], [], []
        maxd = p["max_depth"]

        self._build_hist(ds, ws, 0, 0, m, g, h, c)
        self._best_split(ds, ws, 0, tot[0], best_gain, best)
        for new in range(1, L):
            l = int(np.argmax(best_gain[:new]))
            gain = best_gain[l]
            if not gain > 0:
                break
            fi, thr, lg, lh, lc = best[l]
            fi, thr = int(fi), int(thr)
            kind, gi = ds.feat_kind[fi], ds.feat_gidx[fi]
            start, nb = ds.feat_start[fi], ds.feat_nb[fi]
            b, e = lb[l], le[l]
            # 1) partition the rows of leaf l (right rows -> leaf `new`)
            if kind == 0:
                nl = K.partition_dense(ws.order, b, e, ds.Xd[gi], start, nb, thr, ws.tmp,
                                       ws.leaf_of, new, ws.mark, has_sparse)
            else:
                bb = ws.sbeg[gi, l]
                K.mark_sparse(ws.sp_rows, ws.sp_bins, bb, bb + ws.scnt[gi, l], start, nb, thr,
                              ws.mark)
                nl = K.partition_rows(ws.order, b, e, ws.mark, ws.tmp, ws.leaf_of, new)
            # 2) keep the per-group sparse tables ordered by leaf
            if has_sparse:
                K.partition_sparse(ws.sp_rows, ws.sp_bins, ws.tmp_rows, ws.tmp_bins, ws.sbeg,
                                   ws.scnt, l, new, ws.mark)
                K.clear_mark(ws.order, b + nl, e, ws.mark)
            lb[new], le[new], le[l] = b + nl, e, b + nl
            # 3) record the node
            node = new - 1
            f_orig = int(ds.feat_ids[fi])
            sf.append(f_orig); sfi.append(fi); sb.append(thr)
            sv.append(ds.split_value(f_orig, thr))
            lft.append(~l); rgt.append(~new); gains.append(gain)
            if parent_node[l] >= 0:
                if parent_is_left[l]:
                    lft[parent_node[l]] = node
                else:
                    rgt[parent_node[l]] = node
            parent_node[l], parent_is_left[l] = node, True
            parent_node[new], parent_is_left[new] = node, False
            depth[l] += 1
            depth[new] = depth[l]
            parent_tot = tot[l].copy()
            tot[l] = (lg, lh, lc)
            tot[new] = parent_tot - tot[l]
            # 4) histograms: build the smaller child, subtract for the larger one
            if (le[l] - lb[l]) <= (le[new] - lb[new]):
                ws.hist[new][:] = ws.hist[l]
                self._build_hist(ds, ws, l, lb[l], le[l], g, h, c)
                K.hist_subtract(ws.hist[new], ws.hist[l])
            else:
                self._build_hist(ds, ws, new, lb[new], le[new], g, h, c)
                K.hist_subtract(ws.hist[l], ws.hist[new])
            for leaf in (l, new):
                if maxd > 0 and depth[leaf] >= maxd:
                    best_gain[leaf] = -np.inf
                else:
                    self._best_split(ds, ws, leaf, tot[leaf], best_gain, best)

        nleaf = len(sf) + 1
        t = Tree()
        t.split_feat = np.array(sf, np.int64)
        t.split_fi = np.array(sfi, np.int64)
        t.split_bin = np.array(sb, np.int64)
        t.split_val = np.array(sv, np.float64)
        t.left = np.array(lft, np.int64)
        t.right = np.array(rgt, np.int64)
        t.gain = np.array(gains)
        lv = -tot[:nleaf, 0] / (tot[:nleaf, 1] + p["lambda_l2"]) * p["learning_rate"]
        t.leaf_value = lv.astype(np.float64)
        return t

    def _update_train_score(self, ds, ws, tree, score, used, unused):
        if used is None:
            K.add_leaf_values(score, ws.leaf_of, tree.leaf_value)
            return
        K.add_leaf_values_rows(score, used, ws.leaf_of, tree.leaf_value)
        # rows not sampled in this iteration: traverse the tree on the binned data
        K.predict_binned_rows(unused, ds.Xd, ds.sp_rows, ds.sp_bins, ds.sp_ptr, ds.feat_kind,
                              ds.feat_gidx, ds.feat_start, ds.feat_nb, tree.split_fi,
                              tree.split_bin, tree.left, tree.right, tree.leaf_value, score)

    # ------------------------------------------------------------------ sampling
    def _sample(self, it, g, h, rng):
        """Algorithm 2 (GOSS) / SGB. Returns (sorted used row ids, per-row weight) or
        (None, None) when no sampling is done in this iteration."""
        p, n = self.p, len(g)
        if p["sampling"] == "goss" and it >= p["goss_warmup"]:
            a, b = p["top_rate"], p["other_rate"]
            top_n, rand_n = int(a * n), int(b * n)
            key = np.abs(g) if p["goss_rank"] == "grad" else np.abs(g * h)
            top = np.argpartition(key, n - top_n)[n - top_n:]
            is_top = np.zeros(n, bool)
            is_top[top] = True
            rest = np.flatnonzero(~is_top)
            rand = rng.choice(rest, rand_n, replace=False)
            w = np.zeros(n)
            w[top] = 1.0
            w[rand] = (1.0 - a) / b if p["goss_reweight"] else 1.0
            return np.sort(np.concatenate([top, rand])).astype(np.int64), w
        if p["sampling"] == "sgb":
            m = int(p["bagging_fraction"] * n)
            used = np.sort(rng.choice(n, m, replace=False)).astype(np.int64)
            w = np.zeros(n)
            w[used] = 1.0
            return used, w
        return None, None

    # ------------------------------------------------------------------ public API
    def fit(self, X, y, qptr=None, eval_sets=(), dataset_params=None, verbose=10,
            dataset=None, time_budget=None, early_stopping_rounds=None):
        """X: ndarray / scipy sparse (or pass a prebuilt BinnedDataset via `dataset`).
        eval_sets: list of (name, X, y, qptr_or_None). Per-iteration training time and the
        eval metrics are stored in self.history (evaluation time is not counted)."""
        p = self.p
        ds = dataset if dataset is not None else BinnedDataset(X, **(dataset_params or {}))
        self.ds = ds
        y = np.asarray(y, np.float64)
        obj = BinaryLogloss() if p["objective"] == "binary" else LambdaRank(
            qptr, y, p["truncation"])
        self.init = obj.init_score(y)
        score = np.full(ds.n, self.init)
        g = np.empty(ds.n); h = np.empty(ds.n); ones = np.ones(ds.n)
        ws = _Workspace(ds, p["num_leaves"])
        rng = np.random.default_rng(p["seed"])
        evals = []
        for name, Xe, ye, qe in eval_sets:
            Xe = sp.csr_matrix(Xe, dtype=np.float32)  # inputs are float32; saves memory
            Xe.sort_indices()
            evals.append((name, Xe, np.asarray(ye, np.float64), qe,
                          np.full(Xe.shape[0], self.init)))
        self.history = {"iter": [], "time": [], "iter_time": []}
        for name, *_ in evals:
            self.history[name] = []
        total = 0.0
        best_it, best_val = 0, -np.inf
        for it in range(p["num_iterations"]):
            t0 = time.perf_counter()
            obj.grad_hess(score, y, g, h)
            used, w = self._sample(it, g, h, rng)
            if used is None:
                gw, hw, cw, unused = g, h, ones, None
            else:
                gw, hw, cw = g * w, h * w, (w > 0).astype(np.float64)
                unused = np.flatnonzero(w == 0)
            tree = self._grow_tree(ds, ws, gw, hw, cw, used)
            self._update_train_score(ds, ws, tree, score, used, unused)
            dt = time.perf_counter() - t0
            total += dt
            self.trees.append(tree)
            self.history["iter"].append(it + 1)
            self.history["time"].append(total)
            self.history["iter_time"].append(dt)
            msg = []
            for name, Xe, ye, qe, se in evals:
                tree.predict_csr(Xe, se)
                m = auc(ye, se) if qe is None else ndcg(ye, se, qe, p["eval_at"])
                self.history[name].append(m)
                msg.append(f"{name}={m:.5f}")
            if verbose and ((it + 1) % verbose == 0 or it == 0):
                print(f"    iter {it+1:4d}  {dt:6.3f}s/iter  total {total:7.1f}s  " + " ".join(msg),
                      flush=True)
            if time_budget is not None and total > time_budget:
                break
            if early_stopping_rounds and evals:  # monitor the first eval set (higher = better)
                v = self.history[evals[0][0]][-1]
                if v > best_val:
                    best_val, best_it = v, it
                elif it - best_it >= early_stopping_rounds:
                    break
        self.best_iteration = best_it + 1
        self.train_score = score
        return self

    def predict_raw(self, X, num_iteration=None):
        X = sp.csr_matrix(X, dtype=np.float64)
        X.sort_indices()
        out = np.full(X.shape[0], self.init)
        for t in self.trees[:num_iteration]:
            t.predict_csr(X, out)
        return out
