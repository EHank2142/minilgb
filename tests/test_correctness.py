"""Sanity checks for minilgb.  Run:  python -m pytest tests -q   (or python tests/test_correctness.py)"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilgb import GBDT, BinnedDataset, auc, greedy_bundling  # noqa: E402


def make_onehot(n=20000, fields=(5, 30, 100), seed=0):
    rng = np.random.default_rng(seed)
    blocks, logit = [rng.normal(size=(n, 2))], np.zeros(n)
    logit += blocks[0][:, 0] - 0.5 * blocks[0][:, 1] ** 2
    for card in fields:
        codes = rng.integers(0, card, n)
        w = rng.normal(size=card)
        logit += w[codes]
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, card)))
    X = sp.hstack([sp.csr_matrix(blocks[0])] + blocks[1:], format="csr")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, y


def test_bundling_finds_fields():
    X, _ = make_onehot()
    ds = BinnedDataset(X, efb=True, verbose=False)
    # 2 dense numeric features + one bundle per one-hot field
    assert len(ds.groups) == 2 + 3, ds.groups
    assert ds.total_conflicts == 0


def test_efb_exact_when_gamma_zero():
    X, y = make_onehot()
    kw = dict(num_iterations=20, num_leaves=31, learning_rate=0.1)
    m0 = GBDT(**kw).fit(X, y, verbose=0, dataset=BinnedDataset(X, efb=False, verbose=False))
    m1 = GBDT(**kw).fit(X, y, verbose=0, dataset=BinnedDataset(X, efb=True, verbose=False))
    for a, b in zip(m0.trees, m1.trees):
        assert np.array_equal(a.split_feat, b.split_feat)
        assert np.array_equal(a.split_bin, b.split_bin)
        np.testing.assert_allclose(a.leaf_value, b.leaf_value, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(m0.predict_raw(X), m1.predict_raw(X), rtol=1e-6, atol=1e-8)


def test_training_prediction_consistent():
    """score maintained during training (via leaf assignment) == raw-value prediction."""
    X, y = make_onehot(5000)
    for s in ("none", "goss", "sgb"):
        m = GBDT(num_iterations=10, num_leaves=15, sampling=s).fit(X, y, verbose=0)
        np.testing.assert_allclose(m.train_score, m.predict_raw(X), rtol=1e-9, atol=1e-9)


def test_close_to_official_lightgbm():
    import lightgbm as lgb
    X, y = make_onehot(40000, seed=1)
    X, Xt, y, yt = X[:30000], X[30000:], y[:30000], y[30000:]
    ours = GBDT(num_iterations=50, num_leaves=31).fit(X, y, verbose=0)
    a_ours = auc(yt, ours.predict_raw(Xt))
    ref = lgb.train(dict(objective="binary", num_leaves=31, learning_rate=0.1, verbose=-1,
                         min_data_in_leaf=20, min_sum_hessian_in_leaf=1e-3, lambda_l2=0.0),
                    lgb.Dataset(X, y), 50)
    a_ref = auc(yt, ref.predict(Xt, raw_score=True))
    print(f"AUC ours={a_ours:.4f} official={a_ref:.4f}")
    assert abs(a_ours - a_ref) < 0.01


def test_greedy_bundling_conflicts():
    # features 0,1 exclusive; feature 2 overlaps both on 1 row
    nz = [np.array([0, 1, 2]), np.array([3, 4]), np.array([2, 4, 5])]
    assert len(greedy_bundling(nz, 6, [2, 2, 2], gamma=0.0)) == 2
    assert len(greedy_bundling(nz, 6, [2, 2, 2], gamma=2 / 6)) == 1


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
