"""Parity of the numpy StructDirSAGE with the PyTorch Geometric model (DirGNNConv(SAGEConv) x3)."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_v2 import StructDirSAGE, sage_weights_to_npz, agg_features, log_transform

import pytest
torch = pytest.importorskip("torch")          # parity tests need PyTorch Geometric; skipped in the backend image
pytest.importorskip("torch_geometric")
import torch.nn as nn, torch.nn.functional as F_
from torch_geometric.nn import SAGEConv, DirGNNConv


class SNet(nn.Module):        # same as the tuning notebook
    def __init__(self, d_in, h=32, L=3, dropout=0.3):
        super().__init__()
        dims = [d_in] + [h] * L
        self.convs = nn.ModuleList([DirGNNConv(SAGEConv(dims[i], dims[i + 1]), alpha=0.5, root_weight=True) for i in range(L)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(L)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(h + d_in, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 2))
    def forward(self, x, ei):
        h = x
        for i, (c, n) in enumerate(zip(self.convs, self.norms)):
            h = torch.relu(n(c(h, ei)))
            if i < len(self.convs) - 1:
                h = F_.dropout(h, self.dropout, self.training)
        return self.head(torch.cat([h, x], 1))


def _setup(seed=0, n=3000, m=12000):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m); dst = rng.integers(0, n, m)
    hub = rng.integers(0, n, 5); src[:800] = hub[0]; dst[800:1400] = hub[1]       # a couple of hubs
    keep = src != dst
    e = np.unique(np.stack([src[keep], dst[keep]], 1), axis=0); src, dst = e[:, 0], e[:, 1]
    S = np.abs(rng.lognormal(0, 1.5, (n, 12))).astype(np.float32); S[rng.random(n) < 0.1] = 0
    torch.manual_seed(seed)
    net = SNet(12)
    for nm in net.norms:                                  # non-trivial BatchNorm statistics
        nm.running_mean.uniform_(-0.5, 0.5); nm.running_var.uniform_(0.5, 2.0)
        nm.weight.data.uniform_(0.5, 1.5); nm.bias.data.uniform_(-0.2, 0.2)
    net.eval()
    mu = log_transform(S).mean(0); sd = log_transform(S).std(0) + 1e-6
    X = ((log_transform(S) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        z = net(torch.tensor(X), torch.tensor(np.vstack([src, dst]), dtype=torch.long)).numpy()
    logit_t = z[:, 1] - z[:, 0]
    path = os.path.join(tempfile.mkdtemp(), "w.npz")
    sage_weights_to_npz({k: v.numpy() for k, v in net.state_dict().items()}, {"layers": 3}, mu, sd,
                        [f"f{i}" for i in range(12)], path)
    m_ = StructDirSAGE.load(path); m_.prepare(S, src, dst)
    return m_, logit_t, src, dst


def test_forward_all_matches_torch():
    m, lt, _, _ = _setup()
    ln = m.forward_all()
    assert np.max(np.abs(ln - lt)) < 1e-4, np.max(np.abs(ln - lt))


def test_local_logit_matches_full():
    m, lt, src, dst = _setup(1)
    deg = np.bincount(np.concatenate([src, dst]), minlength=m.N)
    nodes = [int(np.argmax(deg)), int(np.argmin(deg)), 7, 123, 2999]
    for t in nodes:
        assert abs(m.local_logit(t) - lt[t]) < 1e-4, (t, m.local_logit(t), lt[t])


def test_explain_runs_and_is_consistent():
    m, lt, _, _ = _setup(2)
    e = m.explain(42)
    assert abs(e["logit"] - lt[42]) < 1e-4 and len(e["features"]) == 12
    assert all(np.isfinite(f["effect"]) for f in e["features"])
    assert all(d["direction"] in ("sender", "receiver") for d in e["neighbours"])
    # occluding a feature equals local_logit with that feature set to 0
    xo = m.X[42].copy(); xo[3] = 0
    f3 = [f for f in e["features"] if f["feature"] == "f3"][0]
    assert abs(f3["effect"] - (lt[42] - m.local_logit(42, x_t=xo))) < 1e-6


def test_agg_features_shape():
    rng = np.random.default_rng(0)
    Xb = np.abs(rng.normal(size=(50, 4))).astype(np.float32)
    src = rng.integers(0, 50, 200); dst = rng.integers(0, 50, 200)
    M = agg_features(Xb, src, dst, np.zeros(4, np.float32), np.ones(4, np.float32))
    assert M.shape == (50, 16) and np.isfinite(M).all()
