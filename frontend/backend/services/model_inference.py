"""
On-the-fly scoring for the Elliptic++ models (no PyTorch needed).

Files expected in `model_dir` (exported by ellipticpp_train.ipynb):
  model_meta.json, calibration.json, xgb_model.json, [tx_xgb_model.json], [gnn_weights.npz]

Usage:
    bundle = ModelBundle("backend/data/ellipticpp/model")
    wallet = bundle.score_wallet(features_dict)                 # XGBoost + SHAP
    tx = bundle.score_tx(tx_features_dict)                      # XGBoost for transactions
    gnn = GraphSAGEScorer(bundle, X_raw, src, dst)              # once at startup
    gnn.score([node_index])                                     # GraphSAGE for one/many wallets
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

BN_EPS = 1e-5


def _best_range(booster: xgb.Booster):
    """Trees up to the early-stopping best iteration (the saved model also contains later trees)."""
    best = booster.attr("best_iteration")
    return (0, int(best) + 1) if best is not None else (0, 0)


def _calibrate(raw, table):
    """sklearn IsotonicRegression(out_of_bounds='clip').predict == linear interpolation with clipping."""
    return np.interp(raw, table["x"], table["y"])


@dataclass
class ModelBundle:
    model_dir: str
    meta: dict = field(init=False)
    calibration: dict = field(init=False)
    wallet_booster: xgb.Booster = field(init=False)
    tx_booster: xgb.Booster | None = field(init=False, default=None)
    gnn_weights: dict | None = field(init=False, default=None)

    def __post_init__(self):
        p = lambda name: os.path.join(self.model_dir, name)
        self.meta = json.load(open(p("model_meta.json")))
        self.calibration = json.load(open(p("calibration.json")))
        self.wallet_features = self.meta["features"]
        self.tx_features = self.meta.get("tx_features") or []
        self.thresholds = self.meta["thresholds"]
        self.wallet_booster = xgb.Booster()
        self.wallet_booster.load_model(p("xgb_model.json"))
        if os.path.exists(p("tx_xgb_model.json")) and self.tx_features:
            self.tx_booster = xgb.Booster()
            self.tx_booster.load_model(p("tx_xgb_model.json"))
        if os.path.exists(p("gnn_weights.npz")):
            z = np.load(p("gnn_weights.npz"))
            self.gnn_weights = {k.replace("__", "."): z[k] for k in z.files}

    # ---------------- XGBoost ----------------
    def _xgb_score(self, booster, names, rows, cal_key, thr_key, top_k):
        X = np.asarray([[float(r.get(f, 0.0) or 0.0) for f in names] for r in rows], dtype=np.float32)
        d = xgb.DMatrix(X, feature_names=names)
        rng = _best_range(booster)
        raw = booster.predict(d, iteration_range=rng)
        contrib = booster.predict(d, pred_contribs=True, iteration_range=rng)[:, :-1] if top_k else None
        out = []
        for i in range(len(rows)):
            res = {"prob": float(_calibrate(raw[i], self.calibration[cal_key])), "raw": float(raw[i]),
                   "flag": bool(raw[i] >= self.thresholds[thr_key]),
                   "model_version": self.meta["model_version"]}
            if top_k:
                order = np.argsort(-np.abs(contrib[i]))[:top_k]
                res["explanations"] = [{"feature": names[j], "value": float(X[i, j]),
                                        "shap": float(contrib[i, j])} for j in order]
            out.append(res)
        return out

    def score_wallets(self, feature_rows: list[dict], top_k: int = 5) -> list[dict]:
        return self._xgb_score(self.wallet_booster, self.wallet_features, feature_rows, "xgb", "xgb_raw", top_k)

    def score_wallet(self, features: dict, top_k: int = 5) -> dict:
        return self.score_wallets([features], top_k)[0]

    def score_tx(self, features: dict, top_k: int = 0) -> dict | None:
        if self.tx_booster is None:
            return None
        return self._xgb_score(self.tx_booster, self.tx_features, [features], "tx", "tx_raw", top_k)[0]


def build_csr(src: np.ndarray, dst: np.ndarray, n: int):
    """Undirected, de-duplicated adjacency without self-loops (same as PyG to_undirected)."""
    keep = (src >= 0) & (dst >= 0) & (src != dst)
    s, d = src[keep].astype(np.int64), dst[keep].astype(np.int64)
    a = np.concatenate([s, d]); b = np.concatenate([d, s])
    key = np.unique(a * n + b)
    a, b = key // n, key % n
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    return np.cumsum(indptr), b


class GraphSAGEScorer:
    """Exact numpy re-implementation of the notebook's 2-layer GraphSAGE (mean aggregation, eval mode).

    A node's score depends only on its 2-hop neighbourhood, so it is computed on demand.
    New wallets/edges can be added by rebuilding the scorer with the new arrays.
    """

    def __init__(self, bundle: ModelBundle, X_raw: np.ndarray, src: np.ndarray, dst: np.ndarray,
                 max_neighbors: int | None = 5000, seed: int = 0):
        if bundle.gnn_weights is None:
            raise ValueError("gnn_weights.npz not found")
        self.w = bundle.gnn_weights
        self.bundle = bundle
        self.X_raw = X_raw.astype(np.float32, copy=False)
        self.n = len(X_raw)
        self.indptr, self.indices = build_csr(src, dst, self.n)
        self.max_neighbors = max_neighbors
        self.rng = np.random.default_rng(seed)
        self.mu, self.sd = self.w["mu"], self.w["sd"]

    def _x(self, idx):
        a = self.X_raw[idx]
        return ((np.sign(a) * np.log1p(np.abs(a))) - self.mu) / self.sd

    def _nbrs(self, i):
        nb = self.indices[self.indptr[i]:self.indptr[i + 1]]
        if self.max_neighbors and len(nb) > self.max_neighbors:
            return self.rng.choice(nb, self.max_neighbors, replace=False), True
        return nb, False

    def _bn(self, h, p):
        w = self.w
        return (h - w[p + ".running_mean"]) / np.sqrt(w[p + ".running_var"] + BN_EPS) * w[p + ".weight"] + w[p + ".bias"]

    def _sage(self, h_self, h_mean, p):
        w = self.w
        return h_mean @ w[p + ".lin_l.weight"].T + w[p + ".lin_l.bias"] + h_self @ w[p + ".lin_r.weight"].T

    def score(self, targets) -> dict:
        t0 = time.perf_counter()
        targets = np.asarray(targets, dtype=np.int64)
        approx = False
        nb1 = {}
        for t in targets:
            nb1[t], a = self._nbrs(t); approx |= a
        s1 = np.unique(np.concatenate([targets] + [nb1[t] for t in targets]))
        nb2 = {}
        for s in s1:
            nb2[s], a = self._nbrs(s); approx |= a
        need = np.unique(np.concatenate([s1] + [nb2[s] for s in s1]))
        pos = {v: i for i, v in enumerate(need)}
        x = self._x(need)

        # layer 1 for targets and their neighbours
        mean1 = np.stack([x[[pos[j] for j in nb2[s]]].mean(0) if len(nb2[s]) else np.zeros(x.shape[1], np.float32)
                          for s in s1])
        h1 = np.maximum(self._bn(self._sage(x[[pos[s] for s in s1]], mean1, "c1"), "n1"), 0)
        p1 = {v: i for i, v in enumerate(s1)}
        # layer 2 for targets
        mean2 = np.stack([h1[[p1[j] for j in nb1[t]]].mean(0) if len(nb1[t]) else np.zeros(h1.shape[1], np.float32)
                          for t in targets])
        h2 = np.maximum(self._bn(self._sage(h1[[p1[t] for t in targets]], mean2, "c2"), "n2"), 0)
        z = np.concatenate([h2, x[[pos[t] for t in targets]]], axis=1)
        z = np.maximum(z @ self.w["head.0.weight"].T + self.w["head.0.bias"], 0)
        logits = z @ self.w["head.3.weight"].T + self.w["head.3.bias"]
        e = np.exp(logits - logits.max(1, keepdims=True))
        raw = e[:, 1] / e.sum(1)
        cal = self.bundle.calibration.get("gnn")
        thr = self.bundle.thresholds.get("gnn_raw")
        return {"raw": raw.tolist(),
                "prob": (_calibrate(raw, cal) if cal else raw).tolist(),
                "flag": [bool(r >= thr) for r in raw] if thr is not None else [False] * len(raw),
                "approximate": approx, "nodes_used": int(len(need)),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "model_version": self.bundle.meta["model_version"]}
