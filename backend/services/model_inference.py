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

    def explain_tx(self, features: dict, top_k: int = 8) -> dict | None:
        """Compute TreeSHAP explanations for a transaction, grouped by feature type."""
        if self.tx_booster is None:
            return None

        names = self.tx_features
        X = np.asarray([float(features.get(f, 0.0) or 0.0) for f in names], dtype=np.float32)
        d = xgb.DMatrix(X[np.newaxis, :], feature_names=names)
        rng = _best_range(self.tx_booster)

        # Score and SHAP (use margin for SHAP values to match)
        raw_margin = float(self.tx_booster.predict(d, output_margin=True, iteration_range=rng)[0])
        contrib_full = self.tx_booster.predict(d, pred_contribs=True, iteration_range=rng)[0]
        contrib = contrib_full[:-1]
        base_value = float(contrib_full[-1])
        prob = float(_calibrate(raw_margin, self.calibration["tx"]))
        flag = bool(raw_margin >= self.thresholds["tx_raw"])

        # Categorize features
        interpretable = [
            "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
            "num_input_addresses", "num_output_addresses",
            "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
            "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
        ]
        groups_dict = {}
        for i, name in enumerate(names):
            if name in interpretable:
                group = "Elliptic++ (interpretable)"
            elif name.startswith("Local_"):
                group = "Local (anonymised)"
            elif name.startswith("Aggregate_"):
                group = "Neighbour aggregates (anonymised)"
            else:
                group = "Other"
            groups_dict.setdefault(group, [])
            groups_dict[group].append((i, name, contrib[i]))

        # Top-k features overall
        order = np.argsort(-np.abs(contrib))[:top_k]
        top_features = []
        for j in order:
            name = names[j]
            if name in interpretable:
                label = _TX_FEATURE_LABELS.get(name, name)
            elif name.startswith("Local_"):
                label = f"{name} (anonymised)"
            elif name.startswith("Aggregate_"):
                label = f"{name} (anonymised)"
            else:
                label = name
            top_features.append({
                "feature": name,
                "label": label,
                "group": next((g for g, items in groups_dict.items() if any(f[1] == name for f in items)), "Other"),
                "value": float(X[j]),
                "shap": float(contrib[j])
            })

        # Group sums
        groups = []
        for group_name in ["Local (anonymised)", "Neighbour aggregates (anonymised)", "Elliptic++ (interpretable)"]:
            if group_name in groups_dict:
                shap_sum = float(np.sum([c for _, _, c in groups_dict[group_name]]))
                groups.append({"group": group_name, "shap_sum": shap_sum})

        return {
            "prob": prob,
            "flag": flag,
            "raw": raw_margin,
            "base_value": base_value,
            "top": top_features,
            "groups": groups,
            "model_version": self.meta["model_version"]
        }


_TX_FEATURE_LABELS = {
    "in_txs_degree": "Number of inputs",
    "out_txs_degree": "Number of outputs",
    "total_BTC": "Total BTC",
    "fees": "Fee (BTC)",
    "size": "Transaction size (bytes)",
    "num_input_addresses": "Number of input addresses",
    "num_output_addresses": "Number of output addresses",
    "in_BTC_min": "Min input BTC",
    "in_BTC_max": "Max input BTC",
    "in_BTC_mean": "Mean input BTC",
    "in_BTC_median": "Median input BTC",
    "in_BTC_total": "Total input BTC",
    "out_BTC_min": "Min output BTC",
    "out_BTC_max": "Max output BTC",
    "out_BTC_mean": "Mean output BTC",
    "out_BTC_median": "Median output BTC",
    "out_BTC_total": "Total output BTC",
}


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
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mu, self.sd = self.w["mu"], self.w["sd"]

    def _x(self, idx):
        a = self.X_raw[idx]
        return ((np.sign(a) * np.log1p(np.abs(a))) - self.mu) / self.sd

    def _nbrs(self, i):
        nb = self.indices[self.indptr[i]:self.indptr[i + 1]]
        if self.max_neighbors and len(nb) > self.max_neighbors:
            # Use per-node seeded RNG for deterministic sampling
            rng = np.random.default_rng((self.seed, int(i)))
            return rng.choice(nb, self.max_neighbors, replace=False), True
        return nb, False

    def _bn(self, h, p):
        w = self.w
        return (h - w[p + ".running_mean"]) / np.sqrt(w[p + ".running_var"] + BN_EPS) * w[p + ".weight"] + w[p + ".bias"]

    def _sage(self, h_self, h_mean, p):
        w = self.w
        return h_mean @ w[p + ".lin_l.weight"].T + w[p + ".lin_l.bias"] + h_self @ w[p + ".lin_r.weight"].T

    def _neighbourhood(self, targets):
        """Build 2-hop neighbourhood. Returns (nb1, s1, nb2, need, pos) where:
        - nb1: dict mapping target -> 1-hop neighbours
        - s1: all nodes at distance <= 1
        - nb2: dict mapping s1 node -> 2-hop neighbours
        - need: all nodes needed (distance <= 2)
        - pos: mapping from node_idx -> position in need array
        """
        targets = np.asarray(targets, dtype=np.int64)
        approx = False
        nb1 = {}
        for t in targets:
            nb1[t], a = self._nbrs(t)
            approx |= a
        s1 = np.unique(np.concatenate([targets] + [nb1[t] for t in targets]))
        nb2 = {}
        for s in s1:
            nb2[s], a = self._nbrs(s)
            approx |= a
        need = np.unique(np.concatenate([s1] + [nb2[s] for s in s1]))
        pos = {v: i for i, v in enumerate(need)}
        return nb1, s1, nb2, need, pos, approx

    def _forward(self, x_need, s1, nb1, nb2, pos, targets):
        """Forward pass through 2-layer GraphSAGE. Returns (raw, h1, h2)."""
        # layer 1 for s1 nodes
        mean1 = np.stack([x_need[[pos[j] for j in nb2[s]]].mean(0) if len(nb2[s]) else np.zeros(x_need.shape[1], np.float32)
                          for s in s1])
        h1 = np.maximum(self._bn(self._sage(x_need[[pos[s] for s in s1]], mean1, "c1"), "n1"), 0)
        p1 = {v: i for i, v in enumerate(s1)}

        # layer 2 for targets
        mean2 = np.stack([h1[[p1[j] for j in nb1[t]]].mean(0) if len(nb1[t]) else np.zeros(h1.shape[1], np.float32)
                          for t in targets])
        h2 = np.maximum(self._bn(self._sage(h1[[p1[t] for t in targets]], mean2, "c2"), "n2"), 0)

        # head
        z = np.concatenate([h2, x_need[[pos[t] for t in targets]]], axis=1)
        z = np.maximum(z @ self.w["head.0.weight"].T + self.w["head.0.bias"], 0)
        logits = z @ self.w["head.3.weight"].T + self.w["head.3.bias"]
        e = np.exp(logits - logits.max(1, keepdims=True))
        raw = e[:, 1] / e.sum(1)
        return raw, h1, h2, p1

    def score(self, targets) -> dict:
        t0 = time.perf_counter()
        targets = np.asarray(targets, dtype=np.int64)
        nb1, s1, nb2, need, pos, approx = self._neighbourhood(targets)
        x = self._x(need)
        raw, h1, h2, p1 = self._forward(x, s1, nb1, nb2, pos, targets)

        cal = self.bundle.calibration.get("gnn")
        thr = self.bundle.thresholds.get("gnn_raw")
        return {"raw": raw.tolist(),
                "prob": (_calibrate(raw, cal) if cal else raw).tolist(),
                "flag": [bool(r >= thr) for r in raw] if thr is not None else [False] * len(raw),
                "approximate": approx, "nodes_used": int(len(need)),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "model_version": self.bundle.meta["model_version"]}

    def explain(self, target_idx: int, top_features: int = 8, top_neighbours: int = 10) -> dict:
        """GraphSAGE explanations via occlusion (features set to mean, neighbours removed).

        Returns:
        - graph_effect: {prob_full, prob_isolated, delta_logodds}
        - features: [{feature, label, value, effect}]
        - neighbours: [{address, effect, model_risk, known_label}]
        - approximate: bool (if neighbourhood was sampled)
        - ms: computation time
        """
        t0 = time.perf_counter()

        def _logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        # Full score
        nb1, s1, nb2, need, pos, approx = self._neighbourhood([target_idx])
        x = self._x(need)
        raw_full, h1, h2, p1 = self._forward(x, s1, nb1, nb2, pos, [target_idx])
        raw_full = raw_full[0]

        cal = self.bundle.calibration.get("gnn")
        prob_full = float(_calibrate(raw_full, cal) if cal else raw_full)

        # (a) Isolated score (no neighbours at either layer)
        nb1_iso = {target_idx: np.array([], dtype=np.int64)}
        s1_iso = np.array([target_idx])
        nb2_iso = {target_idx: np.array([], dtype=np.int64)}
        need_iso = np.array([target_idx])
        pos_iso = {target_idx: 0}
        x_iso = self._x(need_iso)
        raw_iso, _, _, _ = self._forward(x_iso, s1_iso, nb1_iso, nb2_iso, pos_iso, [target_idx])
        raw_iso = raw_iso[0]
        prob_isolated = float(_calibrate(raw_iso, cal) if cal else raw_iso)
        delta_logodds = float(_logit(raw_full) - _logit(raw_iso))

        # (b) Feature occlusion
        feature_effects = []
        target_pos_in_need = pos[target_idx]
        for feat_idx in range(x.shape[1]):
            x_occluded = x.copy()
            # Set feature to 0 (training mean in standardized space) wherever target appears
            for i, node in enumerate(need):
                if node == target_idx:
                    x_occluded[i, feat_idx] = 0.0
            raw_occ, _, _, _ = self._forward(x_occluded, s1, nb1, nb2, pos, [target_idx])
            effect = float(_logit(raw_full) - _logit(raw_occ[0]))
            feature_effects.append((feat_idx, effect, x[target_pos_in_need, feat_idx]))

        # Top features by |effect|
        feature_effects.sort(key=lambda f: abs(f[1]), reverse=True)
        top_feat = []
        for feat_idx, effect, value in feature_effects[:top_features]:
            feat_name = self.bundle.wallet_features[feat_idx]
            from services.model_service import _FEATURE_LABELS
            label = _FEATURE_LABELS.get(feat_name, feat_name.replace("_", " ").capitalize())
            top_feat.append({
                "feature": feat_name,
                "label": label,
                "value": float(value),
                "effect": effect
            })

        # (c) Neighbour leave-one-out
        direct_neighbours = nb1[target_idx]
        evaluate_limit = 300
        evaluate_nbrs = direct_neighbours
        if len(direct_neighbours) > evaluate_limit:
            # Sample most-sampled neighbours (deterministic by frequency in sampling)
            approx = True
            # For simplicity, take first 300 (they are already randomly sampled in _nbrs)
            evaluate_nbrs = direct_neighbours[:evaluate_limit]

        neighbour_effects = []
        target_p1 = p1[target_idx]
        for nbr_idx in evaluate_nbrs:
            if nbr_idx not in p1:
                continue
            # Recompute layer-2 mean without this neighbour
            other_nbrs = [j for j in nb1[target_idx] if j != nbr_idx]
            if other_nbrs:
                mean2_without = h1[[p1[j] for j in other_nbrs]].mean(0, keepdims=True)
            else:
                mean2_without = np.zeros((1, h1.shape[1]), np.float32)

            h2_without = np.maximum(self._bn(
                self._sage(h1[[target_p1]], mean2_without, "c2"), "n2"), 0)
            z = np.concatenate([h2_without, x[[pos[target_idx]]]], axis=1)
            z = np.maximum(z @ self.w["head.0.weight"].T + self.w["head.0.bias"], 0)
            logits = z @ self.w["head.3.weight"].T + self.w["head.3.bias"]
            e = np.exp(logits - logits.max(1, keepdims=True))
            raw_without = (e[:, 1] / e.sum(1))[0]

            effect = float(_logit(raw_full) - _logit(raw_without))
            neighbour_effects.append((nbr_idx, effect))

        # Top neighbours by |effect|
        neighbour_effects.sort(key=lambda n: abs(n[1]), reverse=True)
        top_nbrs = []
        for nbr_idx, effect in neighbour_effects[:top_neighbours]:
            # Get neighbour's own GNN score and label
            nbr_result = self.score([nbr_idx])
            top_nbrs.append({
                "address": int(nbr_idx),  # Will be mapped to address string in service
                "effect": float(effect),
                "model_risk": float(nbr_result["prob"][0]),
                "known_label": None  # Will be filled in service layer
            })

        ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "graph_effect": {
                "prob_full": float(prob_full),
                "prob_isolated": float(prob_isolated),
                "delta_logodds": float(delta_logodds)
            },
            "features": top_feat,
            "neighbours": top_nbrs,
            "approximate": bool(approx),
            "ms": float(ms)
        }
