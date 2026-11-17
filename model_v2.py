"""Numpy inference for the v2 wallet models (no PyTorch needed in the backend).

* StructDirSAGE  — tuned GraphSAGE on 12 structural features: 3 layers of DirGNNConv(SAGEConv, alpha=0.5,
                   root_weight=True) (mean over senders and over receivers separately), BatchNorm (eval), ReLU,
                   skip head on concat(h, x). Exact re-implementation of the PyTorch Geometric model.
* agg_features   — XGBoost v2 inputs: 51 base wallet features + neighbour aggregates (mean / max over 1 hop,
                   mean over 2 hops) of the standardised log base features, on the undirected graph.

Edges are directed (src -> dst = sender -> receiver) and must be the same edge list the model was trained on
(deduplicated AddrAddr edges, self-loops removed).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

STRUCT_FEATURES = ["deg_in", "deg_out", "deg", "share_in", "reciprocity", "triangles", "clustering",
                   "kcore", "pagerank", "nbr_deg_mean", "nbr_deg_max", "two_hop_reach"]


def log_transform(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _mean_adj(rows, cols, n):
    """Row-normalised adjacency: (A @ H)[i] = mean of H over i's neighbours (multi-edges counted)."""
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0).astype(np.float32)
    return (sp.diags(inv) @ A).tocsr()


class StructDirSAGE:
    def __init__(self, weights: dict):
        w = {k: np.asarray(v) for k, v in weights.items()}
        self.L = int(w["n_layers"]); self.alpha = float(w["alpha"])
        self.mu, self.sd = w["mu"].astype(np.float32), w["sd"].astype(np.float32)
        self.layers = []
        for l in range(self.L):
            p = f"convs__{l}__"
            b = f"norms__{l}__"
            self.layers.append({
                "Win": w[p + "conv_in__lin_l__weight"].T.astype(np.float32), "bin": w[p + "conv_in__lin_l__bias"].astype(np.float32),
                "Wout": w[p + "conv_out__lin_l__weight"].T.astype(np.float32), "bout": w[p + "conv_out__lin_l__bias"].astype(np.float32),
                "Wr": w[p + "lin__weight"].T.astype(np.float32), "br": w[p + "lin__bias"].astype(np.float32),
                "scale": (w[b + "weight"] / np.sqrt(w[b + "running_var"] + float(w["bn_eps"]))).astype(np.float32),
                "shift": (w[b + "bias"] - w[b + "running_mean"] * w[b + "weight"] / np.sqrt(w[b + "running_var"] + float(w["bn_eps"]))).astype(np.float32),
            })
        self.W1, self.b1 = w["head__0__weight"].T.astype(np.float32), w["head__0__bias"].astype(np.float32)
        self.W2, self.b2 = w["head__3__weight"].T.astype(np.float32), w["head__3__bias"].astype(np.float32)
        self.features = [str(f) for f in w["features"]]

    @classmethod
    def load(cls, path: str) -> "StructDirSAGE":
        return cls(dict(np.load(path, allow_pickle=False)))

    # ---- graph -------------------------------------------------------------------------------------
    def prepare(self, S_raw: np.ndarray, src: np.ndarray, dst: np.ndarray):
        """S_raw: (N, 12) raw structural features in self.features order; src/dst: directed edges."""
        self.N = S_raw.shape[0]
        self.X = ((log_transform(S_raw.astype(np.float32)) - self.mu) / self.sd).astype(np.float32)
        self.A_in = _mean_adj(dst, src, self.N)    # row i: mean over senders j (j -> i)
        self.A_out = _mean_adj(src, dst, self.N)   # row i: mean over receivers j (i -> j)
        self.nbrs = ((self.A_in + self.A_out) > 0).tocsr()

    def _layer(self, l, agg_in, agg_out, h_self):
        p = self.layers[l]
        z = (self.alpha * (agg_out @ p["Wout"] + p["bout"]) + (1 - self.alpha) * (agg_in @ p["Win"] + p["bin"])
             + h_self @ p["Wr"] + p["br"])
        return np.maximum(z * p["scale"] + p["shift"], 0).astype(np.float32)

    def _head(self, h, x):
        z = np.maximum(np.concatenate([h, x], 1) @ self.W1 + self.b1, 0) @ self.W2 + self.b2
        return (z[:, 1] - z[:, 0]).astype(np.float64)                      # logit of class 1

    @staticmethod
    def prob(logit):
        return 1.0 / (1.0 + np.exp(-np.asarray(logit, np.float64)))

    def forward_all(self) -> np.ndarray:
        """Logits for all nodes (exact, full graph)."""
        H = self.X
        for l in range(self.L):
            H = self._layer(l, self.A_in @ H, self.A_out @ H, H)
        return self._head(H, self.X)

    # ---- local, exact computation for one node (used by explanations) ------------------------------
    def _rings(self, t):
        """R[k] = nodes within k hops of t (undirected), k = 0..L-1, as sorted index arrays."""
        R = [np.array([t])]
        for _ in range(self.L - 1):
            cur = R[-1]
            R.append(np.union1d(cur, self.nbrs[cur].indices))
        return R

    def _hidden_rows(self, rows, n_layers, t=None, x_t=None):
        """Outputs of the first `n_layers` layers at `rows` (exact, computed locally).
        If x_t is given, node t's input features are replaced by x_t."""
        if n_layers == 0:
            Xr = self.X[rows]
            if x_t is not None:
                Xr = Xr.copy(); Xr[rows == t] = x_t
            return Xr
        cols = np.union1d(rows, self.nbrs[rows].indices)
        Hc = self._hidden_rows(cols, n_layers - 1, t, x_t)
        ai = self.A_in[rows][:, cols] @ Hc
        ao = self.A_out[rows][:, cols] @ Hc
        return self._layer(n_layers - 1, ai, ao, Hc[np.searchsorted(cols, rows)])

    def local_logit(self, t, x_t=None):
        """Logit of node t from its L-hop neighbourhood only (equals forward_all()[t])."""
        rows = np.array([t])
        h = self._hidden_rows(rows, self.L, t, x_t)
        x = (self.X[t] if x_t is None else x_t)[None, :]
        return float(self._head(h, x.astype(np.float32))[0])

    def explain(self, t, labels=None, top_neighbours=10, max_neighbours=300):
        """Occlusion of t's structural features (set to training mean = 0), isolated score and the effect of
        removing each direct neighbour from t's last-layer aggregation (senders and receivers separately)."""
        t0 = time.time()
        base = self.local_logit(t)
        feats = []
        for j, name in enumerate(self.features):
            xo = self.X[t].copy(); xo[j] = 0.0
            feats.append({"feature": name, "label": (labels or {}).get(name, name),
                          "value_std": float(self.X[t, j]), "effect": base - self.local_logit(t, x_t=xo)})
        feats.sort(key=lambda d: -abs(d["effect"]))

        # isolated: no neighbours at any layer (aggregations = 0)
        h = self.X[t][None, :]
        for l in range(self.L):
            z0 = np.zeros((1, h.shape[1]), np.float32)
            h = self._layer(l, z0, z0, h)
        isolated = float(self._head(h, self.X[t][None, :])[0])

        # neighbour leave-one-out at the last layer
        R = self._rings(t)
        prev_idx = R[1] if self.L > 1 else np.array([t])
        H = self._hidden_rows(prev_idx, self.L - 1)            # layer L-2 outputs (input of last layer) at R[1]
        pos = {int(n): i for i, n in enumerate(prev_idx)}
        ai_row, ao_row = self.A_in[t], self.A_out[t]
        agg_in = (ai_row[:, prev_idx] @ H).ravel(); agg_out = (ao_row[:, prev_idx] @ H).ravel()
        h_self = H[pos[int(t)]][None, :]
        effects = []
        for direction, row, agg_v in (("sender", ai_row, agg_in), ("receiver", ao_row, agg_out)):
            js, wj = row.indices, row.data
            approx = len(js) > max_neighbours
            if approx:
                order = np.argsort(-wj)[:max_neighbours]; js, wj = js[order], wj[order]
            if len(js) == 0:
                continue
            Hj = H[[pos[int(j)] for j in js]]
            with np.errstate(divide="ignore", invalid="ignore"):
                new = np.where((wj < 1)[:, None], (agg_v[None, :] - wj[:, None] * Hj) / (1 - wj[:, None]), 0.0)
            AI = np.repeat(agg_in[None, :], len(js), 0); AO = np.repeat(agg_out[None, :], len(js), 0)
            if direction == "sender":
                AI = new.astype(np.float32)
            else:
                AO = new.astype(np.float32)
            hL = self._layer(self.L - 1, AI, AO, np.repeat(h_self, len(js), 0))
            lj = self._head(hL, np.repeat(self.X[t][None, :], len(js), 0))
            for j, l_wo in zip(js, lj):
                effects.append({"node": int(j), "direction": direction, "effect": base - float(l_wo), "approximate": approx})
        effects.sort(key=lambda d: -abs(d["effect"]))
        return {"logit": base, "isolated_logit": isolated, "features": feats,
                "neighbours": effects[:top_neighbours], "ms": int((time.time() - t0) * 1000)}


def agg_features(X_base_raw: np.ndarray, src: np.ndarray, dst: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """XGBoost v2 input matrix: [base (raw) | nbr_mean | nbr_max | nbr2_mean] of standardised log base features.

    Identical to graph_features.neighbour_aggregates (undirected, deduplicated, self-loops removed)."""
    n = X_base_raw.shape[0]
    Xs = ((log_transform(X_base_raw.astype(np.float32)) - mu) / sd).astype(np.float32)
    keep = src != dst
    e = np.unique(np.stack([src[keep], dst[keep]], 1), axis=0)
    U = sp.csr_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n, n))
    U = ((U + U.T) > 0).astype(np.float32).tocsr()
    deg = np.asarray(U.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0).astype(np.float32)
    P = sp.diags(inv) @ U
    mean1 = P @ Xs
    mean2 = P @ mean1
    max1 = np.zeros_like(Xs)
    nz = np.diff(U.indptr) > 0
    starts = U.indptr[:-1][nz]
    for j in range(Xs.shape[1]):
        max1[nz, j] = np.maximum.reduceat(Xs[U.indices, j], starts)
    return np.concatenate([X_base_raw.astype(np.float32), mean1, max1, mean2], 1)


def agg_feature_names(base: list[str]) -> list[str]:
    return list(base) + [f"{t}__{b}" for t in ("nbr_mean", "nbr_max", "nbr2_mean") for b in base]


def agg_feature_label(name: str, base_labels: dict) -> str:
    for tag, txt in (("nbr_mean__", "Neighbours' mean · "), ("nbr_max__", "Neighbours' max · "), ("nbr2_mean__", "2-hop mean · ")):
        if name.startswith(tag):
            b = name[len(tag):]
            return txt + base_labels.get(b, b)
    return base_labels.get(name, name)


def sage_weights_to_npz(state_dict: dict, cfg: dict, mu, sd, features, path: str, bn_eps: float = 1e-5):
    """Save a trained SNet (tuning notebook) as an .npz StructDirSAGE can load.
    state_dict values must be numpy arrays (e.g. {k: v.cpu().numpy() for k, v in model.state_dict().items()})."""
    out = {k.replace(".", "__"): np.asarray(v) for k, v in state_dict.items() if not k.endswith("num_batches_tracked")}
    out.update(n_layers=np.array(cfg["layers"]), alpha=np.array(0.5), bn_eps=np.array(bn_eps),
               mu=np.asarray(mu, np.float32), sd=np.asarray(sd, np.float32), features=np.array(list(features)))
    np.savez(path, **out)
