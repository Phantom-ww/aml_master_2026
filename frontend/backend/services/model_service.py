"""On-the-fly scoring: XGBoost and GraphSAGE for Elliptic++ addresses."""
import time
import threading
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional
from services.model_inference import ModelBundle, GraphSAGEScorer

DATA_DIR = Path(__file__).parent.parent / "data" / "ellipticpp"
MODEL_DIR = DATA_DIR / "model"


class ModelService:
    def __init__(self):
        self._bundle: Optional[ModelBundle] = None
        self._gnn_scorer: Optional[GraphSAGEScorer] = None
        self._address_idx: Optional[dict] = None
        self._in_sample: Optional[set] = None
        self._X_raw: Optional[np.ndarray] = None
        self._load_error: Optional[str] = None
        self._ready = False
        self._tx_cache: dict = {}          # txId -> {prob, flag} (None if tx has no features)
        self._db_lock = threading.Lock()   # DuckDB connection is not thread-safe
        self._con = None

    def load(self, epp_service):
        try:
            t0 = time.perf_counter()

            # Check required files
            required = [
                MODEL_DIR / "model_meta.json",
                MODEL_DIR / "calibration.json",
                MODEL_DIR / "xgb_model.json",
                DATA_DIR / "wallets.parquet",
                DATA_DIR / "addr_edges.parquet",
                MODEL_DIR / "predictions.parquet"
            ]
            missing = [f.name for f in required if not f.exists()]
            if missing:
                self._load_error = f"Missing files: {', '.join(missing)}"
                print(f"⚠ Model service: {self._load_error}")
                return

            # Load bundle
            self._bundle = ModelBundle(str(MODEL_DIR))
            print(f"Model bundle loaded: v{self._bundle.meta['model_version']}")

            # Share address index from EllipticPPService
            self._address_idx = epp_service.address_index
            print(f"Address index: {len(self._address_idx)} wallets")

            # Build X_raw from wallets.parquet (order must match address_idx)
            # Vectorized: one float32 matrix, NaN -> 0, no extra pandas copy.
            import pyarrow.parquet as pq
            cols = list(self._bundle.wallet_features)
            table = pq.read_table(DATA_DIR / "wallets.parquet", columns=["address"] + cols)
            addrs = table.column("address").to_pylist()
            # Reorder rows to match the shared address index (wallets.parquet order)
            pos = {a: i for i, a in enumerate(addrs)}
            order = np.array([pos.get(a, -1) for a in self._address_idx.keys()], dtype=np.int64)
            present = order >= 0
            X_raw = np.zeros((len(order), len(cols)), dtype=np.float32)
            if present.any():
                data = table.select(cols).to_pandas().to_numpy(dtype=np.float64, copy=False)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                X_raw[present] = data[order[present]].astype(np.float32, copy=False)
            self._X_raw = X_raw
            x_mb = self._X_raw.nbytes / 1024**2
            print(f"X_raw: {self._X_raw.shape[0]} × {self._X_raw.shape[1]}, {x_mb:.1f} MB")

            # Load GNN graph from addr_edges.parquet
            import pyarrow.parquet as pq
            edge_tbl = pq.read_table(DATA_DIR / "addr_edges.parquet",
                                     columns=["input_address", "output_address"])
            src_list = edge_tbl.column("input_address").to_pylist()
            dst_list = edge_tbl.column("output_address").to_pylist()
            src_idx = np.fromiter((self._address_idx.get(a, -1) for a in src_list),
                                  dtype=np.int32, count=len(src_list))
            dst_idx = np.fromiter((self._address_idx.get(a, -1) for a in dst_list),
                                  dtype=np.int32, count=len(dst_list))
            del src_list, dst_list, edge_tbl
            edge_mb = (src_idx.nbytes + dst_idx.nbytes) / 1024**2
            print(f"GNN edges: {len(src_idx)} edges, {edge_mb:.1f} MB")

            # Build GraphSAGE scorer
            if (MODEL_DIR / "gnn_weights.npz").exists():
                try:
                    t_gnn = time.perf_counter()
                    print(f"Building GraphSAGE scorer for {len(src_idx)} edges...")
                    self._gnn_scorer = GraphSAGEScorer(self._bundle, self._X_raw, src_idx, dst_idx)
                    print(f"GraphSAGE scorer ready in {time.perf_counter() - t_gnn:.2f}s")
                except Exception as e:
                    print(f"⚠ GraphSAGE initialization failed: {e}, GNN scoring disabled")
            else:
                print("⚠ gnn_weights.npz not found, GNN scoring disabled")

            # Load in-sample addresses from predictions.parquet
            preds_tbl = pq.read_table(MODEL_DIR / "predictions.parquet", columns=["address", "split"])
            preds_addr = preds_tbl.column("address").to_pylist()
            preds_split = preds_tbl.column("split").to_pylist()
            self._in_sample = {a for a, s in zip(preds_addr, preds_split) if s in ("train", "val")}
            print(f"In-sample: {len(self._in_sample)} addresses")

            self._ready = True
            elapsed = time.perf_counter() - t0
            print(f"Model service ready in {elapsed:.2f}s")

        except Exception as e:
            import traceback
            self._load_error = f"Failed to load model service: {e}"
            print(f"⚠ {self._load_error}")
            traceback.print_exc()

    def reload(self):
        """Reload models (for updated data)."""
        from services.ellipticpp_service import get_service as get_epp_service
        self.__init__()
        self.load(get_epp_service())

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def get_in_sample(self, address: str) -> bool:
        """Check if address was in training/val set."""
        return address in self._in_sample if self._in_sample else False

    def score_wallet_xgb(self, address: str, wallet_features: tuple) -> Optional[dict]:
        """Score one wallet with XGBoost + SHAP explanations."""
        if not self._ready or not self._bundle:
            return None

        try:
            feature_dict = dict(zip(self._bundle.wallet_features, wallet_features))
            result = self._bundle.score_wallet(feature_dict, top_k=5)

            # Enrich explanations with human-readable labels
            for exp in result.get("explanations", []):
                exp["label"] = self._feature_label(exp["feature"])

            return result
        except Exception as e:
            print(f"XGBoost scoring error for {address}: {e}")
            return None

    def score_wallet_gnn(self, address: str) -> Optional[dict]:
        """Score one wallet with GraphSAGE."""
        if not self._ready or not self._gnn_scorer or address not in self._address_idx:
            return None

        try:
            idx = self._address_idx[address]
            result = self._gnn_scorer.score([idx])
            return {
                "prob": result["prob"][0],
                "flag": result["flag"][0],
                "approximate": result["approximate"]
            }
        except Exception as e:
            print(f"GNN scoring error for {address}: {e}")
            return None

    # ------------------------------------------------------------------ batch helpers
    def score_txs(self, tx_ids) -> dict:
        """Score many transactions at once: one DuckDB query + one XGBoost call. Returns {txId: {prob, flag}}."""
        out = {}
        if not self._ready or not self._bundle or not self._bundle.tx_booster:
            return out
        ids = []
        for t in tx_ids:
            try:
                t = int(t)
            except (TypeError, ValueError):
                continue
            if t in self._tx_cache:
                if self._tx_cache[t] is not None:
                    out[t] = self._tx_cache[t]
            else:
                ids.append(t)
        ids = list(dict.fromkeys(ids))
        if not ids:
            return out
        rows = self._fetch_tx_features(ids)
        if rows:
            scored = self._bundle._xgb_score(self._bundle.tx_booster, self._bundle.tx_features,
                                             [r for _, r in rows], "tx", "tx_raw", 0)
            for (t, _), res in zip(rows, scored):
                block = {"prob": res["prob"], "flag": res["flag"]}
                self._tx_cache[t] = block
                out[t] = block
        found = {t for t, _ in rows}
        for t in ids:
            if t not in found:
                self._tx_cache[t] = None
        if len(self._tx_cache) > 50_000:
            self._tx_cache.clear()
        return out

    def _fetch_tx_features(self, ids: list) -> list:
        """[(txId, {feature: value})] for ids present in tx_features.parquet (explicit column order)."""
        tx_path = DATA_DIR / "tx_features.parquet"
        if not tx_path.exists():
            return []
        cols = self._bundle.tx_features
        col_sql = ", ".join(f'"{c}"' for c in cols)
        id_list = ", ".join(str(int(i)) for i in ids)
        with self._db_lock:
            if self._con is None:
                self._con = duckdb.connect(":memory:")
            res = self._con.execute(
                f"SELECT txId, {col_sql} FROM read_parquet('{tx_path}') WHERE txId IN ({id_list})"
            ).fetchall()
        return [(int(r[0]), {c: (float(v) if v is not None else 0.0) for c, v in zip(cols, r[1:])}) for r in res]

    def score_risk(self, addresses) -> dict:
        """Compact risk for graph coloring: {address: prob} (GraphSAGE, fallback XGBoost). No SHAP."""
        out = {}
        if not self._ready or not self._bundle:
            return out
        known = [a for a in dict.fromkeys(addresses) if a in self._address_idx]
        if not known:
            return out
        idx = np.array([self._address_idx[a] for a in known], dtype=np.int64)
        if self._gnn_scorer is not None:
            try:
                res = self._gnn_scorer.score(idx)
                return {a: float(p) for a, p in zip(known, res["prob"])}
            except Exception as e:
                print(f"GNN batch scoring error: {e}")
        rows = [dict(zip(self._bundle.wallet_features, self._X_raw[i])) for i in idx]
        res = self._bundle.score_wallets(rows, top_k=0)
        return {a: r["prob"] for a, r in zip(known, res)}

    def score_tx(self, txId: int) -> Optional[dict]:
        """Score one transaction with XGBoost."""
        if not self._ready or not self._bundle or not self._bundle.tx_booster:
            return None

        try:
            return self.score_txs([txId]).get(int(txId))
        except Exception as e:
            print(f"Tx scoring error for {txId}: {e}")
            return None

    def _feature_label(self, feature: str) -> str:
        """Human-readable feature labels."""
        return _FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())

    def get_model_meta(self) -> dict:
        """Return model metadata."""
        if not self._ready or not self._bundle:
            return
        return {
            "model_version": self._bundle.meta["model_version"],
            "metrics": self._bundle.meta.get("metrics", {}),
            "thresholds": self._bundle.thresholds
        }


# Human-readable labels for the 51 wallet features used by the XGBoost model.
_FEATURE_LABELS = {
    "num_txs_as_sender": "Transactions Sent",
    "num_txs_as receiver": "Transactions Received",
    "lifetime_in_blocks": "Wallet Lifetime (blocks)",
    "total_txs": "Total Transactions",
    "num_timesteps_appeared_in": "Active Time Steps",
    "btc_transacted_total": "Total BTC Transacted",
    "btc_transacted_min": "Min BTC Transacted",
    "btc_transacted_max": "Max BTC Transacted",
    "btc_transacted_mean": "Mean BTC Transacted",
    "btc_transacted_median": "Median BTC Transacted",
    "btc_sent_total": "Total BTC Sent",
    "btc_sent_min": "Min BTC Sent",
    "btc_sent_max": "Max BTC Sent",
    "btc_sent_mean": "Mean BTC Sent",
    "btc_sent_median": "Median BTC Sent",
    "btc_received_total": "Total BTC Received",
    "btc_received_min": "Min BTC Received",
    "btc_received_max": "Max BTC Received",
    "btc_received_mean": "Mean BTC Received",
    "btc_received_median": "Median BTC Received",
    "fees_total": "Total Fees Paid",
    "fees_min": "Min Fee Paid",
    "fees_max": "Max Fee Paid",
    "fees_mean": "Mean Fee Paid",
    "fees_median": "Median Fee Paid",
    "fees_as_share_total": "Fees as Share of Total",
    "fees_as_share_min": "Min Fee Share",
    "fees_as_share_max": "Max Fee Share",
    "fees_as_share_mean": "Mean Fee Share",
    "fees_as_share_median": "Median Fee Share",
    "blocks_btwn_txs_total": "Blocks Between Transactions (total)",
    "blocks_btwn_txs_min": "Min Blocks Between Transactions",
    "blocks_btwn_txs_max": "Max Blocks Between Transactions",
    "blocks_btwn_txs_mean": "Mean Blocks Between Transactions",
    "blocks_btwn_txs_median": "Median Blocks Between Transactions",
    "blocks_btwn_input_txs_total": "Blocks Between Input Transactions (total)",
    "blocks_btwn_input_txs_min": "Min Blocks Between Inputs",
    "blocks_btwn_input_txs_max": "Max Blocks Between Inputs",
    "blocks_btwn_input_txs_mean": "Mean Blocks Between Inputs",
    "blocks_btwn_input_txs_median": "Median Blocks Between Inputs",
    "blocks_btwn_output_txs_total": "Blocks Between Output Transactions (total)",
    "blocks_btwn_output_txs_min": "Min Blocks Between Outputs",
    "blocks_btwn_output_txs_max": "Max Blocks Between Outputs",
    "blocks_btwn_output_txs_mean": "Mean Blocks Between Outputs",
    "blocks_btwn_output_txs_median": "Median Blocks Between Outputs",
    "num_addr_transacted_multiple": "Addresses Transacted Multiple Times",
    "transacted_w_address_total": "Total Counterparties",
    "transacted_w_address_min": "Min Counterparties",
    "transacted_w_address_max": "Max Counterparties",
    "transacted_w_address_mean": "Mean Counterparties",
    "transacted_w_address_median": "Median Counterparties",
}


_service = ModelService()


def get_service() -> ModelService:
    return _service
