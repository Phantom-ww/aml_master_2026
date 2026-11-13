"""On-the-fly scoring: XGBoost and GraphSAGE for Elliptic++ addresses."""
import time
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional
from services.model_inference import ModelBundle, GraphSAGEScorer
from functools import lru_cache

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

    def score_risk(self, addresses: list[str]) -> dict[str, float]:
        """Batch score addresses and return {address: risk_prob}.

        Uses GraphSAGE for all addresses if available, otherwise XGBoost as fallback.
        """
        if not self._ready or not self._address_idx:
            return {}

        # Filter to addresses we have
        valid = [a for a in addresses if a in self._address_idx]
        if not valid:
            return {}

        result = {}

        # Try GraphSAGE first (batch scoring)
        if self._gnn_scorer:
            try:
                indices = [self._address_idx[a] for a in valid]
                gnn_result = self._gnn_scorer.score(indices)
                for addr, prob in zip(valid, gnn_result["prob"]):
                    result[addr] = float(prob)
                return result
            except Exception as e:
                print(f"GNN batch scoring failed: {e}, falling back to XGBoost")

        # Fallback to XGBoost (individual scoring, slower)
        for addr in valid:
            try:
                idx = self._address_idx[addr]
                features = tuple(self._X_raw[idx])
                xgb_result = self.score_wallet_xgb(addr, features)
                if xgb_result:
                    result[addr] = xgb_result["prob"]
            except Exception as e:
                print(f"XGBoost scoring error for {addr}: {e}")

        return result

    def score_tx(self, txId: int) -> Optional[dict]:
        """Score one transaction with XGBoost."""
        if not self._ready or not self._bundle or not self._bundle.tx_booster:
            return None

        try:
            # Fetch transaction features from tx_features.parquet via DuckDB (cached)
            tx_features = self._get_tx_features(txId)
            if not tx_features:
                return None

            result = self._bundle.score_tx(tx_features, top_k=0)
            return result
        except Exception as e:
            print(f"Tx scoring error for {txId}: {e}")
            return None

    def explain_tx(self, txId: int, top_k: int = 8) -> Optional[dict]:
        """Explain one transaction with TreeSHAP."""
        if not self._ready or not self._bundle or not self._bundle.tx_booster:
            return None

        try:
            tx_features = self._get_tx_features(txId)
            if not tx_features:
                return None

            result = self._bundle.explain_tx(tx_features, top_k=top_k)
            return result
        except Exception as e:
            print(f"Tx explanation error for {txId}: {e}")
            return None

    def explain_wallet_gnn(self, address: str, top_features: int = 8, top_neighbours: int = 10) -> Optional[dict]:
        """Explain one wallet with GraphSAGE occlusion."""
        if not self._ready or not self._gnn_scorer or address not in self._address_idx:
            return None

        try:
            idx = self._address_idx[address]
            result = self._gnn_scorer.explain(idx, top_features=top_features, top_neighbours=top_neighbours)

            # Map neighbour indices back to addresses and add known labels
            idx_to_addr = {v: k for k, v in self._address_idx.items()}
            from services.ellipticpp_service import get_service as get_epp_service
            epp_svc = get_epp_service()

            for nbr in result["neighbours"]:
                nbr_idx = nbr["address"]  # Already an int now
                nbr_addr = idx_to_addr.get(nbr_idx)
                if nbr_addr:
                    nbr["address"] = nbr_addr
                    # Get known label from EllipticPP service
                    if epp_svc.is_ready:
                        wallet = epp_svc.get_wallet(nbr_addr)
                        if wallet and wallet["class"] != 3:
                            nbr["known_label"] = 1 if wallet["class"] == 1 else 0
                else:
                    # Fallback: keep as string index if address not found
                    nbr["address"] = str(nbr_idx)

            return result
        except Exception as e:
            print(f"GNN explanation error for {address}: {e}")
            import traceback
            traceback.print_exc()
            return None

    @lru_cache(maxsize=1024)
    def _get_tx_features(self, txId: int) -> Optional[dict]:
        """Fetch transaction features from tx_features.parquet (LRU cached)."""
        tx_path = DATA_DIR / "tx_features.parquet"
        if not tx_path.exists():
            return None

        try:
            con = duckdb.connect(":memory:")
            row = con.execute(f"""
                SELECT * FROM read_parquet('{tx_path}') WHERE txId = {txId}
            """).fetchone()
            con.close()

            if not row:
                return None

            # Convert row to dict (first column is txId, rest are features)
            cols = self._bundle.tx_features
            return {cols[i]: float(row[i+1]) if row[i+1] is not None else 0.0 for i in range(len(cols))}
        except Exception as e:
            print(f"Error fetching tx features for {txId}: {e}")
            return None

    def _feature_label(self, feature: str) -> str:
        """Human-readable feature labels."""
        return _FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())

    def get_model_meta(self) -> dict:
        """Return model metadata."""
        if not self._ready or not self._bundle:
            return None
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
