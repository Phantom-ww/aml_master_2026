"""Elliptic++ dataset service: load parquet at startup, index by address."""
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "ellipticpp"

class EllipticPPService:
    def __init__(self):
        self._wallets: Optional[pd.DataFrame] = None
        self._edges: Optional[pd.DataFrame] = None
        self._address_index: Optional[dict] = None
        self._load_error: Optional[str] = None
        self._ready = False

    def load(self):
        import time
        try:
            t0 = time.perf_counter()
            wallets_path = DATA_DIR / "wallets.parquet"
            edges_path = DATA_DIR / "tx_edges.parquet"

            if not wallets_path.exists() or not edges_path.exists():
                self._load_error = f"Parquet files not found. Run prepare_ellipticpp.py first."
                return

            self._wallets = pd.read_parquet(wallets_path).set_index("address")
            self._address_index = {addr: i for i, addr in enumerate(self._wallets.index)}
            wallet_mb = self._wallets.memory_usage(deep=True).sum() / 1024**2
            print(f"Wallets: {len(self._wallets)} rows, {wallet_mb:.1f} MB")

            self._edges = pd.read_parquet(edges_path)
            edge_mb = self._edges.memory_usage(deep=True).sum() / 1024**2
            print(f"Edges: {len(self._edges)} rows, {edge_mb:.1f} MB")

            self._ready = True
            print(f"EllipticPPService loaded in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            self._load_error = f"Failed to load Elliptic++: {e}"

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def address_index(self) -> dict:
        """Shared address → index mapping for ModelService."""
        return self._address_index

    def get_wallet(self, address: str) -> Optional[dict]:
        if not self._ready or address not in self._wallets.index:
            return None
        row = self._wallets.loc[address]
        return row.to_dict()

    def get_transactions(self, address: str, offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        """Return (transactions, total_count)."""
        if not self._ready:
            return [], 0

        edges = self._edges[
            (self._edges["src"] == address) | (self._edges["dst"] == address)
        ].copy()

        edges["direction"] = edges.apply(
            lambda r: "out" if r["src"] == address else "in", axis=1
        )

        total = len(edges)
        page = edges.iloc[offset:offset+limit]

        # Import model_service for tx scoring
        from services.model_service import get_service as get_model_service
        model_svc = get_model_service()

        result = []
        for _, r in page.iterrows():
            tx_model_block = None
            if model_svc.is_ready and model_svc._bundle and model_svc._bundle.tx_booster:
                try:
                    tx_result = model_svc.score_tx(int(r["txId"]))
                    if tx_result:
                        tx_model_block = {"prob": tx_result["prob"], "flag": tx_result["flag"]}
                except (ValueError, TypeError):
                    pass

            result.append({
                "txId": r["txId"],
                "direction": r["direction"],
                "total_BTC": r["total_BTC"],
                "time_step": r["time_step"],
                "tx_class": r["tx_class"],
                "tx_model": tx_model_block
            })

        return result, total

    def get_tx_edges(self, txId: str, focal: Optional[str], max_edges: int) -> dict:
        if not self._ready:
            return {"nodes": [], "edges": [], "total_counterparties": 0, "truncated": False}

        # Convert txId to int if needed
        try:
            txId_val = int(txId)
        except (ValueError, TypeError):
            txId_val = txId

        edges = self._edges[self._edges["txId"] == txId_val].copy()
        total = len(edges)

        if focal:
            # prioritize edges involving focal
            focal_edges = edges[(edges["src"] == focal) | (edges["dst"] == focal)]
            other_edges = edges[(edges["src"] != focal) & (edges["dst"] != focal)]
            edges = pd.concat([focal_edges, other_edges]).head(max_edges)
        else:
            edges = edges.head(max_edges)

        truncated = len(edges) < total

        addresses = set(edges["src"]) | set(edges["dst"])

        # Score all addresses in batch
        from services.model_service import get_service as get_model_service
        model_svc = get_model_service()
        risks = model_svc.score_risk(list(addresses)) if model_svc.is_ready else {}

        nodes = []
        for addr in addresses:
            wallet = self.get_wallet(addr)
            nodes.append({
                "id": addr,
                "label": addr[:8],
                "address": addr,
                "data_source": "ellipticpp",
                "known_label": None if not wallet or wallet["class"] == 3 else (1 if wallet["class"] == 1 else 0),
                "model_risk": risks.get(addr)
            })

        # Score transaction with model
        tx_model_block = None
        if model_svc.is_ready and model_svc._bundle and model_svc._bundle.tx_booster:
            try:
                tx_result = model_svc.score_tx(txId_val)
                if tx_result:
                    tx_model_block = {"prob": tx_result["prob"], "flag": tx_result["flag"]}
            except (ValueError, TypeError):
                pass

        edge_list = [
            {
                "id": f"{r['txId']}:{r['src']}:{r['dst']}",
                "source": r["src"],
                "target": r["dst"],
                "data_source": "ellipticpp",
                "tx_address": str(r["txId"]),
                "tx_class": r["tx_class"],
                "tx_model": tx_model_block,
                "tx_risk": tx_model_block["prob"] if tx_model_block else None
            }
            for _, r in edges.iterrows()
        ]

        return {
            "nodes": nodes,
            "edges": edge_list,
            "total_counterparties": total,
            "truncated": truncated
        }

    def get_tx_details(self, txId: str) -> Optional[dict]:
        if not self._ready:
            return None

        # Convert txId to int if needed
        try:
            txId_val = int(txId)
        except (ValueError, TypeError):
            txId_val = txId

        edges = self._edges[self._edges["txId"] == txId_val]
        if edges.empty:
            return None

        row = edges.iloc[0]
        inputs = edges["src"].unique().tolist()
        outputs = edges["dst"].unique().tolist()

        return {
            "txId": str(txId_val),
            "time_step": int(row["time_step"]),
            "class": int(row["tx_class"]),
            "total_BTC": float(row["total_BTC"]),
            "fees": float(row["fees"]),
            "input_addresses": inputs,
            "output_addresses": outputs
        }

    def get_examples(self) -> dict:
        if not self._ready:
            return {"illicit": [], "licit": [], "unknown": []}

        illicit = self._wallets[self._wallets["class"] == 1].head(3).index.tolist()
        licit = self._wallets[self._wallets["class"] == 2].head(3).index.tolist()
        unknown = self._wallets[self._wallets["class"] == 3].head(3).index.tolist()

        return {"illicit": illicit, "licit": licit, "unknown": unknown}

_service = EllipticPPService()

def get_service() -> EllipticPPService:
    return _service
