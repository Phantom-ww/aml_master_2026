from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from functools import lru_cache
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import csv
import os

from database import get_db, init_db, Investigation
from schemas import (
    SubgraphResponse,
    NodeDetailsResponse,
    EdgeDetailsResponse,
    SaveInvestigationRequest,
    InvestigationResponse,
    ChainStatusResponse,
    TransactionPageResponse,
    ChainTxDetailsResponse,
    TxInputSummary,
    TxOutputSummary
)
from config import settings
from ellipticpp import get_label as get_ellipticpp_label

# Elliptic++ service
from services.ellipticpp_service import get_service as get_ellipticpp_service

# Chain imports
if settings.chain_enabled:
    from chain.esplora import EsploraSource
    from chain.cache import ChainCache
    from chain.exceptions import InvalidAddress, NotFound, ChainUnavailable

# Global chain source
chain_source: Optional[EsploraSource] = None
chain_cache: Optional[ChainCache] = None

# ---------------------------------------------------------------------------
# OFAC sanctioned BTC addresses
# ---------------------------------------------------------------------------
_ofac_btc: dict = {}      # address -> {"entity": str, "program": str}
_ofac_available: bool = False


def _load_ofac_btc() -> None:
    global _ofac_btc, _ofac_available
    csv_path = settings.ofac_btc_csv
    if not os.path.exists(csv_path):
        print(f"OFAC BTC list not found at {csv_path} — sanctions checking unavailable")
        return
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _ofac_btc[row["address"]] = {
                    "entity": row.get("entity", ""),
                    "program": row.get("program", ""),
                }
        _ofac_available = True
        print(f"Loaded {len(_ofac_btc)} OFAC sanctioned BTC addresses")
    except Exception as exc:
        print(f"Failed to load OFAC BTC list: {exc} — sanctions checking unavailable")


def _ofac_check(addresses: List[str]) -> dict:
    """Return sanctions summary for a list of addresses."""
    if not _ofac_available:
        return {"available": False}
    hits = [
        {"address": a, "entity": _ofac_btc[a]["entity"]}
        for a in addresses
        if a in _ofac_btc
    ]
    return {"available": True, "hits": hits}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, close on shutdown"""
    global chain_source, chain_cache

    # Initialize database
    init_db()
    print("Database initialized")

    # Load OFAC BTC sanctions list
    _load_ofac_btc()

    # Initialize Elliptic++ service
    epp_svc = get_ellipticpp_service()
    epp_svc.load()
    if epp_svc.is_ready:
        print("Elliptic++ dataset loaded")
    else:
        print(f"Elliptic++ not ready: {epp_svc.load_error}")

    # Initialize model service (after EllipticPP for shared address index)
    from services.model_service import get_service as get_model_service
    model_svc = get_model_service()
    model_svc.load(epp_svc)
    if model_svc.is_ready:
        print("Model service ready")
    else:
        print(f"Model service not ready: {model_svc.load_error}")

    # Initialize live chain model service
    from services.live_model import get_service as get_live_model_service
    live_svc = get_live_model_service()
    live_svc.load(settings.live_model_dir)
    if live_svc.is_ready:
        print(f"Live model service ready: {live_svc.model_version}")
    else:
        print(f"Live model not ready: {live_svc.load_error}")

    # Initialize chain services if enabled
    if settings.chain_enabled:
        try:
            chain_cache = ChainCache(settings.chain_cache_path)
            await chain_cache.init()
            chain_source = EsploraSource(
                base_url=settings.esplora_url,
                cache=chain_cache,
                network=settings.btc_network,
                fallback_url=settings.esplora_fallback_url,
                max_txs_per_address=settings.max_txs_per_address,
                max_nodes=settings.max_nodes,
                timeout=settings.http_timeout,
                max_retries=settings.max_retries,
                max_concurrent=settings.max_concurrent_requests
            )
            await chain_source.init()
            print(f"Chain integration enabled (network: {settings.btc_network}, esplora: {settings.esplora_url})")
        except Exception as e:
            import traceback
            print(f"Failed to initialize chain services: {e}")
            traceback.print_exc()
            print("Chain endpoints will be unavailable")

    yield

    # Cleanup
    if chain_source:
        await chain_source.close()
    if chain_cache:
        await chain_cache.close()
    print("Services closed")

app = FastAPI(title="AML Investigation Platform API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Elliptic++ router
from routers.ellipticpp import router as epp_router
app.include_router(epp_router)

# ============================================================================
# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "aml-api"}

# ============================================================================
# MODEL ENDPOINTS
# ============================================================================

@app.get("/api/model/info")
def get_global_model_info():
    """Return model metadata (version, metrics, thresholds)."""
    from services.model_service import get_service as get_model_service
    model_svc = get_model_service()
    if not model_svc.is_ready:
        raise HTTPException(status_code=503, detail=f"Model not ready: {model_svc.load_error}")
    return model_svc.get_model_meta()

# ============================================================================
# INVESTIGATION ENDPOINTS
# ============================================================================

# LRU cache for investigation snapshots
@lru_cache(maxsize=50)
def get_investigation_cached(investigation_id: str, db_session_id: int):
    """Cached investigation retrieval (cache key includes session id for invalidation)"""
    # This is called by the actual endpoint with a session identifier
    pass

_cache_version = 0

def invalidate_cache():
    """Increment cache version to invalidate all cached investigations"""
    global _cache_version
    _cache_version += 1
    get_investigation_cached.cache_clear()

@app.post("/api/investigations")
async def save_investigation(
    request: SaveInvestigationRequest,
    analyst_id: str = "default",
    db: Session = Depends(get_db)
):
    """Save a graph snapshot to PostgreSQL"""
    investigation = Investigation(
        analyst_id=analyst_id,
        name=request.name,
        description=request.description,
        snapshot=request.snapshot.dict(),
        meta={
            "analyst_notes": request.analyst_notes,
            "node_count": len(request.snapshot.nodes),
            "edge_count": len(request.snapshot.edges),
        }
    )
    db.add(investigation)
    db.commit()
    db.refresh(investigation)

    # Invalidate cache
    invalidate_cache()

    return {
        "id": investigation.id,
        "message": "Investigation saved successfully",
        "created_at": investigation.created_at.isoformat()
    }

@app.get("/api/investigations/{investigation_id}")
async def load_investigation(investigation_id: str, db: Session = Depends(get_db)):
    """Retrieve a saved investigation snapshot"""
    investigation = db.query(Investigation).filter(
        Investigation.id == investigation_id
    ).first()

    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")

    return {
        "id": investigation.id,
        "snapshot": investigation.snapshot,
        "created_at": investigation.created_at.isoformat()
    }

@app.get("/api/investigations", response_model=List[InvestigationResponse])
async def list_investigations(analyst_id: str = "default", db: Session = Depends(get_db)):
    """List all investigations for an analyst"""
    investigations = db.query(Investigation).filter(
        Investigation.analyst_id == analyst_id
    ).order_by(Investigation.created_at.desc()).limit(50).all()

    return [
        InvestigationResponse(
            id=inv.id,
            name=inv.name,
            created_at=inv.created_at.isoformat(),
            analyst_notes=inv.meta.get("analyst_notes", "") if inv.meta else "",
            node_count=inv.meta.get("node_count", 0) if inv.meta else 0,
            edge_count=inv.meta.get("edge_count", 0) if inv.meta else 0
        )
        for inv in investigations
    ]

@app.delete("/api/investigations/{investigation_id}")
async def delete_investigation(investigation_id: str, db: Session = Depends(get_db)):
    """Delete an investigation"""
    investigation = db.query(Investigation).filter(
        Investigation.id == investigation_id
    ).first()

    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")

    db.delete(investigation)
    db.commit()

    # Invalidate cache
    invalidate_cache()

    return {"message": "Investigation deleted successfully"}

# ============================================================================
# CHAIN ENDPOINTS
# ============================================================================

def require_chain_source():
    """Dependency to ensure chain source is available"""
    if not chain_source:
        raise HTTPException(
            status_code=503,
            detail="Chain integration not enabled or failed to initialize"
        )
    return chain_source

@app.get("/api/chain/status", response_model=ChainStatusResponse)
async def get_chain_status():
    """Get chain service status"""
    if not settings.chain_enabled or not chain_source:
        return ChainStatusResponse(
            network=settings.btc_network,
            esplora_reachable=False,
            cache_stats={}
        )

    try:
        start = time.time()
        tip_height = await chain_source.get_tip_height()
        latency = (time.time() - start) * 1000

        cache_stats = await chain_cache.stats() if chain_cache else {}

        return ChainStatusResponse(
            network=settings.btc_network,
            tip_height=tip_height,
            esplora_reachable=True,
            cache_stats=cache_stats,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        return ChainStatusResponse(
            network=settings.btc_network,
            esplora_reachable=False,
            cache_stats={}
        )

@app.get("/api/chain/subgraph/{address}", response_model=SubgraphResponse)
async def get_chain_subgraph(
    address: str,
    k: int = 1,
    max_txs: Optional[int] = None,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get k-hop subgraph from live blockchain"""
    if k > settings.max_k_hops:
        raise HTTPException(status_code=400, detail=f"k must be <= {settings.max_k_hops}")

    try:
        start = time.time()
        subgraph = await source.get_k_hop_subgraph(
            address,
            k=k
        )
        latency = (time.time() - start) * 1000
        print(f"Chain subgraph fetched in {latency:.0f}ms (truncated: {subgraph.truncated})")

        # Transform addresses to GraphNodes
        nodes = []
        for addr, addr_data in subgraph.addresses.items():
            ellipticpp_class = get_ellipticpp_label(addr)
            nodes.append({
                "id": addr,
                "label": addr[:8] + "...",
                "x": 0.0,  # Frontend will position
                "y": 0.0,
                "node_type": "wallet",
                "address": addr,
                "balance_crypto": addr_data.balance_sat / 100_000_000,
                "balance_sat": addr_data.balance_sat,
                "data_source": "chain",
                "tx_count": addr_data.tx_count,
                "ellipticpp_class": ellipticpp_class
            })

        # Transform transactions to GraphEdges
        edges = []
        for txid, tx in subgraph.transactions.items():
            from chain.coinjoin import detect_coinjoin
            cj = detect_coinjoin(tx)
            is_cj = cj.get("is_coinjoin", False)
            cj_type = cj.get("type")
            # Each tx creates edges between input addresses and output addresses
            input_addrs = [inp.address for inp in tx.inputs if inp.address]
            output_addrs = [out.address for out in tx.outputs if out.address]

            # Create edges from each input to each output
            for inp_addr in input_addrs:
                for out_addr in output_addrs:
                    if inp_addr != out_addr:  # Skip self-loops in graph view
                        edges.append({
                            "id": f"{txid}_{inp_addr[:8]}_{out_addr[:8]}",
                            "source": inp_addr,
                            "target": out_addr,
                            "tx_address": txid,
                            "amount": sum(out.value_sat for out in tx.outputs if out.address == out_addr) / 100_000_000,
                            "timestamp": datetime.fromtimestamp(tx.block_time, tz=timezone.utc).isoformat() if tx.block_time else None,
                            "data_source": "chain",
                            "block_height": tx.block_height,
                            "confirmed": tx.block_height is not None,
                            "is_coinjoin": is_cj,
                            "coinjoin_type": cj_type,
                        })

        return SubgraphResponse(
            nodes=nodes,
            edges=edges,
            center_node=address,
            truncated=subgraph.truncated,
            source="chain"
        )
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Chain subgraph error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chain/wallet/{address}", response_model=SubgraphResponse)
async def get_chain_wallet(
    address: str,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get single wallet node from live blockchain"""
    try:
        addr_data = await source.get_address_with_txs(address)
        ellipticpp_class = get_ellipticpp_label(address)

        # Score with model if available
        from services.model_service import get_service as get_model_service
        model_svc = get_model_service()
        risks = model_svc.score_risk([address]) if model_svc.is_ready else {}
        model_risk = risks.get(address)

        node = {
            "id": address,
            "label": address[:8] + "...",
            "x": 0.0,
            "y": 0.0,
            "node_type": "wallet",
            "address": address,
            "balance_crypto": addr_data.balance_sat / 100_000_000,
            "balance_sat": addr_data.balance_sat,
            "data_source": "chain",
            "tx_count": addr_data.tx_count,
            "ellipticpp_class": ellipticpp_class,
            "model_risk": model_risk
        }

        # Collect all addresses appearing in this wallet's loaded transactions
        all_addrs: set = set()
        for tx in addr_data.txs:
            for inp in tx.inputs:
                if inp.address and inp.address != address:
                    all_addrs.add(inp.address)
            for out in tx.outputs:
                if out.address and out.address != address:
                    all_addrs.add(out.address)

        # Sanctions
        self_check = _ofac_check([address])
        cp_check = _ofac_check(list(all_addrs))
        sanctions = {
            "available": _ofac_available,
            "listed": bool(self_check.get("hits")),
            "entity": self_check["hits"][0]["entity"] if self_check.get("hits") else None,
            "counterparties_listed": cp_check.get("hits", []),
        }

        return SubgraphResponse(
            nodes=[node],
            edges=[],
            center_node=address,
            source="chain",
            sanctions=sanctions,
        )
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Chain wallet error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/chain/wallet/{address}/live-score")
async def get_chain_wallet_live_score(
    address: str,
    source: EsploraSource = Depends(require_chain_source),
):
    """Experimental live-chain wallet risk score.

    Fetches the full confirmed transaction history, computes scale-free
    features, scores them with the live XGBoost model, and runs the OOD check.
    Returns 503 if the live model is not loaded.
    """
    import time as _time
    from chain.history_fetch import fetch_confirmed_txs
    from chain.live_features import live_features_from_txs
    from services.live_model import get_service as get_live_model_service

    t0 = _time.monotonic()

    live_svc = get_live_model_service()
    if not live_svc.is_ready:
        raise HTTPException(status_code=503, detail="Live model not available")

    mode = settings.live_feature_mode
    window = settings.live_feature_window_txs

    try:
        source.validate_address(address)
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        all_txs, truncated = await fetch_confirmed_txs(
            address, source, max_txs=settings.live_max_txs
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chain unavailable: {e}")

    # Select txs for feature computation according to the feature mode.
    if mode == "window":
        feature_txs = all_txs[:window]  # Esplora returns most-recent first
    else:
        feature_txs = all_txs

    if not feature_txs:
        raise HTTPException(
            status_code=422,
            detail="Address has no confirmed transactions — cannot score",
        )

    features, base_stats = live_features_from_txs(address, feature_txs)

    scored = live_svc.score(features, use_recalibration=settings.live_use_recalibration)
    ood_result = live_svc.ood(features, base_stats)

    ms = round((_time.monotonic() - t0) * 1000, 1)

    return {
        "model_version": live_svc.model_version,
        "experimental": True,
        "mode": mode,
        "n_txs_fetched": len(all_txs),
        "n_txs_used": len(feature_txs),
        "truncated": truncated,
        "prob": scored["prob"],
        "raw": scored["raw"],
        "flag": scored["flag"],
        "explanations": scored["explanations"],
        "decomposition": scored["decomposition"],
        "ood": ood_result,
        "ms": ms,
    }


@app.get("/api/chain/address/{address}", response_model=NodeDetailsResponse)
async def get_chain_address(
    address: str,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get address details from live blockchain"""
    try:
        print(f"Fetching address data for {address}")
        addr_data = await source.get_address_with_txs(address)
        print(f"Got address data: balance={addr_data.balance_sat}, tx_count={addr_data.tx_count}, txs_len={len(addr_data.txs)}")

        # No connected_edges for chain data
        connected_edges = []

        # Build transaction summary list for all transactions
        transactions = []
        for tx in addr_data.txs:
            # Calculate sent, received, and to_others amounts for this address
            sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
            received = sum(out.value_sat for out in tx.outputs if out.address == address)
            to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)

            # Determine direction and amount
            is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

            if not is_in_inputs:
                # not in inputs → "in"
                direction = "in"
                amount_sat = received
                change_returned = False
            elif to_others > 0:
                # in inputs and to_others > 0 → "out"
                direction = "out"
                amount_sat = sent - received  # address's OWN contribution
                change_returned = received > 0
            else:
                # in inputs and to_others == 0 → "self" (consolidation)
                direction = "self"
                amount_sat = sent - received  # the fee paid
                change_returned = False

            # Format timestamp
            from datetime import datetime
            timestamp = datetime.fromtimestamp(tx.block_time).isoformat() if tx.block_time else None

            from chain.coinjoin import detect_coinjoin
            cj = detect_coinjoin(tx)

            transactions.append({
                "txid": tx.txid,
                "amount_btc": amount_sat / 100_000_000,
                "timestamp": timestamp,
                "confirmed": tx.block_height is not None,
                "block_height": tx.block_height,
                "is_self_send": direction == "self",
                "direction": direction,
                "coinjoin": cj,
            })

        # Calculate next_cursor: last confirmed transaction's txid, or None
        next_cursor = None
        confirmed_txs = [tx for tx in addr_data.txs if tx.block_height is not None]
        if confirmed_txs and len(addr_data.txs) >= 25:  # Esplora page size
            next_cursor = confirmed_txs[-1].txid

        ellipticpp_class = get_ellipticpp_label(address)

        return NodeDetailsResponse(
            node_id=address,
            node_type="wallet",
            address=address,
            balance_crypto=addr_data.balance_sat / 100_000_000,
            balance_usd=0.0,  # No USD conversion for chain data
            risk_score=None,
            connected_edges=connected_edges,
            balance_sat=addr_data.balance_sat,
            tx_count=addr_data.tx_count,
            confirmed_tx_count=addr_data.confirmed_tx_count,
            mempool_tx_count=addr_data.mempool_tx_count,
            transactions=transactions,
            next_cursor=next_cursor,
            ellipticpp_class=ellipticpp_class
        )
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Chain address error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chain/address/{address}/transactions", response_model=TransactionPageResponse)
async def get_chain_address_txs_page(
    address: str,
    cursor: Optional[str] = None,
    page_size: int = 25,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get paginated transaction history for a chain address"""
    try:
        print(f"Fetching transaction page for {address}, cursor={cursor}, page_size={page_size}")
        txs, next_cursor, has_more = await source.get_address_txs_page(address, cursor, page_size)

        # Build transaction summary list
        transactions = []
        for tx in txs:
            # Calculate sent, received, and to_others amounts for this address
            sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
            received = sum(out.value_sat for out in tx.outputs if out.address == address)
            to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)

            # Determine direction and amount
            is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

            if not is_in_inputs:
                # not in inputs → "in"
                direction = "in"
                amount_sat = received
                change_returned = False
            elif to_others > 0:
                # in inputs and to_others > 0 → "out"
                direction = "out"
                amount_sat = sent - received  # address's OWN contribution
                change_returned = received > 0
            else:
                # in inputs and to_others == 0 → "self" (consolidation)
                direction = "self"
                amount_sat = sent - received  # the fee paid
                change_returned = False

            # Format timestamp
            from datetime import datetime
            timestamp = datetime.fromtimestamp(tx.block_time).isoformat() if tx.block_time else None

            from schemas import TransactionSummary
            from chain.coinjoin import detect_coinjoin
            cj = detect_coinjoin(tx)
            transactions.append(TransactionSummary(
                txid=tx.txid,
                amount_btc=amount_sat / 100_000_000,
                timestamp=timestamp,
                confirmed=tx.block_height is not None,
                block_height=tx.block_height,
                is_self_send=direction == "self",
                direction=direction,
                coinjoin=cj,
            ))

        return TransactionPageResponse(
            transactions=transactions,
            next_cursor=next_cursor,
            has_more=has_more
        )
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Chain address transactions page error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chain/tx/{txid}")
async def get_chain_tx(
    txid: str,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get transaction details from live blockchain"""
    try:
        tx = await source.get_transaction(txid)

        # Calculate totals
        total_in = sum(inp.value_sat for inp in tx.inputs if not inp.is_coinbase)
        total_out = sum(out.value_sat for out in tx.outputs)

        # Calculate fee rate (sat/vB)
        fee_rate_sat_vb = None
        if tx.vsize and tx.vsize > 0 and tx.fee_sat > 0:
            fee_rate_sat_vb = tx.fee_sat / tx.vsize

        # Determine change addresses: output addresses that also appear in inputs
        input_addresses = {inp.address for inp in tx.inputs if inp.address and not inp.is_coinbase}

        # Build inputs summary
        inputs_summary = [
            TxInputSummary(
                address=inp.address,
                value_btc=inp.value_sat / 1e8,
                is_coinbase=inp.is_coinbase
            )
            for inp in tx.inputs
        ]

        # Build outputs summary with change detection
        outputs_summary = [
            TxOutputSummary(
                address=out.address,
                value_btc=out.value_sat / 1e8,
                vout=out.vout,
                is_change=out.address is not None and out.address in input_addresses
            )
            for out in tx.outputs
        ]

        # Format block time as ISO
        from datetime import datetime, timezone
        from chain.coinjoin import detect_coinjoin
        block_time_iso = None
        if tx.block_time:
            block_time_iso = datetime.fromtimestamp(tx.block_time, tz=timezone.utc).isoformat()

        cj = detect_coinjoin(tx)

        return ChainTxDetailsResponse(
            txid=txid,
            confirmed=tx.confirmations > 0,
            block_height=tx.block_height,
            block_time=block_time_iso,
            confirmations=tx.confirmations,
            total_in_btc=total_in / 1e8,
            total_out_btc=total_out / 1e8,
            fee_btc=tx.fee_sat / 1e8,
            fee_rate_sat_vb=fee_rate_sat_vb,
            input_count=len(tx.inputs),
            output_count=len(tx.outputs),
            inputs=inputs_summary,
            outputs=outputs_summary,
            has_op_return=tx.has_op_return,
            op_return_data=tx.op_return_data,
            burned_value_btc=tx.burned_value_sat / 1e8,
            coinjoin=cj,
        )
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Chain tx error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chain/tx/{txid}/edges")
async def get_chain_tx_edges(
    txid: str,
    focal: str,
    max: int = 10,
    source: EsploraSource = Depends(require_chain_source)
):
    """Get transaction edges and counterparty nodes for a focal address"""
    try:
        from chain.transform import transaction_to_edges
        from schemas import GraphNode, GraphEdge

        # Fetch transaction
        tx = await source.get_transaction(txid)

        # CoinJoin detection for this tx (cached)
        from chain.coinjoin import detect_coinjoin
        cj = detect_coinjoin(tx)
        is_cj = cj.get("is_coinjoin", False)
        cj_type = cj.get("type")

        # Convert to edges with focal address
        edge_data_list = transaction_to_edges(tx, focal_address=focal, include_change=False)

        # Filter to only edges where focal is source or target
        focal_edges = [e for e in edge_data_list if e["source"] == focal or e["target"] == focal]

        # Collect counterparties with amounts
        counterparty_amounts = {}
        for edge in focal_edges:
            counterparty = edge["target"] if edge["source"] == focal else edge["source"]
            if counterparty == focal:
                continue  # Skip self-loops
            if counterparty not in counterparty_amounts:
                counterparty_amounts[counterparty] = 0
            counterparty_amounts[counterparty] += edge["amount_sat"]

        # Keep top `max` counterparties by amount
        sorted_counterparties = sorted(counterparty_amounts.items(), key=lambda x: x[1], reverse=True)
        top_counterparties = dict(sorted_counterparties[:max])
        total_counterparties = len(counterparty_amounts)
        truncated = total_counterparties > max

        # Build nodes (counterparty addresses + coinbase if needed)
        nodes = []
        for counterparty in top_counterparties.keys():
            if counterparty == "coinbase":
                nodes.append(GraphNode(
                    id="coinbase",
                    label="Coinbase",
                    x=0.0,
                    y=0.0,
                    node_type="coinbase",
                    risk_score=None,
                    pinned=False,
                    address="coinbase",
                    owner=None,
                    balance_crypto=None,
                    balance_usd=None,
                    data_source="chain",
                    balance_sat=None,
                    tx_count=0,
                ))
            else:
                ellipticpp_class = get_ellipticpp_label(counterparty)
                nodes.append(GraphNode(
                    id=counterparty,
                    label=counterparty[:12] + "...",
                    x=0.0,
                    y=0.0,
                    node_type="wallet",
                    risk_score=None,
                    pinned=False,
                    address=counterparty,
                    owner=None,
                    balance_crypto=None,  # Lazily loaded
                    balance_usd=None,
                    data_source="chain",
                    balance_sat=None,
                    tx_count=None,
                    ellipticpp_class=ellipticpp_class
                ))

        # Deduplicate and aggregate focal_edges by (source, target)
        edge_map = {}
        for edge_data in focal_edges:
            key = (edge_data["source"], edge_data["target"])
            if key not in edge_map:
                edge_map[key] = edge_data
            else:
                edge_map[key]["amount_sat"] += edge_data["amount_sat"]

        # Build edges (only those involving top counterparties)
        edges = []
        from datetime import datetime
        for edge_data in edge_map.values():
            counterparty = edge_data["target"] if edge_data["source"] == focal else edge_data["source"]
            if counterparty not in top_counterparties:
                continue

            edge_id = f"{txid}:{edge_data['source']}:{edge_data['target']}"
            timestamp_iso = None
            if edge_data.get("block_time"):
                timestamp_iso = datetime.utcfromtimestamp(edge_data["block_time"]).isoformat()

            edges.append(GraphEdge(
                id=edge_id,
                source=edge_data["source"],
                target=edge_data["target"],
                amount=edge_data["amount_sat"] / 1e8,
                risk_score=None,
                risk_color="#6b7280",
                time_step=0,
                tx_address=txid,
                from_wallet=edge_data["source"],
                to_wallet=edge_data["target"],
                timestamp=timestamp_iso,
                amount_crypto=edge_data["amount_sat"] / 1e8,
                amount_usd=None,
                data_source="chain",
                block_height=edge_data.get("block_height"),
                confirmed=edge_data.get("confirmed"),
                is_change=edge_data.get("is_change"),
                large_tx=edge_data.get("large_tx"),
                is_coinjoin=is_cj,
                coinjoin_type=cj_type,
            ))

        return {
            "nodes": nodes,
            "edges": edges,
            "total_counterparties": total_counterparties,
            "truncated": truncated
        }
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChainUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Chain tx edges error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Mixing & sanctions endpoint
# ---------------------------------------------------------------------------

@app.get("/api/chain/address/{address}/mixing")
async def get_chain_address_mixing(
    address: str,
    source: EsploraSource = Depends(require_chain_source),
):
    """
    Compute wallet-level CoinJoin exposure and OFAC sanctions summary.
    Loads at most 2 pages (≈ 50 txs) of address history and fetches
    at most 20 parent transactions to check post-mix funding.
    """
    from chain.coinjoin import detect_coinjoin
    from chain.exceptions import NotFound as _NotFound, ChainUnavailable as _Unavail
    import asyncio as _asyncio

    t0 = time.time()
    try:
        source.validate_address(address)
    except InvalidAddress as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ---- fetch up to 2 pages of address history ----
    try:
        txs_p1, cursor1, has_more1 = await source.get_address_txs_page(address, None, 25)
    except (_NotFound, _Unavail) as e:
        raise HTTPException(status_code=503, detail=str(e))

    txs = list(txs_p1)
    truncated = False

    if has_more1 and cursor1:
        try:
            txs_p2, _cur2, _more2 = await source.get_address_txs_page(address, cursor1, 25)
            txs.extend(txs_p2)
            if _more2:
                truncated = True
        except Exception:
            truncated = True  # partial result

    # ---- analyse own transactions ----
    cj_participations = 0
    cj_types: dict = {}
    sent_to_coinjoin = 0
    # collect parent txids needed for received_from_coinjoin and consolidations
    # parent_need[parent_txid] = set of "use" tags ("post_mix" | "consolidation")
    parent_need: dict = {}

    for tx in txs:
        cj = detect_coinjoin(tx)
        if not cj["is_coinjoin"]:
            continue
        cj_type = cj["type"]
        is_input = any(i.address == address for i in tx.inputs if i.address)
        is_output = any(o.address == address for o in tx.outputs if o.address)
        if is_input or is_output:
            cj_participations += 1
            cj_types[cj_type] = cj_types.get(cj_type, 0) + 1
        if is_input:
            sent_to_coinjoin += 1

    # For post-mix and consolidation checks we need parent txs.
    # Identify inputs to this address's txs where the address is an OUTPUT
    # (to check if the payer was in a CoinJoin — post-mix).
    for tx in txs:
        is_output = any(o.address == address for o in tx.outputs if o.address)
        if is_output:
            for inp in tx.inputs:
                if inp.txid and not inp.is_coinbase:
                    parent_need.setdefault(inp.txid, set()).add("post_mix")

        # For consolidation: txs where address is an INPUT
        is_input = any(i.address == address for i in tx.inputs if i.address)
        if is_input:
            addr_inputs = [i for i in tx.inputs if i.address == address and i.txid]
            if len(addr_inputs) >= 2:
                for inp in addr_inputs:
                    if inp.txid:
                        parent_need.setdefault(inp.txid, set()).add("consolidation")

    # Fetch up to 20 parents
    MAX_PARENTS = 20
    parent_txids = list(parent_need.keys())[:MAX_PARENTS]
    if len(parent_need) > MAX_PARENTS:
        truncated = True

    parent_cj: dict = {}  # txid -> bool (is CoinJoin)

    async def _fetch_parent(ptxid: str):
        try:
            ptx = await source.get_transaction(ptxid)
            cj = detect_coinjoin(ptx)
            return ptxid, cj["is_coinjoin"]
        except Exception:
            return ptxid, None  # failed fetch treated as unknown

    results = await _asyncio.gather(*[_fetch_parent(p) for p in parent_txids])
    for ptxid, is_cj in results:
        parent_cj[ptxid] = is_cj

    # ---- post-mix received ----
    received_from_coinjoin = 0
    seen_post_mix_txs: set = set()
    for tx in txs:
        is_output = any(o.address == address for o in tx.outputs if o.address)
        if not is_output:
            continue
        funded_by_cj = any(
            parent_cj.get(i.txid) is True
            for i in tx.inputs
            if i.txid and not i.is_coinbase and "post_mix" in parent_need.get(i.txid, set())
        )
        if funded_by_cj and tx.txid not in seen_post_mix_txs:
            received_from_coinjoin += 1
            seen_post_mix_txs.add(tx.txid)

    # ---- consolidations ----
    consolidations = 0
    for tx in txs:
        addr_inputs = [i for i in tx.inputs if i.address == address and i.txid]
        if len(addr_inputs) < 2:
            continue
        cj_input_count = sum(
            1 for i in addr_inputs
            if parent_cj.get(i.txid) is True
        )
        if cj_input_count >= 2:
            consolidations += 1

    # ---- sanctions ----
    all_counterparty_addrs: set = set()
    for tx in txs:
        for inp in tx.inputs:
            if inp.address and inp.address != address:
                all_counterparty_addrs.add(inp.address)
        for out in tx.outputs:
            if out.address and out.address != address:
                all_counterparty_addrs.add(out.address)

    self_check = _ofac_check([address])
    cp_check = _ofac_check(list(all_counterparty_addrs))
    sanctions = {
        "available": _ofac_available,
        "listed": bool(self_check.get("hits")),
        "entity": self_check["hits"][0]["entity"] if self_check.get("hits") else None,
        "counterparties_listed": cp_check.get("hits", []),
    }

    ms = round((time.time() - t0) * 1000)
    return {
        "cj_participations": cj_participations,
        "cj_types": cj_types,
        "received_from_coinjoin": received_from_coinjoin,
        "consolidations": consolidations,
        "sent_to_coinjoin": sent_to_coinjoin,
        "checked_txs": len(txs),
        "parents_checked": len(results),
        "truncated": truncated,
        "ms": ms,
        "sanctions": sanctions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
