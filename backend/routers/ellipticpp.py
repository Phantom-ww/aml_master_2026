"""Elliptic++ API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import time

from services.ellipticpp_service import get_service
from services.model_service import get_service as get_model_service
from schemas import SubgraphResponse, GraphNode, GraphEdge

router = APIRouter(prefix="/api/epp", tags=["ellipticpp"])


def _check_ready():
    svc = get_service()
    if not svc.is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Elliptic++ not ready: {svc.load_error or 'Not loaded'}"
        )
    return svc


def _score_model_block(address: str, wallet: dict, source: str | None = None):
    """Build the shared `model` block: {xgb, gnn, ms, model_version, in_sample}.

    Returns None if the model service is not ready.
    """
    model_svc = get_model_service()
    if not model_svc.is_ready or not model_svc._bundle:
        return None

    t0 = time.perf_counter()
    wallet_features = tuple(wallet.get(f, 0.0) for f in model_svc._bundle.wallet_features)
    xgb_result = model_svc.score_wallet_xgb(address, wallet_features)
    gnn_result = model_svc.score_wallet_gnn(address)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    in_sample = model_svc.get_in_sample(address)

    block = {
        "xgb": {
            "prob": xgb_result["prob"],
            "flag": xgb_result["flag"],
            "explanations": xgb_result["explanations"],
        } if xgb_result else None,
        "gnn": gnn_result if gnn_result else None,
        "ms": ms,
        "model_version": model_svc._bundle.meta["model_version"],
        "in_sample": in_sample,
    }
    if source:
        block["source"] = source
    return block


@router.get("/wallet/{address}", response_model=SubgraphResponse)
def get_wallet_subgraph(address: str):
    """Return a SubgraphResponse with one node for the given address."""
    svc = _check_ready()
    wallet = svc.get_wallet(address)

    if wallet is None:
        raise HTTPException(status_code=404, detail=f"Address {address} not found")

    label_value = None if wallet["class"] == 3 else (1 if wallet["class"] == 1 else 0)

    # Get model risk for this address
    model_svc = get_model_service()
    risks = model_svc.score_risk([address]) if model_svc.is_ready else {}
    model_risk = risks.get(address)

    node = GraphNode(
        id=address,
        label=address[:8],
        x=0.0,
        y=0.0,
        address=address,
        data_source="ellipticpp",
        known_label=label_value,
        model_risk=model_risk
    )

    model_block = _score_model_block(address, wallet)

    return SubgraphResponse(
        nodes=[node],
        edges=[],
        center_node=address,
        truncated=False,
        source="ellipticpp",
        model=model_block
    )


@router.get("/address/{address}")
def get_address_details(
    address: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """Return address details: class, first/last time step, key features, transactions, model scores."""
    svc = _check_ready()
    wallet = svc.get_wallet(address)

    if wallet is None:
        raise HTTPException(status_code=404, detail=f"Address {address} not found")

    txs, total = svc.get_transactions(address, offset, limit)
    model_block = _score_model_block(address, wallet)

    return {
        "address": address,
        "class": wallet["class"],
        "first_time_step": wallet["first_ts"],
        "last_time_step": wallet["last_ts"],
        "first_block_appeared_in": int(wallet.get("first_block_appeared_in", 0)),
        "last_block_appeared_in": int(wallet.get("last_block_appeared_in", 0)),
        "key_features": {
            "btc_sent_total": float(wallet.get("btc_sent_total", 0)),
            "btc_received_total": float(wallet.get("btc_received_total", 0)),
            "total_txs": int(wallet.get("total_txs", 0)),
            "lifetime_in_blocks": int(wallet.get("lifetime_in_blocks", 0)),
            "fees_total": float(wallet.get("fees_total", 0))
        },
        "transactions": txs,
        "total_transactions": total,
        "offset": offset,
        "limit": limit,
        "model": model_block
    }


@router.get("/tx/{txId}/edges")
def get_tx_edges(
    txId: str,
    focal: Optional[str] = None,
    max: int = Query(10, ge=1, le=100)
):
    """Return nodes and edges for a transaction, with optional focal address priority."""
    svc = _check_ready()
    result = svc.get_tx_edges(txId, focal, max)

    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"Transaction {txId} not found")

    return result


@router.get("/tx/{txId}")
def get_tx_details(txId: str):
    """Return transaction details: time step, class, total BTC, fees, input/output addresses, tx model score."""
    svc = _check_ready()
    details = svc.get_tx_details(txId)

    if details is None:
        raise HTTPException(status_code=404, detail=f"Transaction {txId} not found")

    # Score transaction with model and add explanation
    model_svc = get_model_service()
    tx_model_block = None
    if model_svc.is_ready and model_svc._bundle and model_svc._bundle.tx_booster:
        try:
            tx_result = model_svc.explain_tx(int(txId))
            if tx_result:
                tx_model_block = {
                    "prob": tx_result["prob"],
                    "flag": tx_result["flag"],
                    "explanation": {
                        "raw": tx_result["raw"],
                        "base_value": tx_result["base_value"],
                        "top": tx_result["top"],
                        "groups": tx_result["groups"]
                    }
                }
        except (ValueError, TypeError):
            pass

    return {**details, "tx_model": tx_model_block}


@router.get("/wallet/{address}/explain/gnn")
def explain_wallet_gnn(address: str):
    """Return GraphSAGE explanations for a wallet: graph effect, feature occlusion, neighbour leave-one-out."""
    import logging
    logger = logging.getLogger(__name__)

    svc = _check_ready()
    wallet = svc.get_wallet(address)

    if wallet is None:
        raise HTTPException(status_code=404, detail=f"Address {address} not found")

    model_svc = get_model_service()
    if not model_svc.is_ready or not model_svc._gnn_scorer:
        raise HTTPException(status_code=503, detail="GNN model not available")

    try:
        result = model_svc.explain_wallet_gnn(address)
        if result is None:
            logger.exception("explain_wallet_gnn returned None")
            raise HTTPException(status_code=500, detail="Failed to generate explanation")
        return result
    except Exception as e:
        logger.exception(f"GNN explanation error for {address}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@router.get("/examples")
def get_examples():
    """Return a few example addresses per class (illicit, licit, unknown)."""
    svc = _check_ready()
    return svc.get_examples()


@router.post("/model/reload")
def reload_models():
    """Reload model data and weights (for when data grows)."""
    model_svc = get_model_service()
    model_svc.reload()
    return {"message": "Models reloaded", "ready": model_svc.is_ready, "error": model_svc.load_error}


@router.get("/model/info")
def get_model_info():
    """Return model metadata (also available at /api/model/info)."""
    model_svc = get_model_service()
    if not model_svc.is_ready:
        raise HTTPException(status_code=503, detail=f"Model not ready: {model_svc.load_error}")
    return model_svc.get_model_meta()
