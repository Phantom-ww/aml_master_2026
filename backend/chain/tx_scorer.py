"""Batch tx risk scoring for Live Chain endpoints (experimental).

Scores raw Esplora tx dicts with the live tx XGBoost bundle.  Results are cached
by txid so the same transaction is never scored twice within one process lifetime.

If the tx bundle is not ready (model files absent), all calls return {} silently.

``score_txs_batch`` accepts both raw Esplora dicts (with ``vin``/``vout``) and
``chain.protocol.Transaction`` dataclasses — it converts dataclasses automatically.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level result cache: txid -> score result dict
_tx_score_cache: dict[str, dict] = {}


def _tx_to_esplora_dict(tx) -> dict:
    """Convert a ``chain.protocol.Transaction`` dataclass to a minimal Esplora-style dict.

    Only the fields consumed by ``tx_base_from_esplora`` are populated.
    """
    vin = []
    for inp in tx.inputs:
        if inp.is_coinbase:
            vin.append({"is_coinbase": True})
        else:
            vin.append({
                "is_coinbase": False,
                "prevout": {
                    "scriptpubkey_address": inp.address,
                    "value": inp.value_sat,
                },
            })
    vout = []
    for out in tx.outputs:
        entry: dict = {"value": out.value_sat}
        if out.address:
            entry["scriptpubkey_address"] = out.address
        vout.append(entry)

    return {
        "txid": tx.txid,
        "vin": vin,
        "vout": vout,
        "fee": tx.fee_sat,
        "size": getattr(tx, "size", None) or getattr(tx, "vsize", None) or 0,
    }


def score_txs_batch(
    raw_txs: list,
    full_detail_txid: Optional[str] = None,
) -> dict[str, dict]:
    """Score a list of raw Esplora tx dicts.

    Parameters
    ----------
    raw_txs : list
        Esplora transaction dicts (with ``vin``/``vout``) or
        ``chain.protocol.Transaction`` dataclasses — both are accepted.
        Items that cannot be featurised are silently skipped.
    full_detail_txid : str or None
        When set, the result for this txid is extended with ``explanations``
        (top-5 TreeSHAP), ``decomposition``, and the full ``ood`` block including
        the ``out_of_range`` list.  For all other txids only the compact summary
        (prob, flag, ood_flag, model_version, experimental) is stored.

    Returns
    -------
    dict[txid, result_dict]
        Only the txids that were successfully scored are present.  Returns {}
        immediately if the tx bundle is not ready.
    """
    from services.live_model import get_service as get_live_model_service
    from chain.live_features import tx_live_features_from_esplora

    svc = get_live_model_service()
    if not svc.is_bundle_ready("tx"):
        return {}

    # Normalise: convert dataclasses to Esplora-style dicts
    normalised: list[dict] = []
    for tx in raw_txs:
        if isinstance(tx, dict):
            normalised.append(tx)
        else:
            try:
                normalised.append(_tx_to_esplora_dict(tx))
            except Exception as exc:
                logger.debug("tx_scorer: cannot convert tx dataclass: %s", exc)

    # Separate cached from new
    results: dict[str, dict] = {}
    to_score: list[dict] = []

    for tx in normalised:
        txid = tx.get("txid") or tx.get("tx_id")
        if not txid:
            continue
        if txid in _tx_score_cache and txid != full_detail_txid:
            # Return cached compact result; detail will be re-scored below
            results[txid] = _tx_score_cache[txid]
        else:
            to_score.append(tx)

    if not to_score:
        return results

    # Featurise all new txids
    feat_rows: list[tuple[str, dict, dict]] = []  # (txid, feats, base)
    for tx in to_score:
        txid = tx.get("txid") or tx.get("tx_id")
        try:
            feats, base = tx_live_features_from_esplora(tx)
            feat_rows.append((txid, feats, base))
        except Exception as exc:
            logger.debug("tx_scorer: cannot featurise %s: %s", txid, exc)

    if not feat_rows:
        return results

    # --- Batch predict ---
    import xgboost as xgb

    b = svc._bundles["tx"]
    feat_names: list[str] = b.meta["features"]
    best_iter: int = b.meta["best_iteration"]
    threshold_raw: float = b.meta["threshold_raw"]
    feature_labels: dict = b.meta["feature_labels"]
    calibration: dict = b.meta["calibration"]
    model_version: str = b.meta.get("model_version", "unknown")

    from services.live_model import _calibrate

    X = np.asarray(
        [[float(feats.get(f, 0.0) or 0.0) for f in feat_names] for _, feats, _ in feat_rows],
        dtype=np.float32,
    )
    dm = xgb.DMatrix(X, feature_names=feat_names)
    iter_range = (0, best_iter + 1)

    raw_scores = b.booster.predict(dm, iteration_range=iter_range)

    # SHAP contribs only for the detail tx (expensive; skip for bulk)
    contrib_full_all = None
    detail_row_idx = None
    if full_detail_txid is not None:
        for idx, (txid, _, _) in enumerate(feat_rows):
            if txid == full_detail_txid:
                detail_row_idx = idx
                break
        if detail_row_idx is not None:
            contrib_full_all = b.booster.predict(
                dm, pred_contribs=True, iteration_range=iter_range
            )

    for i, (txid, feats, base) in enumerate(feat_rows):
        raw = float(raw_scores[i])
        prob = _calibrate(raw, calibration)
        flag = bool(raw >= threshold_raw)

        # Compact OOD (flag only for non-detail txids)
        ood_result = svc.ood(feats, base_stats=None, bundle="tx")
        ood_flag = ood_result["flag"]

        result: dict = {
            "prob": prob,
            "flag": flag,
            "ood_flag": ood_flag,
            "model_version": model_version,
            "experimental": True,
        }

        # Full detail for the designated txid
        if full_detail_txid is not None and txid == full_detail_txid and contrib_full_all is not None:
            contrib_full = contrib_full_all[i]
            contrib = contrib_full[:-1]
            base_value = float(contrib_full[-1])

            order = np.argsort(-np.abs(contrib))[:5]
            shown_sum = float(contrib[order].sum())
            other_indices = np.setdiff1d(np.arange(len(contrib)), order)
            other_sum = float(contrib[other_indices].sum())
            logit = float(base_value + shown_sum + other_sum)
            raw_prob = float(1.0 / (1.0 + np.exp(-logit)))
            threshold_calibrated = _calibrate(threshold_raw, calibration)

            explanations = [
                {
                    "feature": feat_names[j],
                    "label": feature_labels.get(feat_names[j], feat_names[j]),
                    "value": float(X[i, j]),
                    "shap": float(contrib[j]),
                }
                for j in order
            ]
            decomposition = {
                "base_value": base_value,
                "shown_sum": shown_sum,
                "other_sum": other_sum,
                "other_count": int(len(other_indices)),
                "logit": logit,
                "raw_prob": raw_prob,
                "prob": prob,
                "threshold_raw": threshold_raw,
                "threshold_calibrated": threshold_calibrated,
            }
            result["explanations"] = explanations
            result["decomposition"] = decomposition
            result["ood"] = ood_result  # full block with out_of_range list

        # Cache compact result (without explanations/decomposition)
        compact = {k: result[k] for k in ("prob", "flag", "ood_flag", "model_version", "experimental")}
        _tx_score_cache[txid] = compact

        results[txid] = result

    return results
