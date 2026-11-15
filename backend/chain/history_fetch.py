"""Fetch confirmed transaction history for a live address.

Pages through /address/{a}/txs and /address/{a}/txs/chain/{last_seen_txid} on the
configured Esplora instance. Returns raw Esplora tx dicts (the same format that
chain.live_features.base_stats_from_txs consumes).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chain.esplora import EsploraSource

logger = logging.getLogger(__name__)


async def fetch_confirmed_txs(
    address: str,
    source: "EsploraSource",
    max_txs: int = 500,
) -> tuple[list[dict], bool]:
    """Fetch up to *max_txs* confirmed transactions for *address* from Esplora.

    Pages through /address/{a}/txs (first page, may include mempool txs) and then
    /address/{a}/txs/chain/{last_seen_txid} for subsequent pages.  Unconfirmed txs
    (no block_height) are silently dropped from the returned list.

    Returns
    -------
    txs : list[dict]
        Raw Esplora tx dicts, most-recent first, confirmed only.
    truncated : bool
        True when the full history exceeds *max_txs* and was cut off.
    """
    txs: list[dict] = []
    last_seen_txid: str | None = None
    truncated = False

    while True:
        path = (
            f"/address/{address}/txs/chain/{last_seen_txid}"
            if last_seen_txid
            else f"/address/{address}/txs"
        )
        try:
            page: list[dict] = await source._fetch_json(path)
        except Exception as exc:
            logger.warning("history_fetch: %s for %s: %s", path, address, exc)
            break

        if not page:
            break

        # Esplora returns mempool txs first on the first call; skip them.
        for tx in page:
            status = tx.get("status") or {}
            if status.get("block_height") is None:
                continue          # unconfirmed — ignore
            txs.append(tx)
            if len(txs) >= max_txs:
                truncated = True
                break

        if truncated:
            break

        # Esplora paginates confirmed txs in pages of 25.
        # If the page has fewer than 25 entries (after including mempool txs on page 1)
        # we may have exhausted the history.  Use the last confirmed tx as next cursor.
        confirmed_in_page = [t for t in page if (t.get("status") or {}).get("block_height") is not None]
        if len(confirmed_in_page) < 25:
            break  # last page

        last_seen_txid = confirmed_in_page[-1]["txid"]

    return txs, truncated
