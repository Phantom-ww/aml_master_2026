"""Wallet features for the Live Chain model.

Single source of truth, used by:
  * the training notebook (derive_live_features on Elliptic++ wallets_features.csv),
  * the backend (base_stats_from_txs on Esplora/mempool transactions + derive_live_features).

Elliptic++ base-feature definitions, checked against the dataset (tx_edges built from
AddrTx x TxAddr, self-pairs removed):
  * num_txs_as_sender / num_txs_as receiver: txs where the address is an input / an output.  [verified 99.9%]
  * fees_*: stats over the FULL fee of every tx the address takes part in (one entry per tx). [verified 100%]
  * fees_as_share_*: per (tx, role): tx_fee * address_amount / tx_input_total, i.e. the part of the
    fee proportional to the address's amount -- an absolute BTC value, not a ratio.          [verified on 1-tx wallets]
  * btc_sent / btc_received: per tx, sum of the address's input / output values;
    btc_transacted = the union of both lists (btc_transacted_total == sent_total + received_total). [total verified]
  * transacted_w_address_*: for every counterparty address (outputs when the address sends,
    inputs when it receives, the address itself excluded) count the txs shared with it;
    stats over counterparties; *_total == sum; num_addr_transacted_multiple == count > 1.      [verified 100%]
  * lifetime_in_blocks = last_block - first_block.                                              [verified 100%]
  * blocks_btwn_*: stats over gaps between consecutive tx blocks (all / as input / as output). [not verified]
  * num_timesteps_appeared_in: Elliptic time steps are ~2 weeks; approximated by 2016-block windows. [approximate]

Important: Elliptic++ computes these over the txs that are IN the dataset (a sampled subgraph;
median wallet has 1 tx), while Live mode sees the full history. Counts and totals are therefore
not comparable; the live model uses only scale-free features (shares, ratios, relative fees).
"""
from __future__ import annotations

import numpy as np

SATS = 1e8
EPS = 1e-8          # 1 satoshi, in BTC
BLOCKS_PER_STEP = 2016

STAT_SUFFIXES = ("total", "min", "max", "mean", "median")

# Scale-free features used by the live model (order matters: it is the model's column order).
LIVE_FEATURES = [
    # behaviour shares and counterparty structure
    "share_txs_as_sender", "share_txs_as_receiver", "share_addr_transacted_multiple",
    "transacted_w_address_mean", "transacted_w_address_median",
    "transacted_w_address_min", "transacted_w_address_max",
    # amount / fee shape (ratios are invariant to the BTC price and the fee level)
    "ratio_sent_received", "ratio_amount_max_min", "ratio_amount_mean_median",
    "ratio_fee_max_min", "ratio_fee_mean_median",
    "ratio_received_max_mean", "ratio_sent_max_mean",
    # fee relative to the moved value, in basis points (log1p)
    "rel_fee_total", "rel_fee_max", "rel_fee_min", "rel_fee_median",
]

# Context features: not used by the model, only for the out-of-distribution check and drift report.
CONTEXT_FEATURES = ["total_txs", "lifetime_in_blocks", "btc_transacted_mean", "fees_mean"]

LIVE_FEATURE_LABELS = {
    "share_txs_as_sender": "Share of txs as sender",
    "share_txs_as_receiver": "Share of txs as receiver",
    "share_addr_transacted_multiple": "Share of repeat counterparties",
    "transacted_w_address_mean": "Mean txs per counterparty",
    "transacted_w_address_median": "Median txs per counterparty",
    "transacted_w_address_min": "Min txs per counterparty",
    "transacted_w_address_max": "Max txs per counterparty",
    "ratio_sent_received": "Sent / received (log)",
    "ratio_amount_max_min": "Largest / smallest amount (log)",
    "ratio_amount_mean_median": "Mean / median amount (log)",
    "ratio_fee_max_min": "Largest / smallest fee (log)",
    "ratio_fee_mean_median": "Mean / median fee (log)",
    "ratio_received_max_mean": "Largest / mean received (log)",
    "ratio_sent_max_mean": "Largest / mean sent (log)",
    "rel_fee_total": "Fee share of value, overall (log bps)",
    "rel_fee_max": "Fee share of largest amount (log bps)",
    "rel_fee_min": "Fee share of smallest amount (log bps)",
    "rel_fee_median": "Fee share of median amount (log bps)",
    "total_txs": "Total transactions",
    "lifetime_in_blocks": "Lifetime (blocks)",
    "btc_transacted_mean": "Mean BTC transacted",
    "fees_mean": "Mean fee (BTC)",
}

LIVE_FEATURE_FORMULAS = {
    "share_txs_as_sender": "num_txs_as_sender / max(total_txs, 1)",
    "share_txs_as_receiver": "num_txs_as receiver / max(total_txs, 1)",
    "share_addr_transacted_multiple": "num_addr_transacted_multiple / max(transacted_w_address_total, 1)",
    "transacted_w_address_mean": "as in Elliptic++",
    "transacted_w_address_median": "as in Elliptic++",
    "transacted_w_address_min": "as in Elliptic++",
    "transacted_w_address_max": "as in Elliptic++",
    "ratio_sent_received": "log1p(btc_sent_total / (btc_received_total + eps))",
    "ratio_amount_max_min": "log1p(btc_transacted_max / (btc_transacted_min + eps))",
    "ratio_amount_mean_median": "log1p(btc_transacted_mean / (btc_transacted_median + eps))",
    "ratio_fee_max_min": "log1p(fees_max / (fees_min + eps))",
    "ratio_fee_mean_median": "log1p(fees_mean / (fees_median + eps))",
    "ratio_received_max_mean": "log1p(btc_received_max / (btc_received_mean + eps))",
    "ratio_sent_max_mean": "log1p(btc_sent_max / (btc_sent_mean + eps))",
    "rel_fee_total": "log1p(1e4 * fees_as_share_total / (btc_transacted_total + eps))",
    "rel_fee_max": "log1p(1e4 * fees_as_share_max / (btc_transacted_max + eps))",
    "rel_fee_min": "log1p(1e4 * fees_as_share_min / (btc_transacted_min + eps))",
    "rel_fee_median": "log1p(1e4 * fees_as_share_median / (btc_transacted_median + eps))",
}


def _L(x):
    return np.log1p(np.clip(x, 0, None))


def derive_live_features(b):
    """Scale-free features from Elliptic++ base features.

    `b` is a pandas DataFrame (vectorised, used in the notebook) or a dict of floats
    (one wallet, used by the backend). Returns a dict {feature: value or Series}.
    """
    g = lambda k: b[k]
    total_txs = np.maximum(g("total_txs"), 1)
    cp_total = np.maximum(g("transacted_w_address_total"), 1)
    out = {
        "share_txs_as_sender": g("num_txs_as_sender") / total_txs,
        "share_txs_as_receiver": g("num_txs_as receiver") / total_txs,
        "share_addr_transacted_multiple": g("num_addr_transacted_multiple") / cp_total,
        "transacted_w_address_mean": g("transacted_w_address_mean"),
        "transacted_w_address_median": g("transacted_w_address_median"),
        "transacted_w_address_min": g("transacted_w_address_min"),
        "transacted_w_address_max": g("transacted_w_address_max"),
        "ratio_sent_received": _L(g("btc_sent_total") / (g("btc_received_total") + EPS)),
        "ratio_amount_max_min": _L(g("btc_transacted_max") / (g("btc_transacted_min") + EPS)),
        "ratio_amount_mean_median": _L(g("btc_transacted_mean") / (g("btc_transacted_median") + EPS)),
        "ratio_fee_max_min": _L(g("fees_max") / (g("fees_min") + EPS)),
        "ratio_fee_mean_median": _L(g("fees_mean") / (g("fees_median") + EPS)),
        "ratio_received_max_mean": _L(g("btc_received_max") / (g("btc_received_mean") + EPS)),
        "ratio_sent_max_mean": _L(g("btc_sent_max") / (g("btc_sent_mean") + EPS)),
        "rel_fee_total": _L(1e4 * g("fees_as_share_total") / (g("btc_transacted_total") + EPS)),
        "rel_fee_max": _L(1e4 * g("fees_as_share_max") / (g("btc_transacted_max") + EPS)),
        "rel_fee_min": _L(1e4 * g("fees_as_share_min") / (g("btc_transacted_min") + EPS)),
        "rel_fee_median": _L(1e4 * g("fees_as_share_median") / (g("btc_transacted_median") + EPS)),
    }
    return {k: out[k] for k in LIVE_FEATURES}


def _stats(prefix, values, out):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        for s in STAT_SUFFIXES:
            out[f"{prefix}_{s}"] = 0.0
        return
    out[f"{prefix}_total"] = float(v.sum())
    out[f"{prefix}_min"] = float(v.min())
    out[f"{prefix}_max"] = float(v.max())
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_median"] = float(np.median(v))


def base_stats_from_txs(address: str, txs: list[dict]) -> dict:
    """Elliptic++ wallet base features from Esplora/mempool.space transactions of `address`.

    `txs` are Esplora tx objects (as returned by /address/{a}/txs); unconfirmed txs are ignored.
    Amounts are converted from satoshis to BTC.
    """
    sent, received, fees, fee_share = [], [], [], []
    blocks, in_blocks, out_blocks = [], [], []
    cp_counts: dict[str, int] = {}
    n_sender = n_receiver = n_txs = 0
    seen = set()

    for tx in txs:
        txid = tx.get("txid")
        if txid in seen:
            continue
        seen.add(txid)
        height = (tx.get("status") or {}).get("block_height")
        if height is None:
            continue

        ins = [((v.get("prevout") or {}).get("scriptpubkey_address"), (v.get("prevout") or {}).get("value", 0))
               for v in tx.get("vin", []) if not v.get("is_coinbase")]
        outs = [(o.get("scriptpubkey_address"), o.get("value", 0)) for o in tx.get("vout", [])]
        is_s = any(a == address for a, _ in ins)
        is_r = any(a == address for a, _ in outs)
        if not (is_s or is_r):
            continue

        fee = tx.get("fee", 0) / SATS
        input_total = sum(v for _, v in ins) / SATS or sum(v for _, v in outs) / SATS
        n_txs += 1
        fees.append(fee)
        blocks.append(height)

        cps = set()
        if is_s:
            n_sender += 1
            amt = sum(v for a, v in ins if a == address) / SATS
            sent.append(amt)
            fee_share.append(fee * amt / input_total if input_total > 0 else 0.0)
            in_blocks.append(height)
            cps |= {a for a, _ in outs if a and a != address}
        if is_r:
            n_receiver += 1
            amt = sum(v for a, v in outs if a == address) / SATS
            received.append(amt)
            fee_share.append(fee * amt / input_total if input_total > 0 else 0.0)
            out_blocks.append(height)
            cps |= {a for a, _ in ins if a and a != address}
        for a in cps:
            cp_counts[a] = cp_counts.get(a, 0) + 1

    out: dict[str, float] = {
        "num_txs_as_sender": float(n_sender),
        "num_txs_as receiver": float(n_receiver),
        "total_txs": float(n_txs),
    }
    if blocks:
        first, last = min(blocks), max(blocks)
        out.update(first_block_appeared_in=float(first), last_block_appeared_in=float(last),
                   lifetime_in_blocks=float(last - first),
                   num_timesteps_appeared_in=float(len({h // BLOCKS_PER_STEP for h in blocks})))
    else:
        out.update(first_block_appeared_in=0.0, last_block_appeared_in=0.0,
                   lifetime_in_blocks=0.0, num_timesteps_appeared_in=0.0)

    _stats("btc_transacted", sent + received, out)
    _stats("btc_sent", sent, out)
    _stats("btc_received", received, out)
    _stats("fees", fees, out)
    _stats("fees_as_share", fee_share, out)
    for name, hs in (("blocks_btwn_txs", blocks), ("blocks_btwn_input_txs", in_blocks),
                     ("blocks_btwn_output_txs", out_blocks)):
        _stats(name, np.diff(sorted(set(hs))) if len(set(hs)) > 1 else [], out)
    counts = list(cp_counts.values())
    _stats("transacted_w_address", counts, out)
    out["num_addr_transacted_multiple"] = float(sum(c > 1 for c in counts))
    return out


def live_features_from_txs(address: str, txs: list[dict]) -> tuple[dict, dict]:
    """(live_features, base_stats) for one live address."""
    base = base_stats_from_txs(address, txs)
    feats = {k: float(v) for k, v in derive_live_features(base).items()}
    return feats, base
