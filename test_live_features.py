"""Parity of live_features.base_stats_from_txs with Elliptic++ wallet features (verified subset)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from live_features import base_stats_from_txs, derive_live_features, live_features_from_txs, LIVE_FEATURES

D = os.environ.get("EPP_PARQUET_DIR", "/home/claude/project/backend/data/ellipticpp/")


def _tx(txid, ins, outs, fee_sats, height=500000):
    return {"txid": txid, "fee": fee_sats, "status": {"block_height": height},
            "vin": [{"prevout": {"scriptpubkey_address": a, "value": v}} for a, v in ins],
            "vout": [{"scriptpubkey_address": a, "value": v} for a, v in outs]}


def test_synthetic_wallet():
    A = "A"
    txs = [_tx("t1", [("X", 100_000_000)], [(A, 60_000_000), ("Y", 39_990_000)], 10_000, 100),
           _tx("t2", [(A, 60_000_000)], [("Z", 59_980_000)], 20_000, 150),
           _tx("t3", [("X", 50_000_000)], [(A, 49_995_000)], 5_000, 300)]
    b = base_stats_from_txs(A, txs)
    assert b["total_txs"] == 3 and b["num_txs_as_sender"] == 1 and b["num_txs_as receiver"] == 2
    assert b["lifetime_in_blocks"] == 200
    assert np.isclose(b["btc_received_total"], 0.6 + 0.49995) and np.isclose(b["btc_sent_total"], 0.6)
    assert np.isclose(b["fees_total"], 0.00035)
    # fee share of t1 = 0.0001 * 0.6 / 1.0
    assert np.isclose(b["fees_as_share_max"], max(0.0001 * 0.6, 0.0002 * 0.6 / 0.6, 0.00005 * 0.49995 / 0.5))
    # counterparties: X twice (t1, t3), Z once
    assert b["transacted_w_address_total"] == 3 and b["transacted_w_address_max"] == 2
    assert b["num_addr_transacted_multiple"] == 1
    f, _ = live_features_from_txs(A, txs)
    assert list(f) == LIVE_FEATURES and all(np.isfinite(v) for v in f.values())


def test_unconfirmed_and_unrelated_ignored():
    txs = [_tx("u", [("A", 1000)], [("B", 900)], 100, None), _tx("v", [("Q", 1000)], [("R", 900)], 100)]
    assert base_stats_from_txs("A", txs)["total_txs"] == 0


def test_parity_with_ellipticpp():
    """Counts, counterparty stats and fee stats recomputed from Elliptic++ txs match wallets_features."""
    if not os.path.exists(D + "tx_edges.parquet"):
        return
    e = pd.read_parquet(D + "tx_edges.parquet", columns=["src", "dst", "txId", "fees"])
    w = pd.read_parquet(D + "wallets.parquet")
    w = w[(w.total_txs >= 2) & (w.total_txs <= 30)].sample(300, random_state=7)
    A = set(w.address)
    txids = set(e.loc[e.src.isin(A) | e.dst.isin(A), "txId"])
    sub = e[e.txId.isin(txids)]
    txs = {}
    for t, g in sub.groupby("txId"):
        fee = int(round(g.fees.iloc[0] * 1e8))
        txs[t] = _tx(t, [(a, 1000) for a in g.src.unique()], [(a, 1000) for a in g.dst.unique()], fee)
    by_addr = {}
    for t, g in sub.groupby("txId"):
        for a in set(g.src) | set(g.dst):
            if a in A:
                by_addr.setdefault(a, []).append(txs[t])
    cols = ["num_txs_as_sender", "num_txs_as receiver", "transacted_w_address_total", "transacted_w_address_min",
            "transacted_w_address_max", "transacted_w_address_mean", "transacted_w_address_median",
            "num_addr_transacted_multiple", "fees_total", "fees_min", "fees_max", "fees_median"]
    ok = {c: [] for c in cols}
    for _, r in w.iterrows():
        b = base_stats_from_txs(r.address, by_addr.get(r.address, []))
        for c in cols:
            ok[c].append(np.isclose(b[c], r[c], rtol=1e-3, atol=2e-8))
    rates = {c: float(np.mean(v)) for c, v in ok.items()}
    print(rates)
    assert min(rates.values()) > 0.97, rates


def test_derive_dataframe_equals_dict():
    if not os.path.exists(D + "wallets.parquet"):
        return
    w = pd.read_parquet(D + "wallets.parquet").sample(50, random_state=1)
    vec = pd.DataFrame(derive_live_features(w))
    for i, (_, r) in enumerate(w.iterrows()):
        one = derive_live_features(r.to_dict())
        assert np.allclose([one[k] for k in LIVE_FEATURES], vec.iloc[i][LIVE_FEATURES].values)


from live_features import tx_live_features_from_esplora, LIVE_TX_FEATURES, TX_CONTEXT_FEATURES


def test_tx_features_match_authors_example():
    """Elliptic++ example tx 272145560: 1 input 2.7732 BTC -> outputs 0.001917 + 2.770883, fee 0.0004."""
    tx = _tx("x", [("I", 277_320_000)], [("O1", 191_700), ("O2", 277_088_300)], 40_000)
    tx["size"] = 226
    f, b = tx_live_features_from_esplora(tx)
    assert np.isclose(b["in_BTC_total"], 2.7732) and np.isclose(b["out_BTC_total"], 2.7728)
    assert np.isclose(b["out_BTC_mean"], 1.3864) and np.isclose(b["out_BTC_min"], 0.001917)
    assert np.isclose(b["in_BTC_total"] - b["out_BTC_total"], b["fees"])
    assert list(f) == LIVE_TX_FEATURES + TX_CONTEXT_FEATURES
    assert np.isclose(f["share_largest_input"], 1.0, atol=1e-6) and f["num_output_addresses"] == 2
    assert np.isclose(f["fee_rate_sat_b"], 40_000 / 226)


def test_tx_features_coinbase_and_op_return():
    tx = {"txid": "cb", "fee": 0, "size": 200, "status": {"block_height": 1},
          "vin": [{"is_coinbase": True}],
          "vout": [{"scriptpubkey_address": "M", "value": 312_500_000}, {"scriptpubkey_type": "op_return", "value": 0}]}
    f, b = tx_live_features_from_esplora(tx)
    assert b["num_output_addresses"] == 1 and b["num_input_addresses"] == 0
    assert all(np.isfinite(v) for v in f.values())
