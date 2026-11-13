"""Parity check: on-the-fly inference (model_inference.py) == notebook outputs (predictions parquet).
Usage: python scripts/check_model_parity.py <data_dir with CSVs> [model_dir]"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from services.model_inference import ModelBundle, GraphSAGEScorer

data = sys.argv[1]
md = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data, "model")
b = ModelBundle(md)
pred = pd.read_parquet(os.path.join(md, "predictions.parquet"))

feat = pd.read_csv(os.path.join(data, "wallets_features.csv")).rename(columns={"Time step": "ts"})
w = feat.sort_values("ts").drop_duplicates("address", keep="last").set_index("address").reindex(pred["address"])
X = w[b.wallet_features].astype("float32").fillna(0).values

# 1) XGBoost wallets
rows = [dict(zip(b.wallet_features, r)) for r in X]
t0 = time.perf_counter(); res = b.score_wallets(rows, top_k=5); ms = (time.perf_counter() - t0) * 1000
raw = np.array([r["raw"] for r in res]); prob = np.array([r["prob"] for r in res])
print(f"xgb  max|raw-ref|={np.abs(raw - pred.xgb_raw).max():.2e}  max|prob-ref|={np.abs(prob - pred.xgb_score).max():.2e}  ({ms/len(rows):.3f} ms/wallet)")
assert np.allclose(raw, pred.xgb_raw, atol=1e-5) and np.allclose(prob, pred.xgb_score, atol=1e-5)
shap_ref = pd.read_parquet(os.path.join(md, "shap_top.parquet"))
a0 = pred.address.iloc[0]
ref = shap_ref[shap_ref.address == a0].sort_values("rank")
assert list(ref.feature) == [e["feature"] for e in res[0]["explanations"]], "SHAP features differ"
assert np.allclose(ref.shap.values, [e["shap"] for e in res[0]["explanations"]], atol=1e-4)
print("shap top-5 matches for", a0)

# 2) GraphSAGE wallets (numpy, 2-hop on demand) vs notebook full-graph torch inference
aa = pd.read_csv(os.path.join(data, "AddrAddr_edgelist.csv"))
idx = pd.Index(pred.address)
g = GraphSAGEScorer(b, X, idx.get_indexer(aa.input_address), idx.get_indexer(aa.output_address), max_neighbors=None)
sample = np.random.default_rng(0).choice(len(pred), 400, replace=False)
out = g.score(sample)
d_raw = np.abs(np.array(out["raw"]) - pred.gnn_raw.values[sample]).max()
d_prob = np.abs(np.array(out["prob"]) - pred.gnn_score.values[sample]).max()
one = g.score([int(sample[0])])
print(f"gnn  max|raw-ref|={d_raw:.2e}  max|prob-ref|={d_prob:.2e}  (one wallet: {one['ms']} ms, {one['nodes_used']} nodes)")
assert d_raw < 1e-4 and d_prob < 1e-4

# 3) XGBoost transactions
if b.tx_booster is not None:
    txp = pd.read_parquet(os.path.join(md, "tx_predictions.parquet"))
    txf = pd.read_csv(os.path.join(data, "txs_features.csv")).set_index("txId").reindex(txp.txId)
    trows = [dict(zip(b.tx_features, r)) for r in txf[b.tx_features].astype("float32").fillna(0).values[:500]]
    tr = np.array([b.score_tx(r)["raw"] for r in trows])
    d = np.abs(tr - txp.tx_raw.values[:500]).max()
    print(f"tx   max|raw-ref|={d:.2e}"); assert d < 1e-5
print("ALL PARITY CHECKS PASSED")
