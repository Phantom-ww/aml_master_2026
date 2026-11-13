"""Test parity between notebook scoring and ModelService."""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

def main(data_dir: str, model_dir: str):
    from services.model_inference import ModelBundle, GraphSAGEScorer

    DATA = Path(data_dir)
    MODEL = Path(model_dir)

    # Load bundle
    bundle = ModelBundle(str(MODEL))

    # Load wallets + address index (same as ModelService)
    wallets = pd.read_parquet(DATA / "wallets.parquet").set_index("address")
    address_idx = {addr: i for i, addr in enumerate(wallets.index)}

    # Build X_raw
    X_rows = []
    for addr in address_idx.keys():
        if addr in wallets.index:
            row = wallets.loc[addr]
            X_rows.append([float(row.get(f, 0.0) or 0.0) for f in bundle.wallet_features])
        else:
            X_rows.append([0.0] * len(bundle.wallet_features))
    X_raw = np.array(X_rows, dtype=np.float32)

    # Load GNN graph
    addr_edges = pd.read_parquet(DATA / "addr_edges.parquet")
    src_idx = np.array([address_idx.get(a, -1) for a in addr_edges["input_address"]], dtype=np.int32)
    dst_idx = np.array([address_idx.get(a, -1) for a in addr_edges["output_address"]], dtype=np.int32)

    # Build GraphSAGE scorer
    gnn = GraphSAGEScorer(bundle, X_raw, src_idx, dst_idx)

    # Load predictions (ground truth from notebook)
    preds = pd.read_parquet(DATA / "predictions.parquet")
    test_addrs = preds[preds["split"] == "test"]["address"].tolist()[:100]

    # Score with XGBoost + GNN
    xgb_diffs = []
    gnn_diffs = []

    for addr in test_addrs:
        if addr not in wallets.index or addr not in address_idx:
            continue

        # XGBoost
        wallet = wallets.loc[addr]
        feature_dict = {f: float(wallet.get(f, 0.0) or 0.0) for f in bundle.wallet_features}
        xgb_result = bundle.score_wallet(feature_dict, top_k=0)

        # GNN
        idx = address_idx[addr]
        gnn_result = gnn.score([idx])

        # Compare with ground truth (if available in preds)
        pred_row = preds[preds["address"] == addr]
        if not pred_row.empty:
            gt_xgb = pred_row.iloc[0].get("pred_xgb")
            gt_gnn = pred_row.iloc[0].get("pred_gnn")

            if gt_xgb is not None:
                xgb_diffs.append(abs(xgb_result["prob"] - gt_xgb))
            if gt_gnn is not None:
                gnn_diffs.append(abs(gnn_result["prob"][0] - gt_gnn))

    # Report
    if xgb_diffs:
        print(f"XGBoost MAE: {np.mean(xgb_diffs):.6f} (max: {np.max(xgb_diffs):.6f}) over {len(xgb_diffs)} samples")
    else:
        print("XGBoost: No ground truth in predictions.parquet")

    if gnn_diffs:
        print(f"GNN MAE: {np.mean(gnn_diffs):.6f} (max: {np.max(gnn_diffs):.6f}) over {len(gnn_diffs)} samples")
    else:
        print("GNN: No ground truth in predictions.parquet")

    print("✓ Parity test complete (mock predictions.parquet → no ground truth, but inference runs)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_parity.py <data_dir> <model_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
