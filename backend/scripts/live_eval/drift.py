"""Feature distribution drift report for the live wallet-risk model.

Compares the Elliptic++ training reference sample to the recently collected live
addresses (negatives as population proxy; positives separately). Reports KS statistic,
PSI (Population Stability Index), and histogram overlays for the most drifted features.

PSI interpretation:
  < 0.10  — stable
  0.10–0.25 — moderate drift
  > 0.25  — strong drift

Usage:
    python drift.py [--data-dir DIR] [--model-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from chain.live_features import (
    live_features_from_txs,
    LIVE_FEATURES,
    CONTEXT_FEATURES,
)

WINDOW_TXS = 10
PSI_EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _psi_interp(live_q: str, psi_edges: list, psi_ref_props: list) -> str:
    if live_q < 0.1:
        return "stable"
    elif live_q < 0.25:
        return "moderate"
    else:
        return "strong"


def _compute_psi(live_values: np.ndarray, psi_edges: list, psi_ref_props: list) -> float:
    """Compute PSI using reference bin edges and reference proportions."""
    edges = np.array(psi_edges)
    ref_props = np.array(psi_ref_props)

    # Assign each live value to a bin (same as np.digitize with the reference edges)
    # bins: (-inf, e0], (e0, e1], ..., (e_{n-1}, inf)
    # np.digitize returns 0 for values below edges[0], n for values above edges[-1]
    bins = np.digitize(live_values, edges)
    n_bins = len(ref_props)

    # Count live values per bin
    live_counts = np.bincount(bins, minlength=n_bins)[:n_bins].astype(float)
    n_total = len(live_values)
    if n_total == 0:
        return float("nan")

    live_props = live_counts / n_total
    # Add epsilon to avoid log(0)
    live_props_safe = np.clip(live_props, PSI_EPS, None)
    ref_props_safe = np.clip(ref_props, PSI_EPS, None)
    # Renormalise after clipping
    live_props_safe = live_props_safe / live_props_safe.sum()
    ref_props_safe = ref_props_safe / ref_props_safe.sum()

    psi = float(np.sum((live_props_safe - ref_props_safe) * np.log(live_props_safe / ref_props_safe)))
    return psi


# ---------------------------------------------------------------------------
# Feature extraction from cached tx histories
# ---------------------------------------------------------------------------

def extract_features_from_labels(
    labels: pd.DataFrame,
    raw_dir: Path,
    mode: str,
    all_features: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with one row per address and one column per feature."""
    rows = []
    missing = 0
    for _, row in labels.iterrows():
        addr = row["address"]
        rp = raw_dir / f"{addr}.json.gz"
        if not rp.exists():
            missing += 1
            continue
        txs = _load_json_gz(rp)
        if mode == "window":
            confirmed = [tx for tx in txs if (tx.get("status") or {}).get("block_height")]
            txs_used = confirmed[:WINDOW_TXS]
        else:
            txs_used = txs
        try:
            feats, base = live_features_from_txs(addr, txs_used)
            combined = {**feats, **{f: base.get(f) for f in CONTEXT_FEATURES}}
            combined["address"] = addr
            combined["y"] = int(row["y"])
            rows.append(combined)
        except Exception as e:
            warnings.warn(f"feature error for {addr}: {e}")
            continue

    if missing:
        warnings.warn(f"{missing} addresses skipped — raw file not found")

    if not rows:
        return pd.DataFrame(columns=all_features + ["address", "y"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_drift(
    ref_df: pd.DataFrame,
    live_neg: pd.DataFrame,
    meta: dict,
    mode: str,
) -> list[dict]:
    """Compute KS and PSI for each feature."""
    all_features = meta["features"] + meta.get("context_features", [])
    ref_map = meta.get("reference", {})
    rows = []

    for feat in all_features:
        if feat not in ref_df.columns or feat not in live_neg.columns:
            continue

        ref_vals = ref_df[feat].dropna().values.astype(float)
        live_vals = live_neg[feat].dropna().values.astype(float)

        if len(ref_vals) < 2 or len(live_vals) < 2:
            continue

        ks_stat, ks_pval = ks_2samp(ref_vals, live_vals)

        feat_ref = ref_map.get(feat, {})
        psi_edges = feat_ref.get("psi_edges", [])
        psi_ref_props = feat_ref.get("psi_ref_props", [])

        if psi_edges and psi_ref_props:
            psi = _compute_psi(live_vals, psi_edges, psi_ref_props)
        else:
            psi = float("nan")

        label = meta.get("feature_labels", {}).get(feat, feat)
        interp = "n/a" if np.isnan(psi) else ("stable" if psi < 0.1 else ("moderate" if psi < 0.25 else "strong"))

        rows.append({
            "feature": feat,
            "label": label,
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "psi": round(float(psi), 4) if not np.isnan(psi) else None,
            "interpretation": interp,
            "mode": mode,
        })

    return rows


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    df.to_csv(path, index=False)


def save_md(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Feature | Label | KS stat | KS p-value | PSI | Interpretation | Mode |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        psi_str = f"{row['psi']:.4f}" if row["psi"] is not None else "n/a"
        lines.append(
            f"| `{row['feature']}` | {row['label']} | {row['ks_stat']:.4f} |"
            f" {row['ks_pvalue']:.4g} | {psi_str} | {row['interpretation']} | {row['mode']} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_drift_top(
    ref_df: pd.DataFrame,
    live_neg: pd.DataFrame,
    live_pos: pd.DataFrame,
    rows: list[dict],
    out_path: Path,
    n_top: int = 6,
) -> None:
    """Histogram overlays for the n_top most-drifted features."""
    valid_rows = [r for r in rows if r["psi"] is not None]
    top = sorted(valid_rows, key=lambda r: r["psi"], reverse=True)[:n_top]
    if not top:
        return

    ncols = 3
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, (feat_row, ax) in enumerate(zip(top, axes)):
        feat = feat_row["feature"]
        ref_vals = ref_df[feat].dropna().values.astype(float)
        neg_vals = live_neg[feat].dropna().values.astype(float) if feat in live_neg.columns else np.array([])
        pos_vals = live_pos[feat].dropna().values.astype(float) if feat in live_pos.columns and len(live_pos) > 0 else np.array([])

        all_vals = np.concatenate([ref_vals, neg_vals, pos_vals])
        p1, p99 = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        bins = np.linspace(p1, p99, 30)

        ax.hist(ref_vals, bins=bins, density=True, alpha=0.5, color="gray", label="Reference")
        if len(neg_vals):
            ax.hist(neg_vals, bins=bins, density=True, alpha=0.5, color="steelblue", label="Live (neg)")
        if len(pos_vals):
            ax.hist(pos_vals, bins=bins, density=True, alpha=0.5, color="darkorange", label="Live (pos)")

        ax.set_title(f"{feat_row['label']}\nPSI={feat_row['psi']:.3f}", fontsize=9)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top drifted features (reference vs. live)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_summary(rows_window: list[dict], rows_full: list[dict]) -> None:
    """Print a short text summary of drift results."""
    def _top(rows, n=5):
        return sorted([r for r in rows if r["psi"] is not None], key=lambda r: r["psi"], reverse=True)[:n]

    print("\n=== Drift Summary ===")
    for mode_label, rows in [("window", rows_window), ("full", rows_full)]:
        if not rows:
            continue
        top = _top(rows)
        print(f"\n[{mode_label} mode] Top drifted features:")
        for r in top:
            print(f"  {r['feature']:40s}  PSI={r['psi']:.3f}  ({r['interpretation']})")

    # Compare window vs full: does window mode reduce drift?
    if rows_window and rows_full:
        feat_map_w = {r["feature"]: r["psi"] for r in rows_window if r["psi"] is not None}
        feat_map_f = {r["feature"]: r["psi"] for r in rows_full if r["psi"] is not None}
        common = set(feat_map_w) & set(feat_map_f)
        if common:
            w_avg = np.mean([feat_map_w[f] for f in common])
            f_avg = np.mean([feat_map_f[f] for f in common])
            if w_avg < f_avg:
                print(f"\n→ Window mode reduces mean PSI ({w_avg:.3f} vs {f_avg:.3f} for full): "
                      f"using recent {WINDOW_TXS} txs is closer to the training distribution.")
            else:
                print(f"\n→ Full mode has lower mean PSI ({f_avg:.3f} vs {w_avg:.3f} for window).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _PROJECT_ROOT / "data" / "live_eval"
    model_dir = Path(args.model_dir) if args.model_dir else _PROJECT_ROOT / "backend" / "data" / "ellipticpp"
    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "reports" / "live_eval"
    raw_dir = data_dir / "raw"

    # Load meta
    meta_path = model_dir / "live_model_meta.json"
    if not meta_path.exists():
        print(f"ERROR: live_model_meta.json not found at {meta_path}")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    all_features = meta["features"] + meta.get("context_features", [])

    # Load reference sample
    ref_path = model_dir / "live_reference_sample.parquet"
    if not ref_path.exists():
        print(f"ERROR: live_reference_sample.parquet not found at {ref_path}")
        sys.exit(1)
    ref_df = pd.read_parquet(ref_path, columns=[f for f in all_features if f != "address"])
    print(f"Reference sample: {len(ref_df)} rows")

    # Load labels
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        print(f"ERROR: labels.csv not found at {labels_path}. Run collect.py first.")
        sys.exit(1)
    labels = pd.read_csv(labels_path)
    neg_labels = labels[labels["y"] == 0]
    pos_labels = labels[labels["y"] == 1]
    print(f"Labels: {len(neg_labels)} negatives, {len(pos_labels)} positives")

    all_rows_window, all_rows_full = [], []

    for mode in ["window", "full"]:
        print(f"\n=== Drift computation: {mode} mode ===")

        live_neg = extract_features_from_labels(neg_labels, raw_dir, mode, all_features)
        live_pos = extract_features_from_labels(pos_labels, raw_dir, mode, all_features)
        print(f"  Negatives: {len(live_neg)}, Positives: {len(live_pos)}")

        rows = compute_drift(ref_df, live_neg, meta, mode)

        if mode == "window":
            all_rows_window = rows
        else:
            all_rows_full = rows

        csv_path = out_dir / "drift.csv"
        md_path = out_dir / "drift.md"
        # Append mode rows or overwrite for the first pass
        all_combined = all_rows_window + all_rows_full
        save_csv(all_combined, csv_path)
        save_md(all_combined, md_path)

        # Top plot using most recent mode
        plot_drift_top(ref_df, live_neg, live_pos, rows, out_dir / "drift_top.png")

    print_summary(all_rows_window, all_rows_full)

    # Final save of combined results
    all_combined = all_rows_window + all_rows_full
    save_csv(all_combined, out_dir / "drift.csv")
    save_md(all_combined, out_dir / "drift.md")

    print(f"\nOutputs saved to {out_dir}")
    print(f"  drift.csv, drift.md, drift_top.png")

    # === Tx drift ===
    run_tx_drift(data_dir, model_dir, out_dir)


# ---------------------------------------------------------------------------
# Tx drift section
# ---------------------------------------------------------------------------

def run_tx_drift(data_dir: Path, model_dir: Path, out_dir: Path) -> None:
    """Compute KS + PSI for tx features and append a Transactions section to drift.md."""
    import gzip as _gzip
    from chain.live_features import tx_live_features_from_esplora, LIVE_TX_FEATURES, TX_CONTEXT_FEATURES

    tx_labels_path = data_dir / "tx_labels.csv"
    tx_meta_path = model_dir / "live_tx_model_meta.json"
    tx_ref_path = model_dir / "live_tx_reference_sample.parquet"

    missing_files = [p for p in (tx_labels_path, tx_meta_path, tx_ref_path) if not p.exists()]
    if missing_files:
        print(f"\n[tx drift] Skipping — missing files: {[str(p) for p in missing_files]}")
        return

    with open(tx_meta_path) as f:
        tx_meta = json.load(f)
    all_tx_features = LIVE_TX_FEATURES + TX_CONTEXT_FEATURES

    tx_ref_df = pd.read_parquet(tx_ref_path, columns=[f for f in all_tx_features if f != "txid"])
    print(f"\n[tx drift] Tx reference sample: {len(tx_ref_df)} rows")

    tx_labels = pd.read_csv(tx_labels_path)
    tx_neg_labels = tx_labels[tx_labels["y"] == 0]
    tx_pos_labels = tx_labels[tx_labels["y"] == 1]

    def _extract_tx_features(label_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in label_df.iterrows():
            raw_path = row.get("raw_path", "")
            if not raw_path or not Path(raw_path).exists():
                continue
            with _gzip.open(raw_path, "rt", encoding="utf-8") as f:
                tx = json.load(f)
            try:
                feats, _ = tx_live_features_from_esplora(tx)
                feats["y"] = int(row["y"])
                rows.append(feats)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=all_tx_features + ["y"])
        return pd.DataFrame(rows)

    live_tx_neg = _extract_tx_features(tx_neg_labels)
    live_tx_pos = _extract_tx_features(tx_pos_labels)
    print(f"  [tx drift] Negatives: {len(live_tx_neg)}, Positives: {len(live_tx_pos)}")

    # Compute drift using the same logic as wallet drift
    tx_drift_rows = compute_drift(tx_ref_df, live_tx_neg, tx_meta, mode="tx")
    print(f"  [tx drift] {len(tx_drift_rows)} features computed")

    # Save drift_tx_top.png
    plot_drift_top(tx_ref_df, live_tx_neg, live_tx_pos, tx_drift_rows, out_dir / "drift_tx_top.png")
    print(f"  [tx drift] Saved {out_dir / 'drift_tx_top.png'}")

    # Append "## Transactions" section to drift.md
    drift_md_path = out_dir / "drift.md"
    if tx_drift_rows:
        df_tx_drift = pd.DataFrame(tx_drift_rows).sort_values("psi", ascending=False, na_position="last")
        lines = [
            "",
            "## Transactions",
            "",
            "KS and PSI statistics for the live transaction model features, "
            "comparing the Elliptic++ training reference to live negatives "
            "(random non-coinbase transactions from sampled blocks).",
            "",
            "| Feature | Label | KS stat | KS p-value | PSI | Interpretation |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in df_tx_drift.iterrows():
            psi_str = f"{r['psi']:.4f}" if r["psi"] is not None else "n/a"
            lines.append(
                f"| `{r['feature']}` | {r['label']} | {r['ks_stat']:.4f} |"
                f" {r['ks_pvalue']:.4g} | {psi_str} | {r['interpretation']} |"
            )
        with open(drift_md_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [tx drift] Appended Transactions section to {drift_md_path}")


if __name__ == "__main__":
    main()
