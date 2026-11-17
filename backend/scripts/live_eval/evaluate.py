"""Evaluate the live wallet-risk model on the collected labeled dataset.

Computes ROC-AUC, PR-AUC, precision/recall/F1 at stored threshold, and OOD statistics
for both "window" and "full" feature modes, with 95% bootstrap CIs.

Also evaluates the original Elliptic++ XGBoost model on the same addresses to show
the cross-time degradation compared to the scale-free live model.

Recalibration (Platt scaling) is fitted and saved if --recalibrate is set.

Usage:
    python evaluate.py [--data-dir DIR] [--model-dir DIR] [--mode {window,full,both}] [--n-bootstrap 1000]
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from scipy.special import logit as scipy_logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chain.live_features import (
    base_stats_from_txs,
    live_features_from_txs,
    LIVE_FEATURES,
    CONTEXT_FEATURES,
)

WINDOW_TXS = 10
EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _calibrate(raw, cal_x, cal_y):
    return float(np.interp(raw, cal_x, cal_y))


def _bootstrap_auc(y_true, y_score, n=1000, metric="roc", seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            if metric == "roc":
                vals.append(roc_auc_score(yt, ys))
            else:
                vals.append(average_precision_score(yt, ys))
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan, np.nan
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# OOD check
# ---------------------------------------------------------------------------

def _ood_check(features: dict, base_stats: dict, meta: dict) -> dict:
    """Compute OOD flag per meta.ood_rule."""
    rule = meta["ood_rule"]
    low_q = rule["low_q"]
    high_q = rule["high_q"]
    ref = meta["reference"]
    important = set(rule["important_features"])
    min_oor = rule.get("min_out_of_range", 3)
    context_rule = rule.get("context", {})

    out_of_range = []
    for feat in meta["features"] + meta.get("context_features", []):
        val = features.get(feat) if feat in meta["features"] else base_stats.get(feat)
        if val is None:
            continue
        feat_ref = ref.get(feat)
        if feat_ref is None:
            continue
        lo = feat_ref["quantiles"].get(low_q)
        hi = feat_ref["quantiles"].get(high_q)
        if lo is None or hi is None:
            continue
        if val < lo or val > hi:
            out_of_range.append({
                "feature": feat,
                "label": meta.get("feature_labels", {}).get(feat, feat),
                "value": float(val),
                "low": float(lo),
                "high": float(hi),
            })

    oor_names = {r["feature"] for r in out_of_range}
    important_oor = any(f in oor_names for f in important)
    count_oor = len([r for r in out_of_range if r["feature"] in meta["features"]])
    total_txs_flag = False
    if "total_txs" in context_rule:
        q_key = context_rule["total_txs"]
        tq = ref.get("total_txs", {}).get("quantiles", {}).get(q_key)
        if tq is not None and base_stats.get("total_txs", 0) > tq:
            total_txs_flag = True

    flag = important_oor or count_oor >= min_oor or total_txs_flag
    return {"flag": flag, "out_of_range": out_of_range}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path, raw_dir: Path, mode: str) -> pd.DataFrame:
    """Load labels.csv and compute features for each address."""
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    labels = pd.read_csv(labels_path)
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
            # Keep only the most recent WINDOW_TXS confirmed txs
            confirmed = [tx for tx in txs if (tx.get("status") or {}).get("block_height")]
            txs_used = confirmed[:WINDOW_TXS]
        else:
            txs_used = txs
        try:
            feats, base = live_features_from_txs(addr, txs_used)
        except Exception as e:
            warnings.warn(f"feature error for {addr}: {e}")
            continue
        rows.append({
            "address": addr,
            "y": int(row["y"]),
            "features": feats,
            "base_stats": base,
            "n_txs_used": len(txs_used),
        })
    if missing:
        warnings.warn(f"{missing} addresses skipped — raw file not found")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def score_live_model(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    meta_path = model_dir / "live_model_meta.json"
    model_path = model_dir / "xgb_live_model.json"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Live model files not found in {model_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    best_iter = meta.get("best_iteration", 0)
    iter_range = (0, best_iter + 1) if best_iter else (0, 0)
    threshold_raw = meta["threshold_raw"]
    cal = meta["calibration"]
    cal_x, cal_y = np.array(cal["x"]), np.array(cal["y"])
    features_list = meta["features"]

    raws, probs, flags, oods = [], [], [], []
    for _, row in df.iterrows():
        feats = row["features"]
        base = row["base_stats"]
        X = np.array([[float(feats.get(f, 0.0) or 0.0) for f in features_list]], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=features_list)
        raw = float(booster.predict(dm, iteration_range=iter_range)[0])
        prob = _calibrate(raw, cal_x, cal_y)
        flag = raw >= threshold_raw
        ood = _ood_check(feats, base, meta)
        raws.append(raw)
        probs.append(prob)
        flags.append(flag)
        oods.append(ood["flag"])

    result = df.copy()
    result["live_raw"] = raws
    result["live_prob"] = probs
    result["live_flag"] = flags
    result["live_ood"] = oods
    result["live_threshold_raw"] = threshold_raw
    return result, meta


def score_epp_model(df: pd.DataFrame, epp_model_dir: Path) -> pd.DataFrame:
    """Score with original Elliptic++ XGBoost model on base stats features."""
    xgb_path = epp_model_dir / "xgb_model.json"
    meta_path = epp_model_dir / "model_meta.json"
    cal_path = epp_model_dir / "calibration.json"
    if not xgb_path.exists():
        warnings.warn(f"EPP model not found at {xgb_path}, skipping EPP scoring")
        df["epp_prob"] = np.nan
        df["epp_flag"] = False
        return df

    with open(meta_path) as f:
        epp_meta = json.load(f)
    with open(cal_path) as f:
        epp_cal = json.load(f)

    booster = xgb.Booster()
    booster.load_model(str(xgb_path))

    best_iter_attr = booster.attr("best_iteration")
    best_iter = int(best_iter_attr) if best_iter_attr is not None else 0
    iter_range = (0, best_iter + 1) if best_iter else (0, 0)

    epp_features = epp_meta["features"]
    epp_thr = epp_meta["thresholds"]["xgb_raw"]
    cal_xgb = epp_cal["xgb"]
    cal_x = np.array(cal_xgb["x"])
    cal_y = np.array(cal_xgb["y"])

    probs, flags = [], []
    for _, row in df.iterrows():
        base = row["base_stats"]
        X = np.array([[float(base.get(f, 0.0) or 0.0) for f in epp_features]], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=epp_features)
        raw = float(booster.predict(dm, iteration_range=iter_range)[0])
        prob = _calibrate(raw, cal_x, cal_y)
        probs.append(prob)
        flags.append(raw >= epp_thr)

    result = df.copy()
    result["epp_prob"] = probs
    result["epp_flag"] = flags
    result["epp_threshold_raw"] = epp_thr
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, score_col: str, flag_col: str, ood_col: str, n_bootstrap: int = 1000) -> dict:
    y = df["y"].values.astype(int)
    scores = df[score_col].values
    flags = df[flag_col].values.astype(bool)

    valid = ~np.isnan(scores)
    y, scores, flags = y[valid], scores[valid], flags[valid]
    if len(np.unique(y)) < 2:
        return {"error": "only one class present"}

    roc_mean, roc_lo, roc_hi = _bootstrap_auc(y, scores, n_bootstrap, "roc")
    pr_mean, pr_lo, pr_hi = _bootstrap_auc(y, scores, n_bootstrap, "pr")
    base_rate = float(y.mean())
    prec, rec, f1, _ = precision_recall_fscore_support(y, flags, average="binary", zero_division=0)
    brier = float(brier_score_loss(y, scores))

    # OOD split
    ood_flags = df.loc[valid, ood_col].values.astype(bool) if ood_col in df.columns else np.zeros(len(y), bool)
    ood_share = float(ood_flags.mean())
    in_dist_mask = ~ood_flags
    roc_in = float(roc_auc_score(y[in_dist_mask], scores[in_dist_mask])) if in_dist_mask.sum() > 1 and len(np.unique(y[in_dist_mask])) > 1 else np.nan
    roc_ood = float(roc_auc_score(y[ood_flags], scores[ood_flags])) if ood_flags.sum() > 1 and len(np.unique(y[ood_flags])) > 1 else np.nan

    return {
        "roc_auc": roc_mean, "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi,
        "pr_auc": pr_mean, "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
        "base_rate": base_rate,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "brier": brier,
        "ood_share": ood_share,
        "roc_auc_in_dist": roc_in,
        "roc_auc_ood": roc_ood,
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Recalibration (Platt scaling)
# ---------------------------------------------------------------------------

def recalibrate(df: pd.DataFrame, out_dir: Path) -> dict:
    y = df["y"].values.astype(int)
    raws = df["live_raw"].values
    valid = ~np.isnan(raws)
    y, raws = y[valid], raws[valid]

    # Clip raws away from 0/1 for logit stability
    raws_clipped = np.clip(raws, EPS, 1 - EPS)
    logits = scipy_logit(raws_clipped).reshape(-1, 1)

    brier_before = float(brier_score_loss(y, raws))

    # 5-fold cross-validated logistic regression
    lr = LogisticRegressionCV(cv=5, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(logits, y)

    probs_recal = lr.predict_proba(logits)[:, 1]
    brier_after = float(brier_score_loss(y, probs_recal))

    result = {
        "coef": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "brier_before": brier_before,
        "brier_after": brier_after,
    }

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    for probs, label, color in [(raws, "Before recal.", "steelblue"), (probs_recal, "After recal.", "darkorange")]:
        bin_means, bin_fracs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_means.append(probs[mask].mean())
                bin_fracs.append(y[mask].mean())
        ax.plot(bin_means, bin_fracs, "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "reliability.png", dpi=120)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    linestyles = {"window": "-", "full": "--"}
    colors = {"live": "steelblue", "epp": "darkorange"}
    for key, (df, _) in all_results.items():
        model_type, mode = key.split("_", 1)
        y = df["y"].values.astype(int)
        score_col = "live_prob" if model_type == "live" else "epp_prob"
        scores = df[score_col].values
        valid = ~np.isnan(scores)
        if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[valid], scores[valid])
        auc = roc_auc_score(y[valid], scores[valid])
        ax.plot(fpr, tpr, linestyles.get(mode, "-"), color=colors.get(model_type, "gray"),
                label=f"{model_type} ({mode}) AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curves")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "roc.png", dpi=120)
    plt.close(fig)


def plot_pr(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    linestyles = {"window": "-", "full": "--"}
    colors = {"live": "steelblue", "epp": "darkorange"}
    for key, (df, _) in all_results.items():
        model_type, mode = key.split("_", 1)
        y = df["y"].values.astype(int)
        score_col = "live_prob" if model_type == "live" else "epp_prob"
        scores = df[score_col].values
        valid = ~np.isnan(scores)
        if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y[valid], scores[valid])
        ap = average_precision_score(y[valid], scores[valid])
        ax.plot(rec, prec, linestyles.get(mode, "-"), color=colors.get(model_type, "gray"),
                label=f"{model_type} ({mode}) AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curves")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pr.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_auc(mean, lo, hi):
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {(hi-lo)/2:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"


def metrics_to_md(all_metrics: dict) -> str:
    header = "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | OOD% | Brier |\n"
    header += "|---|---|---|---|---|---|---|---|\n"
    rows = []
    for name, m in all_metrics.items():
        if "error" in m:
            rows.append(f"| {name} | — | — | — | — | — | — | — |")
            continue
        roc = _fmt_auc(m["roc_auc"], m["roc_auc_lo"], m["roc_auc_hi"])
        pr = _fmt_auc(m["pr_auc"], m["pr_auc_lo"], m["pr_auc_hi"])
        rows.append(
            f"| {name} | {roc} | {pr} | {m['precision']:.3f} | {m['recall']:.3f} |"
            f" {m['f1']:.3f} | {m['ood_share']*100:.1f}% | {m['brier']:.4f} |"
        )
    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--mode", choices=["window", "full", "both"], default="both")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--recalibrate", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _PROJECT_ROOT / "data" / "live_eval"
    model_dir = Path(args.model_dir) if args.model_dir else _PROJECT_ROOT / "backend" / "data" / "ellipticpp"
    epp_model_dir = _PROJECT_ROOT / "data" / "ellipticpp" / "model"
    raw_dir = data_dir / "raw"
    out_dir = _PROJECT_ROOT / "reports" / "live_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["window", "full"] if args.mode == "both" else [args.mode]

    all_metrics: dict = {}
    all_results: dict = {}

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        df = load_dataset(data_dir, raw_dir, mode)
        print(f"  Loaded {len(df)} addresses (pos={int(df['y'].sum())}, neg={int((df['y']==0).sum())})")

        df, meta = score_live_model(df, model_dir)
        df = score_epp_model(df, epp_model_dir)

        all_results[f"live_{mode}"] = (df, meta)
        all_results[f"epp_{mode}"] = (df, meta)

        live_m = compute_metrics(df, "live_prob", "live_flag", "live_ood", args.n_bootstrap)
        epp_m = compute_metrics(df, "epp_prob", "epp_flag", "live_ood", args.n_bootstrap)
        all_metrics[f"live_{mode}"] = live_m
        all_metrics[f"epp_xgb_{mode}"] = epp_m

        roc_str = f"{live_m['roc_auc']:.3f}" if "roc_auc" in live_m and not np.isnan(live_m["roc_auc"]) else "n/a"
        epp_str = f"{epp_m['roc_auc']:.3f}" if "roc_auc" in epp_m and not np.isnan(epp_m["roc_auc"]) else "n/a"
        print(f"  live ROC-AUC: {roc_str}  OOD%: {live_m.get('ood_share', 0)*100:.1f}%")
        print(f"  epp  ROC-AUC: {epp_str}")

    # Recalibration
    if args.recalibrate and all_results:
        for _, (first_df, _) in all_results.items():
            if "live_raw" in first_df.columns:
                print("\n=== Recalibration (Platt) ===")
                recal = recalibrate(first_df, out_dir)
                print(f"  Brier before: {recal['brier_before']:.4f}  after: {recal['brier_after']:.4f}")
                recal_path = model_dir / "live_recalibration.json"
                with open(recal_path, "w") as f:
                    json.dump(recal, f, indent=2)
                print(f"  Saved to {recal_path}")
                break

    # Save outputs
    metrics_path = out_dir / "metrics.json"

    def _json_safe(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        raise TypeError(type(obj))

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_safe)
    print(f"\nMetrics saved to {metrics_path}")

    md = metrics_to_md(all_metrics)
    md_path = out_dir / "metrics.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table saved to {md_path}")
    print("\n" + md)

    plot_roc(all_results, out_dir)
    plot_pr(all_results, out_dir)
    print(f"Plots saved to {out_dir}")

    # === Tx model evaluation ===
    run_tx_evaluation(data_dir, model_dir, out_dir, args.n_bootstrap, args.recalibrate)


# ---------------------------------------------------------------------------
# Tx model evaluation
# ---------------------------------------------------------------------------

def run_tx_evaluation(
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
    n_bootstrap: int = 1000,
    recalibrate_flag: bool = True,
) -> None:
    """Score the live tx model on tx_labels.csv and append a Transactions section to metrics.md."""
    from chain.live_features import tx_live_features_from_esplora, LIVE_TX_FEATURES, TX_CONTEXT_FEATURES

    tx_labels_path = data_dir / "tx_labels.csv"
    if not tx_labels_path.exists():
        print(f"\n[tx eval] tx_labels.csv not found at {tx_labels_path} — run collect.py first.")
        return

    tx_meta_path = model_dir / "live_tx_model_meta.json"
    tx_model_path = model_dir / "xgb_live_tx_model.json"
    if not tx_meta_path.exists() or not tx_model_path.exists():
        print(f"\n[tx eval] Tx model files not found in {model_dir} — skipping tx evaluation.")
        return

    with open(tx_meta_path) as f:
        tx_meta = json.load(f)

    tx_booster = xgb.Booster()
    tx_booster.load_model(str(tx_model_path))
    best_iter = tx_meta.get("best_iteration", 0)
    iter_range = (0, best_iter + 1) if best_iter else (0, 0)
    threshold_raw = tx_meta["threshold_raw"]
    cal = tx_meta["calibration"]
    cal_x, cal_y = np.array(cal["x"]), np.array(cal["y"])
    features_list = tx_meta["features"]
    feature_labels = tx_meta.get("feature_labels", {})

    tx_labels = pd.read_csv(tx_labels_path)
    print(f"\n[tx eval] Loaded {len(tx_labels)} tx labels"
          f" (pos={int((tx_labels['y']==1).sum())}, neg={int((tx_labels['y']==0).sum())})")

    raws, probs, flags, oods, ys = [], [], [], [], []
    missing = 0
    for _, row in tx_labels.iterrows():
        raw_path = row.get("raw_path", "")
        if not raw_path or not Path(raw_path).exists():
            missing += 1
            continue
        with gzip.open(raw_path, "rt", encoding="utf-8") as f:
            tx = json.load(f)
        try:
            feats, _ = tx_live_features_from_esplora(tx)
        except Exception as e:
            warnings.warn(f"tx featurise error {row.get('txid')}: {e}")
            continue

        X = np.array([[float(feats.get(ft, 0.0) or 0.0) for ft in features_list]], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=features_list)
        raw = float(tx_booster.predict(dm, iteration_range=iter_range)[0])
        prob = _calibrate(raw, cal_x, cal_y)
        flag = raw >= threshold_raw

        # Compact OOD check using meta.ood_rule
        rule = tx_meta.get("ood_rule", {})
        ref = tx_meta.get("reference", {})
        low_q, high_q = rule.get("low_q", "0.5"), rule.get("high_q", "99.5")
        important = set(rule.get("important_features", []))
        min_oor = rule.get("min_out_of_range", 3)
        oor_count = 0
        important_oor = False
        for ft in features_list:
            val = feats.get(ft)
            if val is None:
                continue
            fr = ref.get(ft, {}).get("quantiles", {})
            lo = fr.get(low_q)
            hi = fr.get(high_q)
            if lo is None or hi is None:
                continue
            if float(val) < lo or float(val) > hi:
                oor_count += 1
                if ft in important:
                    important_oor = True
        ood_flag = important_oor or oor_count >= min_oor

        raws.append(raw)
        probs.append(prob)
        flags.append(flag)
        oods.append(ood_flag)
        ys.append(int(row["y"]))

    if missing:
        print(f"  [tx eval] {missing} txids skipped — raw file not found")

    if not ys or len(np.unique(ys)) < 2:
        print("  [tx eval] Insufficient data for metrics (need both classes)")
        return

    df_tx = pd.DataFrame({"y": ys, "tx_prob": probs, "tx_flag": flags, "tx_ood": oods, "tx_raw": raws})
    tx_metrics = compute_metrics(df_tx, "tx_prob", "tx_flag", "tx_ood", n_bootstrap)

    roc_str = f"{tx_metrics['roc_auc']:.3f}" if not np.isnan(tx_metrics.get("roc_auc", float("nan"))) else "n/a"
    print(f"  [tx eval] ROC-AUC: {roc_str}  OOD%: {tx_metrics.get('ood_share', 0)*100:.1f}%")

    # Platt recalibration
    if recalibrate_flag and len(np.unique(ys)) > 1:
        try:
            df_tx["live_raw"] = raws  # recalibrate() reads "live_raw"
            recal = recalibrate(df_tx, out_dir)
            recal_path = model_dir / "live_tx_recalibration.json"
            with open(recal_path, "w") as f:
                json.dump(recal, f, indent=2)
            print(f"  [tx eval] Tx recalibration saved to {recal_path} "
                  f"(Brier before={recal['brier_before']:.4f}, after={recal['brier_after']:.4f})")
        except Exception as e:
            print(f"  [tx eval] Recalibration failed: {e}")

    # Append "## Transactions" section to metrics.md
    md_path = out_dir / "metrics.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_auc_local(mean, lo, hi):
        if np.isnan(mean):
            return "n/a"
        return f"{mean:.3f} ± {(hi-lo)/2:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"

    tx_md_lines = [
        "",
        "## Transactions",
        "",
        "Live transaction model (experimental). Positives = txs where an OFAC-listed address "
        "is an input (spends sanctioned funds). Negatives = random non-coinbase txs from "
        "sampled blocks in the same block-height range.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ROC-AUC | {_fmt_auc_local(tx_metrics['roc_auc'], tx_metrics['roc_auc_lo'], tx_metrics['roc_auc_hi'])} |",
        f"| PR-AUC  | {_fmt_auc_local(tx_metrics['pr_auc'], tx_metrics['pr_auc_lo'], tx_metrics['pr_auc_hi'])} |",
        f"| Precision | {tx_metrics['precision']:.3f} |",
        f"| Recall    | {tx_metrics['recall']:.3f} |",
        f"| F1        | {tx_metrics['f1']:.3f} |",
        f"| Brier     | {tx_metrics['brier']:.4f} |",
        f"| OOD share | {tx_metrics['ood_share']*100:.1f}% |",
        f"| ROC-AUC (in-dist) | {tx_metrics.get('roc_auc_in_dist', float('nan')):.3f} |",
        f"| N         | {tx_metrics['n']} |",
        "",
        "_Trained on Elliptic++ (2016–2017) using only interpretable features; "
        "the original transaction model relies on anonymised features that cannot "
        "be computed for live transactions._",
    ]

    # Append or create
    with open(md_path, "a") as f:
        f.write("\n".join(tx_md_lines) + "\n")
    print(f"  [tx eval] Appended Transactions section to {md_path}")


if __name__ == "__main__":
    main()
