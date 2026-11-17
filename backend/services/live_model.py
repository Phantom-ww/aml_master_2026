"""Live Chain risk scoring (experimental) — wallet and transaction bundles.

Loads two independent XGBoost bundles from ``settings.live_model_dir``:
  * "wallet"  — xgb_live_model.json + live_model_meta.json
  * "tx"      — xgb_live_tx_model.json + live_tx_model_meta.json

Each bundle is optional: a missing file logs a warning and disables only that
bundle without blocking startup or the other bundle.

Design goals:
  * Load once at startup; never raise on missing files.
  * score() returns raw score, calibrated probability, flag, top-5 TreeSHAP
    explanations, and the same decomposition block used by the Elliptic++ endpoint.
  * ood() implements the OOD rule stored in the meta JSON; skips the context
    check when meta has no "context" key (tx bundle).
  * No new heavy dependencies: only xgboost + numpy (already required).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundle file manifests
# ---------------------------------------------------------------------------

BUNDLE_FILES: dict[str, tuple[str, str]] = {
    "wallet": ("xgb_live_model.json", "live_model_meta.json"),
    "tx":     ("xgb_live_tx_model.json", "live_tx_model_meta.json"),
}
RECAL_FILES: dict[str, str] = {
    "wallet": "live_recalibration.json",
    "tx":     "live_tx_recalibration.json",
}

_service: Optional["LiveModelService"] = None


def get_service() -> "LiveModelService":
    global _service
    if _service is None:
        _service = LiveModelService()
    return _service


def _calibrate(raw: float, calibration: dict) -> float:
    """IsotonicRegression(out_of_bounds='clip').predict — linear interp with clip."""
    return float(np.interp(raw, calibration["x"], calibration["y"]))


# ---------------------------------------------------------------------------
# Bundle state
# ---------------------------------------------------------------------------

@dataclass
class _Bundle:
    is_ready: bool = False
    load_error: Optional[str] = None
    booster: object = None          # xgb.Booster, typed as object to avoid import at class level
    meta: dict = field(default_factory=dict)
    recal: Optional[dict] = None    # Platt recalibration {coef, intercept}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LiveModelService:
    """Thin wrapper around the live-chain XGBoost models (wallet and tx)."""

    def __init__(self) -> None:
        self._bundles: dict[str, _Bundle] = {
            "wallet": _Bundle(),
            "tx":     _Bundle(),
        }

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self, model_dir: str) -> None:
        """Load both bundles from *model_dir*.  Missing files warn; never raise."""
        import xgboost as xgb

        for bundle_name, (model_file, meta_file) in BUNDLE_FILES.items():
            b = self._bundles[bundle_name]
            meta_path  = os.path.join(model_dir, meta_file)
            model_path = os.path.join(model_dir, model_file)

            if not os.path.exists(meta_path) or not os.path.exists(model_path):
                b.load_error = (
                    f"Live model files not found in {model_dir} "
                    f"(expected {meta_file} and {model_file})"
                )
                logger.warning(b.load_error)
                continue

            try:
                with open(meta_path, encoding="utf-8") as f:
                    b.meta = json.load(f)

                booster = xgb.Booster()
                booster.load_model(model_path)
                b.booster = booster
                b.is_ready = True
                logger.info(
                    "Live %s model loaded: version=%s, features=%d",
                    bundle_name,
                    b.meta.get("model_version"),
                    len(b.meta.get("features", [])),
                )

                # Optional Platt recalibration
                recal_path = os.path.join(model_dir, RECAL_FILES[bundle_name])
                if os.path.exists(recal_path):
                    with open(recal_path, encoding="utf-8") as f:
                        b.recal = json.load(f)
                    logger.info("Live %s model Platt recalibration loaded", bundle_name)

            except Exception as exc:
                b.load_error = str(exc)
                b.is_ready = False
                logger.error(
                    "Failed to load live %s model: %s", bundle_name, exc, exc_info=True
                )

    # ------------------------------------------------------------------
    # Bundle access helpers
    # ------------------------------------------------------------------

    def is_bundle_ready(self, bundle: str = "wallet") -> bool:
        return self._bundles[bundle].is_ready

    def get_model_version(self, bundle: str = "wallet") -> str:
        return self._bundles[bundle].meta.get("model_version", "unknown")

    # Keep old property working (wallet bundle)
    @property
    def is_ready(self) -> bool:
        return self._bundles["wallet"].is_ready

    @property
    def load_error(self) -> Optional[str]:
        return self._bundles["wallet"].load_error

    @property
    def model_version(self) -> str:
        return self.get_model_version("wallet")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        features: dict,
        bundle: str = "wallet",
        use_recalibration: bool = False,
    ) -> dict:
        """Score one sample.

        Parameters
        ----------
        features : dict
            Output of ``live_features_from_txs`` (wallet) or
            ``tx_live_features_from_esplora`` (tx) — first element of the tuple.
        bundle : str
            ``"wallet"`` or ``"tx"``.
        use_recalibration : bool
            Apply Platt scaling if available.

        Returns
        -------
        dict with keys: raw, prob, flag, explanations (top-5), decomposition.
        """
        import xgboost as xgb

        b = self._bundles[bundle]
        feat_names: list[str] = b.meta["features"]
        best_iter: int = b.meta["best_iteration"]
        threshold_raw: float = b.meta["threshold_raw"]
        feature_labels: dict = b.meta["feature_labels"]
        calibration: dict = b.meta["calibration"]

        X = np.asarray(
            [[float(features.get(f, 0.0) or 0.0) for f in feat_names]],
            dtype=np.float32,
        )
        dm = xgb.DMatrix(X, feature_names=feat_names)
        iter_range = (0, best_iter + 1)

        raw = float(b.booster.predict(dm, iteration_range=iter_range)[0])

        if use_recalibration and b.recal:
            logit_raw = float(np.log(raw / (1 - raw + 1e-9) + 1e-9))
            coef = b.recal["coef"]
            intercept = b.recal["intercept"]
            prob = float(1.0 / (1.0 + np.exp(-(coef * logit_raw + intercept))))
        else:
            prob = _calibrate(raw, calibration)

        # TreeSHAP — top-5
        contrib_full = b.booster.predict(
            dm, pred_contribs=True, iteration_range=iter_range
        )[0]
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
                "value": float(X[0, j]),
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

        return {
            "raw": raw,
            "prob": prob,
            "flag": bool(raw >= threshold_raw),
            "explanations": explanations,
            "decomposition": decomposition,
        }

    # ------------------------------------------------------------------
    # Out-of-distribution check
    # ------------------------------------------------------------------

    def ood(
        self,
        features: dict,
        base_stats: Optional[dict] = None,
        bundle: str = "wallet",
    ) -> dict:
        """Determine whether the sample is out of training distribution.

        Uses ``ood_rule`` from the bundle's meta JSON:
          - out of range: value outside [reference[f].quantiles[low_q], …high_q]
          - flag = any important_feature OOR
                 OR >= min_out_of_range features OOR
                 OR base_stats.total_txs > reference.total_txs.quantiles[ctx_q]
                    (skipped when meta has no "context" key — tx bundle)

        Returns
        -------
        dict: {flag: bool, out_of_range: list[{feature, label, value, low, high}]}
        """
        b = self._bundles[bundle]
        rule: dict = b.meta["ood_rule"]
        reference: dict = b.meta["reference"]
        feature_labels: dict = b.meta["feature_labels"]
        low_q: str = rule["low_q"]
        high_q: str = rule["high_q"]

        out_of_range: list[dict] = []

        for feat, val_raw in features.items():
            ref_feat = reference.get(feat)
            if ref_feat is None:
                continue
            q = ref_feat.get("quantiles", {})
            lo = q.get(low_q)
            hi = q.get(high_q)
            if lo is None or hi is None:
                continue
            val = float(val_raw or 0.0)
            if val < lo or val > hi:
                out_of_range.append(
                    {
                        "feature": feat,
                        "label": feature_labels.get(feat, feat),
                        "value": val,
                        "low": lo,
                        "high": hi,
                    }
                )

        oor_names = {r["feature"] for r in out_of_range}
        important_oor = any(f in oor_names for f in rule["important_features"])
        min_oor_flag = len(out_of_range) >= rule["min_out_of_range"]

        # Context rule (wallet bundle only; tx bundle has no "context" key)
        total_txs_flag = False
        ctx = rule.get("context")
        if ctx and base_stats is not None:
            ctx_q = ctx.get("total_txs", "99.5")
            total_txs_ref_q = (
                reference.get("total_txs", {})
                .get("quantiles", {})
                .get(ctx_q, float("inf"))
            )
            total_txs_flag = float(base_stats.get("total_txs", 0)) > total_txs_ref_q

        return {
            "flag": bool(important_oor or min_oor_flag or total_txs_flag),
            "out_of_range": out_of_range,
        }
