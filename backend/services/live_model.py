"""Live Chain wallet risk scoring (experimental).

Loads the XGBoost model trained on scale-free Elliptic++ features that are
computable from any address's full on-chain history.  The model and its metadata
live in the directory pointed to by ``settings.live_model_dir``.

Design goals:
  * Load once at startup; never raise on missing files.
  * score() returns raw score, calibrated probability, flag, top-5 TreeSHAP
    explanations, and the same decomposition block used by the Elliptic++ endpoint.
  * ood() implements the OOD rule stored in live_model_meta.json.
  * No new heavy dependencies: only xgboost + numpy (already required by the
    existing model service).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_service: Optional["LiveModelService"] = None


def get_service() -> "LiveModelService":
    global _service
    if _service is None:
        _service = LiveModelService()
    return _service


def _calibrate(raw: float, calibration: dict) -> float:
    """IsotonicRegression(out_of_bounds='clip').predict — linear interp with clip."""
    return float(np.interp(raw, calibration["x"], calibration["y"]))


class LiveModelService:
    """Thin wrapper around the live-chain XGBoost model."""

    def __init__(self) -> None:
        self.is_ready = False
        self.load_error: Optional[str] = None
        self._booster = None
        self._meta: dict = {}
        self._recal: Optional[dict] = None  # Platt recalibration params

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self, model_dir: str) -> None:
        """Load model and metadata.  Logs errors instead of raising."""
        try:
            import xgboost as xgb  # already in requirements

            meta_path = os.path.join(model_dir, "live_model_meta.json")
            model_path = os.path.join(model_dir, "xgb_live_model.json")

            if not os.path.exists(meta_path) or not os.path.exists(model_path):
                self.load_error = (
                    f"Live model files not found in {model_dir} "
                    f"(expected live_model_meta.json and xgb_live_model.json)"
                )
                logger.warning(self.load_error)
                return

            with open(meta_path, encoding="utf-8") as f:
                self._meta = json.load(f)

            self._booster = xgb.Booster()
            self._booster.load_model(model_path)
            self.is_ready = True
            logger.info(
                "Live model loaded: version=%s, features=%d",
                self._meta.get("model_version"),
                len(self._meta.get("features", [])),
            )

            # Optional Platt recalibration (applied only when LIVE_USE_RECALIBRATION=true)
            recal_path = os.path.join(model_dir, "live_recalibration.json")
            if os.path.exists(recal_path):
                with open(recal_path, encoding="utf-8") as f:
                    self._recal = json.load(f)
                logger.info("Live model Platt recalibration loaded")

        except Exception as exc:
            self.load_error = str(exc)
            logger.error("Failed to load live model: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, features: dict, use_recalibration: bool = False) -> dict:
        """Score one wallet.

        Parameters
        ----------
        features : dict
            Output of ``chain.live_features.live_features_from_txs`` (the first
            element of the tuple): a dict keyed by LIVE_FEATURES.
        use_recalibration : bool
            Apply Platt scaling if available (controlled by LIVE_USE_RECALIBRATION).

        Returns
        -------
        dict with keys: raw, prob, flag, explanations, decomposition.
        """
        import xgboost as xgb

        feat_names: list[str] = self._meta["features"]
        best_iter: int = self._meta["best_iteration"]
        threshold_raw: float = self._meta["threshold_raw"]
        feature_labels: dict = self._meta["feature_labels"]
        calibration: dict = self._meta["calibration"]

        X = np.asarray(
            [[float(features.get(f, 0.0) or 0.0) for f in feat_names]],
            dtype=np.float32,
        )
        dm = xgb.DMatrix(X, feature_names=feat_names)
        iter_range = (0, best_iter + 1)

        raw = float(self._booster.predict(dm, iteration_range=iter_range)[0])

        if use_recalibration and self._recal:
            # Platt: sigmoid(coef * logit(raw) + intercept)
            # raw is already in [0,1], so logit it first
            logit_raw = float(np.log(raw / (1 - raw + 1e-9) + 1e-9))
            coef = self._recal["coef"]
            intercept = self._recal["intercept"]
            prob = float(1.0 / (1.0 + np.exp(-(coef * logit_raw + intercept))))
        else:
            prob = _calibrate(raw, calibration)

        # TreeSHAP — top-5
        contrib_full = self._booster.predict(
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

    def ood(self, features: dict, base_stats: dict) -> dict:
        """Determine whether the wallet is out of the training distribution.

        Uses the ``ood_rule`` from live_model_meta.json:
          - a feature is out of range if it lies outside
            [reference[feat].quantiles[low_q], reference[feat].quantiles[high_q]]
          - flag = any important_feature is OOR
                 OR >= min_out_of_range features are OOR
                 OR base_stats.total_txs > reference.total_txs.quantiles["99.5"]

        Returns
        -------
        dict with keys:
          flag       : bool
          out_of_range : list[{feature, label, value, low, high}]
        """
        rule: dict = self._meta["ood_rule"]
        reference: dict = self._meta["reference"]
        feature_labels: dict = self._meta["feature_labels"]
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

        # Context rule: total_txs > reference.total_txs.quantiles["99.5"]
        ctx_total_txs_q = rule.get("context", {}).get("total_txs", "99.5")
        total_txs_ref_q = (
            reference.get("total_txs", {})
            .get("quantiles", {})
            .get(ctx_total_txs_q, float("inf"))
        )
        total_txs_flag = float(base_stats.get("total_txs", 0)) > total_txs_ref_q

        return {
            "flag": bool(important_oor or min_oor_flag or total_txs_flag),
            "out_of_range": out_of_range,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._meta.get("model_version", "unknown")
