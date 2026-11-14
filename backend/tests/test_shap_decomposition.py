"""Tests for SHAP decomposition in XGBoost explanations.

Verifies that the decomposition fields are computed correctly:
- base_value + shown_sum + other_sum == logit (output_margin)
- raw_prob == sigmoid(logit)
- prob == calibrated value
- flag == (raw_prob >= threshold_raw)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DATA_DIR = BACKEND / "data" / "ellipticpp"
MODEL_DIR = DATA_DIR / "model"

REQUIRED = [
    MODEL_DIR / "model_meta.json",
    MODEL_DIR / "calibration.json",
    MODEL_DIR / "xgb_model.json",
    DATA_DIR / "wallets.parquet",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED),
    reason="Elliptic++ model/data files not present",
)


@pytest.fixture(scope="module")
def bundle():
    from services.model_inference import ModelBundle
    return ModelBundle(str(MODEL_DIR))


@pytest.fixture(scope="module")
def sample_features(bundle):
    """Sample wallet features for testing."""
    return {f: 0.1 * i for i, f in enumerate(bundle.wallet_features)}


def test_wallet_decomposition_exact_sum(bundle, sample_features):
    """Verify base_value + shown_sum + other_sum == logit."""
    result = bundle.score_wallet(sample_features, top_k=5)

    assert "decomposition" in result
    decomp = result["decomposition"]

    # Check all required fields exist
    assert "base_value" in decomp
    assert "shown_sum" in decomp
    assert "other_sum" in decomp
    assert "other_count" in decomp
    assert "logit" in decomp
    assert "raw_prob" in decomp
    assert "prob" in decomp
    assert "threshold_raw" in decomp
    assert "threshold_calibrated" in decomp

    # Verify the sum is exact to 1e-3
    computed_logit = decomp["base_value"] + decomp["shown_sum"] + decomp["other_sum"]
    assert abs(computed_logit - decomp["logit"]) < 1e-3, \
        f"Sum mismatch: {computed_logit} != {decomp['logit']}"


def test_wallet_decomposition_sigmoid(bundle, sample_features):
    """Verify raw_prob == sigmoid(logit)."""
    result = bundle.score_wallet(sample_features, top_k=5)
    decomp = result["decomposition"]

    # Compute sigmoid
    expected_raw = 1.0 / (1.0 + np.exp(-decomp["logit"]))
    assert abs(decomp["raw_prob"] - expected_raw) < 1e-6, \
        f"Sigmoid mismatch: {decomp['raw_prob']} != {expected_raw}"


def test_wallet_decomposition_calibration(bundle, sample_features):
    """Verify prob matches the existing calibrated probability."""
    result = bundle.score_wallet(sample_features, top_k=5)
    decomp = result["decomposition"]

    # The prob in decomposition should match the main prob field
    assert abs(decomp["prob"] - result["prob"]) < 1e-9, \
        f"Calibrated prob mismatch: {decomp['prob']} != {result['prob']}"


def test_wallet_decomposition_threshold(bundle, sample_features):
    """Verify flag == (raw_prob >= threshold_raw)."""
    result = bundle.score_wallet(sample_features, top_k=5)
    decomp = result["decomposition"]

    expected_flag = decomp["raw_prob"] >= decomp["threshold_raw"]
    assert result["flag"] == expected_flag, \
        f"Flag mismatch: {result['flag']} != {expected_flag}"


def test_wallet_decomposition_feature_counts(bundle, sample_features):
    """Verify shown + other count == total features."""
    result = bundle.score_wallet(sample_features, top_k=5)
    decomp = result["decomposition"]

    shown_count = len(result["explanations"])
    total_features = len(bundle.wallet_features)

    assert shown_count + decomp["other_count"] == total_features, \
        f"Feature count mismatch: {shown_count} + {decomp['other_count']} != {total_features}"


def test_tx_decomposition_exact_sum(bundle):
    """Verify transaction model decomposition sum."""
    if bundle.tx_booster is None:
        pytest.skip("Transaction model not available")

    tx_features = {f: 0.1 * i for i, f in enumerate(bundle.tx_features)}
    result = bundle.explain_tx(tx_features, top_k=8)

    assert result is not None
    assert "decomposition" in result
    decomp = result["decomposition"]

    # Verify the sum is exact
    computed_logit = decomp["base_value"] + decomp["shown_sum"] + decomp["other_sum"]
    assert abs(computed_logit - decomp["logit"]) < 1e-3, \
        f"TX sum mismatch: {computed_logit} != {decomp['logit']}"


def test_tx_decomposition_sigmoid(bundle):
    """Verify transaction raw_prob == sigmoid(logit)."""
    if bundle.tx_booster is None:
        pytest.skip("Transaction model not available")

    tx_features = {f: 0.1 * i for i, f in enumerate(bundle.tx_features)}
    result = bundle.explain_tx(tx_features, top_k=8)
    decomp = result["decomposition"]

    expected_raw = 1.0 / (1.0 + np.exp(-decomp["logit"]))
    assert abs(decomp["raw_prob"] - expected_raw) < 1e-6, \
        f"TX sigmoid mismatch: {decomp['raw_prob']} != {expected_raw}"


def test_reference_wallet_decomposition(bundle):
    """Test with reference wallet 39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K.

    Expected values from the requirements:
    - base ≈ 0.058
    - shown_sum ≈ 1.63
    - other_sum ≈ 0.08
    - other_count = 51
    - logit ≈ 1.764
    - raw_prob ≈ 0.854
    - prob ≈ 0.415
    - threshold_raw ≈ 0.845
    """
    # Load the wallet features from the service
    from services.ellipticpp_service import get_service as get_epp_service
    epp = get_epp_service()
    if not epp.is_ready:
        epp.load()

    address = "39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K"
    if address not in epp.address_index:
        pytest.skip(f"Reference wallet {address} not in dataset")

    # Get wallet features
    wallet_data = epp.get_wallet(address)
    features = {f: wallet_data.get(f, 0.0) for f in bundle.wallet_features}

    result = bundle.score_wallet(features, top_k=5)
    decomp = result["decomposition"]

    # Verify approximate values (with tolerance for small variations)
    assert abs(decomp["base_value"] - 0.058) < 0.01, \
        f"Base value mismatch: {decomp['base_value']} (expected ≈ 0.058)"
    assert abs(decomp["shown_sum"] - 1.63) < 0.1, \
        f"Shown sum mismatch: {decomp['shown_sum']} (expected ≈ 1.63)"
    assert abs(decomp["other_sum"] - 0.08) < 0.05, \
        f"Other sum mismatch: {decomp['other_sum']} (expected ≈ 0.08)"
    assert decomp["other_count"] == 51, \
        f"Other count mismatch: {decomp['other_count']} (expected 51)"
    assert abs(decomp["logit"] - 1.764) < 0.05, \
        f"Logit mismatch: {decomp['logit']} (expected ≈ 1.764)"
    assert abs(decomp["raw_prob"] - 0.854) < 0.01, \
        f"Raw prob mismatch: {decomp['raw_prob']} (expected ≈ 0.854)"
