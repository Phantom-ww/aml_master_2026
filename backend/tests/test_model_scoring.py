"""Smoke tests for on-the-fly model scoring.

Skipped automatically when the model files are absent.
Run inside the container:
    pytest tests/test_model_scoring.py -v
"""
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DATA_DIR = BACKEND / "data" / "ellipticpp"
MODEL_DIR = DATA_DIR / "model"

REQUIRED = [
    MODEL_DIR / "model_meta.json",
    MODEL_DIR / "calibration.json",
    MODEL_DIR / "xgb_model.json",
    MODEL_DIR / "predictions.parquet",
    DATA_DIR / "wallets.parquet",
    DATA_DIR / "addr_edges.parquet",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED),
    reason="Elliptic++ model/data files not present",
)


@pytest.fixture(scope="module")
def services():
    from services.ellipticpp_service import get_service as get_epp_service
    from services.model_service import get_service as get_model_service

    epp = get_epp_service()
    if not epp.is_ready:
        epp.load()
    model = get_model_service()
    if not model.is_ready:
        model.load(epp)
    assert model.is_ready, model.load_error
    return epp, model


@pytest.fixture(scope="module")
def address():
    """Pick a known address from the wallet index."""
    from services.ellipticpp_service import get_service as get_epp_service
    epp = get_epp_service()
    if not epp.is_ready:
        epp.load()
    return next(iter(epp.address_index))


def test_model_ready(services):
    epp, model = services
    assert epp.is_ready
    assert model.is_ready
    assert model._bundle is not None
    assert model._gnn_scorer is not None
    assert len(model._in_sample) > 0
    assert model._X_raw.dtype.name == "float32"


def test_xgb_scores_wallet(services, address):
    _, model = services
    wallet = dict(zip(model._bundle.wallet_features, [0.0] * len(model._bundle.wallet_features)))
    res = model.score_wallet_xgb(address, tuple(wallet.values()))
    assert res is not None
    assert 0.0 <= res["prob"] <= 1.0
    assert isinstance(res["flag"], bool)
    assert len(res["explanations"]) == 5
    for exp in res["explanations"]:
        assert exp["label"] and exp["label"] != exp["feature"].replace("_", " ").capitalize()


def test_gnn_scores_wallet(services, address):
    _, model = services
    res = model.score_wallet_gnn(address)
    assert res is not None
    assert 0.0 <= res["prob"] <= 1.0
    assert "approximate" in res


def test_in_sample_flag(services, address):
    _, model = services
    assert isinstance(model.get_in_sample(address), bool)


def test_model_info(services):
    _, model = services
    meta = model.get_model_meta()
    assert meta["model_version"]
    assert meta["metrics"]
    assert "xgb_raw" in meta["thresholds"]
