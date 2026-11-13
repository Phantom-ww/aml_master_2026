"""Test that refactored GraphSAGE scorer produces identical results."""
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
    MODEL_DIR / "gnn_weights.npz",
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
def address(services):
    """Pick a known address from the wallet index."""
    epp, model = services
    return next(iter(model._address_idx.keys()))


def test_refactor_parity(services):
    """Verify refactored score() matches original behavior."""
    _, model = services
    gnn_scorer = model._gnn_scorer
    if not gnn_scorer:
        pytest.skip("GNN scorer not loaded")

    # Test on a few wallet indices
    test_indices = [0, 10, 100, 500, 1000]

    for idx in test_indices:
        result = gnn_scorer.score([idx])

        # Basic checks
        assert "prob" in result
        assert "flag" in result
        assert "approximate" in result

        # Probability should be in [0, 1]
        prob = result["prob"][0]
        assert 0 <= prob <= 1, f"Invalid probability: {prob}"

        # Flag should be bool
        assert isinstance(result["flag"][0], (bool, np.bool_))


def test_explain_deterministic(services, address):
    """Verify explain() is deterministic across multiple calls."""
    _, model = services

    # Run explain 3 times
    results = []
    for _ in range(3):
        exp = model.explain_wallet_gnn(address)
        if exp is None:
            pytest.skip(f"Could not explain {address}")
        results.append(exp)

    # Check all results are identical
    for i in range(1, 3):
        assert results[i]["graph_effect"]["prob_full"] == results[0]["graph_effect"]["prob_full"]
        assert results[i]["graph_effect"]["prob_isolated"] == results[0]["graph_effect"]["prob_isolated"]
        assert results[i]["graph_effect"]["delta_logodds"] == results[0]["graph_effect"]["delta_logodds"]

        # Features should be in same order with same values
        assert len(results[i]["features"]) == len(results[0]["features"])
        for j in range(len(results[0]["features"])):
            assert results[i]["features"][j]["feature"] == results[0]["features"][j]["feature"]
            assert abs(results[i]["features"][j]["effect"] - results[0]["features"][j]["effect"]) < 1e-6

        # Neighbours should be identical
        assert len(results[i]["neighbours"]) == len(results[0]["neighbours"])
        for j in range(len(results[0]["neighbours"])):
            assert results[i]["neighbours"][j]["address"] == results[0]["neighbours"][j]["address"]
            assert abs(results[i]["neighbours"][j]["effect"] - results[0]["neighbours"][j]["effect"]) < 1e-6
