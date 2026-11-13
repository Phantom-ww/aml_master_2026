"""Test GNN explain endpoint for robustness and edge cases."""
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
    """Initialize services once for all tests."""
    from services.ellipticpp_service import get_service as get_epp_service
    from services.model_service import get_service as get_model_service

    epp = get_epp_service()
    if not epp.is_ready:
        epp.load()
    model = get_model_service()
    if not model.is_ready:
        model.load(epp)
    assert model.is_ready, model.load_error
    assert model._gnn_scorer is not None, "GNN scorer not loaded"
    return epp, model


@pytest.fixture(scope="module")
def real_addresses(services):
    """Get real addresses from the dataset for testing."""
    epp, model = services

    # Find addresses with different neighbor counts
    isolated = None
    low_degree = None
    high_degree = None

    for addr, idx in list(model._address_idx.items())[:10000]:
        nb_count = len(model._gnn_scorer._nbrs(idx)[0])

        if nb_count == 0 and isolated is None:
            isolated = addr
        elif 1 <= nb_count <= 10 and low_degree is None:
            low_degree = addr
        elif nb_count > 1000 and high_degree is None:
            high_degree = addr

        if isolated and low_degree and high_degree:
            break

    return {
        "isolated": isolated,
        "low_degree": low_degree,
        "high_degree": high_degree,
        "target_a": "39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K",
        "target_b": "39nz1urm6Vbkhzdu6FYedBys1Wzc3JFWbE",
    }


def test_explain_target_addresses(services, real_addresses):
    """Test explain() on the original failing addresses."""
    _, model = services

    for key in ["target_a", "target_b"]:
        address = real_addresses[key]
        if address is None:
            pytest.skip(f"Address {key} not found")

        result = model.explain_wallet_gnn(address)
        assert result is not None, f"explain_wallet_gnn returned None for {address}"

        # Validate structure
        assert "graph_effect" in result
        assert "features" in result
        assert "neighbours" in result
        assert "approximate" in result
        assert "ms" in result

        # Validate graph_effect
        ge = result["graph_effect"]
        assert "prob_full" in ge
        assert "prob_isolated" in ge
        assert "delta_logodds" in ge
        assert 0 <= ge["prob_full"] <= 1
        assert 0 <= ge["prob_isolated"] <= 1

        # Validate features
        assert len(result["features"]) > 0
        for feat in result["features"]:
            assert "feature" in feat
            assert "label" in feat
            assert "value" in feat
            assert "effect" in feat
            assert isinstance(feat["effect"], (int, float))

        # Validate neighbours
        for nbr in result["neighbours"]:
            assert "address" in nbr
            assert "effect" in nbr
            assert "model_risk" in nbr
            assert isinstance(nbr["address"], str)
            assert 0 <= nbr["model_risk"] <= 1


def test_explain_isolated_wallet(services, real_addresses):
    """Test explain() on a wallet with 0 neighbours."""
    _, model = services

    address = real_addresses["isolated"]
    if address is None:
        pytest.skip("No isolated wallet found in test sample")

    result = model.explain_wallet_gnn(address)
    assert result is not None, "explain_wallet_gnn returned None for isolated wallet"

    # Isolated wallet should have empty neighbour list
    assert len(result["neighbours"]) == 0

    # graph_effect should show zero delta (no graph structure to contribute)
    ge = result["graph_effect"]
    assert ge["delta_logodds"] == 0.0 or abs(ge["delta_logodds"]) < 0.01


def test_explain_hub_adjacent(services, real_addresses):
    """Test explain() on a wallet next to a hub (many neighbours)."""
    _, model = services

    address = real_addresses["high_degree"]
    if address is None:
        pytest.skip("No high-degree wallet found in test sample")

    result = model.explain_wallet_gnn(address)
    assert result is not None, "explain_wallet_gnn returned None for high-degree wallet"

    # Should mark as approximate
    assert result["approximate"] is True

    # Should still return valid results
    assert len(result["features"]) > 0
    assert len(result["neighbours"]) >= 0


def test_explain_deterministic(services, real_addresses):
    """Verify explain() produces identical results across 2 calls."""
    _, model = services

    address = real_addresses["target_a"]
    if address is None:
        pytest.skip("Target address not found")

    result1 = model.explain_wallet_gnn(address)
    result2 = model.explain_wallet_gnn(address)

    assert result1 is not None
    assert result2 is not None

    # graph_effect should be identical
    assert result1["graph_effect"]["prob_full"] == result2["graph_effect"]["prob_full"]
    assert result1["graph_effect"]["prob_isolated"] == result2["graph_effect"]["prob_isolated"]
    assert result1["graph_effect"]["delta_logodds"] == result2["graph_effect"]["delta_logodds"]

    # Features should be identical
    assert len(result1["features"]) == len(result2["features"])
    for f1, f2 in zip(result1["features"], result2["features"]):
        assert f1["feature"] == f2["feature"]
        assert f1["effect"] == f2["effect"]

    # Neighbours should be identical
    assert len(result1["neighbours"]) == len(result2["neighbours"])
    for n1, n2 in zip(result1["neighbours"], result2["neighbours"]):
        assert n1["address"] == n2["address"]
        assert n1["effect"] == n2["effect"]


def test_explain_json_serializable(services, real_addresses):
    """Verify explain() returns JSON-serializable data (no numpy types)."""
    import json
    _, model = services

    address = real_addresses["target_a"]
    if address is None:
        pytest.skip("Target address not found")

    result = model.explain_wallet_gnn(address)
    assert result is not None

    # Should serialize without error
    try:
        json_str = json.dumps(result)
        roundtrip = json.loads(json_str)
        assert isinstance(roundtrip, dict)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Result not JSON-serializable: {e}")


def test_explain_all_addresses_work(services, real_addresses):
    """Smoke test: all test addresses should work without errors."""
    _, model = services

    for key, address in real_addresses.items():
        if address is None:
            continue

        result = model.explain_wallet_gnn(address)
        assert result is not None, f"explain_wallet_gnn failed for {key}: {address}"
        assert "graph_effect" in result
        assert "features" in result
        assert "neighbours" in result
