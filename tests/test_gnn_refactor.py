"""Test that refactored GraphSAGE scorer produces identical results."""
import pytest
import numpy as np
from backend.services.model_inference import ModelBundle, GraphSAGEScorer
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "backend" / "data" / "ellipticpp" / "model"


@pytest.fixture
def gnn_scorer():
    """Create a GraphSAGE scorer with small test data."""
    if not (MODEL_DIR / "gnn_weights.npz").exists():
        pytest.skip("GNN weights not found")

    bundle = ModelBundle(str(MODEL_DIR))

    # Create small test graph: 10 nodes, some edges
    np.random.seed(42)
    X_raw = np.random.randn(10, 51).astype(np.float32)
    src = np.array([0, 1, 2, 3, 4, 1, 2, 5], dtype=np.int32)
    dst = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)

    scorer = GraphSAGEScorer(bundle, X_raw, src, dst, max_neighbors=None, seed=42)
    return scorer


def test_score_deterministic(gnn_scorer):
    """Test that score() is deterministic across multiple calls."""
    targets = [0, 1, 5]

    result1 = gnn_scorer.score(targets)
    result2 = gnn_scorer.score(targets)
    result3 = gnn_scorer.score(targets)

    assert result1["raw"] == result2["raw"] == result3["raw"]
    assert result1["prob"] == result2["prob"] == result3["prob"]
    assert result1["nodes_used"] == result2["nodes_used"] == result3["nodes_used"]


def test_explain_deterministic(gnn_scorer):
    """Test that explain() is deterministic across multiple calls."""
    target = 1

    result1 = gnn_scorer.explain(target, top_features=5, top_neighbours=5)
    result2 = gnn_scorer.explain(target, top_features=5, top_neighbours=5)
    result3 = gnn_scorer.explain(target, top_features=5, top_neighbours=5)

    # Graph effect should be identical
    assert result1["graph_effect"] == result2["graph_effect"] == result3["graph_effect"]

    # Feature effects should be identical
    for f1, f2, f3 in zip(result1["features"], result2["features"], result3["features"]):
        assert f1["feature"] == f2["feature"] == f3["feature"]
        assert abs(f1["effect"] - f2["effect"]) < 1e-6
        assert abs(f1["effect"] - f3["effect"]) < 1e-6


def test_explain_shap_property(gnn_scorer):
    """Test that isolated + graph effect ≈ full score (in log-odds space)."""
    target = 1
    result = gnn_scorer.explain(target)

    # delta_logodds = logit(full) - logit(isolated)
    # So: logit(full) ≈ logit(isolated) + delta_logodds
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    expected = logit(result["graph_effect"]["prob_isolated"]) + result["graph_effect"]["delta_logodds"]
    actual = logit(result["graph_effect"]["prob_full"])

    assert abs(expected - actual) < 0.01, f"Expected {expected}, got {actual}"


def test_explain_feature_count(gnn_scorer):
    """Test that explain returns the requested number of features."""
    result = gnn_scorer.explain(1, top_features=3, top_neighbours=2)

    assert len(result["features"]) == 3
    assert len(result["neighbours"]) <= 2  # Could be fewer if node has <2 neighbours
    assert all("effect" in f for f in result["features"])
    assert all("effect" in n for n in result["neighbours"])
