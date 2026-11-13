"""Integration tests for chain API endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from main import app
from chain.protocol import Address, Transaction, TxInput, TxOutput
from datetime import datetime


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_chain_source():
    """Mock EsploraSource for testing"""
    with patch("main.chain_source") as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


def test_chain_status_enabled(client):
    """Test /api/chain/status when chain is enabled"""
    with patch("main.settings") as mock_settings, patch("main.chain_source") as mock_source:
        mock_settings.chain_enabled = True
        mock_settings.btc_network = "signet"

        mock_source_instance = AsyncMock()
        mock_source_instance.get_tip_height = AsyncMock(return_value=185000)
        mock_source.return_value = mock_source_instance

        # Need to mock chain_cache too
        with patch("main.chain_cache") as mock_cache:
            mock_cache.get_stats = AsyncMock(return_value={"tx_count": 100, "address_count": 50})
            mock_source.__bool__ = lambda self: True  # chain_source is truthy

            response = client.get("/api/chain/status")

    assert response.status_code == 200
    data = response.json()
    assert data["network"] == "signet"


def test_chain_subgraph_invalid_address(client):
    """Test /api/chain/subgraph with invalid address format"""
    with patch("main.settings") as mock_settings:
        mock_settings.chain_enabled = True
        mock_settings.max_k = 2

        from chain.exceptions import InvalidAddress

        mock_source_instance = AsyncMock()
        mock_source_instance.get_k_hop_subgraph = AsyncMock(side_effect=InvalidAddress("Invalid address format"))

        with patch("main.chain_source", mock_source_instance):
            response = client.get("/api/chain/subgraph/invalid_addr?k=1")

    assert response.status_code == 400
    assert "Invalid address" in response.json()["detail"]


def test_chain_subgraph_not_found(client):
    """Test /api/chain/subgraph with non-existent address"""
    with patch("main.settings") as mock_settings:
        mock_settings.chain_enabled = True
        mock_settings.max_k = 2

        from chain.exceptions import NotFound

        mock_source_instance = AsyncMock()
        mock_source_instance.get_k_hop_subgraph = AsyncMock(side_effect=NotFound("Address not found"))

        with patch("main.chain_source", mock_source_instance):
            response = client.get("/api/chain/subgraph/bc1qnonexistent?k=1")

    assert response.status_code == 404


def test_chain_address_endpoint(client):
    """Test /api/chain/address endpoint"""
    with patch("main.settings") as mock_settings:
        mock_settings.chain_enabled = True

        # Mock address data
        tx1 = Transaction(
            txid="tx1",
            inputs=[TxInput(txid="prev", vout=0, address="bc1qother", value_sat=100000, is_coinbase=False)],
            outputs=[TxOutput(vout=0, address="bc1qtest", value_sat=99000)],
            block_height=100,
            block_time=int(datetime(2024, 1, 1).timestamp()),
            confirmations=10,
            fee_sat=1000,
        )

        mock_addr = Address(
            address="bc1qtest",
            balance_sat=100000,
            tx_count=5,
            txs=[tx1],
        )

        mock_source_instance = AsyncMock()
        mock_source_instance.get_address_with_txs = AsyncMock(return_value=mock_addr)

        with patch("main.chain_source", mock_source_instance):
            response = client.get("/api/chain/address/bc1qtest")

    assert response.status_code == 200
    data = response.json()
    assert data["address"] == "bc1qtest"
    assert data["balance_sat"] == 100000
    assert data["tx_count"] == 5
    assert data["risk_score"] is None  # Chain data has no ML scores


def test_chain_tx_endpoint(client):
    """Test /api/chain/tx endpoint"""
    with patch("main.settings") as mock_settings:
        mock_settings.chain_enabled = True

        # Mock transaction data
        mock_tx = Transaction(
            txid="abc123",
            inputs=[TxInput(txid="prev", vout=0, address="bc1qsrc", value_sat=200000, is_coinbase=False)],
            outputs=[TxOutput(vout=0, address="bc1qdst", value_sat=199000)],
            block_height=100,
            block_time=int(datetime(2024, 1, 1).timestamp()),
            confirmations=10,
            fee_sat=1000,
        )

        mock_source_instance = AsyncMock()
        mock_source_instance.get_transaction = AsyncMock(return_value=mock_tx)

        with patch("main.chain_source", mock_source_instance):
            response = client.get("/api/chain/tx/abc123")

    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == "abc123"
    assert data["source"] == "bc1qsrc"
    assert data["target"] == "bc1qdst"
    assert data["block_height"] == 100
    assert data["confirmed"] is True
    assert data["risk_score"] is None
    assert data["gnn_score"] is None


def test_k_limit_enforcement(client):
    """Test that k values above max_k are rejected"""
    with patch("main.settings") as mock_settings, patch("main.chain_source") as mock_source:
        mock_settings.chain_enabled = True
        mock_settings.max_k = 2
        mock_source.__bool__ = lambda self: True

        with patch("main.require_chain_source"):
            response = client.get("/api/chain/subgraph/bc1qtest?k=5")

    assert response.status_code == 400
    assert "k must be" in response.json()["detail"]
