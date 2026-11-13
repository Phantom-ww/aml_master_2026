"""Test Elliptic++ service and endpoints."""
import pytest
from services.ellipticpp_service import EllipticPPService
from tests.fixtures_ellipticpp import tiny_ellipticpp_data

def test_service_load(tiny_ellipticpp_data, monkeypatch):
    """Test service loads parquet correctly."""
    import services.ellipticpp_service as svc_module
    monkeypatch.setattr(svc_module, "DATA_DIR", tiny_ellipticpp_data)

    svc = EllipticPPService()
    svc.load()

    assert svc.is_ready
    assert svc.load_error is None

def test_get_wallet(tiny_ellipticpp_data, monkeypatch):
    """Test wallet retrieval."""
    import services.ellipticpp_service as svc_module
    monkeypatch.setattr(svc_module, "DATA_DIR", tiny_ellipticpp_data)

    svc = EllipticPPService()
    svc.load()

    wallet = svc.get_wallet("addr_ill")
    assert wallet is not None
    assert wallet["class"] == 1

    assert svc.get_wallet("nonexistent") is None

def test_get_transactions(tiny_ellipticpp_data, monkeypatch):
    """Test transaction retrieval for an address."""
    import services.ellipticpp_service as svc_module
    monkeypatch.setattr(svc_module, "DATA_DIR", tiny_ellipticpp_data)

    svc = EllipticPPService()
    svc.load()

    txs, total = svc.get_transactions("addr_lic", offset=0, limit=50)
    assert total == 2  # addr_lic appears in both tx1 (in) and tx2 (out)
    assert len(txs) == 2

def test_get_tx_edges(tiny_ellipticpp_data, monkeypatch):
    """Test fetching edges for a transaction."""
    import services.ellipticpp_service as svc_module
    monkeypatch.setattr(svc_module, "DATA_DIR", tiny_ellipticpp_data)

    svc = EllipticPPService()
    svc.load()

    result = svc.get_tx_edges("tx1", focal=None, max_edges=10)
    assert len(result["nodes"]) == 2  # src and dst
    assert len(result["edges"]) == 1
    assert result["edges"][0]["id"] == "tx1:addr_ill:addr_lic"

def test_get_examples(tiny_ellipticpp_data, monkeypatch):
    """Test examples endpoint returns addresses per class."""
    import services.ellipticpp_service as svc_module
    monkeypatch.setattr(svc_module, "DATA_DIR", tiny_ellipticpp_data)

    svc = EllipticPPService()
    svc.load()

    examples = svc.get_examples()
    assert "addr_ill" in examples["illicit"]
    assert "addr_lic" in examples["licit"]
    assert "addr_unk" in examples["unknown"]
