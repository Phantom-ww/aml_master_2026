"""Fixtures for Elliptic++ tests."""
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture
def tiny_ellipticpp_data(tmp_path):
    """Create minimal parquet files for testing."""
    # 3 wallets
    wallets = pd.DataFrame({
        "address": ["addr_ill", "addr_lic", "addr_unk"],
        "class": [1, 2, 3],
        "time_step": [10, 15, 20],
        "first_ts": [10, 15, 20],
        "last_ts": [10, 15, 20],
        **{f"Feature_{i}": [0.1, 0.2, 0.3] for i in range(1, 57)}
    })

    # 2 transactions: addr_ill -> addr_lic, addr_lic -> addr_unk
    edges = pd.DataFrame({
        "txId": ["tx1", "tx2"],
        "src": ["addr_ill", "addr_lic"],
        "dst": ["addr_lic", "addr_unk"],
        "time_step": [10, 15],
        "tx_class": [1, 2],
        "total_BTC": [0.5, 1.0],
        "fees": [0.001, 0.002]
    })

    wallets.to_parquet(tmp_path / "wallets.parquet")
    edges.to_parquet(tmp_path / "tx_edges.parquet")

    return tmp_path
