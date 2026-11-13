"""Tests for chain/esplora.py with HTTP mocking"""

import pytest
import respx
from httpx import Response
from datetime import datetime
from chain.esplora import EsploraSource
from chain.cache import ChainCache
from chain.exceptions import InvalidAddress, NotFound, ChainUnavailable
import tempfile
from pathlib import Path


@pytest.fixture
async def mock_cache():
    """Create temporary cache for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = ChainCache(str(cache_path), short_ttl_seconds=600, address_ttl_seconds=300)
        await cache.init()
        yield cache
        await cache.close()


@pytest.fixture
def esplora(mock_cache):
    """Create EsploraSource with mock cache"""
    return EsploraSource(
        base_url="https://mempool.space/api",
        cache=mock_cache,
        network="mainnet",
        max_txs_per_address=50,
        max_nodes=300,
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_get_tip_height():
    """Test fetching current blockchain tip height"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = ChainCache(str(cache_path))
        await cache.init()

        esplora = EsploraSource(
            base_url="https://mempool.space/api",
            cache=cache,
            network="mainnet",
            max_txs_per_address=50,
            max_nodes=300,
            timeout=10.0,
        )

        with respx.mock:
            respx.get("https://mempool.space/api/blocks/tip/height").mock(
                return_value=Response(200, text="850000")
            )

            await esplora.init()
            tip = await esplora._get_tip_height()
            await esplora.close()

        await cache.close()
        assert tip == 850000


@pytest.mark.asyncio
async def test_get_transaction():
    """Test fetching transaction data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = ChainCache(str(cache_path))
        await cache.init()

        esplora = EsploraSource(
            base_url="https://mempool.space/api",
            cache=cache,
            network="mainnet",
            timeout=10.0,
        )

        with respx.mock:
            respx.get("https://mempool.space/api/blocks/tip/height").mock(
                return_value=Response(200, text="850000")
            )
            respx.get("https://mempool.space/api/tx/abc123").mock(
                return_value=Response(
                    200,
                    json={
                        "txid": "abc123",
                        "vin": [
                            {
                                "txid": "prev1",
                                "vout": 0,
                                "prevout": {
                                    "scriptpubkey_address": "bc1qsrc",
                                    "value": 100000,
                                    "scriptpubkey_type": "v0_p2wpkh",
                                },
                            }
                        ],
                        "vout": [
                            {
                                "n": 0,
                                "scriptpubkey_address": "bc1qdst",
                                "value": 99000,
                                "scriptpubkey_type": "v0_p2wpkh",
                            }
                        ],
                        "status": {"confirmed": True, "block_height": 850000, "block_time": 1700000000},
                        "fee": 1000,
                    },
                )
            )

            await esplora.init()
            tx = await esplora.get_transaction("abc123")
            await esplora.close()

        await cache.close()

        assert tx.txid == "abc123"
        assert len(tx.inputs) == 1
        assert tx.inputs[0].address == "bc1qsrc"
        assert len(tx.outputs) == 1
        assert tx.outputs[0].address == "bc1qdst"
        assert tx.fee_sat == 1000


@pytest.mark.asyncio
async def test_coinbase_transaction():
    """Test parsing coinbase transaction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = ChainCache(str(cache_path))
        await cache.init()

        esplora = EsploraSource(
            base_url="https://mempool.space/api",
            cache=cache,
            network="mainnet",
            timeout=10.0,
        )

        with respx.mock:
            respx.get("https://mempool.space/api/blocks/tip/height").mock(
                return_value=Response(200, text="850000")
            )
            respx.get("https://mempool.space/api/tx/coinbase123").mock(
                return_value=Response(
                    200,
                    json={
                        "txid": "coinbase123",
                        "vin": [
                            {
                                "txid": "0000000000000000000000000000000000000000000000000000000000000000",
                                "vout": 4294967295,
                                "is_coinbase": True,
                                "scriptsig": "03abc123",
                            }
                        ],
                        "vout": [
                            {
                                "n": 0,
                                "scriptpubkey_address": "bc1qminer",
                                "value": 625000000,
                                "scriptpubkey_type": "v0_p2wpkh",
                            }
                        ],
                        "status": {"confirmed": True, "block_height": 850000, "block_time": 1700000000},
                        "fee": 0,
                    },
                )
            )

            await esplora.init()
            tx = await esplora.get_transaction("coinbase123")
            await esplora.close()

        await cache.close()

        assert tx.inputs[0].is_coinbase == True
        assert tx.inputs[0].address is None
        assert tx.fee_sat == 0


@pytest.mark.asyncio
async def test_op_return_output():
    """Test parsing OP_RETURN output"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = ChainCache(str(cache_path))
        await cache.init()

        esplora = EsploraSource(
            base_url="https://mempool.space/api",
            cache=cache,
            network="mainnet",
            timeout=10.0,
        )

        with respx.mock:
            respx.get("https://mempool.space/api/blocks/tip/height").mock(
                return_value=Response(200, text="850000")
            )
            respx.get("https://mempool.space/api/tx/opreturn123").mock(
                return_value=Response(
                    200,
                    json={
                        "txid": "opreturn123",
                        "vin": [
                            {
                                "txid": "prev",
                                "vout": 0,
                                "prevout": {
                                    "scriptpubkey_address": "bc1qsrc",
                                    "value": 100000,
                                    "scriptpubkey_type": "v0_p2wpkh",
                                },
                            }
                        ],
                        "vout": [
                            {
                                "n": 0,
                                "scriptpubkey_address": "bc1qdst",
                                "value": 95000,
                                "scriptpubkey_type": "v0_p2wpkh",
                            },
                            {
                                "n": 1,
                                "scriptpubkey": "6a0c48656c6c6f20776f726c64",
                                "scriptpubkey_type": "op_return",
                                "value": 0,
                            },
                        ],
                        "status": {"confirmed": True, "block_height": 850000, "block_time": 1700000000},
                        "fee": 5000,
                    },
                )
            )

            await esplora.init()
            tx = await esplora.get_transaction("opreturn123")
            await esplora.close()

        await cache.close()

        assert len(tx.outputs) == 1  # OP_RETURN not included in outputs
        assert tx.has_op_return == True
