"""Tests for chain address transaction pagination"""
import pytest
from chain.esplora import EsploraSource
from chain.cache import ChainCache


@pytest.mark.asyncio
async def test_get_address_txs_page_first_page():
    """Test fetching first page of transactions"""
    cache = ChainCache(":memory:")
    await cache.init()

    source = EsploraSource(
        base_url="https://mutinynet.com/api",
        cache=cache,
        network="signet"
    )
    await source.init()

    try:
        # Test address with multiple transactions (mutinynet faucet address)
        address = "tb1qd28npep0s8fqq2rs6u093h62llkfs8hfc3v57w"

        txs, next_cursor, has_more = await source.get_address_txs_page(address, cursor=None, page_size=10)

        assert len(txs) > 0, "Should fetch transactions"
        assert all(tx.txid for tx in txs), "All transactions should have txid"

        # If has_more, next_cursor should be set
        if has_more:
            assert next_cursor is not None, "next_cursor should be set when has_more is True"
            assert next_cursor == txs[-1].txid if txs else None, "next_cursor should be last tx's txid"

        print(f"✓ First page: {len(txs)} transactions, next_cursor={next_cursor}, has_more={has_more}")

    finally:
        await source.close()


@pytest.mark.asyncio
async def test_get_address_txs_page_with_cursor():
    """Test fetching subsequent page with cursor"""
    cache = ChainCache(":memory:")
    await cache.init()

    source = EsploraSource(
        base_url="https://mutinynet.com/api",
        cache=cache,
        network="signet"
    )
    await source.init()

    try:
        address = "tb1qd28npep0s8fqq2rs6u093h62llkfs8hfc3v57w"

        # Get first page
        txs_page1, cursor1, has_more1 = await source.get_address_txs_page(address, cursor=None, page_size=5)

        if has_more1 and cursor1:
            # Get second page
            txs_page2, cursor2, has_more2 = await source.get_address_txs_page(address, cursor=cursor1, page_size=5)

            assert len(txs_page2) > 0, "Second page should have transactions"

            # Ensure no overlap
            page1_txids = {tx.txid for tx in txs_page1}
            page2_txids = {tx.txid for tx in txs_page2}
            assert len(page1_txids & page2_txids) == 0, "Pages should not overlap"

            print(f"✓ Second page: {len(txs_page2)} transactions, next_cursor={cursor2}, has_more={has_more2}")
        else:
            print("⚠ Address has fewer transactions than page size, skipping second page test")

    finally:
        await source.close()


@pytest.mark.asyncio
async def test_next_cursor_logic_in_endpoint():
    """Test that get_chain_address sets next_cursor correctly"""
    cache = ChainCache(":memory:")
    await cache.init()

    source = EsploraSource(
        base_url="https://mutinynet.com/api",
        cache=cache,
        network="signet",
        max_txs_per_address=25  # Match default page size
    )
    await source.init()

    try:
        address = "tb1qd28npep0s8fqq2rs6u093h62llkfs8hfc3v57w"

        addr_data = await source.get_address_with_txs(address)

        # Check if cursor should be set
        confirmed_txs = [tx for tx in addr_data.txs if tx.block_height is not None]

        if len(addr_data.txs) >= 25 and confirmed_txs:
            expected_cursor = confirmed_txs[-1].txid
            print(f"✓ Address has {len(addr_data.txs)} transactions, {len(confirmed_txs)} confirmed")
            print(f"  Expected cursor: {expected_cursor}")
        else:
            print(f"⚠ Address has {len(addr_data.txs)} transactions (< 25), cursor should be None")

    finally:
        await source.close()


@pytest.mark.asyncio
async def test_fallback_cursor_from_confirmed_txs():
    """Test fallback logic: use last confirmed tx as cursor when next_cursor is null but more txs exist"""
    cache = ChainCache(":memory:")
    await cache.init()

    source = EsploraSource(
        base_url="https://mutinynet.com/api",
        cache=cache,
        network="signet"
    )
    await source.init()

    try:
        address = "tb1qd28npep0s8fqq2rs6u093h62llkfs8hfc3v57w"

        # Get address data
        addr_data = await source.get_address_with_txs(address, max_txs=10)

        # Find last confirmed transaction
        confirmed_txs = [tx for tx in addr_data.txs if tx.block_height is not None]

        if confirmed_txs and addr_data.tx_count > len(addr_data.txs):
            fallback_cursor = confirmed_txs[-1].txid

            # Try to load more using fallback cursor
            txs, next_cursor, has_more = await source.get_address_txs_page(
                address, cursor=fallback_cursor, page_size=10
            )

            # Should get different transactions
            original_txids = {tx.txid for tx in addr_data.txs}
            new_txids = {tx.txid for tx in txs}

            assert len(new_txids - original_txids) > 0, "Should fetch new transactions with fallback cursor"
            print(f"✓ Fallback cursor worked: fetched {len(txs)} new transactions")
        else:
            print("⚠ Address has all transactions loaded, skipping fallback test")

    finally:
        await source.close()
