#!/usr/bin/env python3
"""
Step-by-step debugging script for chain API endpoint
Tests each component in isolation before testing the full flow
"""

import asyncio
import httpx
import sys
from pathlib import Path

# Test addresses for different networks
TEST_ADDRESSES = {
    "mainnet": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",  # Real mainnet address
    "testnet4": "tb1q9pvjqz5u5sdgpatg3wn0ce438u5cyv85lly0pc",  # Testnet4 address
    "signet": "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",  # Signet address
}

MEMPOOL_URLS = {
    "mainnet": "https://mempool.space/api",
    "testnet4": "https://mempool.space/testnet/api",
    "signet": "https://mempool.space/signet/api",
}


async def test_step_1_mempool_connectivity():
    """Test 1: Can we reach Mempool.space API?"""
    print("\n" + "="*60)
    print("TEST 1: Mempool.space API Connectivity")
    print("="*60)

    for network, base_url in MEMPOOL_URLS.items():
        print(f"\n[{network}] Testing {base_url}...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/blocks/tip/height")
                response.raise_for_status()
                tip_height = int(response.text)
                print(f"  ✓ Connected! Tip height: {tip_height}")
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")


async def test_step_2_address_fetch(network="mainnet"):
    """Test 2: Can we fetch a specific address?"""
    print("\n" + "="*60)
    print(f"TEST 2: Fetch Address Data ({network})")
    print("="*60)

    base_url = MEMPOOL_URLS[network]
    address = TEST_ADDRESSES[network]

    print(f"\nAddress: {address}")
    print(f"Endpoint: {base_url}/address/{address}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch address summary
            response = await client.get(f"{base_url}/address/{address}")
            response.raise_for_status()
            data = response.json()

            funded = data["chain_stats"]["funded_txo_sum"]
            spent = data["chain_stats"]["spent_txo_sum"]
            balance = funded - spent
            tx_count = data["chain_stats"]["tx_count"]

            print(f"\n  ✓ Address Summary:")
            print(f"    Balance: {balance / 100_000_000:.8f} BTC ({balance} sats)")
            print(f"    TX Count: {tx_count}")
            print(f"    Funded: {funded / 100_000_000:.8f} BTC")
            print(f"    Spent: {spent / 100_000_000:.8f} BTC")

            return True
    except httpx.HTTPStatusError as e:
        print(f"\n  ✗ HTTP Error {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        return False


async def test_step_3_address_transactions(network="mainnet", max_txs=5):
    """Test 3: Can we fetch address transactions?"""
    print("\n" + "="*60)
    print(f"TEST 3: Fetch Address Transactions ({network})")
    print("="*60)

    base_url = MEMPOOL_URLS[network]
    address = TEST_ADDRESSES[network]

    print(f"\nFetching first {max_txs} transactions for {address}...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/address/{address}/txs")
            response.raise_for_status()
            txs = response.json()

            print(f"\n  ✓ Fetched {len(txs)} transactions (page 1)")

            for i, tx in enumerate(txs[:max_txs]):
                txid = tx["txid"]
                inputs = len(tx["vin"])
                outputs = len(tx["vout"])
                confirmed = "status" in tx and tx["status"].get("confirmed", False)

                print(f"\n    TX {i+1}: {txid[:16]}...")
                print(f"      Inputs: {inputs}, Outputs: {outputs}")
                print(f"      Confirmed: {confirmed}")

                # Show input/output addresses
                input_addrs = []
                for vin in tx["vin"][:3]:  # First 3 inputs
                    if vin.get("prevout") and vin["prevout"].get("scriptpubkey_address"):
                        input_addrs.append(vin["prevout"]["scriptpubkey_address"][:20])

                output_addrs = []
                for vout in tx["vout"][:3]:  # First 3 outputs
                    if vout.get("scriptpubkey_address"):
                        output_addrs.append(vout["scriptpubkey_address"][:20])

                if input_addrs:
                    print(f"      Sample inputs: {', '.join(input_addrs)}...")
                if output_addrs:
                    print(f"      Sample outputs: {', '.join(output_addrs)}...")

            return True
    except Exception as e:
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        return False


async def test_step_4_cache_layer():
    """Test 4: Can we initialize the cache?"""
    print("\n" + "="*60)
    print("TEST 4: SQLite Cache Layer")
    print("="*60)

    try:
        from chain.cache import ChainCache

        cache_path = "./data/chain_cache_test.db"
        print(f"\nInitializing cache at {cache_path}...")

        cache = ChainCache(cache_path)
        await cache.init()

        print("  ✓ Cache initialized successfully")

        # Test cache operations
        test_key = "test_address_1"
        test_data = {
            "address": "bc1test",
            "balance_sat": 1000000,
            "tx_count": 5,
            "txs": []
        }

        print("\n  Testing cache write...")
        await cache.set_address(test_key, test_data)
        print("    ✓ Write successful")

        print("\n  Testing cache read...")
        result = await cache.get_address(test_key)
        if result and result["balance_sat"] == 1000000:
            print("    ✓ Read successful, data matches")
        else:
            print("    ✗ Read failed or data mismatch")

        await cache.close()
        print("\n  ✓ Cache closed")

        # Cleanup test db
        Path(cache_path).unlink(missing_ok=True)
        Path(cache_path + "-shm").unlink(missing_ok=True)
        Path(cache_path + "-wal").unlink(missing_ok=True)

        return True
    except Exception as e:
        import traceback
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


async def test_step_5_esplora_source(network="mainnet"):
    """Test 5: Can we initialize EsploraSource?"""
    print("\n" + "="*60)
    print(f"TEST 5: EsploraSource Initialization ({network})")
    print("="*60)

    try:
        from chain.esplora import EsploraSource
        from chain.cache import ChainCache

        cache_path = "./data/chain_cache_test.db"
        cache = ChainCache(cache_path)
        await cache.init()

        base_url = MEMPOOL_URLS[network]
        print(f"\nInitializing EsploraSource with {base_url}...")

        source = EsploraSource(
            base_url=base_url,
            cache=cache,
            network=network,
            timeout=30.0,
            max_retries=3,
            max_concurrent=3,
            max_txs_per_address=10,
            max_nodes=50
        )

        await source.init()
        print("  ✓ EsploraSource initialized")

        # Test address validation
        test_addr = TEST_ADDRESSES[network]
        print(f"\n  Testing address validation for {test_addr}...")
        try:
            source.validate_address(test_addr)
            print("    ✓ Address validation passed")
        except Exception as e:
            print(f"    ✗ Address validation failed: {e}")

        # Test fetching address with transactions
        print(f"\n  Fetching address data with transactions...")
        addr_data = await source.get_address_with_txs(test_addr, max_txs=5)
        print(f"    ✓ Got address data:")
        print(f"      Balance: {addr_data.balance_sat / 100_000_000:.8f} BTC")
        print(f"      TX Count: {addr_data.tx_count}")
        print(f"      TXs fetched: {len(addr_data.txs)}")

        await source.close()
        await cache.close()

        # Cleanup
        Path(cache_path).unlink(missing_ok=True)
        Path(cache_path + "-shm").unlink(missing_ok=True)
        Path(cache_path + "-wal").unlink(missing_ok=True)

        return True
    except Exception as e:
        import traceback
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


async def test_step_6_subgraph_fetch(network="mainnet", k=1):
    """Test 6: Can we fetch a k-hop subgraph?"""
    print("\n" + "="*60)
    print(f"TEST 6: K-Hop Subgraph Fetch ({network}, k={k})")
    print("="*60)

    try:
        from chain.esplora import EsploraSource
        from chain.cache import ChainCache

        cache_path = "./data/chain_cache_test.db"
        cache = ChainCache(cache_path)
        await cache.init()

        base_url = MEMPOOL_URLS[network]
        source = EsploraSource(
            base_url=base_url,
            cache=cache,
            network=network,
            timeout=30.0,
            max_retries=3,
            max_concurrent=3,
            max_txs_per_address=10,
            max_nodes=50
        )

        await source.init()

        test_addr = TEST_ADDRESSES[network]
        print(f"\nFetching {k}-hop subgraph for {test_addr}...")
        print("(This may take 30-60 seconds...)")

        subgraph = await source.get_k_hop_subgraph(test_addr, k=k)

        print(f"\n  ✓ Subgraph fetched successfully:")
        print(f"    Center: {subgraph.center_address}")
        print(f"    Addresses: {len(subgraph.addresses)}")
        print(f"    Transactions: {len(subgraph.transactions)}")
        print(f"    Truncated: {subgraph.truncated}")

        # Show some addresses
        print(f"\n    Sample addresses:")
        for i, (addr, addr_data) in enumerate(list(subgraph.addresses.items())[:5]):
            print(f"      {i+1}. {addr[:20]}... (balance: {addr_data.balance_sat} sats, txs: {addr_data.tx_count})")

        await source.close()
        await cache.close()

        # Cleanup
        Path(cache_path).unlink(missing_ok=True)
        Path(cache_path + "-shm").unlink(missing_ok=True)
        Path(cache_path + "-wal").unlink(missing_ok=True)

        return True
    except Exception as e:
        import traceback
        print(f"\n  ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CHAIN API DEBUG - Step-by-Step Testing")
    print("="*60)

    # Check if we're in the right directory
    if not Path("chain").exists():
        print("\n✗ Error: 'chain' module not found. Run this script from backend/ directory:")
        print("  cd backend && python3 test_chain_debug.py")
        sys.exit(1)

    # Allow user to select network
    print("\nSelect network to test:")
    print("1. mainnet (default)")
    print("2. testnet4")
    print("3. signet")

    choice = input("\nEnter choice (1-3) or press Enter for mainnet: ").strip()
    network_map = {"1": "mainnet", "2": "testnet4", "3": "signet", "": "mainnet"}
    network = network_map.get(choice, "mainnet")

    print(f"\nTesting with network: {network}")
    print(f"Test address: {TEST_ADDRESSES[network]}")

    # Run tests sequentially
    results = {}

    # Test 1: Mempool connectivity
    await test_step_1_mempool_connectivity()

    # Test 2: Address fetch
    results["address_fetch"] = await test_step_2_address_fetch(network)

    # Test 3: Transaction fetch
    results["tx_fetch"] = await test_step_3_address_transactions(network, max_txs=3)

    # Test 4: Cache layer
    results["cache"] = await test_step_4_cache_layer()

    # Test 5: EsploraSource
    results["esplora_source"] = await test_step_5_esplora_source(network)

    # Test 6: Subgraph (only if previous tests passed)
    if all(results.values()):
        print("\n✓ All component tests passed! Testing full subgraph fetch...")
        results["subgraph"] = await test_step_6_subgraph_fetch(network, k=1)
    else:
        print("\n✗ Some component tests failed. Skipping subgraph test.")
        results["subgraph"] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! The chain API should work.")
        print("\nNext steps:")
        print("  1. Start backend: cd backend && uvicorn main:app --reload")
        print(f"  2. Test endpoint: curl http://localhost:8000/api/chain/status")
        print(f"  3. Test subgraph: curl 'http://localhost:8000/api/chain/subgraph/{TEST_ADDRESSES[network]}?k=1'")
    else:
        print("\n✗ Some tests failed. Check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
