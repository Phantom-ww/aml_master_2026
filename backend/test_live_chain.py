#!/usr/bin/env python3
"""Test live chain subgraph fetching step by step"""

import asyncio
import sys
from chain.esplora import EsploraSource
from chain.cache import ChainCache

async def main():
    # Use a well-known Bitcoin address with activity
    test_addresses = [
        "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",  # Known mainnet address
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block coinbase
    ]

    cache = ChainCache('./data/chain_cache.db')
    await cache.init()

    source = EsploraSource(
        base_url='https://mempool.space/api',
        cache=cache,
        network='mainnet',
        max_nodes=50,
        max_txs_per_address=10,
        timeout=30.0
    )
    await source.init()

    for address in test_addresses:
        print(f'\n{"="*60}')
        print(f'Testing address: {address}')
        print(f'{"="*60}')

        try:
            # Step 1: Test address validation
            print(f'\n1. Validating address...')
            is_valid = source.validate_address(address)
            print(f'   ✓ Valid: {is_valid}')

            if not is_valid:
                print(f'   ✗ Skipping invalid address')
                continue

            # Step 2: Fetch address data
            print(f'\n2. Fetching address data...')
            addr_data = await source.get_address_with_txs(address, limit=5)
            print(f'   ✓ Balance: {addr_data.balance_sat} sat')
            print(f'   ✓ TX count: {addr_data.tx_count}')
            print(f'   ✓ Fetched {len(addr_data.txids)} transaction IDs')

            if len(addr_data.txids) == 0:
                print(f'   ⚠ No transactions found, trying next address...')
                continue

            # Step 3: Fetch subgraph
            print(f'\n3. Fetching 1-hop subgraph...')
            subgraph = await source.get_k_hop_subgraph(address, k=1)
            print(f'   ✓ Addresses: {len(subgraph.addresses)}')
            print(f'   ✓ Transactions: {len(subgraph.transactions)}')
            print(f'   ✓ Truncated: {subgraph.truncated}')

            # Step 4: Show sample data
            print(f'\n4. Sample addresses:')
            for addr, data in list(subgraph.addresses.items())[:3]:
                print(f'   - {addr[:20]}... (balance: {data.balance_sat} sat, txs: {data.tx_count})')

            print(f'\n5. Sample transactions:')
            for txid, tx in list(subgraph.transactions.items())[:3]:
                print(f'   - {txid[:16]}... (inputs: {len(tx.inputs)}, outputs: {len(tx.outputs)})')

            print(f'\n✅ SUCCESS! This address works for testing.')
            break

        except Exception as e:
            print(f'\n✗ Error: {e}')
            import traceback
            traceback.print_exc()
            continue

    await source.close()
    await cache.close()

if __name__ == '__main__':
    asyncio.run(main())
