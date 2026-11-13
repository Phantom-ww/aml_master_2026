"""Quick test script to verify Esplora connectivity"""
import asyncio
import httpx

async def test_esplora():
    """Test basic Esplora API connectivity"""
    base_url = "https://mempool.space/api"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Get tip height
        print("Testing tip height...")
        try:
            resp = await client.get(f"{base_url}/blocks/tip/height")
            print(f"✓ Tip height: {resp.text}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return
        
        # Test 2: Get a well-known address with few transactions (Satoshi's genesis coinbase)
        test_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        print(f"\nTesting address lookup for {test_addr}...")
        try:
            resp = await client.get(f"{base_url}/address/{test_addr}")
            data = resp.json()
            print(f"✓ Balance: {(data['chain_stats']['funded_txo_sum'] - data['chain_stats']['spent_txo_sum']) / 1e8} BTC")
            print(f"✓ TX count: {data['chain_stats']['tx_count']}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return
        
        # Test 3: Get transaction history (paginated)
        print(f"\nTesting transaction history...")
        try:
            resp = await client.get(f"{base_url}/address/{test_addr}/txs")
            txs = resp.json()
            print(f"✓ Got {len(txs)} transactions")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return
        
        print("\n✓ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_esplora())
