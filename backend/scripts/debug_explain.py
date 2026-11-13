#!/usr/bin/env python3
"""Debug script to reproduce GNN explain failures outside the web server."""
import sys
import traceback
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ellipticpp_service import get_service as get_epp_service
from services.model_service import get_service as get_model_service

def main():
    # Load services exactly as main.py does
    epp_svc = get_epp_service()
    epp_svc.load()
    if not epp_svc.is_ready:
        print(f"❌ EllipticPP not ready: {epp_svc.load_error}")
        return 1
    print(f"✓ EllipticPP ready")

    model_svc = get_model_service()
    model_svc.load(epp_svc)
    if not model_svc.is_ready:
        print(f"❌ Model service not ready: {model_svc.load_error}")
        return 1
    print(f"✓ Model service ready")

    if not model_svc._gnn_scorer:
        print("❌ GNN scorer not available")
        return 1
    print(f"✓ GNN scorer ready")

    # Test addresses
    test_addresses = [
        "39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K",
        "39nz1urm6Vbkhzdu6FYedBys1Wzc3JFWbE",
    ]

    for address in test_addresses:
        print(f"\n{'='*60}")
        print(f"Testing: {address}")
        print(f"{'='*60}")

        try:
            result = model_svc.explain_wallet_gnn(address)
            if result is None:
                print(f"❌ explain_wallet_gnn returned None")
            else:
                print(f"✓ Success!")
                print(f"  graph_effect: {result['graph_effect']}")
                print(f"  features: {len(result['features'])} features")
                print(f"  neighbours: {len(result['neighbours'])} neighbours")
                print(f"  approximate: {result['approximate']}")
                print(f"  ms: {result['ms']}")
        except Exception as e:
            print(f"❌ Exception: {e}")
            traceback.print_exc()

    return 0

if __name__ == "__main__":
    sys.exit(main())
