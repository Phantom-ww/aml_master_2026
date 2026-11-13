#!/usr/bin/env python3
"""Verify GNN explain endpoint works for both target addresses."""
import json
import sys
import requests

ADDRESSES = [
    "39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K",
    "39nz1urm6Vbkhzdu6FYedBys1Wzc3JFWbE",
]

BASE_URL = "http://localhost:8000"

def test_address(address):
    """Test explain endpoint for a given address."""
    url = f"{BASE_URL}/api/epp/wallet/{address}/explain/gnn"
    resp = requests.get(url, timeout=60)

    if resp.status_code != 200:
        print(f"❌ {address}: HTTP {resp.status_code}")
        print(f"   Response: {resp.text}")
        return False

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"❌ {address}: Invalid JSON - {e}")
        return False

    # Validate structure
    required_keys = ["graph_effect", "features", "neighbours", "approximate", "ms"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"❌ {address}: Missing keys: {missing}")
        return False

    # Validate graph_effect
    ge = data["graph_effect"]
    required_ge = ["prob_full", "prob_isolated", "delta_logodds"]
    missing_ge = [k for k in required_ge if k not in ge]
    if missing_ge:
        print(f"❌ {address}: Missing graph_effect keys: {missing_ge}")
        return False

    if not (0 <= ge["prob_full"] <= 1):
        print(f"❌ {address}: Invalid prob_full: {ge['prob_full']}")
        return False

    if not (0 <= ge["prob_isolated"] <= 1):
        print(f"❌ {address}: Invalid prob_isolated: {ge['prob_isolated']}")
        return False

    # Validate features
    if len(data["features"]) == 0:
        print(f"❌ {address}: No features returned")
        return False

    for feat in data["features"]:
        required_feat = ["feature", "label", "value", "effect"]
        missing_feat = [k for k in required_feat if k not in feat]
        if missing_feat:
            print(f"❌ {address}: Feature missing keys: {missing_feat}")
            return False

    # Validate neighbours
    for nbr in data["neighbours"]:
        required_nbr = ["address", "effect", "model_risk"]
        missing_nbr = [k for k in required_nbr if k not in nbr]
        if missing_nbr:
            print(f"❌ {address}: Neighbour missing keys: {missing_nbr}")
            return False

    print(f"✓ {address}: OK ({len(data['neighbours'])} neighbours)")
    return True

def main():
    """Run tests."""
    print("Testing GNN explain endpoint...")
    print()

    success_count = 0
    for addr in ADDRESSES:
        if test_address(addr):
            success_count += 1
        print()

    print(f"Results: {success_count}/{len(ADDRESSES)} passed")
    return 0 if success_count == len(ADDRESSES) else 1

if __name__ == "__main__":
    sys.exit(main())
