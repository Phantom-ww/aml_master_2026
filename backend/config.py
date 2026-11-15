from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./test.db"

    # Data directory
    data_dir: str = "./data"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Blockchain integration
    chain_enabled: bool = True
    esplora_url: str = "https://blockstream.info/api"
    esplora_fallback_url: str = "https://mempool.space/api"
    btc_network: Literal["signet", "testnet4", "mainnet"] = "mainnet"
    chain_cache_path: str = "./data/chain_cache.db"

    # Chain query limits
    max_k_hops: int = 2
    max_txs_per_address: int = 50
    max_nodes: int = 300
    max_fanout: int = 200

    # HTTP client
    http_timeout: float = 30.0
    max_retries: int = 3
    max_concurrent_requests: int = 3  # Conservative for public APIs

    # Cache TTLs (seconds)
    cache_short_ttl: int = 600  # 10 min for 1-5 confirmations
    cache_address_ttl: int = 300  # 5 min for address data
    cache_confirmed_threshold: int = 6  # ≥6 conf = permanent
    cache_max_entries: int = 10000

    # OFAC sanctioned BTC addresses
    ofac_btc_csv: str = "./data/labels/ofac_btc.csv"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()


# ---------------------------------------------------------------------------
# CoinJoin detection constants — tune here, referenced by chain/coinjoin.py.
# Rules: Schnoering & Vazirgiannis 2023 (arXiv 2311.12491);
#        post-2024 Wasabi thresholds from Gavenda et al. ESORICS 2025
#        (arXiv 2510.17284).
# ---------------------------------------------------------------------------
def _gen_wasabi2_denoms(max_sat: int = 2_100_000_000):
    """Wasabi 2.x standard denomination set (>= 5000 sats).

    Powers of 2, powers of 3, 2*3^n, powers of 10, 2*10^n, 5*10^n.
    Keep in sync with chain/coinjoin.py; this is the user-tunable copy.
    """
    d: set = set()
    for base, multipliers in [
        (2, [2 ** n for n in range(1, 32)]),
        (3, [3 ** n for n in range(1, 21)]),
        (6, [2 * 3 ** n for n in range(1, 21)]),
        (10, [10 ** n for n in range(1, 11)]),
        (20, [2 * 10 ** n for n in range(1, 11)]),
        (50, [5 * 10 ** n for n in range(1, 11)]),
    ]:
        for v in multipliers:
            if 5_000 <= v <= max_sat:
                d.add(v)
    return frozenset(d)


COINJOIN_CONFIG = {
    "whirlpool_pool_denoms": [100_000, 1_000_000, 5_000_000, 50_000_000],
    "whirlpool_eps_max": 200_000,
    "whirlpool_tx0_max_premix": 70,
    "whirlpool_tx0_eps_min": 100,
    "whirlpool_tx0_eps_max": 100_000,
    "wasabi2_min_inputs": 50,
    "wasabi2_min_inputs_post2024": 20,
    "wasabi2_post2024_ts": 1_714_521_600,  # 2024-05-01 UTC
    "wasabi2_min_input_value": 5_000,
    "wasabi2_denominations": _gen_wasabi2_denoms(),
    "wasabi1_denom_center_sat": 10_000_000,  # 0.1 BTC
    "wasabi1_denom_tolerance": 0.10,
    "joinmarket_min_equal_outputs": 3,
    "generic_min_equal_outputs": 3,
    "generic_min_distinct_inputs": 3,
    "reuse_downgrade_threshold": 0.70,
}
