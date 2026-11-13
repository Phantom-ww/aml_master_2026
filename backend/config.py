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

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
