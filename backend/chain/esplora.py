"""Esplora HTTP API client for Bitcoin blockchain data"""

import httpx
import asyncio
import re
import random
from typing import List, Optional, Literal
from .protocol import Transaction, TxInput, TxOutput, Address, Subgraph
from .cache import ChainCache
from .exceptions import InvalidAddress, NotFound, ChainUnavailable


class EsploraSource:
    """
    Bitcoin blockchain data source via Esplora HTTP API (electrs).
    Handles address validation, pagination, caching, retries.
    """

    def __init__(
        self,
        base_url: str,
        cache: ChainCache,
        network: Literal["signet", "testnet4", "mainnet"] = "signet",
        fallback_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_concurrent: int = 10,
        max_txs_per_address: int = 50,
        max_nodes: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.fallback_url = fallback_url.rstrip("/") if fallback_url else None
        self.cache = cache
        self.network = network
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_txs_per_address = max_txs_per_address
        self.max_nodes = max_nodes

        self.client: Optional[httpx.AsyncClient] = None
        self._cached_tip_height: Optional[int] = None
        self._tip_height_cache_time: float = 0.0

    async def init(self):
        """Initialize HTTP client with connection pooling"""
        limits = httpx.Limits(
            max_connections=self.max_concurrent,
            max_keepalive_connections=self.max_concurrent,
            keepalive_expiry=30.0
        )
        # Note: http2=True requires httpx[http2] package
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            follow_redirects=True
        )

        # Test connectivity
        try:
            print(f"Testing connectivity to {self.base_url}...")
            response = await self.client.get(f"{self.base_url}/blocks/tip/height")
            response.raise_for_status()
            print(f"✓ Connected to Esplora API (tip height: {response.text})")
        except Exception as e:
            print(f"✗ Failed to connect to {self.base_url}: {type(e).__name__}: {e}")
            if self.fallback_url:
                print(f"Trying fallback URL: {self.fallback_url}")
                self.base_url = self.fallback_url
                self.fallback_url = None
                try:
                    response = await self.client.get(f"{self.base_url}/blocks/tip/height")
                    response.raise_for_status()
                    print(f"✓ Connected to fallback Esplora API (tip height: {response.text})")
                except Exception as e2:
                    print(f"✗ Fallback also failed: {type(e2).__name__}: {e2}")
                    raise

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None

    def validate_address(self, address: str):
        """
        Validate address format and network prefix.
        Raises InvalidAddress if format is wrong or network mismatch.
        """
        # Bech32/Bech32m (segwit v0/v1+)
        if address.startswith(("bc1", "tb1")):
            if self.network in ["signet", "testnet4"] and not address.startswith("tb1"):
                raise InvalidAddress(
                    f"Address {address} has mainnet prefix, but network is {self.network}"
                )
            if self.network == "mainnet" and not address.startswith("bc1"):
                raise InvalidAddress(
                    f"Address {address} has testnet prefix, but network is mainnet"
                )

            # Basic bech32 format check (lowercased, valid charset)
            if not re.match(r"^(bc|tb)1[ac-hj-np-z02-9]{11,71}$", address.lower()):
                raise InvalidAddress(f"Invalid bech32 format: {address}")

        # Base58 (P2PKH/P2SH)
        elif re.match(r"^[13mn2][a-km-zA-HJ-NP-Z1-9]{25,34}$", address):
            mainnet_prefixes = ["1", "3"]
            testnet_prefixes = ["m", "n", "2"]

            if self.network in ["signet", "testnet4"]:
                if address[0] in mainnet_prefixes:
                    raise InvalidAddress(
                        f"Address {address} has mainnet prefix, but network is {self.network}"
                    )
            elif self.network == "mainnet":
                if address[0] in testnet_prefixes:
                    raise InvalidAddress(
                        f"Address {address} has testnet prefix, but network is mainnet"
                    )
        else:
            raise InvalidAddress(f"Unrecognized address format: {address}")

    async def get_k_hop_subgraph(
        self, center_address: str, k: int = 2
    ) -> Subgraph:
        """
        Extract k-hop subgraph via BFS with limits.
        Returns Subgraph with addresses and transactions.
        """
        print(f"[DEBUG] get_k_hop_subgraph called: address={center_address}, k={k}")
        self.validate_address(center_address)
        print(f"[DEBUG] Address validation passed")

        addresses = {}
        transactions = {}
        visited_addresses = set()
        visited_txs = set()
        queue = [(center_address, 0)]  # (address, hop_distance)
        truncated = False

        while queue and not truncated:
            address, hop = queue.pop(0)

            if address in visited_addresses or hop > k:
                continue

            if len(addresses) >= self.max_nodes:
                truncated = True
                break

            visited_addresses.add(address)

            # Fetch address data
            print(f"[DEBUG] Fetching address data for: {address} (hop {hop})")
            try:
                addr_data = await self.get_address_with_txs(
                    address, max_txs=self.max_txs_per_address
                )
                print(f"[DEBUG] Fetched address {address}: balance={addr_data.balance_sat}, tx_count={addr_data.tx_count}")
            except NotFound:
                # Address exists in tx outputs but has no history yet
                addr_data = Address(
                    address=address,
                    balance_sat=0,
                    tx_count=0,
                    txs=[],
                    funded_sat=0,
                    spent_sat=0,
                )

            addresses[address] = addr_data

            # Process transactions
            for tx in addr_data.txs:
                if tx.txid in visited_txs:
                    continue

                visited_txs.add(tx.txid)
                transactions[tx.txid] = tx

                # Add connected addresses to queue if within hop limit
                if hop < k:
                    # From inputs
                    for inp in tx.inputs:
                        if inp.address and inp.address not in visited_addresses:
                            if not inp.is_coinbase:
                                queue.append((inp.address, hop + 1))

                    # From outputs
                    for out in tx.outputs:
                        if out.address and out.address not in visited_addresses:
                            queue.append((out.address, hop + 1))

                if len(transactions) >= self.max_txs_per_address * self.max_nodes:
                    truncated = True
                    break

        return Subgraph(
            addresses=addresses,
            transactions=transactions,
            center_address=center_address,
            k=k,
            truncated=truncated,
            source="chain",
        )

    async def get_address_with_txs(
        self, address: str, max_txs: int = 50
    ) -> Address:
        """
        Get address summary and transaction history (paginated).
        """
        self.validate_address(address)

        # Check cache first
        cached = await self.cache.get_address(address)
        if cached:
            return Address(
                address=cached["address"],
                balance_sat=cached["balance_sat"],
                tx_count=cached["tx_count"],
                txs=[self._dict_to_tx(tx) for tx in cached["txs"]],
                funded_sat=cached.get("funded_sat", 0),
                spent_sat=cached.get("spent_sat", 0),
                confirmed_tx_count=cached.get("confirmed_tx_count", cached["tx_count"]),
                mempool_tx_count=cached.get("mempool_tx_count", 0),
            )

        # Fetch address summary
        addr_summary = await self._fetch_json(f"/address/{address}")

        balance_sat = addr_summary["chain_stats"]["funded_txo_sum"] - addr_summary["chain_stats"]["spent_txo_sum"]
        confirmed_tx_count = addr_summary["chain_stats"]["tx_count"]
        mempool_tx_count = addr_summary["mempool_stats"]["tx_count"]
        tx_count = confirmed_tx_count + mempool_tx_count
        funded_sat = addr_summary["chain_stats"]["funded_txo_sum"]
        spent_sat = addr_summary["chain_stats"]["spent_txo_sum"]

        # Fetch transactions with pagination
        txs = await self._fetch_address_txs(address, max_txs)

        addr = Address(
            address=address,
            balance_sat=balance_sat,
            tx_count=tx_count,
            txs=txs,
            funded_sat=funded_sat,
            spent_sat=spent_sat,
            confirmed_tx_count=confirmed_tx_count,
            mempool_tx_count=mempool_tx_count,
        )

        # Cache result
        await self.cache.set_address(
            address,
            {
                "address": addr.address,
                "balance_sat": addr.balance_sat,
                "tx_count": addr.tx_count,
                "txs": [self._tx_to_dict(tx) for tx in addr.txs],
                "funded_sat": addr.funded_sat,
                "spent_sat": addr.spent_sat,
                "confirmed_tx_count": addr.confirmed_tx_count,
                "mempool_tx_count": addr.mempool_tx_count,
            },
        )

        return addr

    async def get_address_txs_page(
        self, address: str, cursor: Optional[str] = None, page_size: int = 25
    ) -> tuple[List[Transaction], Optional[str], bool]:
        """
        Get paginated transaction history for an address.
        Returns (transactions, next_cursor, has_more).
        cursor is the txid to start after.
        """
        self.validate_address(address)

        # Construct URL with pagination
        if cursor:
            url = f"/address/{address}/txs/chain/{cursor}"
        else:
            url = f"/address/{address}/txs"

        # Fetch from API
        tx_list = await self._fetch_json(url)

        if not tx_list:
            return ([], None, False)

        # Parse transactions
        txs = []
        for tx_json in tx_list[:page_size]:
            tx = await self._parse_transaction(tx_json)
            txs.append(tx)

        # Determine next cursor and has_more
        # Esplora returns 25 transactions per page by default for confirmed txs
        has_more = len(tx_list) >= 25
        next_cursor = tx_list[-1]["txid"] if has_more and txs else None

        return (txs, next_cursor, has_more)

    async def _fetch_address_txs(
        self, address: str, max_txs: int
    ) -> List[Transaction]:
        """Fetch address transactions with pagination (avoid N+1 queries)"""
        txs = []
        last_seen_txid = None
        fetched = 0

        while fetched < max_txs:
            # Construct URL with pagination
            if last_seen_txid:
                url = f"/address/{address}/txs/chain/{last_seen_txid}"
            else:
                url = f"/address/{address}/txs"

            # electrs returns full tx objects with vin/vout resolved
            tx_list = await self._fetch_json(url)

            if not tx_list:
                break  # No more transactions

            # Parse transactions directly (no N+1 re-fetch)
            for tx_json in tx_list:
                if fetched >= max_txs:
                    break

                tx = await self._parse_transaction(tx_json)
                txs.append(tx)
                fetched += 1

            # Pagination continuation
            if len(tx_list) < 25:  # electrs default page size for confirmed
                break

            last_seen_txid = tx_list[-1]["txid"]

        return txs

    async def get_transaction(self, txid: str) -> Transaction:
        """Get single transaction by ID"""
        # Check cache
        cached = await self.cache.get_tx(txid)
        if cached:
            return self._dict_to_tx(cached)

        # Fetch from API
        tx_json = await self._fetch_json(f"/tx/{txid}")
        tx = await self._parse_transaction(tx_json)

        # Cache result
        await self.cache.set_tx(txid, self._tx_to_dict(tx), tx.confirmations)

        return tx

    async def _parse_transaction(self, tx_json: dict) -> Transaction:
        """Parse Esplora transaction JSON into Transaction dataclass"""
        txid = tx_json["txid"]

        # Parse inputs
        inputs = []
        for vin in tx_json["vin"]:
            if vin.get("is_coinbase") or vin.get("prevout") is None:
                # Coinbase input
                inputs.append(
                    TxInput(
                        txid=None,
                        vout=None,
                        address=None,
                        value_sat=0,
                        is_coinbase=True,
                    )
                )
            else:
                prevout = vin["prevout"]
                inputs.append(
                    TxInput(
                        txid=vin["txid"],
                        vout=vin["vout"],
                        address=prevout.get("scriptpubkey_address"),
                        value_sat=prevout["value"],
                        is_coinbase=False,
                    )
                )

        # Parse outputs
        outputs = []
        op_return_data = None
        burned_value_sat = 0

        for vout_idx, vout in enumerate(tx_json["vout"]):
            address = vout.get("scriptpubkey_address")
            value_sat = vout["value"]

            # OP_RETURN detection
            if vout.get("scriptpubkey_type") == "op_return":
                op_return_data = vout.get("scriptpubkey", "")
                if value_sat > 0:
                    burned_value_sat += value_sat
                # Don't create an output node for OP_RETURN
                continue

            outputs.append(
                TxOutput(
                    vout=vout_idx,
                    address=address,
                    value_sat=value_sat,
                    spent=False,  # Updated if we fetch outspends
                )
            )

        # Status and confirmations
        status = tx_json.get("status", {})
        block_height = status.get("block_height")
        block_time = status.get("block_time")
        confirmed = status.get("confirmed", False)

        # Fetch tip height for confirmations
        if confirmed and block_height:
            tip = await self._get_tip_height()
            confirmations = max(0, tip - block_height + 1)
        else:
            confirmations = 0

        # Fee calculation
        total_in = sum(inp.value_sat for inp in inputs if not inp.is_coinbase)
        total_out = sum(out.value_sat for out in outputs)
        fee_sat = max(0, total_in - total_out)

        # Parse weight/size for fee rate calculation
        weight = tx_json.get("weight")
        size = tx_json.get("size")
        # vsize = ceil(weight / 4)
        vsize = None
        if weight:
            vsize = (weight + 3) // 4  # Integer ceiling division

        return Transaction(
            txid=txid,
            inputs=inputs,
            outputs=outputs,
            fee_sat=fee_sat,
            block_height=block_height,
            block_time=block_time,
            confirmations=confirmations,
            weight=weight,
            size=size,
            vsize=vsize,
            op_return_data=op_return_data,
            has_op_return=op_return_data is not None,
            burned_value_sat=burned_value_sat,
        )

    async def _get_tip_height(self) -> int:
        """Get current blockchain tip height (cached for 60s)"""
        import time

        # Return cached value if fresh (under 60 seconds old)
        now = time.time()
        if self._cached_tip_height is not None and (now - self._tip_height_cache_time) < 60:
            return self._cached_tip_height

        # Simple GET to /blocks/tip/height returns plain integer
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.get(f"{self.base_url}/blocks/tip/height")
                    response.raise_for_status()
                    height = int(response.text.strip())

                    # Update cache
                    self._cached_tip_height = height
                    self._tip_height_cache_time = now

                    return height
                except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    if attempt == self.max_retries - 1:
                        raise ChainUnavailable(f"Cannot reach {self.base_url}: {e}")
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(backoff)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        raise NotFound(f"Tip height not found (is electrs synced?)")
                    elif e.response.status_code == 429:
                        retry_after = int(e.response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)
                    elif e.response.status_code >= 500:
                        if attempt == self.max_retries - 1:
                            raise ChainUnavailable(f"Server error: {e.response.status_code}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise ChainUnavailable(f"HTTP {e.response.status_code}")

    async def _fetch_json(self, path: str) -> dict:
        """Fetch JSON from Esplora API with retries and rate limit handling"""
        async with self.semaphore:
            url = f"{self.base_url}{path}"

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.get(url)
                    response.raise_for_status()
                    return response.json()

                except (httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                    # RemoteProtocolError = server disconnected, treat as retriable
                    error_msg = f"Connection error on attempt {attempt + 1}/{self.max_retries} for {path}: {type(e).__name__}: {e}"
                    print(error_msg)
                    if attempt == self.max_retries - 1:
                        raise ChainUnavailable(f"Connection error fetching {path}: {e}")
                    # Exponential backoff with jitter
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(backoff)

                except httpx.ConnectError as e:
                    error_msg = f"Connect error on attempt {attempt + 1}/{self.max_retries} for {path}: {e}"
                    print(error_msg)
                    if attempt == self.max_retries - 1:
                        # Try fallback URL if available
                        if self.fallback_url and self.base_url != self.fallback_url:
                            print(f"Switching to fallback URL: {self.fallback_url}")
                            old_base = self.base_url
                            self.base_url = self.fallback_url
                            self.fallback_url = None  # Prevent infinite fallback loop
                            try:
                                return await self._fetch_json(path)  # Retry with fallback
                            except Exception:
                                self.base_url = old_base  # Restore on failure
                                raise ChainUnavailable(f"Cannot connect to {old_base} or {self.fallback_url}: {e}")
                        raise ChainUnavailable(f"Cannot connect to {self.base_url}: {e}")
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(backoff)

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        raise NotFound(f"Resource not found: {path}")

                    elif e.response.status_code == 429:
                        # Rate limited - respect Retry-After
                        retry_after = int(e.response.headers.get("Retry-After", 5))
                        await asyncio.sleep(retry_after)

                    elif e.response.status_code >= 500:
                        # Server error - retry with backoff
                        if attempt == self.max_retries - 1:
                            raise ChainUnavailable(
                                f"Server error {e.response.status_code} on {path}"
                            )
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(backoff)

                    elif e.response.status_code == 400:
                        # Bad request - likely invalid address format
                        raise InvalidAddress(f"Invalid request: {path}")

                    else:
                        raise ChainUnavailable(
                            f"HTTP {e.response.status_code} on {path}"
                        )

            raise ChainUnavailable(f"Max retries exceeded for {path}")

    def _tx_to_dict(self, tx: Transaction) -> dict:
        """Convert Transaction to dict for caching"""
        return {
            "txid": tx.txid,
            "inputs": [
                {
                    "txid": inp.txid,
                    "vout": inp.vout,
                    "address": inp.address,
                    "value_sat": inp.value_sat,
                    "is_coinbase": inp.is_coinbase,
                }
                for inp in tx.inputs
            ],
            "outputs": [
                {
                    "vout": out.vout,
                    "address": out.address,
                    "value_sat": out.value_sat,
                    "spent": out.spent,
                }
                for out in tx.outputs
            ],
            "fee_sat": tx.fee_sat,
            "block_height": tx.block_height,
            "block_time": tx.block_time,
            "confirmations": tx.confirmations,
            "op_return_data": tx.op_return_data,
            "has_op_return": tx.has_op_return,
            "burned_value_sat": tx.burned_value_sat,
        }

    def _dict_to_tx(self, d: dict) -> Transaction:
        """Convert dict back to Transaction"""
        return Transaction(
            txid=d["txid"],
            inputs=[TxInput(**inp) for inp in d["inputs"]],
            outputs=[TxOutput(**out) for out in d["outputs"]],
            fee_sat=d["fee_sat"],
            block_height=d.get("block_height"),
            block_time=d.get("block_time"),
            confirmations=d["confirmations"],
            op_return_data=d.get("op_return_data"),
            has_op_return=d.get("has_op_return", False),
            burned_value_sat=d.get("burned_value_sat", 0),
        )
