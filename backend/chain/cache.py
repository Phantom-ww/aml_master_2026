"""Async SQLite cache for chain data with TTL policies"""

import aiosqlite
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict


class ChainCache:
    """
    Async SQLite cache with confirmation-based TTL:
    - Unconfirmed (0 conf): never cached
    - 1-5 confirmations: short TTL (configurable, default 10min)
    - ≥6 confirmations: no expiry (permanent)
    - Address data: TTL (configurable, default 5min)
    """

    def __init__(
        self,
        db_path: str,
        short_ttl_seconds: int = 600,  # 10 minutes
        address_ttl_seconds: int = 300,  # 5 minutes
        confirmed_threshold: int = 6,
        max_cache_entries: int = 10000,
    ):
        self.db_path = Path(db_path)
        self.short_ttl = short_ttl_seconds
        self.address_ttl = address_ttl_seconds
        self.confirmed_threshold = confirmed_threshold
        self.max_entries = max_cache_entries
        self.conn: Optional[aiosqlite.Connection] = None

        # Stats
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    async def init(self):
        """Initialize database connection and schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = await aiosqlite.connect(str(self.db_path))
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA synchronous=NORMAL")

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at INTEGER,
                created_at INTEGER NOT NULL
            )
        """)
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)"
        )
        await self.conn.commit()

    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def get_tx(self, txid: str) -> Optional[Dict[str, Any]]:
        """Get cached transaction"""
        return await self._get(f"tx:{txid}")

    async def set_tx(self, txid: str, tx_data: Dict[str, Any], confirmations: int):
        """
        Cache transaction with TTL based on confirmations:
        - 0 confirmations: not cached
        - 1-5 confirmations: short TTL
        - ≥6 confirmations: permanent (no expiry)
        """
        if confirmations == 0:
            return  # Never cache unconfirmed

        if confirmations < self.confirmed_threshold:
            ttl = self.short_ttl
        else:
            ttl = None  # Permanent

        await self._set(f"tx:{txid}", tx_data, ttl)

    async def get_address(self, address: str) -> Optional[Dict[str, Any]]:
        """Get cached address data"""
        return await self._get(f"addr:{address}")

    async def set_address(self, address: str, addr_data: Dict[str, Any]):
        """Cache address data with TTL"""
        await self._set(f"addr:{address}", addr_data, self.address_ttl)

    async def _get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if not expired"""
        if not self.conn:
            return None

        now = int(time.time())
        async with self._lock:
            cursor = await self.conn.execute(
                """
                SELECT value, expires_at FROM cache
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (key, now),
            )
            row = await cursor.fetchone()

        if row:
            self._hits += 1
            return json.loads(row[0])

        self._misses += 1
        return None

    async def _set(self, key: str, value: Dict[str, Any], ttl: Optional[int]):
        """Set value in cache with optional TTL"""
        if not self.conn:
            return

        now = int(time.time())
        expires_at = (now + ttl) if ttl else None
        value_json = json.dumps(value)

        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (key, value_json, expires_at, now),
            )
            await self.conn.commit()

            # Eviction check (batch cleanup when limit exceeded)
            await self._evict_if_needed()

    async def _evict_if_needed(self):
        """Evict expired entries and enforce size limit"""
        now = int(time.time())

        # Delete expired entries
        await self.conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )

        # Enforce size limit (keep most recently created)
        cursor = await self.conn.execute("SELECT COUNT(*) FROM cache")
        count = (await cursor.fetchone())[0]

        if count > self.max_entries:
            to_delete = count - self.max_entries
            await self.conn.execute(
                """
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache
                    WHERE expires_at IS NOT NULL
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (to_delete,),
            )

        await self.conn.commit()

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.conn:
            return {"error": "not_initialized"}

        cursor = await self.conn.execute("SELECT COUNT(*) FROM cache")
        total = (await cursor.fetchone())[0]

        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at IS NULL"
        )
        permanent = (await cursor.fetchone())[0]

        hit_rate = (
            self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0.0
        )

        return {
            "total_entries": total,
            "permanent_entries": permanent,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }

    async def clear(self):
        """Clear all cache entries (for testing)"""
        if self.conn:
            await self.conn.execute("DELETE FROM cache")
            await self.conn.commit()
            self._hits = 0
            self._misses = 0
