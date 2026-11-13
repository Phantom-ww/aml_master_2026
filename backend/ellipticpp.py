"""Elliptic++ label lookup."""
import sqlite3
from pathlib import Path
from typing import Optional
from functools import lru_cache

DB_PATH = Path(__file__).parent / "ellipticpp_labels.db"

@lru_cache(maxsize=10000)
def get_label(address: str) -> Optional[int]:
    """Get Elliptic++ class for an address (1=illicit, 2=licit, None=unknown)."""
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT class FROM ellipticpp_labels WHERE address = ?", (address,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None
