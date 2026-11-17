"""Collect a labeled evaluation dataset for the live wallet-risk model.

Positives: BTC addresses from data/labels/ofac_btc.csv (OFAC sanctioned addresses).
Negatives: 3× as many random addresses sampled from blocks time-matched to the positives.

NOTE: The OFAC list includes exchanges and other sanctioned entities, not only fraud.
These are imperfect positive labels — treat evaluation results accordingly.

Usage:
    python collect.py [--limit N] [--esplora-url URL] [--max-txs N] [--throttle SECS] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — HTTP
# ---------------------------------------------------------------------------

def _get_json(url: str, retries: int = 3, throttle: float = 1.0) -> dict | list | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aml-live-eval/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            time.sleep(throttle)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.debug("404 for %s", url)
                return None
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 10))
                log.warning("Rate-limited, sleeping %ss", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %s for %s (attempt %d)", e.code, url, attempt + 1)
        except Exception as e:
            log.warning("Error fetching %s (attempt %d): %s", url, attempt + 1, e)
        if attempt < retries - 1:
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return None


# ---------------------------------------------------------------------------
# Tx history fetch
# ---------------------------------------------------------------------------

STANDARD_TYPES = {"p2pkh", "p2sh", "v0_p2wpkh", "v0_p2wsh", "p2tr"}


def fetch_address_txs(address: str, esplora_url: str, max_txs: int, throttle: float) -> tuple[list[dict], bool]:
    """Fetch up to max_txs confirmed txs for an address. Returns (txs, truncated)."""
    txs: list[dict] = []
    last_seen = None
    truncated = False

    while True:
        if last_seen:
            url = f"{esplora_url}/address/{address}/txs/chain/{last_seen}"
        else:
            url = f"{esplora_url}/address/{address}/txs"

        page = _get_json(url, throttle=throttle)
        if not page:
            break

        # Filter confirmed only
        confirmed = [tx for tx in page if (tx.get("status") or {}).get("block_height")]
        txs.extend(confirmed)

        if len(txs) >= max_txs:
            txs = txs[:max_txs]
            truncated = True
            break

        if len(page) < 25:
            break

        last_seen = page[-1]["txid"]

    return txs, truncated


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(address: str, raw_dir: Path) -> Path:
    return raw_dir / f"{address}.json.gz"


def load_cached(address: str, raw_dir: Path) -> list[dict] | None:
    p = cache_path(address, raw_dir)
    if p.exists():
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached(address: str, txs: list[dict], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = cache_path(address, raw_dir)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(txs, f)


# ---------------------------------------------------------------------------
# First block height from tx list
# ---------------------------------------------------------------------------

def first_height(txs: list[dict]) -> int | None:
    heights = [
        tx["status"]["block_height"]
        for tx in txs
        if (tx.get("status") or {}).get("block_height")
    ]
    return min(heights) if heights else None


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(
    n_needed: int,
    height_range: tuple[int, int],
    esplora_url: str,
    raw_dir: Path,
    ofac_set: set[str],
    max_txs: int,
    throttle: float,
    already_collected: set[str],
) -> list[dict]:
    """Sample n_needed negative addresses from random blocks in height_range."""
    negatives: list[dict] = []
    lo, hi = height_range
    attempts = 0
    max_attempts = n_needed * 20

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        height = random.randint(lo, hi)

        # Get block hash
        block_hash = _get_json(f"{esplora_url}/block-height/{height}", throttle=throttle)
        if not isinstance(block_hash, str):
            continue

        # Get block txids
        txids = _get_json(f"{esplora_url}/block/{block_hash}/txids", throttle=throttle)
        if not txids:
            continue

        # Pick a few random txids from that block (skip coinbase = first tx)
        candidates = txids[1:]
        random.shuffle(candidates)
        for txid in candidates[:5]:
            tx_data = _get_json(f"{esplora_url}/tx/{txid}", throttle=throttle)
            if not tx_data:
                continue

            for vout in tx_data.get("vout", []):
                addr = vout.get("scriptpubkey_address")
                stype = vout.get("scriptpubkey_type", "")
                if (
                    addr
                    and stype in STANDARD_TYPES
                    and addr not in ofac_set
                    and addr not in already_collected
                    and addr not in {r["address"] for r in negatives}
                ):
                    # Fetch and cache tx history
                    cached = load_cached(addr, raw_dir)
                    if cached is not None:
                        txs, truncated = cached, len(cached) >= max_txs
                    else:
                        try:
                            txs, truncated = fetch_address_txs(addr, esplora_url, max_txs, throttle)
                            save_cached(addr, txs, raw_dir)
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            log.warning("Failed to fetch negative %s: %s", addr, e)
                            continue

                    fh = first_height(txs)
                    negatives.append({
                        "address": addr,
                        "y": 0,
                        "source": "block_sample",
                        "entity": "",
                        "first_height": fh or 0,
                        "n_txs": len(txs),
                        "truncated": truncated,
                    })
                    log.info("  negative %d/%d: %s (n_txs=%d)", len(negatives), n_needed, addr, len(txs))

                    if len(negatives) >= n_needed:
                        break

            if len(negatives) >= n_needed:
                break

    if len(negatives) < n_needed:
        log.warning("Only collected %d/%d negatives after %d block attempts", len(negatives), n_needed, attempts)
    return negatives


# ---------------------------------------------------------------------------
# Labels CSV
# ---------------------------------------------------------------------------

def load_existing_labels(labels_path: Path) -> dict[str, dict]:
    if not labels_path.exists():
        return {}
    with labels_path.open(newline="", encoding="utf-8") as f:
        return {row["address"]: row for row in csv.DictReader(f)}


def write_labels(labels_path: Path, rows: list[dict]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["address", "y", "source", "entity", "first_height", "n_txs", "truncated"]
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max total addresses")
    parser.add_argument("--esplora-url", default="https://mempool.space/api", help="Esplora base URL")
    parser.add_argument("--max-txs", type=int, default=500, help="Max confirmed txs per address")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds between requests")
    parser.add_argument("--out-dir", default=None, help="Project root directory")
    args = parser.parse_args()

    # Resolve project root
    project_root = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[3]
    ofac_csv = project_root / "data" / "labels" / "ofac_btc.csv"
    raw_dir = project_root / "data" / "live_eval" / "raw"
    labels_path = project_root / "data" / "live_eval" / "labels.csv"

    if not ofac_csv.exists():
        log.error("OFAC CSV not found at %s", ofac_csv)
        log.error("Run:  python backend/scripts/update_ofac_btc.py")
        sys.exit(1)

    # Load existing labels (for resumability)
    existing = load_existing_labels(labels_path)
    log.info("Existing labels: %d addresses", len(existing))

    # Load OFAC positives
    with ofac_csv.open(newline="", encoding="utf-8") as f:
        ofac_rows = list(csv.DictReader(f))
    ofac_set = {r["address"] for r in ofac_rows}
    log.info("OFAC BTC addresses: %d", len(ofac_rows))

    # Apply limit
    if args.limit:
        ofac_rows = ofac_rows[:max(1, args.limit // 4)]

    rows: list[dict] = list(existing.values())
    done_addrs = {r["address"] for r in rows}

    # --- Step 2: Fetch positives ---
    log.info("=== Fetching positives ===")
    new_positives: list[dict] = []
    try:
        for entry in ofac_rows:
            addr = entry["address"]
            if addr in done_addrs:
                log.info("  skip (cached): %s", addr)
                continue

            cached = load_cached(addr, raw_dir)
            if cached is not None:
                txs, truncated = cached, len(cached) >= args.max_txs
                log.info("  loaded from cache: %s (n_txs=%d)", addr, len(txs))
            else:
                log.info("  fetching: %s", addr)
                try:
                    txs, truncated = fetch_address_txs(addr, args.esplora_url, args.max_txs, args.throttle)
                    save_cached(addr, txs, raw_dir)
                except Exception as e:
                    log.warning("  SKIP %s: %s", addr, e)
                    continue

            fh = first_height(txs)
            rec = {
                "address": addr,
                "y": 1,
                "source": "ofac",
                "entity": entry.get("entity", ""),
                "first_height": fh or 0,
                "n_txs": len(txs),
                "truncated": truncated,
            }
            new_positives.append(rec)
            rows.append(rec)
            done_addrs.add(addr)
            log.info("  positive %d: %s (n_txs=%d, fh=%s)", len(new_positives), addr, len(txs), fh)

    except KeyboardInterrupt:
        log.info("Interrupted — saving progress...")
        write_labels(labels_path, rows)
        sys.exit(0)

    # Collect all positives (new + existing) for height range
    all_positives = [r for r in rows if r["y"] == "1" or r.get("y") == 1]
    heights = [int(r["first_height"]) for r in all_positives if int(r.get("first_height", 0)) > 0]
    if not heights:
        log.error("No positive addresses with valid first_height; cannot sample negatives.")
        write_labels(labels_path, rows)
        sys.exit(1)

    # --- Step 3: Sample negatives ---
    n_positives = len(all_positives)
    n_negatives_needed = n_positives * 3
    existing_negatives = sum(1 for r in rows if str(r.get("y")) == "0")
    still_needed = max(0, n_negatives_needed - existing_negatives)

    if args.limit:
        still_needed = min(still_needed, args.limit - len(rows))

    log.info("=== Sampling negatives (need %d more) ===", still_needed)
    if still_needed > 0:
        try:
            neg_rows = sample_negatives(
                n_needed=still_needed,
                height_range=(min(heights), max(heights)),
                esplora_url=args.esplora_url,
                raw_dir=raw_dir,
                ofac_set=ofac_set,
                max_txs=args.max_txs,
                throttle=args.throttle,
                already_collected=done_addrs,
            )
            rows.extend(neg_rows)
        except KeyboardInterrupt:
            log.info("Interrupted — saving progress...")
            write_labels(labels_path, rows)
            sys.exit(0)

    # --- Step 4: Write labels.csv ---
    write_labels(labels_path, rows)
    n_pos = sum(1 for r in rows if str(r.get("y")) == "1")
    n_neg = sum(1 for r in rows if str(r.get("y")) == "0")
    log.info("Done. labels.csv: %d total (%d pos, %d neg) → %s", len(rows), n_pos, n_neg, labels_path)

    # --- Step 5: Collect transaction-level labels for the live tx model ---
    log.info("=== Collecting transaction labels ===")
    try:
        collect_tx_labels(
            ofac_set=ofac_set,
            raw_dir=raw_dir,
            tx_raw_dir=project_root / "data" / "live_eval" / "raw_tx",
            tx_labels_path=project_root / "data" / "live_eval" / "tx_labels.csv",
            esplora_url=args.esplora_url,
            throttle=args.throttle,
            limit=args.limit,
            height_range=(min(heights), max(heights)),
        )
    except KeyboardInterrupt:
        log.info("Interrupted during tx label collection — saving progress...")
    except Exception as e:
        log.warning("Tx label collection failed: %s", e)


# ---------------------------------------------------------------------------
# Transaction-level label collection (for the live tx model)
# ---------------------------------------------------------------------------

def _tx_raw_path(txid: str, tx_raw_dir: Path) -> Path:
    return tx_raw_dir / f"{txid}.json.gz"


def _load_tx_raw(txid: str, tx_raw_dir: Path) -> dict | None:
    p = _tx_raw_path(txid, tx_raw_dir)
    if p.exists():
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_tx_raw(txid: str, tx: dict, tx_raw_dir: Path) -> None:
    tx_raw_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(_tx_raw_path(txid, tx_raw_dir), "wt", encoding="utf-8") as f:
        json.dump(tx, f)


def _fetch_tx_raw(txid: str, esplora_url: str, throttle: float) -> dict | None:
    return _get_json(f"{esplora_url}/tx/{txid}", throttle=throttle)


def collect_tx_labels(
    ofac_set: set[str],
    raw_dir: Path,
    tx_raw_dir: Path,
    tx_labels_path: Path,
    esplora_url: str,
    throttle: float,
    limit: int | None,
    height_range: tuple[int, int],
) -> None:
    """Collect tx_labels.csv for the live tx model.

    Positives: txs where an OFAC-listed address is an INPUT (spends sanctioned funds).
    Negatives: random non-coinbase txs from random blocks in height_range.

    Raw tx dicts are cached under data/live_eval/raw_tx/{txid}.json.gz.
    """
    # Resume from existing tx_labels.csv
    existing: dict[str, dict] = {}
    if tx_labels_path.exists():
        with tx_labels_path.open(newline="", encoding="utf-8") as f:
            existing = {r["txid"]: r for r in csv.DictReader(f)}

    rows: list[dict] = list(existing.values())
    done_txids = {r["txid"] for r in rows}

    # ---- Positives: scan cached OFAC address histories for txs where the
    #      OFAC address appears in vin (is an input / spends sanctioned funds) ----
    pos_rows: list[dict] = []
    for addr in sorted(ofac_set):
        if limit and len(pos_rows) >= limit:
            break
        cached = load_cached(addr, raw_dir)
        if not cached:
            continue
        for tx in cached:
            txid = tx.get("txid")
            if not txid or txid in done_txids:
                continue
            # Is the OFAC address an input? (is it in vin[].prevout.scriptpubkey_address?)
            in_inputs = any(
                (v.get("prevout") or {}).get("scriptpubkey_address") == addr
                for v in tx.get("vin", [])
                if not v.get("is_coinbase")
            )
            if not in_inputs:
                continue
            # Cache the raw tx dict (the address-txs payload already has vin/vout)
            _save_tx_raw(txid, tx, tx_raw_dir)
            rec = {"txid": txid, "y": 1, "source": "ofac_input", "address": addr, "raw_path": str(_tx_raw_path(txid, tx_raw_dir))}
            pos_rows.append(rec)
            rows.append(rec)
            done_txids.add(txid)
            if limit and len(pos_rows) >= limit:
                break

    n_pos = len(pos_rows)
    log.info("  tx positives: %d", n_pos)

    # ---- Negatives: random non-coinbase txs from random blocks in height_range ----
    n_neg_needed = n_pos * 3
    if limit:
        n_neg_needed = min(n_neg_needed, max(0, limit - n_pos))
    neg_rows: list[dict] = []
    lo, hi = height_range
    attempts = 0
    max_attempts = n_neg_needed * 10

    while len(neg_rows) < n_neg_needed and attempts < max_attempts:
        attempts += 1
        height = random.randint(lo, hi)
        block_hash = _get_json(f"{esplora_url}/block-height/{height}", throttle=throttle)
        if not isinstance(block_hash, str):
            continue
        txids = _get_json(f"{esplora_url}/block/{block_hash}/txids", throttle=throttle)
        if not txids:
            continue
        candidates = txids[1:]  # skip coinbase (first tx)
        random.shuffle(candidates)
        for txid in candidates[:5]:
            if txid in done_txids:
                continue
            raw = _load_tx_raw(txid, tx_raw_dir) or _fetch_tx_raw(txid, esplora_url, throttle)
            if not raw:
                continue
            # Skip coinbase / op_return-only txs
            if any(v.get("is_coinbase") for v in raw.get("vin", [])):
                continue
            _save_tx_raw(txid, raw, tx_raw_dir)
            rec = {"txid": txid, "y": 0, "source": "block_sample", "address": "", "raw_path": str(_tx_raw_path(txid, tx_raw_dir))}
            neg_rows.append(rec)
            rows.append(rec)
            done_txids.add(txid)
            log.info("  tx negative %d/%d: %s", len(neg_rows), n_neg_needed, txid)
            if len(neg_rows) >= n_neg_needed:
                break

    log.info("  tx negatives: %d", len(neg_rows))

    # Write tx_labels.csv
    tx_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with tx_labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["txid", "y", "source", "address", "raw_path"])
        w.writeheader()
        w.writerows(rows)
    n_pos_total = sum(1 for r in rows if str(r.get("y")) == "1")
    n_neg_total = sum(1 for r in rows if str(r.get("y")) == "0")
    log.info("Done. tx_labels.csv: %d total (%d pos, %d neg) → %s", len(rows), n_pos_total, n_neg_total, tx_labels_path)


if __name__ == "__main__":
    main()
