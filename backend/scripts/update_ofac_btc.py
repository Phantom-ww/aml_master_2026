#!/usr/bin/env python3
"""
Download the OFAC SDN list and extract Bitcoin (XBT) addresses.

Run manually on the host — not on startup:
    python3 backend/scripts/update_ofac_btc.py

Output: data/labels/ofac_btc.csv (address, entity, program)

OFAC SDN CSV is published at:
  https://www.treasury.gov/ofac/downloads/sdn.csv
  (mirror: https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV)

Format of SDN.CSV (pipe-delimited-ish, but typically comma-delimited with 12 fields):
  ent_num, SDN_name, SDN_type, program, title, call_sign, vess_type, tonnage,
  GRT, vess_flag, vess_owner, remarks
The "remarks" field may contain semicolon-delimited entries such as:
  Digital Currency Address - XBT 1FfmbHfnpaZjKFvyi1okTjJJusN455paPH;
"""

import csv
import io
import os
import re
import sys
import urllib.request
from pathlib import Path

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
FALLBACK_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"

# Regex: "Digital Currency Address - XBT <address>"
_XBT_RE = re.compile(r"Digital Currency Address\s*-\s*XBT\s+([A-Za-z0-9]+)", re.IGNORECASE)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "labels"
OUT_FILE = OUT_DIR / "ofac_btc.csv"


def _download(url: str) -> str:
    """Download URL and return content as string."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (AML-research-tool)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("latin-1")


def _parse_sdn(raw: str) -> list[dict]:
    """Parse SDN.CSV rows, extracting XBT addresses from the remarks field."""
    entries = []
    reader = csv.reader(io.StringIO(raw))
    for row in reader:
        if len(row) < 12:
            continue
        entity_name = row[1].strip().strip('"')
        program = row[3].strip().strip('"')
        remarks = row[11].strip().strip('"')
        for m in _XBT_RE.finditer(remarks):
            addr = m.group(1).strip()
            if addr:
                entries.append(
                    {"address": addr, "entity": entity_name, "program": program}
                )
    return entries


def main() -> None:
    print(f"Downloading OFAC SDN list from {SDN_URL} ...")
    raw = None
    for url in (SDN_URL, FALLBACK_URL):
        try:
            raw = _download(url)
            print(f"  Downloaded {len(raw):,} bytes from {url}")
            break
        except Exception as exc:
            print(f"  Failed ({url}): {exc}", file=sys.stderr)

    if raw is None:
        print("ERROR: could not download SDN list from any source.", file=sys.stderr)
        sys.exit(1)

    entries = _parse_sdn(raw)
    print(f"Found {len(entries)} XBT address entries.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["address", "entity", "program"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
