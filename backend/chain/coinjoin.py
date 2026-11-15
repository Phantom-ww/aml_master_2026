"""
CoinJoin transaction detector — pure functions, no I/O.

Rules follow:
  Schnoering & Vazirgiannis 2023 (arXiv 2311.12491)
  Gavenda et al. ESORICS 2025 (arXiv 2510.17284)
"""

from __future__ import annotations
from collections import Counter
from typing import Union, Optional, List, Tuple

# ---------------------------------------------------------------------------
# Constants (also mirrored in config.COINJOIN_CONFIG for tunability)
# ---------------------------------------------------------------------------

def _gen_wasabi2_denoms(max_sat: int = 2_100_000_000) -> frozenset:
    """Generate Wasabi 2.x standard denomination set >= 5000 sats."""
    d: set = set()
    for base, multipliers in [
        (2, [2 ** n for n in range(1, 32)]),   # powers of 2
        (3, [3 ** n for n in range(1, 21)]),   # powers of 3
        (6, [2 * 3 ** n for n in range(1, 21)]),  # 2 * 3^n
        (10, [10 ** n for n in range(1, 11)]),   # powers of 10
        (20, [2 * 10 ** n for n in range(1, 11)]),  # 2 * 10^n
        (50, [5 * 10 ** n for n in range(1, 11)]),  # 5 * 10^n
    ]:
        for v in multipliers:
            if 5_000 <= v <= max_sat:
                d.add(v)
    return frozenset(d)


WASABI2_DENOMS: frozenset = _gen_wasabi2_denoms()

# Tune these in config.COINJOIN_CONFIG
_DEFAULT_CFG = {
    "whirlpool_pool_denoms": [100_000, 1_000_000, 5_000_000, 50_000_000],
    "whirlpool_eps_max": 200_000,
    "whirlpool_tx0_max_premix": 70,
    "whirlpool_tx0_eps_min": 100,
    "whirlpool_tx0_eps_max": 100_000,
    "wasabi2_min_inputs": 50,
    "wasabi2_min_inputs_post2024": 20,
    "wasabi2_post2024_ts": 1_714_521_600,  # 2024-05-01 UTC
    "wasabi2_min_input_value": 5_000,
    "wasabi2_denominations": WASABI2_DENOMS,
    "wasabi1_denom_center_sat": 10_000_000,  # 0.1 BTC
    "wasabi1_denom_tolerance": 0.10,
    "joinmarket_min_equal_outputs": 3,
    "generic_min_equal_outputs": 3,
    "generic_min_distinct_inputs": 3,
    "reuse_downgrade_threshold": 0.70,
}

# ---------------------------------------------------------------------------
# Cache (txid → result, simple FIFO eviction)
# ---------------------------------------------------------------------------

_cache: dict = {}
_CACHE_MAX = 10_000


def _cache_put(txid: str, result: dict) -> None:
    if len(_cache) >= _CACHE_MAX:
        # evict oldest (insertion-order guaranteed in Python 3.7+)
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[txid] = result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

InputList = List[Tuple[Optional[str], int]]   # [(address, value_sat), ...]
OutputList = List[Tuple[Optional[str], int]]  # [(address, value_sat), ...]


def _extract(tx) -> Tuple[str, InputList, OutputList, bool, Optional[int]]:
    """
    Normalise a Transaction dataclass or Esplora-shaped dict into
    (txid, inputs, outputs, has_op_return, block_time).
    """
    try:
        from chain.protocol import Transaction  # local import to stay pure
    except ImportError:
        Transaction = None

    if Transaction is not None and isinstance(tx, Transaction):
        txid = tx.txid
        inputs = [(i.address, i.value_sat) for i in tx.inputs if not i.is_coinbase]
        outputs = [(o.address, o.value_sat) for o in tx.outputs]
        return txid, inputs, outputs, tx.has_op_return, tx.block_time

    # Esplora dict
    txid = tx.get("txid", "")
    inputs = []
    for vin in tx.get("vin", []):
        if vin.get("is_coinbase") or vin.get("prevout") is None:
            continue
        prevout = vin["prevout"]
        inputs.append((prevout.get("scriptpubkey_address"), prevout.get("value", 0)))
    outputs = []
    has_op_return = False
    for vout in tx.get("vout", []):
        if vout.get("scriptpubkey_type") == "op_return":
            has_op_return = True
            continue
        outputs.append((vout.get("scriptpubkey_address"), vout.get("value", 0)))
    block_time = tx.get("status", {}).get("block_time")
    return txid, inputs, outputs, has_op_return, block_time


def _null_result() -> dict:
    return {
        "is_coinjoin": False,
        "type": None,
        "confidence": None,
        "denomination_btc": None,
        "participants_est": None,
        "reasons": [],
    }


def _make_result(cj_type: str, confidence: str,
                 denomination_btc: Optional[float],
                 participants_est: Optional[int],
                 reasons: list) -> dict:
    return {
        "is_coinjoin": True,
        "type": cj_type,
        "confidence": confidence,
        "denomination_btc": denomination_btc,
        "participants_est": participants_est,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Type detectors
# ---------------------------------------------------------------------------

def _detect_whirlpool_mix(inp: InputList, out: OutputList, cfg: dict) -> Optional[dict]:
    denoms = set(cfg["whirlpool_pool_denoms"])
    eps_max = cfg["whirlpool_eps_max"]
    if len(inp) != 5 or len(out) != 5:
        return None
    in_addrs = [a for a, _ in inp]
    out_addrs = [a for a, _ in out]
    if len(set(in_addrs)) != 5 or len(set(out_addrs)) != 5:
        return None
    out_vals = [v for _, v in out]
    if len(set(out_vals)) != 1:
        return None
    d = out_vals[0]
    if d not in denoms:
        return None
    above = sum(1 for _, v in inp if v > d)
    if above < 1 or above > 4:
        return None
    if any(v < d or v > d + eps_max for _, v in inp):
        return None
    return _make_result(
        "whirlpool", "high",
        d / 1e8, 5,
        [
            f"5 inputs and 5 outputs to distinct addresses",
            f"all outputs equal {d} sats (pool denomination)",
            f"every input in [{d}, {d + eps_max}] sats",
            f"{above} input(s) above denomination",
        ],
    )


def _detect_whirlpool_tx0(inp: InputList, out: OutputList,
                          has_op_return: bool, cfg: dict) -> Optional[dict]:
    if not has_op_return:
        return None
    denoms = cfg["whirlpool_pool_denoms"]
    eps_min = cfg["whirlpool_tx0_eps_min"]
    eps_max = cfg["whirlpool_tx0_eps_max"]
    max_premix = cfg["whirlpool_tx0_max_premix"]
    # group outputs by value
    val_counts = Counter(v for _, v in out)
    best = None
    for v, k in val_counts.items():
        for d in denoms:
            eps = v - d
            if eps_min <= eps <= eps_max:
                if best is None or k > best[0]:
                    best = (k, v, d)
    if best is None:
        return None
    k, v, d = best
    if k < 1 or k > max_premix:
        return None
    # other (non-premix) outputs in tx.outputs; OP_RETURN counts as 1 extra "other"
    others = len(out) - k
    if others + 1 > 3:  # +1 for the OP_RETURN
        return None
    return _make_result(
        "whirlpool_tx0", "medium",
        d / 1e8, k,
        [
            "has OP_RETURN output",
            f"{k} pre-mix output(s) of {v} sats (pool {d} + ε={v - d})",
            f"at most {others + 1} other output(s) (coordinator/change/OP_RETURN)",
        ],
    )


def _detect_wasabi2(inp: InputList, out: OutputList,
                    block_time: Optional[int], cfg: dict) -> Optional[dict]:
    denoms = cfg["wasabi2_denominations"]
    min_in = cfg["wasabi2_min_inputs"]
    if block_time is not None and block_time >= cfg["wasabi2_post2024_ts"]:
        min_in = cfg["wasabi2_min_inputs_post2024"]
    if len(inp) < min_in:
        return None
    if any(v < cfg["wasabi2_min_input_value"] for _, v in inp):
        return None
    out_addrs = [a for a, _ in out]
    if len(set(out_addrs)) != len(out_addrs):
        return None
    n_denom = sum(1 for _, v in out if v in denoms)
    if n_denom < (len(out) - 1) / 2:
        return None
    return _make_result(
        "wasabi2", "high",
        None, len(inp),
        [
            f"{len(inp)} inputs (≥ {min_in})",
            "all inputs ≥ 5_000 sats",
            f"{n_denom} output(s) in Wasabi 2.x standard denomination set",
            "all output addresses distinct",
        ],
    )


def _detect_wasabi1(inp: InputList, out: OutputList, cfg: dict) -> Optional[dict]:
    center = cfg["wasabi1_denom_center_sat"]
    tol = cfg["wasabi1_denom_tolerance"]
    out_addrs = [a for a, _ in out]
    if len(set(out_addrs)) != len(out_addrs):
        return None
    # most frequent value within ±tol of center
    val_counts = Counter(v for _, v in out)
    most_val, n = val_counts.most_common(1)[0]
    if abs(most_val - center) > center * tol:
        return None
    if n < (len(out) - 1) / 2:
        return None
    in_addrs = {a for a, _ in inp if a}
    if len(in_addrs) < n:
        return None
    return _make_result(
        "wasabi1", "medium",
        most_val / 1e8, n,
        [
            f"most frequent output {most_val} sats (within ±{int(tol * 100)}% of 0.1 BTC)",
            f"{n} output(s) at that value",
            f"{len(in_addrs)} distinct input addresses (≥ {n})",
            "all output addresses distinct",
        ],
    )


def _detect_joinmarket(inp: InputList, out: OutputList, cfg: dict) -> Optional[dict]:
    out_addrs = [a for a, _ in out]
    if len(set(out_addrs)) != len(out_addrs):
        return None
    val_counts = Counter(v for _, v in out)
    most_val, n = val_counts.most_common(1)[0]
    if n < cfg["joinmarket_min_equal_outputs"]:
        return None
    if len(out) > 2 * n:
        return None
    in_addrs = {a for a, _ in inp if a}
    if len(in_addrs) < n:
        return None
    return _make_result(
        "joinmarket", "medium",
        most_val / 1e8, n,
        [
            f"{n} equal output(s) of {most_val} sats (≥ 3)",
            f"≤ {2 * n} total outputs",
            f"{len(in_addrs)} distinct input addresses (≥ {n})",
            "all output addresses distinct",
        ],
    )


def _detect_generic(inp: InputList, out: OutputList, cfg: dict) -> Optional[dict]:
    out_addrs = [a for a, _ in out]
    val_counts = Counter(v for _, v in out)
    most_val, n = val_counts.most_common(1)[0]
    if n < cfg["generic_min_equal_outputs"]:
        return None
    # distinct addresses receiving that value
    distinct_recip = len({a for a, v in out if v == most_val and a})
    if distinct_recip < 3:
        return None
    in_addrs = {a for a, _ in inp if a}
    if len(in_addrs) < cfg["generic_min_distinct_inputs"]:
        return None
    if n < len(out) / 3:
        return None
    return _make_result(
        "generic", "low",
        most_val / 1e8, n,
        [
            f"{n} equal-value output(s) ({most_val} sats) to {distinct_recip} distinct addresses",
            f"{len(in_addrs)} distinct input addresses (≥ 3)",
            f"equal outputs are {n}/{len(out)} of all outputs (≥ 1/3)",
        ],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_coinjoin(tx, cfg: Optional[dict] = None) -> dict:
    """
    Detect whether a Bitcoin transaction is a CoinJoin.

    Args:
        tx: either a Transaction dataclass or an Esplora-shaped dict.
        cfg: optional override of constants; defaults to _DEFAULT_CFG.

    Returns:
        {
          is_coinjoin: bool,
          type: "whirlpool" | "whirlpool_tx0" | "wasabi2" | "wasabi1" |
                "joinmarket" | "generic" | None,
          confidence: "high" | "medium" | "low" | None,
          denomination_btc: float | None,
          participants_est: int | None,
          reasons: [str],
        }
    """
    c = {**_DEFAULT_CFG, **(cfg or {})}
    txid, inp, out, has_op_return, block_time = _extract(tx)

    # Cache lookup
    if txid and txid in _cache:
        return _cache[txid]

    # False-positive pre-filter: all inputs from one address → not a CoinJoin
    in_addrs = [a for a, _ in inp if a]
    result: dict

    if len(inp) > 0 and len(set(in_addrs)) <= 1 and len(inp) > 1:
        # single-address multi-input (exchange payout / consolidation / mining pool)
        result = _null_result()
        result["reasons"] = ["all inputs from a single address — not a CoinJoin"]
        if txid:
            _cache_put(txid, result)
        return result

    # Try detectors in priority order; first match wins.
    detector_chain = [
        ("whirlpool",     lambda: _detect_whirlpool_mix(inp, out, c)),
        ("whirlpool_tx0", lambda: _detect_whirlpool_tx0(inp, out, has_op_return, c)),
        ("wasabi2",       lambda: _detect_wasabi2(inp, out, block_time, c)),
        ("wasabi1",       lambda: _detect_wasabi1(inp, out, c)),
        ("joinmarket",    lambda: _detect_joinmarket(inp, out, c)),
        ("generic",       lambda: _detect_generic(inp, out, c)),
    ]

    result = _null_result()
    for _name, fn in detector_chain:
        r = fn()
        if r is not None:
            result = r
            break

    # Address-reuse downgrade: if input addresses reappear as outputs > 70%, downgrade.
    if result["is_coinjoin"]:
        out_addrs = {a for a, _ in out if a}
        in_addr_set = set(in_addrs)
        if in_addr_set:
            reuse = len(in_addr_set & out_addrs) / len(in_addr_set)
            if reuse > c["reuse_downgrade_threshold"]:
                old = result["confidence"]
                result["confidence"] = "low"
                result["reasons"].append(
                    f"address reuse {int(reuse * 100)}% between inputs/outputs — "
                    f"downgraded from {old}"
                )

    if txid:
        _cache_put(txid, result)
    return result


def coinjoin_badge_label(cj: dict) -> Optional[str]:
    """Short human label for a coinjoin result, for UI badges."""
    if not cj or not cj.get("is_coinjoin"):
        return None
    t = cj.get("type")
    denom = cj.get("denomination_btc")
    conf = cj.get("confidence")
    if t == "whirlpool" and denom:
        return f"Whirlpool {denom:g} BTC"
    if t == "whirlpool_tx0":
        return "Whirlpool Tx0"
    if t == "wasabi2":
        return "Wasabi 2.x"
    if t == "wasabi1":
        return "Wasabi 1.x"
    if t == "joinmarket":
        return "JoinMarket"
    if t == "generic":
        return "Possible CoinJoin"
    return "CoinJoin"
