"""Test transaction direction and amount calculation"""

import pytest
from chain.protocol import Transaction, TxInput, TxOutput


def test_incoming_tx():
    """Test: not in inputs → 'in', amount = received"""
    tx = Transaction(
        txid="incoming_tx",
        inputs=[
            TxInput(txid="prev", vout=0, address="other_address", value_sat=100000, is_coinbase=False)
        ],
        outputs=[
            TxOutput(vout=0, address="my_address", value_sat=50000, spent=False),
            TxOutput(vout=1, address="another_address", value_sat=50000, spent=False)
        ],
        fee_sat=0,
        block_height=100,
        block_time=1609459200,
        confirmations=10
    )

    address = "my_address"
    sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
    received = sum(out.value_sat for out in tx.outputs if out.address == address)
    to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)
    is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

    assert not is_in_inputs, "Should not be in inputs"
    assert received == 50000, "Should receive 50000 sat"
    direction = "in"
    amount_sat = received

    assert direction == "in"
    assert amount_sat == 50000


def test_outgoing_with_change():
    """Test: in inputs and to_others > 0 → 'out', amount = sent - received (own contribution)"""
    tx = Transaction(
        txid="outgoing_tx",
        inputs=[
            TxInput(txid="prev", vout=0, address="my_address", value_sat=100000, is_coinbase=False)
        ],
        outputs=[
            TxOutput(vout=0, address="recipient", value_sat=70000, spent=False),
            TxOutput(vout=1, address="my_address", value_sat=29000, spent=False)  # change
        ],
        fee_sat=1000,
        block_height=100,
        block_time=1609459200,
        confirmations=10
    )

    address = "my_address"
    sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
    received = sum(out.value_sat for out in tx.outputs if out.address == address)
    to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)
    is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

    assert is_in_inputs, "Should be in inputs"
    assert sent == 100000, "Should send 100000 sat"
    assert received == 29000, "Should receive 29000 sat as change"
    assert to_others == 70000, "70000 sat goes to others"

    direction = "out"
    amount_sat = sent - received  # Own contribution: 100000 - 29000 = 71000 (includes fee)

    assert direction == "out"
    assert amount_sat == 71000  # Own contribution


def test_multi_input_tx_small_contribution():
    """Test: multi-input tx where my address contributes a small input"""
    tx = Transaction(
        txid="multi_input_tx",
        inputs=[
            TxInput(txid="prev1", vout=0, address="other_address_1", value_sat=5000000, is_coinbase=False),
            TxInput(txid="prev2", vout=0, address="my_address", value_sat=10000, is_coinbase=False),
            TxInput(txid="prev3", vout=0, address="other_address_2", value_sat=7000000, is_coinbase=False)
        ],
        outputs=[
            TxOutput(vout=0, address="recipient", value_sat=12000000, spent=False),
            TxOutput(vout=1, address="my_address", value_sat=5000, spent=False)  # change
        ],
        fee_sat=5000,
        block_height=100,
        block_time=1609459200,
        confirmations=10
    )

    address = "my_address"
    sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
    received = sum(out.value_sat for out in tx.outputs if out.address == address)
    to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)
    is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

    assert is_in_inputs, "Should be in inputs"
    assert sent == 10000, "Should contribute 10000 sat"
    assert received == 5000, "Should receive 5000 sat as change"
    assert to_others == 12000000, "12M sat goes to others (NOT my contribution)"

    direction = "out"
    amount_sat = sent - received  # Own contribution: 10000 - 5000 = 5000

    assert direction == "out"
    assert amount_sat == 5000  # Only my contribution, NOT the total to_others


def test_consolidation():
    """Test: in inputs and to_others == 0 → 'self', amount = fee paid"""
    tx = Transaction(
        txid="consolidation_tx",
        inputs=[
            TxInput(txid="prev1", vout=0, address="my_address", value_sat=10000, is_coinbase=False),
            TxInput(txid="prev2", vout=0, address="my_address", value_sat=15000, is_coinbase=False)
        ],
        outputs=[
            TxOutput(vout=0, address="my_address", value_sat=24500, spent=False)
        ],
        fee_sat=500,
        block_height=100,
        block_time=1609459200,
        confirmations=10
    )

    address = "my_address"
    sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
    received = sum(out.value_sat for out in tx.outputs if out.address == address)
    to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)
    is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

    assert is_in_inputs, "Should be in inputs"
    assert sent == 25000, "Should send 25000 sat total"
    assert received == 24500, "Should receive 24500 sat"
    assert to_others == 0, "No outputs to others"

    direction = "self"
    amount_sat = sent - received  # Fee paid: 25000 - 24500 = 500

    assert direction == "self"
    assert amount_sat == 500  # Fee paid


def test_coinbase_tx():
    """Test: coinbase transaction"""
    tx = Transaction(
        txid="coinbase_tx",
        inputs=[
            TxInput(txid=None, vout=None, address=None, value_sat=0, is_coinbase=True)
        ],
        outputs=[
            TxOutput(vout=0, address="my_address", value_sat=5000000000, spent=False)
        ],
        fee_sat=0,
        block_height=100,
        block_time=1609459200,
        confirmations=10
    )

    address = "my_address"
    sent = sum(inp.value_sat for inp in tx.inputs if inp.address == address)
    received = sum(out.value_sat for out in tx.outputs if out.address == address)
    to_others = sum(out.value_sat for out in tx.outputs if out.address and out.address != address)
    is_in_inputs = any(inp.address == address for inp in tx.inputs if inp.address)

    assert not is_in_inputs, "Coinbase has no regular inputs"
    assert received == 5000000000, "Should receive coinbase reward"

    direction = "in"
    amount_sat = received

    assert direction == "in"
    assert amount_sat == 5000000000
