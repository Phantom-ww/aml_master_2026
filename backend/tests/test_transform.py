"""Tests for chain/transform.py UTXO to graph transformation"""

import pytest
from datetime import datetime
from chain.protocol import Transaction, TxInput, TxOutput, Address
from chain.transform import (
    largest_remainder_allocation,
    transaction_to_edges,
    subgraph_to_graph_nodes_edges,
)


class TestLargestRemainderAllocation:
    """Test proportional allocation without rounding loss"""

    def test_equal_distribution_zero_values(self):
        result = largest_remainder_allocation([0, 0, 0], 1000)
        assert sum(result) == 1000
        # Should distribute evenly
        assert result == [334, 333, 333] or result == [333, 334, 333] or result == [333, 333, 334]

    def test_proportional_exact(self):
        # 50:30:20 split of 100
        result = largest_remainder_allocation([50, 30, 20], 100)
        assert result == [50, 30, 20]
        assert sum(result) == 100

    def test_proportional_with_remainder(self):
        # 1:1:1 split of 10 satoshis
        result = largest_remainder_allocation([1, 1, 1], 10)
        assert sum(result) == 10
        # Largest remainder goes to first items
        assert sorted(result, reverse=True) == [4, 3, 3]

    def test_zero_total(self):
        result = largest_remainder_allocation([10, 20, 30], 0)
        assert result == [0, 0, 0]

    def test_empty_values(self):
        result = largest_remainder_allocation([], 1000)
        assert result == []


class TestTransactionToEdges:
    """Test UTXO transaction to address graph edge conversion"""

    def test_coinbase_transaction(self):
        # Coinbase: 1 input (coinbase) -> 2 outputs
        tx = Transaction(
            txid="cb123",
            inputs=[
                TxInput(
                    txid="0000000000000000000000000000000000000000000000000000000000000000",
                    vout=4294967295,
                    address=None,
                    value_sat=0,
                    is_coinbase=True,
                )
            ],
            outputs=[
                TxOutput(vout=0, address="miner1", value_sat=5000000000),
                TxOutput(vout=1, address="miner2", value_sat=50000000),
            ],
            block_height=100,
            block_time=datetime(2024, 1, 1),
            confirmations=6,
            fee_sat=0,
        )

        edges = transaction_to_edges(tx)
        assert len(edges) == 2
        assert all(e["source"] == "coinbase" for e in edges)
        assert {e["target"] for e in edges} == {"miner1", "miner2"}
        assert edges[0]["is_coinbase"] == True
        assert edges[0]["amount_sat"] == 5000000000

    def test_simple_p2p_transfer(self):
        # Alice -> Bob (single input, single output)
        tx = Transaction(
            txid="tx123",
            inputs=[TxInput(txid="prev", vout=0, address="alice", value_sat=100000, is_coinbase=False)],
            outputs=[TxOutput(vout=0, address="bob", value_sat=99000)],
            block_height=200,
            block_time=datetime(2024, 2, 1),
            confirmations=3,
            fee_sat=1000,
        )

        edges = transaction_to_edges(tx)
        assert len(edges) == 1
        assert edges[0]["source"] == "alice"
        assert edges[0]["target"] == "bob"
        assert edges[0]["amount_sat"] == 99000
        assert edges[0]["is_change"] == False

    def test_change_detection(self):
        # Alice sends to Bob, change back to Alice
        tx = Transaction(
            txid="tx456",
            inputs=[TxInput(txid="prev", vout=0, address="alice", value_sat=200000, is_coinbase=False)],
            outputs=[
                TxOutput(vout=0, address="bob", value_sat=50000),
                TxOutput(vout=1, address="alice", value_sat=149000),  # change
            ],
            block_height=300,
            block_time=datetime(2024, 3, 1),
            confirmations=10,
            fee_sat=1000,
        )

        edges = transaction_to_edges(tx, include_change=False)
        # Change outputs are excluded (self-loop)
        assert len(edges) == 1
        assert edges[0]["target"] == "bob"

        # With include_change
        edges_with_change = transaction_to_edges(tx, include_change=True)
        assert len(edges_with_change) == 2
        change_edge = [e for e in edges_with_change if e["target"] == "alice"][0]
        assert change_edge["is_change"] == True

    def test_multi_input_proportional_allocation(self):
        # 2 inputs -> 1 output (needs proportional split)
        tx = Transaction(
            txid="tx789",
            inputs=[
                TxInput(txid="p1", vout=0, address="alice", value_sat=100000, is_coinbase=False),
                TxInput(txid="p2", vout=0, address="bob", value_sat=50000, is_coinbase=False),
            ],
            outputs=[TxOutput(vout=0, address="charlie", value_sat=149000)],
            block_height=400,
            block_time=datetime(2024, 4, 1),
            confirmations=1,
            fee_sat=1000,
        )

        edges = transaction_to_edges(tx)
        assert len(edges) == 2

        alice_edge = [e for e in edges if e["source"] == "alice"][0]
        bob_edge = [e for e in edges if e["source"] == "bob"][0]

        # Alice contributed 100k out of 150k total in = 2/3
        # Bob contributed 50k = 1/3
        # Should allocate 149k output proportionally
        assert alice_edge["amount_sat"] + bob_edge["amount_sat"] == 149000
        # Alice gets ~2/3 * 149000 = ~99333
        assert 99000 <= alice_edge["amount_sat"] <= 100000

    def test_op_return_no_edge(self):
        # Output with OP_RETURN should not create edge
        tx = Transaction(
            txid="tx_op",
            inputs=[TxInput(txid="prev", vout=0, address="alice", value_sat=50000, is_coinbase=False)],
            outputs=[
                TxOutput(vout=0, address="bob", value_sat=45000),
                TxOutput(vout=1, address=None, value_sat=0),
            ],
            block_height=500,
            block_time=datetime(2024, 5, 1),
            confirmations=12,
            fee_sat=5000,
        )

        edges = transaction_to_edges(tx)
        assert len(edges) == 1
        assert edges[0]["target"] == "bob"

    def test_large_transaction_focal_truncation(self):
        # 100 inputs * 100 outputs = 10k fan-out, exceeds max_fanout=200
        inputs = [
            TxInput(txid=f"prev{i}", vout=0, address=f"in{i}", value_sat=1000, is_coinbase=False)
            for i in range(100)
        ]
        inputs[0] = TxInput(txid="prev_focal", vout=0, address="focal_addr", value_sat=5000, is_coinbase=False)

        outputs = [
            TxOutput(vout=i, address=f"out{i}", value_sat=500)
            for i in range(100)
        ]

        tx = Transaction(
            txid="large_tx",
            inputs=inputs,
            outputs=outputs,
            block_height=600,
            block_time=datetime(2024, 6, 1),
            confirmations=20,
            fee_sat=1000,
        )

        # With focal_address, should connect only focal edges
        edges = transaction_to_edges(tx, focal_address="focal_addr", max_fanout=200)
        assert len(edges) < 10000  # Truncated
        assert all(e["large_tx"] == True for e in edges)
        assert all(e["source"] == "focal_addr" for e in edges)


class TestSubgraphConversion:
    """Integration test for full subgraph conversion"""

    def test_simple_subgraph_with_layout(self):
        from chain.protocol import Subgraph

        # Build a simple 2-node graph
        tx1 = Transaction(
            txid="tx1",
            inputs=[TxInput(txid="p", vout=0, address="addr1", value_sat=100000, is_coinbase=False)],
            outputs=[TxOutput(vout=0, address="addr2", value_sat=99000)],
            block_height=100,
            block_time=datetime(2024, 1, 1).timestamp(),
            confirmations=10,
            fee_sat=1000,
        )

        addr1 = Address(
            address="addr1",
            balance_sat=100000,
            tx_count=1,
            txs=[tx1],
        )

        addr2 = Address(address="addr2", balance_sat=99000, tx_count=1, txs=[])

        subgraph = Subgraph(
            center_address="addr1",
            addresses={"addr1": addr1, "addr2": addr2},
            transactions={"tx1": tx1},
            truncated=False,
            k=1,
        )

        nodes, edges = subgraph_to_graph_nodes_edges(subgraph, include_change=False, max_fanout=200)

        assert len(nodes) == 2
        assert {n.id for n in nodes} == {"addr1", "addr2"}
        assert len(edges) == 1
        assert edges[0].source == "addr1"
        assert edges[0].target == "addr2"
        assert edges[0].amount == 0.00099  # 99000 sat -> BTC
        assert edges[0].data_source == "chain"

        # Check layout applied (non-zero positions)
        assert any(n.x != 0.0 or n.y != 0.0 for n in nodes)
