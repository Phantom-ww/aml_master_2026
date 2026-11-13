"""Transform UTXO transactions to address-to-address graph edges"""

from typing import List, Dict, Tuple, Set, Optional
from .protocol import Transaction, TxInput, TxOutput
from schemas import GraphNode, GraphEdge


def largest_remainder_allocation(values: List[int], total: int) -> List[int]:
    """
    Allocate `total` across `values` proportionally without rounding loss.
    Uses largest-remainder method to ensure sum(result) == total exactly.
    """
    if total == 0 or not values:
        return [0] * len(values)

    sum_values = sum(values)
    if sum_values == 0:
        # Equal distribution
        base = total // len(values)
        remainder = total % len(values)
        result = [base] * len(values)
        for i in range(remainder):
            result[i] += 1
        return result

    # Proportional allocation with remainders
    proportions = [v / sum_values for v in values]
    allocated = [int(p * total) for p in proportions]
    remainder = total - sum(allocated)

    # Distribute remainder to largest fractional parts
    fractional_parts = [(p * total) % 1 for p in proportions]
    indices_by_fraction = sorted(
        range(len(fractional_parts)), key=lambda i: fractional_parts[i], reverse=True
    )

    for i in range(remainder):
        allocated[indices_by_fraction[i]] += 1

    return allocated


def transaction_to_edges(
    tx: Transaction,
    focal_address: Optional[str] = None,
    include_change: bool = False,
    max_fanout: int = 200,
) -> List[Dict]:
    """
    Convert a UTXO transaction to address-to-address edges.

    Rules:
    - Coinbase: create edges from synthetic "coinbase" node
    - Change detection: outputs to any input address are marked is_change=True
    - Self-loops: not drawn by default (configurable via include_change)
    - OP_RETURN: no edge, recorded in tx metadata
    - Large tx: if inputs×outputs > max_fanout, connect only focal_address
    - Proportional allocation: largest-remainder to avoid rounding loss

    Returns list of edge dicts with:
        source, target, amount_sat, tx_id, is_change, large_tx, block_height,
        block_time, confirmed
    """
    edges = []

    # Detect change (outputs to input addresses)
    input_addresses = {inp.address for inp in tx.inputs if inp.address}
    change_outputs = {
        out.vout for out in tx.outputs if out.address in input_addresses
    }

    # Non-change outputs (or all outputs if include_change=True)
    if include_change:
        spending_outputs = [out for out in tx.outputs if out.address]  # All with address
    else:
        spending_outputs = [out for out in tx.outputs if out.vout not in change_outputs]

    if not spending_outputs:
        # All outputs are change (or OP_RETURN only) - no edges
        return []

    # Check for large transaction
    non_coinbase_inputs = [inp for inp in tx.inputs if not inp.is_coinbase]
    fanout = len(non_coinbase_inputs) * len(spending_outputs)
    is_large_tx = fanout > max_fanout

    # Coinbase handling
    if any(inp.is_coinbase for inp in tx.inputs):
        # All outputs from coinbase
        for out in spending_outputs:
            if out.address:
                edges.append(
                    {
                        "source": "coinbase",
                        "target": out.address,
                        "amount_sat": out.value_sat,
                        "tx_id": tx.txid,
                        "is_change": False,
                        "is_coinbase": True,
                        "large_tx": False,
                        "block_height": tx.block_height,
                        "block_time": tx.block_time,
                        "confirmed": tx.confirmations > 0,
                    }
                )
        return edges

    # Large transaction optimization (only connect focal address)
    if is_large_tx and focal_address:
        focal_inputs = [inp for inp in non_coinbase_inputs if inp.address == focal_address]
        focal_outputs = [out for out in spending_outputs if out.address == focal_address]

        if focal_inputs:
            # focal_address is an input - connect to all outputs
            total_focal_in = sum(inp.value_sat for inp in focal_inputs)
            output_values = [out.value_sat for out in spending_outputs]
            allocated = largest_remainder_allocation(output_values, total_focal_in)

            for out, amount in zip(spending_outputs, allocated):
                if out.address and amount > 0:
                    edges.append(
                        {
                            "source": focal_address,
                            "target": out.address,
                            "amount_sat": amount,
                            "tx_id": tx.txid,
                            "is_change": out.vout in change_outputs,
                            "is_coinbase": False,
                            "large_tx": True,
                            "block_height": tx.block_height,
                            "block_time": tx.block_time,
                            "confirmed": tx.confirmations > 0,
                        }
                    )

        elif focal_outputs:
            # focal_address is an output - connect from all inputs
            total_out = sum(out.value_sat for out in spending_outputs)
            input_values = [inp.value_sat for inp in non_coinbase_inputs]
            allocated = largest_remainder_allocation(input_values, total_out)

            for inp, amount in zip(non_coinbase_inputs, allocated):
                if inp.address:
                    focal_amount = (amount * focal_outputs[0].value_sat) // total_out
                    if focal_amount > 0:
                        edges.append(
                            {
                                "source": inp.address,
                                "target": focal_address,
                                "amount_sat": focal_amount,
                                "tx_id": tx.txid,
                                "is_change": focal_outputs[0].vout in change_outputs,
                                "is_coinbase": False,
                                "large_tx": True,
                                "block_height": tx.block_height,
                                "block_time": tx.block_time,
                                "confirmed": tx.confirmations > 0,
                            }
                        )

        return edges

    # Normal transaction: full input×output mesh with proportional allocation
    for out in spending_outputs:
        if not out.address:
            continue

        # Proportionally allocate this output across all inputs
        input_values = [inp.value_sat for inp in non_coinbase_inputs]
        allocated = largest_remainder_allocation(input_values, out.value_sat)

        for inp, amount in zip(non_coinbase_inputs, allocated):
            if inp.address and amount > 0:
                # Skip self-loops unless include_change is True
                is_self_loop = inp.address == out.address
                is_change_edge = out.vout in change_outputs

                if is_self_loop and not include_change:
                    continue

                edges.append(
                    {
                        "source": inp.address,
                        "target": out.address,
                        "amount_sat": amount,
                        "tx_id": tx.txid,
                        "is_change": is_change_edge,
                        "is_coinbase": False,
                        "large_tx": False,
                        "block_height": tx.block_height,
                        "block_time": tx.block_time,
                        "confirmed": tx.confirmations > 0,
                    }
                )

    return edges


def subgraph_to_graph_nodes_edges(
    subgraph,
    include_change: bool = False,
    max_fanout: int = 200,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """
    Convert chain Subgraph to frontend GraphNode/GraphEdge format.
    Uses igraph layout for positioning.
    """
    import igraph as ig
    from datetime import datetime

    nodes = []
    edges = []

    # Add coinbase node if any coinbase txs
    has_coinbase = any(
        any(inp.is_coinbase for inp in tx.inputs)
        for tx in subgraph.transactions.values()
    )

    if has_coinbase:
        nodes.append(
            GraphNode(
                id="coinbase",
                label="Coinbase",
                x=0.0,
                y=0.0,
                node_type="coinbase",
                risk_score=None,
                pinned=False,
                address="coinbase",
                owner=None,
                balance_crypto=None,
                balance_usd=None,
                data_source="chain",
                balance_sat=None,
                tx_count=0,
            )
        )

    # Address nodes
    for addr_id, addr_data in subgraph.addresses.items():
        nodes.append(
            GraphNode(
                id=addr_id,
                label=addr_id[:12] + "...",
                x=0.0,  # Positioned by layout below
                y=0.0,
                node_type="wallet",
                risk_score=None,  # Chain data has no ML scores
                pinned=False,
                address=addr_id,
                owner=None,
                balance_crypto=addr_data.balance_sat / 1e8 if addr_data.balance_sat is not None else None,
                balance_usd=None,  # No price data
                data_source="chain",
                balance_sat=addr_data.balance_sat,
                tx_count=addr_data.tx_count,
            )
        )

    # Compute layout
    if len(nodes) > 1:
        node_ids = [n.id for n in nodes]
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edge list for layout
        edge_list = []
        for tx in subgraph.transactions.values():
            tx_edges = transaction_to_edges(
                tx,
                focal_address=subgraph.center_address,
                include_change=include_change,
                max_fanout=max_fanout,
            )
            for e in tx_edges:
                if e["source"] in node_id_to_idx and e["target"] in node_id_to_idx:
                    edge_list.append(
                        (node_id_to_idx[e["source"]], node_id_to_idx[e["target"]])
                    )

        # igraph layout
        g = ig.Graph(n=len(node_ids), edges=edge_list, directed=True)
        layout = g.layout_fruchterman_reingold(niter=500)

        for i, node in enumerate(nodes):
            node.x = float(layout[i][0]) * 100
            node.y = float(layout[i][1]) * 100

    # Convert transactions to edges
    for tx in subgraph.transactions.values():
        tx_edges = transaction_to_edges(
            tx,
            focal_address=subgraph.center_address,
            include_change=include_change,
            max_fanout=max_fanout,
        )

        # Deduplicate and aggregate by (txid, source, target)
        edge_map = {}
        for edge_data in tx_edges:
            key = (edge_data['tx_id'], edge_data['source'], edge_data['target'])
            if key not in edge_map:
                edge_map[key] = edge_data
            else:
                # Aggregate: sum amounts
                edge_map[key]['amount_sat'] += edge_data['amount_sat']

        for edge_data in edge_map.values():
            edge_id = f"{edge_data['tx_id']}:{edge_data['source']}:{edge_data['target']}"

            timestamp_iso = None
            if edge_data["block_time"]:
                timestamp_iso = datetime.utcfromtimestamp(edge_data["block_time"]).isoformat()

            edges.append(
                GraphEdge(
                    id=edge_id,
                    source=edge_data["source"],
                    target=edge_data["target"],
                    amount=edge_data["amount_sat"] / 1e8,  # Convert to BTC
                    risk_score=None,
                    risk_color="#6b7280",  # Neutral gray
                    time_step=0,  # No time_step for chain data
                    tx_address=edge_data["tx_id"],
                    from_wallet=edge_data["source"],
                    to_wallet=edge_data["target"],
                    timestamp=timestamp_iso,
                    amount_crypto=edge_data["amount_sat"] / 1e8,
                    amount_usd=None,
                    data_source="chain",
                    block_height=edge_data.get("block_height"),
                    confirmed=edge_data.get("confirmed"),
                    is_change=edge_data.get("is_change"),
                    large_tx=edge_data.get("large_tx"),
                )
            )

    return nodes, edges
