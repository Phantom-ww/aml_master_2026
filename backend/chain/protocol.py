"""Protocol data classes for blockchain data (network-agnostic)"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class TxInput:
    """Single transaction input (prevout reference)"""
    txid: Optional[str]  # None for coinbase
    vout: Optional[int]  # None for coinbase
    address: Optional[str]  # Resolved from prevout
    value_sat: int
    is_coinbase: bool = False


@dataclass
class TxOutput:
    """Single transaction output"""
    vout: int
    address: Optional[str]  # None for OP_RETURN, unspendable scripts
    value_sat: int
    spent: bool = False  # From outspends endpoint


@dataclass
class Transaction:
    """Bitcoin transaction with resolved inputs/outputs"""
    txid: str
    inputs: List[TxInput]
    outputs: List[TxOutput]
    fee_sat: int
    block_height: Optional[int]  # None if unconfirmed
    block_time: Optional[int]  # Unix timestamp
    confirmations: int

    # Transaction weight/size for fee rate calculation
    weight: Optional[int] = None
    size: Optional[int] = None
    vsize: Optional[int] = None

    # Metadata
    op_return_data: Optional[str] = None  # Hex data from OP_RETURN outputs
    has_op_return: bool = False
    burned_value_sat: int = 0  # Value in OP_RETURN outputs with value > 0


@dataclass
class Address:
    """Address summary with transaction history"""
    address: str
    balance_sat: int  # From chain_stats (funded - spent)
    tx_count: int  # Total (confirmed + mempool)
    txs: List[Transaction] = field(default_factory=list)

    # From Esplora chain_stats and mempool_stats
    funded_sat: int = 0
    spent_sat: int = 0
    confirmed_tx_count: int = 0
    mempool_tx_count: int = 0


@dataclass
class Subgraph:
    """K-hop subgraph result with wallet nodes and transaction edges"""

    # Nodes: unique addresses encountered
    addresses: Dict[str, Address]  # address -> Address

    # Edges: transactions between addresses in the subgraph
    transactions: Dict[str, Transaction]  # txid -> Transaction

    # Graph metadata
    center_address: str
    k: int
    truncated: bool = False  # True if hit node/tx limits
    source: str = "chain"
