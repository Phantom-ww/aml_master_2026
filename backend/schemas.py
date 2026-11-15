from pydantic import BaseModel
from typing import List, Optional, Dict, Literal

# ============================================================================
# GRAPH SCHEMAS
# ============================================================================

class GraphNode(BaseModel):
    id: str
    label: str
    x: float
    y: float
    node_type: str = "wallet"
    risk_score: Optional[float] = None  # None for unscored chain data
    pinned: bool = False
    address: Optional[str] = None
    owner: Optional[str] = None
    balance_crypto: Optional[float] = None
    balance_usd: Optional[float] = None

    # Chain-specific fields
    data_source: Literal["ellipticpp", "chain"] = "ellipticpp"
    balance_sat: Optional[int] = None
    tx_count: Optional[int] = None
    ellipticpp_class: Optional[int] = None  # 1=illicit, 2=licit from Elliptic++
    known_label: Optional[int] = None  # 1=illicit, 0=licit, null=unknown (for Elliptic++)
    model_risk: Optional[float] = None  # Model-predicted risk probability (0-1)

class GraphEdge(BaseModel):
    id: str  # transaction ID
    source: str  # source node id
    target: str  # target node id
    amount: float = 0.0
    risk_score: Optional[float] = None  # None for unscored chain data
    risk_color: str = "#999"
    time_step: int = 0
    tx_address: Optional[str] = None
    from_wallet: Optional[str] = None
    to_wallet: Optional[str] = None
    timestamp: Optional[str] = None
    amount_crypto: Optional[float] = None
    amount_usd: Optional[float] = None

    # Chain-specific fields
    data_source: Literal["ellipticpp", "chain"] = "ellipticpp"
    block_height: Optional[int] = None
    confirmed: Optional[bool] = None
    is_change: Optional[bool] = None
    large_tx: Optional[bool] = None
    tx_class: Optional[int] = None  # Elliptic++ transaction class
    tx_risk: Optional[float] = None  # Transaction risk from model (0-1)
    # CoinJoin detection
    is_coinjoin: Optional[bool] = None
    coinjoin_type: Optional[str] = None  # e.g. "whirlpool", "wasabi2", …

class SubgraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_node: str
    truncated: bool = False
    source: Literal["ellipticpp", "chain"] = "ellipticpp"  # data source
    legacy: bool = False  # True for old investigations containing demo data
    model: Optional[Dict] = None  # Model scoring block (xgb/gnn), when available
    sanctions: Optional[Dict] = None  # OFAC sanctions info for chain wallet

class FeatureContribution(BaseModel):
    feature_name: str
    value: float
    shap_value: float
    contribution_type: str  # "increases_risk" or "decreases_risk"

class TransactionSummary(BaseModel):
    txid: str
    amount_btc: float
    timestamp: Optional[str] = None
    confirmed: bool
    block_height: Optional[int] = None
    is_self_send: bool = False
    direction: Literal["in", "out", "self"]
    coinjoin: Optional[Dict] = None  # CoinJoin detection result

class NodeDetailsResponse(BaseModel):
    node_id: str
    node_type: str = "wallet"
    address: str
    owner: Optional[str] = None
    balance_crypto: float
    balance_usd: float
    risk_score: Optional[float] = None
    connected_edges: List[Dict] = []

    # Chain-specific
    balance_sat: Optional[int] = None
    tx_count: Optional[int] = None
    confirmed_tx_count: Optional[int] = None
    mempool_tx_count: Optional[int] = None
    has_op_return: Optional[bool] = None
    transactions: List[TransactionSummary] = []
    next_cursor: Optional[str] = None  # txid for pagination
    ellipticpp_class: Optional[int] = None  # 1=illicit, 2=licit from Elliptic++

class TransactionPageResponse(BaseModel):
    """Paginated transaction response for chain addresses"""
    transactions: List[TransactionSummary]
    next_cursor: Optional[str] = None  # txid of next page start
    has_more: bool  # True if more transactions available

class EdgeDetailsResponse(BaseModel):
    tx_id: str
    tx_address: str
    source: str
    target: str
    from_wallet: str
    to_wallet: str
    timestamp: str
    amount: float
    amount_crypto: float
    amount_usd: float
    risk_score: Optional[float] = None
    gnn_score: Optional[float] = None
    time_step: int
    top_features: List[FeatureContribution]

    # Chain-specific
    block_height: Optional[int] = None
    confirmed: Optional[bool] = None
    is_change: Optional[bool] = None
    large_tx: Optional[bool] = None
    op_return_data: Optional[str] = None
    burned_value: Optional[int] = None

class TxInputSummary(BaseModel):
    """Summary of a transaction input"""
    address: Optional[str] = None
    value_btc: float
    is_coinbase: bool = False

class TxOutputSummary(BaseModel):
    """Summary of a transaction output"""
    address: Optional[str] = None
    value_btc: float
    vout: int
    is_change: bool = False

class ChainTxDetailsResponse(BaseModel):
    """Detailed transaction information for chain transactions"""
    # Identity
    txid: str
    confirmed: bool
    block_height: Optional[int] = None
    block_time: Optional[str] = None  # ISO format
    confirmations: int

    # Amounts
    total_in_btc: float
    total_out_btc: float
    fee_btc: float
    fee_rate_sat_vb: Optional[float] = None
    input_count: int
    output_count: int

    # Inputs and outputs
    inputs: List[TxInputSummary]
    outputs: List[TxOutputSummary]

    # OP_RETURN
    has_op_return: bool = False
    op_return_data: Optional[str] = None
    burned_value_btc: float = 0.0

    # CoinJoin detection
    coinjoin: Optional[Dict] = None

# ============================================================================
# INVESTIGATION SCHEMAS
# ============================================================================

class CameraSnapshot(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0

class InvestigationSnapshot(BaseModel):
    version: int = 1
    timestamp: str
    threshold: float
    selected_node_id: Optional[str] = None
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    camera: CameraSnapshot
    analyst_notes: str = ""

class SaveInvestigationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    snapshot: InvestigationSnapshot
    analyst_notes: str = ""

class InvestigationResponse(BaseModel):
    id: str
    name: str
    created_at: str
    analyst_notes: str
    node_count: int
    edge_count: int = 0

# ============================================================================
# CHAIN SCHEMAS
# ============================================================================

class ChainStatusResponse(BaseModel):
    network: str
    tip_height: Optional[int] = None
    esplora_reachable: bool
    cache_stats: Dict
    latency_ms: Optional[float] = None
