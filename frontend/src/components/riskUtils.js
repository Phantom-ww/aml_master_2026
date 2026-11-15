// Shared helpers for model fraud probabilities (wallets and transactions).

export const RISK_COLORS = {
  none: '#374151',   // not scored
  low: '#6B7280',    // < 20%
  medium: '#FBBF24', // 20–50%
  high: '#F97316',   // 50–80%
  critical: '#EF4444', // >= 80%
};

export const LABEL_COLORS = { illicit: '#EF4444', licit: '#10B981', unknown: '#6B7280' };

export const MIXING_COLORS = {
  cj_edge: '#9333EA',      // purple for CoinJoin edges
  cj_node: '#7E22CE',      // darker purple for CoinJoin nodes
  cj_node_stroke: '#D8B4FE', // light purple stroke
  ofac_ring: '#EF4444',    // red ring for OFAC-listed nodes
  neutral_edge: '#4B5563',
  neutral_node: '#374151',
};

export function riskColor(p) {
  if (p == null || Number.isNaN(p)) return RISK_COLORS.none;
  if (p >= 0.8) return RISK_COLORS.critical;
  if (p >= 0.5) return RISK_COLORS.high;
  if (p >= 0.2) return RISK_COLORS.medium;
  return RISK_COLORS.low;
}

export function formatProb(p) {
  if (p == null || Number.isNaN(p)) return '—';
  if (p < 0.01) return '<1%';
  if (p > 0.99) return '>99%';
  return `${Math.round(p * 100)}%`;
}

export function labelColor(d = {}) {
  if (d.ellipticpp_class === 1 || d.known_label === 1) return LABEL_COLORS.illicit;
  if (d.ellipticpp_class === 2 || d.known_label === 0) return LABEL_COLORS.licit;
  return LABEL_COLORS.unknown;
}

/** True if any edge in txData is a CoinJoin. */
export function nodeHasCoinjoin(nodeId, edges) {
  return edges?.some(e =>
    e.is_coinjoin && (e.source === nodeId || e.target === nodeId)
  ) ?? false;
}

/** True if the coinjoin result object indicates detection. */
export function isCoinJoin(cj) {
  return cj?.is_coinjoin === true;
}

/** Short badge label: "Whirlpool 0.01 BTC", "Wasabi 2.x", "Possible CoinJoin", etc. */
export function coinJoinLabel(cj) {
  if (!isCoinJoin(cj)) return null;
  const t = cj.type;
  const d = cj.denomination_btc;
  if (t === 'whirlpool' && d != null) return `Whirlpool ${+d.toFixed(8).replace(/\.?0+$/, '')} BTC`;
  if (t === 'whirlpool_tx0') return 'Whirlpool Tx0';
  if (t === 'wasabi2') return 'Wasabi 2.x';
  if (t === 'wasabi1') return 'Wasabi 1.x';
  if (t === 'joinmarket') return 'JoinMarket';
  if (t === 'generic') return 'Possible CoinJoin';
  return 'CoinJoin';
}
