// Shared helpers for model fraud probabilities (wallets and transactions).

export const RISK_COLORS = {
  none: '#374151',   // not scored
  low: '#6B7280',    // < 20%
  medium: '#FBBF24', // 20–50%
  high: '#F97316',   // 50–80%
  critical: '#EF4444', // >= 80%
};

export const LABEL_COLORS = { illicit: '#EF4444', licit: '#10B981', unknown: '#6B7280' };

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
