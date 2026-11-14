import { useState } from 'react';
import FeatureChart from './FeatureChart';
import { formatProb, riskColor } from './riskUtils';
import apiClient from '../api/client';

function RiskRow({ name, hint, result, decomposition }) {
  if (!result) {
    return (
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400" title={hint}>{name}</span>
        <span className="text-gray-500">n/a</span>
      </div>
    );
  }
  const color = riskColor(result.prob);
  const thresholdTitle = decomposition
    ? `Threshold ≈ ${formatProb(decomposition.threshold_calibrated)} (best F1 on validation)`
    : undefined;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-300" title={hint}>{name}</span>
        <span className="flex items-center gap-2">
          {result.flag && (
            <span
              className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 border border-red-700/50"
              title={thresholdTitle}
            >
              Above threshold
            </span>
          )}
          <span className="font-bold text-white tabular-nums">{formatProb(result.prob)}</span>
        </span>
      </div>
      <div className="h-2 bg-surface-3 rounded overflow-hidden">
        <div className="h-full" style={{ width: `${Math.max(2, (result.prob || 0) * 100)}%`, backgroundColor: color }} />
      </div>
      {result.approximate && (
        <div className="text-[11px] text-gray-500">Approximate: large neighbourhood was sampled</div>
      )}
    </div>
  );
}

// Fraud probability block for a wallet: XGBoost (own behaviour) + GraphSAGE (behaviour + neighbours).
function ModelRiskBlock({ model, dataSource, address, onAddWallet }) {
  const [explanationTab, setExplanationTab] = useState('xgboost');  // 'xgboost' | 'graphsage'
  const [gnnExplanation, setGnnExplanation] = useState(null);
  const [gnnLoading, setGnnLoading] = useState(false);
  const [gnnError, setGnnError] = useState(null);

  const loadGnnExplanation = async () => {
    if (!address || gnnExplanation || gnnLoading) return;

    setGnnLoading(true);
    setGnnError(null);
    try {
      const result = await apiClient.getEppWalletExplainGnn(address);
      setGnnExplanation(result);
    } catch (err) {
      console.error('Failed to load GNN explanation:', err);
      setGnnError(err.message || 'Failed to load explanation');
    } finally {
      setGnnLoading(false);
    }
  };

  const handleTabSwitch = (tab) => {
    setExplanationTab(tab);
    if (tab === 'graphsage' && !gnnExplanation && !gnnLoading) {
      loadGnnExplanation();
    }
  };

  const handleNeighbourClick = (neighbourAddr) => {
    if (onAddWallet) {
      onAddWallet(neighbourAddr);
    }
  };

  if (!model) {
    return (
      <div>
        <div className="text-gray-400 text-sm mb-1">Fraud risk (model)</div>
        <div className="text-gray-500 text-sm">
          {dataSource === 'chain' ? 'Not scored — address is not in Elliptic++' : 'Model not available'}
        </div>
      </div>
    );
  }

  const hasExplanations = model.xgb?.explanations?.length > 0 || model.gnn;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-gray-400 text-sm">Fraud risk (model)</div>
        {model.source && (
          <span className="text-[11px] px-2 py-0.5 rounded bg-blue-900/40 text-blue-300 border border-blue-700/50">
            Historical (Elliptic++)
          </span>
        )}
      </div>

      <div className="bg-surface-2 rounded-lg p-3 space-y-3">
        <RiskRow name="GraphSAGE (GNN)" hint="Wallet behaviour + its neighbours in the transaction graph" result={model.gnn} />
        <RiskRow name="XGBoost" hint="Wallet's own behaviour only" result={model.xgb} decomposition={model.xgb?.decomposition} />
        <div className="flex items-center justify-between text-[11px] text-gray-500">
          <span>Probability that the wallet is illicit</span>
          {model.ms != null && <span>computed in {model.ms} ms</span>}
        </div>
      </div>

      {model.in_sample && (
        <div className="text-xs text-yellow-300 bg-yellow-900/20 border border-yellow-700/40 rounded p-2">
          In-sample: this wallet was used for training — the score is not an independent check.
        </div>
      )}

      {hasExplanations && (
        <div>
          <div className="flex items-center gap-3 mb-3 border-b border-surface-3">
            <button
              onClick={() => handleTabSwitch('xgboost')}
              className={`text-sm px-3 py-2 transition-colors relative ${
                explanationTab === 'xgboost'
                  ? 'text-white'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              Why: XGBoost
              {explanationTab === 'xgboost' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
              )}
            </button>
            {model.gnn && (
              <button
                onClick={() => handleTabSwitch('graphsage')}
                className={`text-sm px-3 py-2 transition-colors relative ${
                  explanationTab === 'graphsage'
                    ? 'text-white'
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                Why: GraphSAGE
                {explanationTab === 'graphsage' && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
                )}
              </button>
            )}
          </div>

          {explanationTab === 'xgboost' && model.xgb?.explanations?.length > 0 && (
            <div>
              <div className="text-xs text-gray-500 mb-2">SHAP values (log-odds)</div>
              <FeatureChart features={model.xgb.explanations} decomposition={model.xgb.decomposition} />
            </div>
          )}

          {explanationTab === 'graphsage' && (
            <div>
              {gnnLoading && (
                <div className="text-center text-gray-400 py-4">Loading explanation...</div>
              )}
              {gnnError && (
                <div className="text-sm text-red-400 bg-red-900/20 p-3 rounded">{gnnError}</div>
              )}
              {gnnExplanation && (
                <div className="space-y-4">
                  {/* Own behaviour vs neighbours */}
                  <div>
                    <div className="text-sm text-gray-300 font-medium mb-2">Own behaviour vs neighbours</div>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="text-gray-400">Isolated:</span>
                      <span className="font-mono text-white">{formatProb(gnnExplanation.graph_effect.prob_isolated)}</span>
                      <span className="text-gray-500">→</span>
                      <span className="text-gray-400">With graph:</span>
                      <span className="font-mono text-white">{formatProb(gnnExplanation.graph_effect.prob_full)}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {gnnExplanation.graph_effect.delta_logodds > 0
                        ? `Neighbours raise the risk by ${gnnExplanation.graph_effect.delta_logodds.toFixed(2)} log-odds`
                        : gnnExplanation.graph_effect.delta_logodds < -0.01
                        ? `Neighbours lower the risk by ${Math.abs(gnnExplanation.graph_effect.delta_logodds).toFixed(2)} log-odds`
                        : 'Neighbours have minimal effect on the score'}
                    </div>
                  </div>

                  {/* Features (occlusion) */}
                  {gnnExplanation.features && gnnExplanation.features.length > 0 && (
                    <div>
                      <div className="text-sm text-gray-300 font-medium mb-2">Features (occlusion)</div>
                      <div className="text-xs text-gray-500 mb-2">Δ log-odds when feature set to training mean</div>
                      <FeatureChart features={gnnExplanation.features.map(f => ({
                        label: f.label,
                        value: f.value,
                        shap: f.effect  // FeatureChart uses 'shap' field
                      }))} />
                    </div>
                  )}

                  {/* Neighbours driving the score */}
                  {gnnExplanation.neighbours && gnnExplanation.neighbours.length > 0 && (
                    <div>
                      <div className="text-sm text-gray-300 font-medium mb-2">Neighbours driving the score</div>
                      <div className="space-y-2">
                        {gnnExplanation.neighbours.map((nbr, idx) => {
                          const effectUp = nbr.effect > 0;
                          const effectColor = effectUp ? '#EF4444' : '#10B981';
                          const maxAbsEffect = Math.max(...gnnExplanation.neighbours.map(n => Math.abs(n.effect)));
                          const barWidth = (Math.abs(nbr.effect) / maxAbsEffect) * 100;

                          return (
                            <div
                              key={idx}
                              onClick={() => handleNeighbourClick(nbr.address)}
                              className="bg-surface-3 p-2 rounded cursor-pointer hover:bg-surface-4 transition-colors"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs font-mono text-gray-300">{nbr.address.slice(0, 12)}...</span>
                                <div className="flex items-center gap-2">
                                  {nbr.model_risk != null && (
                                    <span
                                      className="text-xs px-1.5 py-0.5 rounded font-medium"
                                      style={{
                                        backgroundColor: `${riskColor(nbr.model_risk)}26`,
                                        color: riskColor(nbr.model_risk),
                                        border: `1px solid ${riskColor(nbr.model_risk)}66`
                                      }}
                                    >
                                      {formatProb(nbr.model_risk)}
                                    </span>
                                  )}
                                  {nbr.known_label != null && (
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                                      nbr.known_label === 1
                                        ? 'bg-red-900/30 text-red-400'
                                        : 'bg-green-900/30 text-green-400'
                                    }`}>
                                      {nbr.known_label === 1 ? 'Illicit' : 'Licit'}
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-2 bg-surface-2 rounded overflow-hidden">
                                  <div
                                    className="h-full transition-all duration-300"
                                    style={{ width: `${barWidth}%`, backgroundColor: effectColor }}
                                  />
                                </div>
                                <span className={`text-xs font-mono w-14 text-right ${effectUp ? 'text-red-400' : 'text-green-400'}`}>
                                  {effectUp ? '+' : ''}{nbr.effect.toFixed(2)}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Footer note */}
                  <div className="text-xs text-gray-500 pt-2 border-t border-surface-3">
                    GraphSAGE explanations use occlusion (feature set to training mean / neighbour removed), not SHAP; values are changes in log-odds.
                    {gnnExplanation.approximate && ' Large neighbourhood was sampled.'}
                    {gnnExplanation.ms && ` Computed in ${gnnExplanation.ms} ms.`}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function TxRiskBadge({ txModel, compact = false }) {
  if (!txModel) return compact ? null : <span className="text-gray-500 text-sm">Not scored</span>;
  const color = riskColor(txModel.prob);
  return (
    <span
      className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-medium"
      style={{ backgroundColor: `${color}26`, color, border: `1px solid ${color}66` }}
      title="Transaction model: probability that the transaction is illicit"
    >
      {compact ? 'Risk' : 'Fraud risk'} {formatProb(txModel.prob)}
      {txModel.flag && ' ⚑'}
    </span>
  );
}

export default ModelRiskBlock;
