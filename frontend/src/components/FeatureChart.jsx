import { formatProb } from './riskUtils';

// SHAP explanation chart. Accepts items as {label|feature|feature_name, value, shap|shap_value}.
function FeatureChart({ features, decomposition }) {
  if (!features || features.length === 0) return null;

  const items = features.map(f => ({
    name: f.label || f.feature_name || f.feature,
    value: f.value,
    shap: f.shap ?? f.shap_value ?? 0,
  }));

  // Include "Other features" if decomposition is provided
  const allItems = decomposition
    ? [...items, {
        name: `Other ${decomposition.other_count} features`,
        value: null,
        shap: decomposition.other_sum,
        isOther: true
      }]
    : items;

  const maxAbs = Math.max(...allItems.map(f => Math.abs(f.shap)), 1e-9);

  const fmtValue = (v) => {
    if (v == null) return '';
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  };

  return (
    <div className="space-y-3">
      {allItems.map((f, idx) => {
        const up = f.shap > 0;
        const width = (Math.abs(f.shap) / maxAbs) * 100;
        return (
          <div key={idx} className="space-y-1">
            <div className="flex justify-between text-xs gap-2">
              <span className={`font-medium truncate ${f.isOther ? 'text-gray-400' : 'text-gray-300'}`} title={f.name}>{f.name}</span>
              <span className="text-gray-500 font-mono shrink-0">{fmtValue(f.value)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-3 bg-surface-3 rounded overflow-hidden">
                <div
                  className="h-full transition-all duration-300"
                  style={{ width: `${width}%`, backgroundColor: f.isOther ? '#6B7280' : (up ? '#EF4444' : '#10B981') }}
                />
              </div>
              <span className={`text-xs font-mono w-14 text-right ${f.isOther ? 'text-gray-400' : (up ? 'text-red-400' : 'text-green-400')}`}>
                {up ? '+' : ''}{f.shap.toFixed(2)}
              </span>
            </div>
          </div>
        );
      })}

      {/* Decomposition explanation */}
      {decomposition && (
        <div className="pt-2 border-t border-surface-3 space-y-2">
          <div className="text-[11px] font-mono text-gray-400 leading-relaxed">
            <span className="text-gray-500">How the score is computed:</span><br />
            base {decomposition.base_value.toFixed(2)} + shown {decomposition.shown_sum.toFixed(2)} + other {decomposition.other_sum.toFixed(2)} = logit {decomposition.logit.toFixed(2)} → raw {formatProb(decomposition.raw_prob)} → calibrated {formatProb(decomposition.prob)}
          </div>
          <div className="flex items-start gap-1.5 text-xs text-gray-500">
            <span className="text-gray-400 shrink-0" title="SHAP values are in log-odds and sum exactly to the model's logit. The logit is converted to a raw probability with the sigmoid; the raw probability is inflated because the model was trained with class weighting, so it is calibrated (isotonic regression on validation data) into the probability shown above. The 'Above threshold' flag compares the raw score with the threshold chosen on validation for best F1 (raw threshold ≈ calibrated threshold).">ⓘ</span>
            <span>SHAP values sum to the logit. Raw prob = sigmoid(logit); calibrated via isotonic regression. Threshold: raw {formatProb(decomposition.threshold_raw)} ≈ {formatProb(decomposition.threshold_calibrated)} calibrated (best F1 on validation).</span>
          </div>
        </div>
      )}

      <div className="pt-2 border-t border-surface-3 text-xs text-gray-500 flex items-center justify-between">
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded bg-red-500 inline-block" />Increases risk</span>
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded bg-green-500 inline-block" />Decreases risk</span>
      </div>
    </div>
  );
}

export default FeatureChart;
