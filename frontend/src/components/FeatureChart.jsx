// SHAP explanation chart. Accepts items as {label|feature|feature_name, value, shap|shap_value}.
function FeatureChart({ features }) {
  if (!features || features.length === 0) return null;

  const items = features.map(f => ({
    name: f.label || f.feature_name || f.feature,
    value: f.value,
    shap: f.shap ?? f.shap_value ?? 0,
  }));
  const maxAbs = Math.max(...items.map(f => Math.abs(f.shap)), 1e-9);

  const fmtValue = (v) => {
    if (v == null) return '';
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  };

  return (
    <div className="space-y-3">
      {items.map((f, idx) => {
        const up = f.shap > 0;
        const width = (Math.abs(f.shap) / maxAbs) * 100;
        return (
          <div key={idx} className="space-y-1">
            <div className="flex justify-between text-xs gap-2">
              <span className="text-gray-300 font-medium truncate" title={f.name}>{f.name}</span>
              <span className="text-gray-500 font-mono shrink-0">{fmtValue(f.value)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-3 bg-surface-3 rounded overflow-hidden">
                <div
                  className="h-full transition-all duration-300"
                  style={{ width: `${width}%`, backgroundColor: up ? '#EF4444' : '#10B981' }}
                />
              </div>
              <span className={`text-xs font-mono w-14 text-right ${up ? 'text-red-400' : 'text-green-400'}`}>
                {up ? '+' : ''}{f.shap.toFixed(2)}
              </span>
            </div>
          </div>
        );
      })}
      <div className="pt-2 border-t border-surface-3 text-xs text-gray-500 flex items-center justify-between">
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded bg-red-500 inline-block" />Increases risk</span>
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded bg-green-500 inline-block" />Decreases risk</span>
      </div>
    </div>
  );
}

export default FeatureChart;
