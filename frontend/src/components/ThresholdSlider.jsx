function ThresholdSlider({ threshold, onChange }) {
  return (
    <div className="bg-surface-2 border-b border-surface-3 px-6 py-4">
      <div className="flex items-center gap-6">
        <label className="text-sm font-medium text-white whitespace-nowrap">
          Risk Threshold:
        </label>
        <div className="flex-1 flex items-center gap-4 max-w-2xl">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-surface-4 rounded-lg appearance-none cursor-pointer accent-accent-primary"
          />
          <div className="w-20 text-right">
            <span className="text-2xl font-bold text-accent-primary font-mono">
              {threshold.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-risk-low"></div>
            <span className="text-gray-400">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-risk-medium"></div>
            <span className="text-gray-400">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-risk-high"></div>
            <span className="text-gray-400">High</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-risk-critical"></div>
            <span className="text-gray-400">Critical</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ThresholdSlider;
