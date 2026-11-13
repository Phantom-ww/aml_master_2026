import { useEffect, useRef, useState } from 'react';
import apiClient from '../api/client';

// Header link "Model vX" → popover with the trained model's test metrics (from /api/model/info).
function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    apiClient.getModelInfo().then(setInfo).catch(err => setError(err.message));
  }, []);

  useEffect(() => {
    const onClick = (e) => { if (open && ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  const fmt = (v) => (v == null ? '—' : v.toFixed(3));

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
          info ? 'border-surface-4 text-gray-300 hover:text-white' : 'border-red-800 text-red-300'
        }`}
        title={error || 'Model information'}
      >
        {info ? `Model ${info.model_version}` : 'Model not loaded'}
      </button>
      {open && (
        <div className="absolute top-full right-0 mt-2 w-[30rem] bg-surface-2 border border-surface-3 rounded-lg shadow-lg z-50 p-4">
          {info ? (
            <>
              <div className="text-white font-semibold mb-1">Elliptic++ models · {info.model_version}</div>
              <div className="text-xs text-gray-400 mb-3">
                Test set: wallets first seen in time steps 35–49 (temporal split). Scores are computed on the fly.
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 text-left">
                    <th className="py-1 font-medium">Model</th>
                    <th className="py-1 font-medium text-right">PR-AUC</th>
                    <th className="py-1 font-medium text-right">F1</th>
                    <th className="py-1 font-medium text-right">Precision</th>
                    <th className="py-1 font-medium text-right">Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {(info.metrics || []).map(m => (
                    <tr key={m.model} className="border-t border-surface-3 text-gray-200">
                      <td className="py-1 pr-2">{m.model}</td>
                      <td className="py-1 text-right tabular-nums">{fmt(m.pr_auc)}</td>
                      <td className="py-1 text-right tabular-nums">{fmt(m.f1)}</td>
                      <td className="py-1 text-right tabular-nums">{fmt(m.precision)}</td>
                      <td className="py-1 text-right tabular-nums">{fmt(m.recall)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-[11px] text-gray-500 mt-3">
                "Authors' protocol" = random split without time order (as in the Elliptic++ paper) — shown for comparison.
              </div>
            </>
          ) : (
            <div className="text-sm text-red-300">{error || 'Loading…'}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default ModelInfo;
