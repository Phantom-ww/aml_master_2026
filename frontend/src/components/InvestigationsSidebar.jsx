function InvestigationsSidebar({ investigations, activeInvestigationId, onLoadInvestigation, onDeleteInvestigation, onRefresh }) {
  return (
    <div className="w-72 bg-surface-1 border-r border-surface-3 flex flex-col">
      {/* Header */}
      <div className="px-4 py-4 border-b border-surface-3">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-white uppercase tracking-wide">
            Saved Investigations
          </h2>
          <button
            onClick={onRefresh}
            className="text-accent-primary hover:text-blue-400 text-sm transition-colors"
            title="Refresh list"
          >
            ↻
          </button>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {investigations.length === 0 ? (
          <div className="p-4 text-center text-gray-500 text-sm">
            No saved investigations yet
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {investigations.map((inv) => (
              <div
                key={inv.id}
                className={`p-3 rounded-lg cursor-pointer transition-all ${
                  activeInvestigationId === inv.id
                    ? 'bg-accent-primary bg-opacity-20 border border-accent-primary'
                    : 'bg-surface-2 hover:bg-surface-3 border border-transparent'
                }`}
                onClick={() => onLoadInvestigation(inv.id)}
              >
                <div className="flex items-start justify-between mb-1">
                  <h3 className="text-sm font-semibold text-white truncate flex-1">
                    {inv.name}
                  </h3>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm('Delete this investigation?')) {
                        onDeleteInvestigation(inv.id);
                      }
                    }}
                    className="text-gray-500 hover:text-red-400 text-xs ml-2 transition-colors"
                    title="Delete"
                  >
                    ✕
                  </button>
                </div>
                <div className="text-xs text-gray-400 space-y-0.5">
                  <div>{new Date(inv.created_at).toLocaleDateString()}</div>
                  <div className="flex items-center gap-3">
                    <span>{inv.node_count} nodes</span>
                    {inv.edge_count > 0 && <span>{inv.edge_count} edges</span>}
                  </div>
                  {inv.analyst_notes && (
                    <div className="text-gray-500 truncate mt-1">
                      {inv.analyst_notes}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default InvestigationsSidebar;
