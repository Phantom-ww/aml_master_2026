import { useEffect, useState, useRef } from 'react';
import apiClient from '../api/client';
import ModelRiskBlock, { TxRiskBadge, TxLiveModelBlock } from './ModelRiskBlock';
import FeatureChart from './FeatureChart';
import { coinJoinLabel, isCoinJoin } from './riskUtils';

// ---------------------------------------------------------------------------
// Mixing & Sanctions Exposure block (chain mode only)
// ---------------------------------------------------------------------------

function MixingExposureBlock({ loading, error, data }) {
  if (loading) {
    return (
      <div>
        <div className="text-gray-400 text-sm mb-2">Mixing &amp; sanctions exposure</div>
        <div className="text-gray-500 text-sm animate-pulse">Loading…</div>
      </div>
    );
  }
  if (error) {
    return (
      <div>
        <div className="text-gray-400 text-sm mb-2">Mixing &amp; sanctions exposure</div>
        <div className="text-red-400 text-xs">{error}</div>
      </div>
    );
  }
  if (!data) return null;

  const { sanctions, cj_participations, cj_types, received_from_coinjoin,
          consolidations, sent_to_coinjoin, checked_txs, truncated } = data;
  const hasCj = cj_participations > 0 || received_from_coinjoin > 0 ||
                consolidations > 0 || sent_to_coinjoin > 0;

  return (
    <div className="space-y-3">
      <div className="text-gray-400 text-sm">Mixing &amp; sanctions exposure</div>

      {/* OFAC sanctions banner */}
      {sanctions?.available && sanctions?.listed && (
        <div className="p-3 rounded-lg bg-red-900/30 border border-red-700/50">
          <div className="text-sm font-semibold text-red-300">
            ⚠ OFAC sanctioned: {sanctions.entity || 'unknown entity'}
          </div>
        </div>
      )}
      {sanctions?.available && !sanctions?.listed &&
        sanctions?.counterparties_listed?.length > 0 && (
        <div className="p-2 rounded bg-orange-900/20 border border-orange-700/40">
          <div className="text-xs font-medium text-orange-300 mb-1">
            Sanctioned counterparties:
          </div>
          <ul className="space-y-1">
            {sanctions.counterparties_listed.map((cp, i) => (
              <li key={i} className="text-xs font-mono text-orange-200 truncate"
                  title={cp.entity}>
                {cp.address.slice(0, 20)}… — {cp.entity}
              </li>
            ))}
          </ul>
        </div>
      )}
      {!sanctions?.available && (
        <div className="text-xs text-gray-500">
          OFAC data unavailable — run scripts/update_ofac_btc.py to load.
        </div>
      )}

      {/* CoinJoin stats */}
      <div className="bg-surface-2 rounded-lg p-3 space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">CoinJoin participations</span>
          <span className={cj_participations > 0 ? 'text-purple-300 font-medium' : 'text-gray-400'}>
            {cj_participations}
          </span>
        </div>
        {cj_participations > 0 && Object.keys(cj_types ?? {}).length > 0 && (
          <div className="ml-3 space-y-1 text-xs text-gray-400">
            {Object.entries(cj_types).map(([t, n]) => (
              <div key={t} className="flex justify-between">
                <span>{t}</span><span>{n}</span>
              </div>
            ))}
          </div>
        )}
        <div className="flex justify-between">
          <span className="text-gray-400">Sent to CoinJoin (pre-mix)</span>
          <span className={sent_to_coinjoin > 0 ? 'text-purple-300' : 'text-gray-400'}>
            {sent_to_coinjoin}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Received from CoinJoin (post-mix)</span>
          <span className={received_from_coinjoin > 0 ? 'text-purple-300' : 'text-gray-400'}>
            {received_from_coinjoin}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">CoinJoin output consolidations</span>
          <span className={consolidations > 0 ? 'text-purple-300' : 'text-gray-400'}>
            {consolidations}
          </span>
        </div>
        <div className="text-xs text-gray-500 pt-1 border-t border-surface-3">
          Checked {checked_txs} transaction{checked_txs !== 1 ? 's' : ''}
          {truncated && ' (partial — full history not loaded)'}
        </div>
      </div>

      <div className="text-xs text-gray-500 leading-relaxed">
        CoinJoin is a privacy technique, not proof of illicit activity.
        Shown as a risk factor only.
      </div>
    </div>
  );
}


function InspectorPanel({ selectedNode, onClose, onDelete, dataSource, onAddTransaction, onSwitchToLiveChain }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMoreTxs, setLoadingMoreTxs] = useState(false);
  const [txLoadError, setTxLoadError] = useState(null);
  const [nextCursor, setNextCursor] = useState(null);
  const [hasMoreTxs, setHasMoreTxs] = useState(false);
  const [loadingTx, setLoadingTx] = useState({});  // Track loading state per txid
  const [onGraphTxs, setOnGraphTxs] = useState(new Set());  // Track which txs are on graph
  const [txNotes, setTxNotes] = useState({});  // per-tx message: "No counterparties" / error
  const [mixingData, setMixingData] = useState(null);
  const [mixingLoading, setMixingLoading] = useState(false);
  const [mixingError, setMixingError] = useState(null);
  const mixingCache = useRef(new Map());

  const handleDelete = () => {
    if (!selectedNode) return;
    onDelete(selectedNode);
    onClose();
  };

  useEffect(() => {
    if (!selectedNode) {
      setDetails(null);
      setNextCursor(null);
      setHasMoreTxs(false);
      setTxLoadError(null);
      setLoadingTx({});
      setOnGraphTxs(new Set());
      setTxNotes({});
      return;
    }

    const resolveTxId = (d) => {
      const raw = d.tx_address ?? d.txId ?? d.tx_id ?? d.edge_id ?? d.id;
      if (!raw) return null;
      return String(raw).split(':')[0];
    };

    const loadDetails = async () => {
      setLoading(true);
      setNextCursor(null);
      setHasMoreTxs(false);
      setTxLoadError(null);
      try {
        if (selectedNode.type === 'edge') {
          const txid = resolveTxId(selectedNode.data);
          const resolvedDataSource = selectedNode.data.data_source;

          if (!txid) {
            setDetails({ type: 'error', message: 'Transaction ID not found' });
            return;
          }

          if (!resolvedDataSource) {
            setDetails({ type: 'error', message: 'Unknown data source for transaction' });
            return;
          }

          console.log('[InspectorPanel] Edge clicked:', { txid, data_source: resolvedDataSource });

          let data;
          if (resolvedDataSource === 'chain') {
            data = await apiClient.getChainTx(txid);
          } else if (resolvedDataSource === 'ellipticpp') {
            data = await apiClient.getEppTx(txid);
          } else {
            setDetails({ type: 'error', message: `Unknown data source: ${resolvedDataSource}` });
            return;
          }
          setDetails({ type: 'transaction', ...data, data_source: resolvedDataSource });
        } else if (selectedNode.type === 'node') {
          const resolvedDataSource = selectedNode.data.data_source;

          if (!resolvedDataSource) {
            setDetails({ type: 'error', message: 'Unknown data source for wallet' });
            return;
          }

          console.log('[InspectorPanel] Node clicked:', {
            id: selectedNode.data.id,
            data_source: resolvedDataSource
          });

          let data;
          if (resolvedDataSource === 'chain') {
            console.log('[InspectorPanel] Fetching chain address:', selectedNode.data.id);
            data = await apiClient.getChainAddress(selectedNode.data.id);
            setNextCursor(data.next_cursor);
            const loadedCount = data.transactions?.length || 0;
            const totalCount = data.tx_count || 0;
            setHasMoreTxs(data.next_cursor != null || loadedCount < totalCount);
          } else if (resolvedDataSource === 'ellipticpp') {
            console.log('[InspectorPanel] Fetching ellipticpp address:', selectedNode.data.id);
            data = await apiClient.getEppAddress(selectedNode.data.id);
          } else {
            setDetails({ type: 'error', message: `Unknown data source: ${resolvedDataSource}` });
            return;
          }
          setDetails({ type: 'wallet', ...data, data_source: resolvedDataSource });
        }
      } catch (err) {
        console.error('Failed to load details:', err);
        setDetails({ type: 'error', message: err.message || 'Failed to load details' });
      } finally {
        setLoading(false);
      }
    };

    loadDetails();
  }, [selectedNode]);

  // Lazy-load mixing & sanctions exposure for chain wallets
  useEffect(() => {
    if (!details || details.type !== 'wallet' || details.data_source !== 'chain') {
      setMixingData(null);
      setMixingError(null);
      setMixingLoading(false);
      return;
    }
    const address = details.address;
    if (!address) return;
    if (mixingCache.current.has(address)) {
      setMixingData(mixingCache.current.get(address));
      setMixingError(null);
      setMixingLoading(false);
      return;
    }
    setMixingLoading(true);
    setMixingError(null);
    setMixingData(null);
    apiClient.getChainAddressMixing(address)
      .then(data => {
        mixingCache.current.set(address, data);
        setMixingData(data);
      })
      .catch(err => {
        setMixingError(err.message || 'Failed to load mixing data');
      })
      .finally(() => setMixingLoading(false));
  }, [details?.address, details?.data_source, details?.type]);

  const handleTransactionClick = async (tx) => {
    if (tx.direction === 'self') {
      return;
    }

    const txid = tx.txId ?? tx.txid;
    setLoadingTx(prev => ({ ...prev, [txid]: true }));
    setTxNotes(prev => ({ ...prev, [txid]: null }));

    try {
      let edgesData;
      const resolvedDataSource = details.data_source;

      if (resolvedDataSource === 'chain') {
        edgesData = await apiClient.getChainTxEdges(txid, details.address, 10);
      } else if (resolvedDataSource === 'ellipticpp') {
        edgesData = await apiClient.getEppTxEdges(txid, details.address, 10);
      } else {
        throw new Error(`Unknown data source: ${resolvedDataSource}`);
      }

      const valid = edgesData && Array.isArray(edgesData.nodes) && Array.isArray(edgesData.edges);
      if (!valid || edgesData.edges.length === 0) {
        setTxNotes(prev => ({ ...prev, [txid]: 'No counterparties' }));
        return;
      }

      onAddTransaction?.(edgesData);
      setOnGraphTxs(prev => new Set([...prev, txid]));
      if (edgesData.truncated) {
        setTxNotes(prev => ({ ...prev, [txid]: `Showing ${edgesData.edges.length} of ${edgesData.total_counterparties} links` }));
      }
    } catch (err) {
      console.error('Failed to load transaction edges:', err);
      setTxNotes(prev => ({ ...prev, [txid]: err.message || 'Failed to load transaction' }));
    } finally {
      setLoadingTx(prev => ({ ...prev, [txid]: false }));
    }
  };

  const loadMoreTransactions = async () => {
    if (!details || !details.address || loadingMoreTxs) return;

    // Determine cursor to use
    let cursor = nextCursor;
    if (!cursor && hasMoreTxs) {
      // Fallback: use last confirmed transaction's txid
      const confirmedTxs = details.transactions.filter(tx => tx.confirmed);
      if (confirmedTxs.length > 0) {
        cursor = confirmedTxs[confirmedTxs.length - 1].txid;
      }
    }

    if (!cursor && !hasMoreTxs) {
      // Nothing to load
      return;
    }

    setLoadingMoreTxs(true);
    setTxLoadError(null);

    try {
      const page = await apiClient.getChainAddressTxsPage(details.address, cursor);

      // Append new transactions
      setDetails(prev => ({
        ...prev,
        transactions: [...(prev.transactions || []), ...page.transactions]
      }));

      // Update pagination state
      setNextCursor(page.next_cursor);

      // Determine if more available
      const newLoadedCount = (details.transactions?.length || 0) + page.transactions.length;
      const totalCount = details.tx_count || 0;
      setHasMoreTxs(page.has_more || (page.next_cursor != null) || (newLoadedCount < totalCount));
    } catch (err) {
      console.error('Failed to load more transactions:', err);
      setTxLoadError(err.message || 'Failed to load transactions');
    } finally {
      setLoadingMoreTxs(false);
    }
  };

  if (!selectedNode) {
    return (
      <div className="w-96 bg-surface-1 border-l border-surface-3 p-6 flex items-center justify-center text-gray-500">
        Select a node or edge to view details
      </div>
    );
  }

  const isWallet = selectedNode.type === 'node';
  const isTransaction = selectedNode.type === 'edge';

  return (
    <div className="w-96 bg-surface-1 border-l border-surface-3 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-surface-3">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-1 rounded text-xs font-medium bg-surface-3 text-gray-300">
                {isWallet ? 'Wallet' : 'Transaction'}
              </span>
            </div>
            <h3 className="text-white font-mono text-sm break-all">
              {isWallet ? selectedNode.data.id : String(selectedNode.data.tx_address ?? selectedNode.data.tx_id ?? '').split(':')[0]}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors ml-2"
          >
            ✕
          </button>
        </div>

        {/* Delete Button */}
        <button
          onClick={handleDelete}
          className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          {isWallet ? 'Delete Node' : 'Remove from graph'}
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="text-gray-400 text-center">Loading...</div>
        ) : details?.type === 'error' ? (
          <div className="text-red-400 text-center p-4 bg-red-900/20 rounded">{details.message}</div>
        ) : details ? (
          <div className="space-y-6">
            {/* Wallet Details */}
            {details.type === 'wallet' && (
              <>
                {/* Wallet Info */}
                <div className="space-y-4">
                  <div>
                    <div className="text-gray-400 text-sm mb-1">Address</div>
                    <div className="text-xs font-mono text-white bg-surface-2 p-2 rounded break-all">
                      {details.address}
                    </div>
                  </div>

                  {/* Known Label */}
                  {details.data_source === 'ellipticpp' && details.class != null && (
                    <div>
                      <div className="text-gray-400 text-sm mb-1">Known Label</div>
                      <div className={`inline-block px-3 py-1 rounded-md text-sm font-semibold ${
                        details.class === 1
                          ? 'bg-red-600/20 text-red-400 border border-red-600/40'
                          : details.class === 2
                          ? 'bg-green-600/20 text-green-400 border border-green-600/40'
                          : 'bg-gray-600/20 text-gray-400 border border-gray-600/40'
                      }`} title="Historical label from the Elliptic++ dataset (KDD'23)">
                        Elliptic++: {details.class === 1 ? 'Illicit' : details.class === 2 ? 'Licit' : 'Unknown'}
                      </div>
                    </div>
                  )}

                  {details.data_source === 'chain' && details.ellipticpp_class != null && (
                    <div>
                      <div className="text-gray-400 text-sm mb-1">Known Label</div>
                      <div className={`inline-block px-3 py-1 rounded-md text-sm font-semibold ${
                        details.ellipticpp_class === 1
                          ? 'bg-red-600/20 text-red-400 border border-red-600/40'
                          : 'bg-green-600/20 text-green-400 border border-green-600/40'
                      }`} title="Historical label from the Elliptic++ dataset (KDD'23)">
                        Elliptic++: {details.ellipticpp_class === 1 ? 'Illicit' : 'Licit'}
                      </div>
                    </div>
                  )}

                  {/* In Dataset (Elliptic++) */}
                  {details.data_source === 'ellipticpp' && details.key_features && (
                    <div>
                      <div className="text-gray-400 text-sm mb-2">In Dataset</div>
                      <div className="space-y-2 text-sm">
                        {details.key_features.btc_received_total != null && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Received total:</span>
                            <span className="text-white font-mono">{details.key_features.btc_received_total.toFixed(4)} BTC</span>
                          </div>
                        )}
                        {details.key_features.btc_sent_total != null && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Sent total:</span>
                            <span className="text-white font-mono">{details.key_features.btc_sent_total.toFixed(4)} BTC</span>
                          </div>
                        )}
                        {details.key_features.total_txs != null && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Transactions:</span>
                            <span className="text-white">{details.key_features.total_txs}</span>
                          </div>
                        )}
                        {details.key_features.fees_total != null && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Fees total:</span>
                            <span className="text-white font-mono">{details.key_features.fees_total.toFixed(6)} BTC</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Key Features (Elliptic++) */}
                  {details.data_source === 'ellipticpp' && details.key_features && (
                    <div>
                      <div className="text-gray-400 text-sm mb-2">Key Features</div>
                      <div className="space-y-2 text-sm">
                        {details.key_features.lifetime_in_blocks != null && (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Lifetime (blocks):</span>
                            <span className="text-white">{details.key_features.lifetime_in_blocks.toLocaleString()}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Time Steps (Elliptic++) */}
                  {details.data_source === 'ellipticpp' && details.first_time_step != null && (
                    <div>
                      <div className="text-gray-400 text-sm mb-1 flex items-center gap-1">
                        Time
                        <span className="text-xs text-gray-500" title="Elliptic time step ≈ 2-week window; real dates not published, block heights are real">ⓘ</span>
                      </div>
                      <div className="text-white text-sm space-y-1">
                        <div>Time step {details.first_time_step} → {details.last_time_step}</div>
                        {details.first_block_appeared_in != null && details.last_block_appeared_in != null && (
                          <div className="text-xs text-gray-400">
                            Blocks {details.first_block_appeared_in.toLocaleString()} → {details.last_block_appeared_in.toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Balance (Chain only) */}
                  {details.data_source === 'chain' && (
                    <div>
                      <div className="text-gray-400 text-sm mb-1">Balance</div>
                      <div className="text-xl font-bold text-white">
                        {details.balance_crypto?.toFixed(4) || '0.0000'} BTC
                      </div>
                      {details.balance_usd != null && (
                        <div className="text-sm text-gray-400">
                          ${details.balance_usd.toLocaleString(undefined, {maximumFractionDigits: 2})} USD
                        </div>
                      )}
                      {details.balance_sat != null && (
                        <div className="text-xs text-gray-500 mt-1">
                          {details.balance_sat.toLocaleString()} sat
                        </div>
                      )}
                      {details.tx_count != null && (
                        <div className="text-xs text-gray-500">
                          {details.confirmed_tx_count != null && details.mempool_tx_count != null ? (
                            <>
                              {details.tx_count} transaction{details.tx_count !== 1 ? 's' : ''}
                              {details.mempool_tx_count > 0 && ` (${details.mempool_tx_count} pending)`}
                            </>
                          ) : (
                            `${details.tx_count} transaction${details.tx_count !== 1 ? 's' : ''}`
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Model fraud risk (on-the-fly XGBoost + GraphSAGE) */}
                  <ModelRiskBlock
                    model={details.model}
                    dataSource={details.data_source}
                    address={details.address}
                    onAddWallet={async (addr) => {
                      // Add wallet to graph
                      try {
                        const walletData = await apiClient.getEppWallet(addr);
                        onAddTransaction?.(walletData);
                      } catch (err) {
                        console.error('Failed to add wallet:', err);
                      }
                    }}
                  />

                  {/* Mixing & sanctions exposure (Live Chain only) */}
                  {details.data_source === 'chain' && (
                    <MixingExposureBlock
                      loading={mixingLoading}
                      error={mixingError}
                      data={mixingData}
                    />
                  )}

                  {/* Open in Live Chain button (Elliptic++ only) */}
                  {details.data_source === 'ellipticpp' && (
                    <button
                      onClick={() => {
                        if (window.confirm(`Switch to Live Chain mode to see current balance and full history for ${details.address}?`)) {
                          // Call parent to switch modes and load this address
                          if (onSwitchToLiveChain) {
                            onSwitchToLiveChain(details.address);
                          }
                        }
                      }}
                      className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                    >
                      Open in Live Chain
                    </button>
                  )}
                </div>

                {/* Transactions */}
                {details.transactions && details.transactions.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-white font-semibold mb-3">
                      Transactions
                      {details.data_source === 'chain' && ` (${details.transactions.length} of ${details.tx_count || details.transactions.length})`}
                      {details.data_source === 'ellipticpp' && ` (${details.transactions.length} of ${details.total_transactions || details.transactions.length})`}
                    </h4>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {details.transactions.map(tx => {
                        const txid = tx.txId ?? tx.txid;
                        const isLoading = loadingTx[txid];
                        const isOnGraph = onGraphTxs.has(txid);
                        const isSelf = tx.direction === 'self';

                        return (
                          <div
                            key={txid}
                            onClick={() => !isSelf && !isLoading && handleTransactionClick(tx)}
                            className={`p-3 bg-surface-2 rounded ${
                              isSelf ? 'opacity-70' : 'hover:bg-surface-3 transition-colors cursor-pointer'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-mono text-gray-400">
                                {txid.toString().slice(0, 12)}...
                              </span>
                              <div className="flex items-center gap-2">
                                <span className={`text-xs font-medium ${
                                  tx.direction === 'self' ? 'text-blue-400' :
                                  tx.direction === 'out' ? 'text-orange-400' : 'text-green-400'
                                }`}>
                                  {tx.direction === 'self' ? '⟲ Self' :
                                   tx.direction === 'out' ? '→ Sent' : '← Received'}
                                </span>
                                {isLoading && (
                                  <span className="text-xs text-blue-400">Loading…</span>
                                )}
                                {isOnGraph && !isLoading && (
                                  <span className="text-xs text-green-400">On graph</span>
                                )}
                              </div>
                            </div>
                            <div className="text-sm text-white mb-1">
                              {(tx.total_BTC ?? tx.amount_btc)?.toFixed(8)} BTC
                              {details.data_source === 'ellipticpp' && (
                                <span className="text-xs text-gray-500 ml-1">tx total</span>
                              )}
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-gray-500">
                                {details.data_source === 'ellipticpp'
                                  ? `Time step ${tx.time_step}`
                                  : tx.timestamp ? new Date(tx.timestamp).toLocaleDateString() : 'Pending'}
                              </span>
                              {details.data_source === 'chain' && (
                                <span className="flex items-center gap-1">
                                  {isCoinJoin(tx.coinjoin) && (
                                    <span
                                      className="text-xs px-2 py-0.5 rounded font-medium bg-purple-900/40 text-purple-300 border border-purple-700/50"
                                      title={tx.coinjoin?.reasons?.join('\n') || 'CoinJoin detected'}
                                    >
                                      {coinJoinLabel(tx.coinjoin)}
                                    </span>
                                  )}
                                  <TxRiskBadge txModel={tx.live_tx_model} compact />
                                  <span className={`text-xs px-2 py-0.5 rounded ${
                                    tx.confirmed
                                      ? 'bg-green-900/30 text-green-400'
                                      : 'bg-yellow-900/30 text-yellow-400'
                                  }`}>
                                    {tx.confirmed ? 'Confirmed' : 'Unconfirmed'}
                                  </span>
                                </span>
                              )}
                              {details.data_source === 'ellipticpp' && (
                                <span className="flex items-center gap-1">
                                  <TxRiskBadge txModel={tx.tx_model} compact />
                                  {tx.tx_class != null && (
                                    <span className={`text-xs px-2 py-0.5 rounded ${
                                      tx.tx_class === 1
                                        ? 'bg-red-900/30 text-red-400'
                                        : tx.tx_class === 2
                                        ? 'bg-green-900/30 text-green-400'
                                        : 'bg-gray-900/30 text-gray-400'
                                    }`} title="Known label (Elliptic)">
                                      {tx.tx_class === 1 ? 'Illicit' : tx.tx_class === 2 ? 'Licit' : 'Unknown'}
                                    </span>
                                  )}
                                </span>
                              )}
                            </div>
                            {isSelf && (
                              <div className="text-xs text-gray-500 mt-1">
                                Consolidation — no counterparties
                              </div>
                            )}
                            {txNotes[txid] && (
                              <div className="text-xs text-gray-400 mt-1">{txNotes[txid]}</div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* Load More Button */}
                    {details.data_source === 'chain' && (
                      <div className="mt-3">
                        {txLoadError && (
                          <div className="text-xs text-red-400 mb-2 p-2 bg-red-900/20 rounded">
                            {txLoadError}
                          </div>
                        )}
                        <button
                          onClick={loadMoreTransactions}
                          disabled={loadingMoreTxs || (!hasMoreTxs && !txLoadError)}
                          className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
                            loadingMoreTxs || (!hasMoreTxs && !txLoadError)
                              ? 'bg-surface-3 text-gray-500 cursor-not-allowed'
                              : 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                          }`}
                        >
                          {loadingMoreTxs
                            ? 'Loading…'
                            : txLoadError
                            ? 'Retry'
                            : hasMoreTxs
                            ? 'Load more'
                            : 'All transactions loaded'}
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Transaction Details */}
            {details.type === 'transaction' && (
              <>
                {details.data_source === 'chain' ? (
                  /* Chain transaction view */
                  <>
                    {/* Transaction Header */}
                    <div className="space-y-4">
                      <div>
                        <div className="text-gray-400 text-sm mb-1">Transaction ID</div>
                        <div className="flex items-center gap-2">
                          <div className="text-xs font-mono text-white bg-surface-2 p-2 rounded break-all flex-1">
                            {details.txid}
                          </div>
                          <button
                            onClick={() => navigator.clipboard.writeText(details.txid)}
                            className="px-2 py-1 bg-surface-3 hover:bg-surface-4 rounded text-xs"
                            title="Copy"
                          >
                            📋
                          </button>
                        </div>
                        <a
                          href={`https://mempool.space/tx/${details.txid}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-400 hover:text-blue-300 mt-1 inline-block"
                        >
                          View on mempool.space →
                        </a>
                      </div>

                      {/* Status and Block Info */}
                      <div className="flex items-center gap-3">
                        <span className={`px-3 py-1 rounded text-sm font-medium ${
                          details.confirmed
                            ? 'bg-green-900/30 text-green-400'
                            : 'bg-yellow-900/30 text-yellow-400'
                        }`}>
                          {details.confirmed ? 'Confirmed' : 'In Mempool'}
                        </span>
                        {details.block_time && (
                          <div className="text-sm text-gray-400">
                            {new Date(details.block_time).toLocaleString()}
                          </div>
                        )}
                      </div>

                      {details.block_height != null && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Block {details.block_height.toLocaleString()}</span>
                          <span className="text-gray-400">{details.confirmations} confirmation{details.confirmations !== 1 ? 's' : ''}</span>
                        </div>
                      )}

                      {/* CoinJoin badge */}
                      {isCoinJoin(details.coinjoin) && (
                        <div>
                          <div className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-900/30 border border-purple-700/50">
                            <span className="text-purple-300 font-medium text-sm">
                              CoinJoin · {coinJoinLabel(details.coinjoin)}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                              details.coinjoin.confidence === 'high'
                                ? 'bg-purple-700/40 text-purple-200'
                                : details.coinjoin.confidence === 'medium'
                                ? 'bg-purple-800/40 text-purple-300'
                                : 'bg-gray-700/40 text-gray-300'
                            }`}>
                              {details.coinjoin.confidence} confidence
                            </span>
                          </div>
                          {details.coinjoin.reasons?.length > 0 && (
                            <details className="mt-2">
                              <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-300">
                                Detection reasons ({details.coinjoin.reasons.length})
                              </summary>
                              <ul className="mt-1 space-y-1 text-xs text-gray-400 bg-surface-2 p-2 rounded">
                                {details.coinjoin.reasons.map((r, i) => (
                                  <li key={i} className="font-mono">• {r}</li>
                                ))}
                              </ul>
                            </details>
                          )}
                        </div>
                      )}
                    </div>

                    {/* This Edge Section */}
                    <div>
                      <h4 className="text-white font-semibold mb-3">This Edge</h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between bg-surface-2 p-2 rounded">
                          <span className="text-xs text-green-400 font-mono truncate flex-1">
                            {selectedNode.data.source_wallet?.slice(0, 16)}...
                          </span>
                          <span className="text-gray-400 mx-2">→</span>
                          <span className="text-xs text-orange-400 font-mono truncate flex-1 text-right">
                            {selectedNode.data.target_wallet?.slice(0, 16)}...
                          </span>
                        </div>
                        {/* Amount on this edge could be shown if we have it */}
                      </div>
                    </div>

                    {/* Transaction Summary */}
                    <div>
                      <h4 className="text-white font-semibold mb-3">Summary</h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-surface-2 p-3 rounded">
                          <div className="text-xs text-gray-400 mb-1">Total In</div>
                          <div className="text-sm font-bold text-white">{details.total_in_btc.toFixed(8)} BTC</div>
                        </div>
                        <div className="bg-surface-2 p-3 rounded">
                          <div className="text-xs text-gray-400 mb-1">Total Out</div>
                          <div className="text-sm font-bold text-white">{details.total_out_btc.toFixed(8)} BTC</div>
                        </div>
                        <div className="bg-surface-2 p-3 rounded">
                          <div className="text-xs text-gray-400 mb-1">Fee</div>
                          <div className="text-sm font-bold text-white">{details.fee_btc.toFixed(8)} BTC</div>
                          {details.fee_rate_sat_vb != null && (
                            <div className="text-xs text-gray-500 mt-0.5">{details.fee_rate_sat_vb.toFixed(1)} sat/vB</div>
                          )}
                        </div>
                        <div className="bg-surface-2 p-3 rounded">
                          <div className="text-xs text-gray-400 mb-1">Inputs / Outputs</div>
                          <div className="text-sm font-bold text-white">{details.input_count} / {details.output_count}</div>
                        </div>
                      </div>
                    </div>

                    {/* Inputs */}
                    <div>
                      <h4 className="text-white font-semibold mb-2">Inputs ({details.input_count})</h4>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {details.inputs.slice(0, 10).map((inp, idx) => (
                          <div key={idx} className="bg-surface-2 p-2 rounded text-xs">
                            <div className="flex items-center justify-between">
                              <span className="font-mono text-gray-400 truncate flex-1" title={inp.address || 'Coinbase'}>
                                {inp.is_coinbase ? '⛏ Coinbase' : (inp.address ? inp.address.slice(0, 20) + '...' : 'Unknown')}
                              </span>
                              <span className="text-white ml-2">{inp.value_btc.toFixed(8)} BTC</span>
                            </div>
                          </div>
                        ))}
                        {details.input_count > 10 && (
                          <div className="text-xs text-gray-500 text-center py-1">
                            + {details.input_count - 10} more inputs
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Outputs */}
                    <div>
                      <h4 className="text-white font-semibold mb-2">Outputs ({details.output_count})</h4>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {details.outputs.slice(0, 10).map((out, idx) => (
                          <div key={idx} className="bg-surface-2 p-2 rounded text-xs">
                            <div className="flex items-center justify-between">
                              <span className="font-mono text-gray-400 truncate flex-1" title={out.address || 'OP_RETURN'}>
                                {out.address ? out.address.slice(0, 20) + '...' : 'OP_RETURN'}
                              </span>
                              <span className="text-white ml-2">{out.value_btc.toFixed(8)} BTC</span>
                            </div>
                            {out.is_change && (
                              <div className="text-xs text-orange-400 mt-0.5">↩ Change</div>
                            )}
                          </div>
                        ))}
                        {details.output_count > 10 && (
                          <div className="text-xs text-gray-500 text-center py-1">
                            + {details.output_count - 10} more outputs
                          </div>
                        )}
                      </div>
                    </div>

                    {/* OP_RETURN */}
                    {details.has_op_return && details.op_return_data && (
                      <div>
                        <h4 className="text-white font-semibold mb-2">OP_RETURN Data</h4>
                        <div className="text-xs font-mono text-purple-400 bg-surface-2 p-2 rounded break-all">
                          {details.op_return_data}
                        </div>
                      </div>
                    )}

                    {/* Risk Score */}
                    <div>
                      <h4 className="text-white font-semibold mb-2">Risk Assessment</h4>
                      <TxLiveModelBlock txModel={details.live_tx_model} />
                    </div>
                  </>
                ) : (
                  /* Ellipticpp transaction view */
                  <>
                    <div className="space-y-4">
                      <div>
                        <div className="text-gray-400 text-sm mb-1">Transaction ID</div>
                        <div className="text-xs font-mono text-white bg-surface-2 p-2 rounded break-all">
                          {details.txId}
                        </div>
                      </div>

                      <div>
                        <div className="text-gray-400 text-sm mb-1">Time Step</div>
                        <div className="text-white">{details.time_step}</div>
                      </div>

                      <div>
                        <div className="text-gray-400 text-sm mb-1">Known label (Elliptic)</div>
                        <span className={`inline-block px-3 py-1 rounded text-sm font-medium ${
                          details.class === 1
                            ? 'bg-red-900/30 text-red-400'
                            : details.class === 2
                            ? 'bg-green-900/30 text-green-400'
                            : 'bg-gray-900/30 text-gray-400'
                        }`}>
                          {details.class === 1 ? 'Illicit' : details.class === 2 ? 'Licit' : 'Unknown'}
                        </span>
                      </div>

                      <div>
                        <div className="text-gray-400 text-sm mb-1">Fraud risk (transaction model)</div>
                        <TxRiskBadge txModel={details.tx_model} />

                        {/* Transaction SHAP explanation */}
                        {details.tx_model?.explanation && (
                          <div className="mt-3 space-y-3">
                            <div className="text-sm text-gray-300 font-medium">Why (transaction model, SHAP)</div>

                            {/* Group sums stacked bar */}
                            {details.tx_model.explanation.groups && details.tx_model.explanation.groups.length > 0 && (
                              <div>
                                <div className="text-xs text-gray-500 mb-1">Signal by feature group:</div>
                                <div className="flex h-6 bg-surface-3 rounded overflow-hidden">
                                  {details.tx_model.explanation.groups.map((group, idx) => {
                                    const totalAbs = details.tx_model.explanation.groups.reduce((sum, g) => sum + Math.abs(g.shap_sum), 0);
                                    const widthPct = totalAbs > 0 ? (Math.abs(group.shap_sum) / totalAbs) * 100 : 0;
                                    const colors = ['#F59E0B', '#3B82F6', '#10B981'];
                                    return (
                                      <div
                                        key={idx}
                                        className="flex items-center justify-center text-[10px] text-white font-medium"
                                        style={{ width: `${widthPct}%`, backgroundColor: colors[idx % colors.length] }}
                                        title={`${group.group}: ${group.shap_sum.toFixed(2)}`}
                                      >
                                        {widthPct > 15 && group.group.split(' ')[0]}
                                      </div>
                                    );
                                  })}
                                </div>
                                <div className="flex flex-wrap gap-2 mt-2 text-[10px]">
                                  {details.tx_model.explanation.groups.map((group, idx) => {
                                    const colors = ['#F59E0B', '#3B82F6', '#10B981'];
                                    return (
                                      <div key={idx} className="flex items-center gap-1">
                                        <div className="w-2 h-2 rounded" style={{ backgroundColor: colors[idx % colors.length] }} />
                                        <span className="text-gray-400">{group.group}: {group.shap_sum > 0 ? '+' : ''}{group.shap_sum.toFixed(2)}</span>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Top features */}
                            {details.tx_model.explanation.top && details.tx_model.explanation.top.length > 0 && (
                              <div>
                                <FeatureChart
                                  features={details.tx_model.explanation.top}
                                  decomposition={details.tx_model.explanation.decomposition}
                                />
                                <div className="text-xs text-gray-500 mt-2">
                                  Most transaction features are anonymised by Elliptic; group totals show where the signal comes from.
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <div className="text-gray-400 text-sm mb-1">Total BTC</div>
                        <div className="text-xl font-bold text-white">
                          {details.total_BTC?.toFixed(4) || '0.0000'} BTC
                        </div>
                      </div>

                      <div>
                        <div className="text-gray-400 text-sm mb-1">Fees</div>
                        <div className="text-white">
                          {details.fees?.toFixed(4) || '0.0000'} BTC
                        </div>
                      </div>
                    </div>

                    {/* Inputs */}
                    {details.input_addresses && details.input_addresses.length > 0 && (
                      <div>
                        <h4 className="text-white font-semibold mb-2">Inputs ({details.input_addresses.length})</h4>
                        <div className="space-y-1 max-h-48 overflow-y-auto">
                          {details.input_addresses.map((addr, idx) => (
                            <div key={idx} className="bg-surface-2 p-2 rounded text-xs font-mono text-gray-400 break-all">
                              {addr}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Outputs */}
                    {details.output_addresses && details.output_addresses.length > 0 && (
                      <div>
                        <h4 className="text-white font-semibold mb-2">Outputs ({details.output_addresses.length})</h4>
                        <div className="space-y-1 max-h-48 overflow-y-auto">
                          {details.output_addresses.map((addr, idx) => (
                            <div key={idx} className="bg-surface-2 p-2 rounded text-xs font-mono text-gray-400 break-all">
                              {addr}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="text-gray-400 text-center">No details available</div>
        )}
      </div>
    </div>
  );
}

export default InspectorPanel;
