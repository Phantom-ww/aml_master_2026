import { useState, useEffect } from 'react';
import GraphCanvas from './components/GraphCanvas';
import InspectorPanel from './components/InspectorPanel';
import SaveInvestigationModal from './components/SaveInvestigationModal';
import ErrorBoundary from './components/ErrorBoundary';
import ModelInfo from './components/ModelInfo';
import apiClient from './api/client';

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [investigations, setInvestigations] = useState([]);
  const [activeInvestigationId, setActiveInvestigationId] = useState(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showLoadMenu, setShowLoadMenu] = useState(false);
  const [searchNodeId, setSearchNodeId] = useState('39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K');
  const [dataSource, setDataSource] = useState('ellipticpp'); // 'ellipticpp' or 'chain'
  const [chainNetwork, setChainNetwork] = useState('signet');
  const [colorMode, setColorMode] = useState('label'); // 'label' | 'model'

  // Update graph nodes (for position changes only - preserves all node data)
  const handleGraphUpdate = (updatedNodes) => {
    console.log('[App] handleGraphUpdate called with:', updatedNodes);
    setGraphData(prev => {
      console.log('[App] Previous graphData:', prev);

      // Only update x/y positions by id, never replace node objects
      const positionMap = new Map(updatedNodes.map(n => [n.id, { x: n.x, y: n.y }]));

      const updated = {
        ...prev,
        nodes: prev.nodes.map(node => {
          const newPos = positionMap.get(node.id);
          return newPos ? { ...node, x: newPos.x, y: newPos.y } : node;
        })
      };
      console.log('[App] New graphData:', updated);
      return updated;
    });
  };

  // Delete node or edge from graph
  const handleDelete = (selectedItem) => {
    console.log('[App] handleDelete called with:', selectedItem);

    if (selectedItem.type === 'node') {
      const nodeId = selectedItem.data.id;
      setGraphData(prev => ({
        nodes: prev.nodes.filter(n => n.id !== nodeId),
        edges: prev.edges.filter(e => e.source !== nodeId && e.target !== nodeId)
      }));
    } else if (selectedItem.type === 'edge') {
      const edgeId = selectedItem.data.edge_id;
      setGraphData(prev => ({
        ...prev,
        edges: prev.edges.filter(e => {
          const eId = e.id ?? `${e.tx_address}:${e.source}:${e.target}`;
          return eId !== edgeId;
        })
      }));
    }

    setSelectedNode(null);
    setActiveInvestigationId(null);
  };

  // Expand graph with transaction data
  const expandGraphWithTransaction = (newData) => {
    if (!newData || !Array.isArray(newData.nodes) || !Array.isArray(newData.edges)) {
      console.warn('[App] expandGraphWithTransaction: invalid data, ignored', newData);
      return;
    }
    setGraphData(prev => {
      console.log('[App] Previous graph state:', { nodeCount: prev.nodes.length, edgeCount: prev.edges.length });

      const existingNodeIds = new Set(prev.nodes.map(n => n.id));
      const existingEdgeIds = new Set(prev.edges.map(e => e.id || `${e.tx_address}:${e.source}:${e.target}`));

      // Dedupe nodes within new batch
      const newNodeMap = new Map();
      newData.nodes.forEach(n => {
        if (!existingNodeIds.has(n.id) && !newNodeMap.has(n.id)) {
          newNodeMap.set(n.id, n);
        }
      });
      const newNodes = Array.from(newNodeMap.values());

      // Dedupe edges within new batch
      const newEdgeMap = new Map();
      newData.edges.forEach(e => {
        const edgeId = e.id || `${e.tx_address}:${e.source}:${e.target}`;
        if (!existingEdgeIds.has(edgeId) && !newEdgeMap.has(edgeId)) {
          newEdgeMap.set(edgeId, e);
        }
      });
      const newEdges = Array.from(newEdgeMap.values());

      console.log('[App] Filtered new nodes:', newNodes);
      console.log('[App] Filtered new edges:', newEdges);

      // Positions are chosen by GraphCanvas (around the wallet the new edges connect to)

      const updatedState = {
        nodes: [...prev.nodes, ...newNodes],
        edges: [...prev.edges, ...newEdges]
      };

      console.log('[App] Updated graph state:', { nodeCount: updatedState.nodes.length, edgeCount: updatedState.edges.length });
      return updatedState;
    });
    setActiveInvestigationId(null);
  };

  // Switch to Live Chain mode and load address
  const handleSwitchToLiveChain = async (address) => {
    setDataSource('chain');
    setSearchNodeId(address);
    setGraphData({ nodes: [], edges: [] });
    setSelectedNode(null);
    try {
      const data = await apiClient.getChainWallet(address);
      setGraphData(data);
      if (data.nodes && data.nodes.length > 0) {
        setSelectedNode({ type: 'node', data: data.nodes[0] });
      }
    } catch (err) {
      console.error('Failed to load chain wallet:', err);
      alert(`Failed to load: ${err.message}`);
    }
  };

  // Load initial subgraph
  const loadSubgraph = async (nodeId, k = 1) => {
    try {
      let data;
      if (dataSource === 'chain') {
        data = await apiClient.getChainWallet(nodeId);
        setGraphData(data);
        setActiveInvestigationId(null);

        // Auto-select the single node
        if (data.nodes && data.nodes.length > 0) {
          setSelectedNode({ type: 'node', data: data.nodes[0] });
        }
      } else {
        data = await apiClient.getEppWallet(nodeId);
        setGraphData(data);
        setActiveInvestigationId(null);

        // Auto-select the single node so its details (and model scores) open
        if (data.nodes && data.nodes.length > 0) {
          setSelectedNode({ type: 'node', data: data.nodes[0] });
        }
      }
    } catch (err) {
      console.error('Failed to load subgraph:', err);
      alert(`Failed to load graph: ${err.message}`);
    }
  };

  // Load investigations list with legacy detection
  const loadInvestigations = async () => {
    try {
      const data = await apiClient.listInvestigations();
      // Mark investigations with demo data as legacy
      setInvestigations(data.map(inv => ({
        ...inv,
        legacy: inv.node_count > 0 &&
                (inv.name?.includes('Demo') ||
                 inv.analyst_notes?.includes('wallet_000000'))
      })));
    } catch (err) {
      console.error('Failed to load investigations:', err);
    }
  };

  // Load saved investigation
  const loadInvestigation = async (id) => {
    try {
      const inv = investigations.find(i => i.id === id);

      // Block legacy demo investigations
      if (inv?.legacy) {
        alert('Legacy demo investigation — cannot be opened');
        return;
      }

      const data = await apiClient.getInvestigation(id);
      const snapshot = data.snapshot;

      // Clear existing graph first to ensure clean load
      setGraphData({ nodes: [], edges: [] });

      // Small delay to ensure graph is cleared before loading new data
      setTimeout(() => {
        setGraphData({
          nodes: snapshot.nodes,
          edges: snapshot.edges
        });

        // Handle selected node or edge from snapshot
        if (snapshot.selected_node_id) {
          const node = snapshot.nodes.find(n => n.id === snapshot.selected_node_id);
          if (node) {
            setSelectedNode({ type: 'node', data: node });
          }
        } else {
          setSelectedNode(null);
        }

        setActiveInvestigationId(id);
      }, 50);
    } catch (err) {
      console.error('Failed to load investigation:', err);
    }
  };

  // Save current investigation
  const saveInvestigation = async (name, description, notes) => {
    // Flatten nodes to match backend schema
    const nodesWithPositions = graphData.nodes.map(node => ({
      id: node.id,
      label: node.data?.label || node.label || node.id.slice(0, 12) + "...",
      x: node.x ?? node.data?.x ?? node.style?.x ?? 0,
      y: node.y ?? node.data?.y ?? node.style?.y ?? 0,
      node_type: node.data?.node_type || node.node_type || "wallet",
      risk_score: node.data?.risk_score ?? node.risk_score ?? 0.0,
      pinned: node.pinned ?? false,
      address: node.data?.address || node.address || node.id,
      owner: node.data?.owner || node.owner || null,
      balance_crypto: node.data?.balance_crypto ?? node.balance_crypto ?? null,
      balance_usd: node.data?.balance_usd ?? node.balance_usd ?? null
    }));

    // Flatten edges to match backend schema
    const flattenedEdges = graphData.edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      amount: edge.data?.amount ?? edge.amount ?? 0.0,
      risk_score: edge.data?.risk_score ?? edge.risk_score ?? 0.0,
      risk_color: edge.risk_color ?? "#999",
      time_step: edge.data?.time_step ?? edge.time_step ?? 0,
      tx_address: edge.data?.tx_address || edge.tx_address || edge.id,
      from_wallet: edge.data?.from_wallet || edge.from_wallet || edge.source,
      to_wallet: edge.data?.to_wallet || edge.to_wallet || edge.target,
      timestamp: edge.data?.timestamp || edge.timestamp || null,
      amount_crypto: edge.data?.amount_crypto ?? edge.amount_crypto ?? null,
      amount_usd: edge.data?.amount_usd ?? edge.amount_usd ?? null
    }));

    const snapshot = {
      version: 1,
      timestamp: new Date().toISOString(),
      selected_node_id: selectedNode?.type === 'node' ? selectedNode.data.id : null,
      nodes: nodesWithPositions,
      edges: flattenedEdges,
      camera: { x: 0, y: 0, zoom: 1 },
      analyst_notes: notes
    };

    try {
      await apiClient.saveInvestigation({
        name,
        description,
        snapshot,
        analyst_notes: notes
      });
      setShowSaveModal(false);
      loadInvestigations();
    } catch (err) {
      console.error('Failed to save investigation:', err);
    }
  };

  // Delete investigation
  const deleteInvestigation = async (id) => {
    try {
      await apiClient.deleteInvestigation(id);
      loadInvestigations();
      if (activeInvestigationId === id) {
        setActiveInvestigationId(null);
      }
    } catch (err) {
      console.error('Failed to delete investigation:', err);
    }
  };

  // Load investigations on mount + fetch chain status
  useEffect(() => {
    loadInvestigations();

    // Fetch chain network setting if available
    if (dataSource === 'chain') {
      apiClient.getChainStatus().then(status => {
        if (status.network) {
          setChainNetwork(status.network);
        }
      }).catch(err => {
        console.warn('Could not fetch chain status:', err);
      });
    }
  }, [dataSource]);

  // Close load menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showLoadMenu && !e.target.closest('.load-menu-container')) {
        setShowLoadMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showLoadMenu]);

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      {/* Top bar */}
      <div className="bg-surface-2 border-b border-surface-3 px-6 py-4 flex items-center gap-4">
        <h1 className="text-xl font-bold text-white">AML Investigation Platform</h1>

        <div className="flex-1 flex items-center gap-3 max-w-3xl">
          {/* Data source toggle */}
          <div className="flex bg-surface-3 border border-surface-4 rounded-lg overflow-hidden">
            <button
              onClick={() => {
                const loadExampleAddress = async () => {
                  try {
                    const examples = await apiClient.getEppExamples();
                    const firstIllicit = examples.illicit?.[0] || '39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K';
                    setDataSource('ellipticpp');
                    setSearchNodeId(firstIllicit);
                    setGraphData({ nodes: [], edges: [] });
                    setSelectedNode(null);
                  } catch (err) {
                    console.warn('Could not fetch examples, using fallback:', err);
                    setDataSource('ellipticpp');
                    setSearchNodeId('39sfuA8pY4UfybgEZi7uvA13jkGzZpsg5K');
                    setGraphData({ nodes: [], edges: [] });
                    setSelectedNode(null);
                  }
                };
                loadExampleAddress();
              }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                dataSource === 'ellipticpp'
                  ? 'bg-accent-primary text-white'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Elliptic++
            </button>
            <button
              onClick={() => {
                setDataSource('chain');
                setSearchNodeId('1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa');
                setGraphData({ nodes: [], edges: [] });
                setSelectedNode(null);
              }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                dataSource === 'chain'
                  ? 'bg-accent-primary text-white'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Live Chain ({chainNetwork})
            </button>
          </div>

          <input
            type="text"
            value={searchNodeId}
            onChange={(e) => setSearchNodeId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                loadSubgraph(searchNodeId);
              }
            }}
            placeholder={
              dataSource === 'chain'
                ? 'Enter BTC address (e.g., bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh)'
                : 'Enter wallet address from Elliptic++'
            }
            className="flex-1 px-4 py-2 bg-surface-3 border border-surface-4 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary"
          />
          <button
            onClick={() => loadSubgraph(searchNodeId)}
            className="px-6 py-2 bg-accent-primary hover:bg-blue-500 text-white font-medium rounded-lg transition-colors"
          >
            Load Graph
          </button>
        </div>

        <ModelInfo />

        {/* Save/Load buttons */}
        <div className="flex items-center gap-3 relative load-menu-container">
          <button
            onClick={() => setShowSaveModal(true)}
            disabled={graphData.nodes.length === 0}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save
          </button>
          <button
            onClick={() => {
              setShowLoadMenu(!showLoadMenu);
              if (!showLoadMenu) loadInvestigations();
            }}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            Load
          </button>

          {/* Load dropdown menu */}
          {showLoadMenu && (
            <div className="absolute top-full right-0 mt-2 w-80 bg-surface-2 border border-surface-3 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
              {investigations.length === 0 ? (
                <div className="p-4 text-gray-400 text-center">No saved investigations</div>
              ) : (
                investigations.map((inv) => (
                  <div
                    key={inv.id}
                    className={`p-4 border-b border-surface-3 hover:bg-surface-3 cursor-pointer ${
                      activeInvestigationId === inv.id ? 'bg-surface-3' : ''
                    }`}
                    onClick={() => {
                      loadInvestigation(inv.id);
                      setShowLoadMenu(false);
                    }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="text-white font-medium">{inv.name}</h3>
                        {inv.description && (
                          <p className="text-gray-400 text-sm mt-1">{inv.description}</p>
                        )}
                        <p className="text-gray-500 text-xs mt-1">
                          {new Date(inv.created_at).toLocaleString()}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteInvestigation(inv.id);
                        }}
                        className="ml-2 text-red-400 hover:text-red-300"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph canvas */}
        <div className="flex-1 relative">
          <ErrorBoundary>
            <GraphCanvas
              graphData={graphData}
              colorMode={colorMode}
              onColorModeChange={setColorMode}
              onNodeClick={setSelectedNode}
              isInvestigationMode={activeInvestigationId !== null}
              onGraphUpdate={handleGraphUpdate}
            />
          </ErrorBoundary>
        </div>

        {/* Inspector panel */}
        <InspectorPanel
          selectedNode={selectedNode}
          onClose={() => setSelectedNode(null)}
          onDelete={handleDelete}
          onAddTransaction={expandGraphWithTransaction}
          onSwitchToLiveChain={handleSwitchToLiveChain}
          dataSource={dataSource}
        />
      </div>

      {/* Save modal */}
      {showSaveModal && (
        <SaveInvestigationModal
          onSave={saveInvestigation}
          onClose={() => setShowSaveModal(false)}
        />
      )}
    </div>
  );
}

export default App;
