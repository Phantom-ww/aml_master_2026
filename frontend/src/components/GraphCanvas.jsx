import { useEffect, useRef } from 'react';
import { Graph } from '@antv/g6';
import { labelColor, riskColor, LABEL_COLORS, RISK_COLORS } from './riskUtils';

function GraphCanvas({ graphData, colorMode = 'label', onColorModeChange, onNodeClick, isInvestigationMode, onGraphUpdate }) {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  // G6 style callbacks are created once; read the current mode through a ref
  const colorModeRef = useRef(colorMode);

  const nodeFill = (d) => (colorModeRef.current === 'model' ? riskColor(d?.model_risk) : labelColor(d || {}));
  const edgeStroke = (d) => {
    if (colorModeRef.current === 'model') {
      return d?.tx_risk != null ? riskColor(d.tx_risk) : '#4B5563';
    }
    // Known label: Elliptic++ illicit transaction (tx_class === 1) → red
    if (d?.data_source === 'ellipticpp' && d?.tx_class === 1) return '#EF4444';
    return '#4B5563';
  };

  // Initialize G6 Graph
  useEffect(() => {
    if (!containerRef.current) return;

    const graph = new Graph({
      container: containerRef.current,
      width: containerRef.current.offsetWidth,
      height: containerRef.current.offsetHeight,
      autoResize: true,
      node: {
        style: {
          size: 30,
          fill: (d) => nodeFill(d.data),
          stroke: '#fff',
          lineWidth: 2,
          labelText: (d) => d.data?.label || d.id.slice(0, 10),
          labelFill: '#fff',
          labelFontSize: 10,
        },
      },
      edge: {
        style: {
          stroke: (d) => edgeStroke(d.data),
          lineWidth: 2,
          endArrow: true,
        },
      },
      layout: {
        type: 'force',
        preventOverlap: true,
        nodeStrength: -30,
        edgeStrength: 0.1,
        linkDistance: 150,
        animated: false,
      },
      transforms: [
        {
          type: 'process-parallel-edges',
          mode: 'bundle',
          distance: 20,
        },
      ],
      behaviors: ['drag-canvas', 'zoom-canvas', 'drag-element'],
    });

    graphRef.current = graph;

    // Handle node click
    graph.on('node:click', (e) => {
      console.log('[GraphCanvas] Node clicked:', e);
      const nodeData = graph.getNodeData(e.target.id);
      onNodeClick({ type: 'node', data: { id: e.target.id, ...nodeData?.data } });
    });

    // Handle edge click
    graph.on('edge:click', (e) => {
      const edgeData = graph.getEdgeData(e.target.id);
      if (edgeData) {
        const sourceData = graph.getNodeData(edgeData.source);
        const targetData = graph.getNodeData(edgeData.target);
        onNodeClick({
          type: 'edge',
          data: {
            ...edgeData.data,
            id: edgeData.id,
            edge_id: edgeData.id,
            source_wallet: sourceData?.id || edgeData.source,
            target_wallet: targetData?.id || edgeData.target,
          },
        });
      }
    });

    // Handle node drag end - save positions back to parent state
    graph.on('node:dragend', (e) => {
      console.log('[GraphCanvas] Node drag ended:', e.target.id);
      // Get all current node positions from the graph
      const allNodes = graph.getNodeData();
      const updatedNodes = allNodes.map(node => {
        const position = graph.getNodeData(node.id);
        return {
          ...node,
          x: position.style?.x ?? node.data?.x,
          y: position.style?.y ?? node.data?.y,
        };
      });
      // Notify parent of position updates
      if (onGraphUpdate) {
        onGraphUpdate(updatedNodes);
      }
    });

    graph.on('afterlayout', () => {
      console.log('[GraphCanvas] Layout complete');
    });

    return () => {
      graph.destroy();
    };
  }, []);

  // Recolor everything when the color mode changes (styles are callbacks → touch data + redraw)
  useEffect(() => {
    colorModeRef.current = colorMode;
    const graph = graphRef.current;
    if (!graph) return;
    const nodes = graph.getNodeData();
    const edges = graph.getEdgeData();
    if (nodes.length === 0) return;
    graph.updateNodeData(nodes.map(n => ({ id: n.id, data: { ...n.data, _colorMode: colorMode } })));
    if (edges.length > 0) {
      graph.updateEdgeData(edges.map(e => ({ id: e.id, data: { ...e.data, _colorMode: colorMode } })));
    }
    graph.draw();
  }, [colorMode]);

  // Update graph data
  useEffect(() => {
    if (!graphRef.current) return;

    const graph = graphRef.current;

    console.log('[GraphCanvas] Update triggered with graphData:', {
      nodes: graphData.nodes.length,
      edges: graphData.edges.length,
      nodeIds: graphData.nodes.map(n => n.id)
    });

    if (!graphData.nodes || graphData.nodes.length === 0) {
      console.log('[GraphCanvas] Empty graph, clearing');
      graph.clear();
      return;
    }

    // Get existing nodes and edges
    const existingNodes = graph.getNodeData();
    const existingEdges = graph.getEdgeData();
    console.log('[GraphCanvas] Existing graph state:', {
      nodes: existingNodes.length,
      edges: existingEdges.length,
      existingNodeIds: existingNodes.map(n => n.id)
    });
    const existingNodeIds = new Set(existingNodes.map(n => n.id));
    const existingEdgeIds = new Set(existingEdges.map(e => e.id));

    // Check if this is a full graph replacement (investigation load)
    // If all incoming nodes are new, it's a full replacement - clear first
    const incomingNodeIds = new Set(graphData.nodes.map(n => n.id));
    const isFullReplacement = existingNodes.length > 0 &&
      ![...incomingNodeIds].some(id => existingNodeIds.has(id));

    if (isFullReplacement) {
      console.log('[GraphCanvas] Full graph replacement detected, clearing and rebuilding');
      graph.clear();

      // Build all nodes first
      const allNodes = graphData.nodes.map((node, index) => {
        const x = node.x ?? 0;
        const y = node.y ?? 0;

        return {
          id: node.id,
          data: {
            label: typeof node.label === 'string' ? node.label : (node.label || node.id.slice(0, 10)),
            x: x,
            y: y,
            risk_score: node.risk_score,
            node_type: node.node_type,
            address: node.address,
            owner: node.owner,
            balance_crypto: node.balance_crypto,
            balance_usd: node.balance_usd,
            data_source: node.data_source,
            ellipticpp_class: node.ellipticpp_class,
            known_label: typeof node.label === 'number' ? node.label : node.known_label,
            model_risk: node.model_risk,
          },
          style: {
            stroke: '#fff',
            lineWidth: 2,
            x: x,
            y: y,
          },
        };
      });

      // Build all edges
      const allEdges = graphData.edges.map((edge) => {
        const edgeId = edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`;
        return {
          id: edgeId,
          source: edge.source,
          target: edge.target,
          data: {
            risk_score: edge.risk_score,
            amount: edge.amount,
            edge_id: edgeId,
            tx_id: edge.tx_address || (edgeId.includes(':') ? edgeId.split(':')[0] : edgeId),
            tx_address: edge.tx_address,
            data_source: edge.data_source,
            tx_class: edge.tx_class,
            tx_risk: edge.tx_risk ?? edge.tx_model?.prob ?? null,
          },
          style: {
            lineWidth: 2,
          },
        };
      });

      // Add all data at once
      graph.addData({
        nodes: allNodes,
        edges: allEdges
      });

      // Stop layout and apply saved positions
      graph.stopLayout();

      setTimeout(() => {
        const positionUpdates = allNodes.map(node => ({
          id: node.id,
          style: {
            x: node.data.x,
            y: node.data.y,
          }
        }));

        graph.updateNodeData(positionUpdates);
        graph.draw();
      }, 0);

      return;
    }

    // Сохраняем текущие данные существующих нод (включая позиции)
    const existingNodesMap = new Map();
    existingNodes.forEach(node => {
      existingNodesMap.set(node.id, node);
    });

    // Find nodes to remove (exist in graph but not in graphData)
    const nodesToRemove = existingNodes.filter(node => !incomingNodeIds.has(node.id));

    // Find edges to remove (exist in graph but not in graphData)
    const incomingEdgeIds = new Set(graphData.edges.map(e => e.id ?? `${e.tx_address}:${e.source}:${e.target}`));
    const edgesToRemove = existingEdges.filter(edge => !incomingEdgeIds.has(edge.id));

    // If there are removals, clear and rebuild the graph
    if (nodesToRemove.length > 0 || edgesToRemove.length > 0) {
      console.log('[GraphCanvas] Removing:', {
        nodes: nodesToRemove.map(n => n.id),
        edges: edgesToRemove.map(e => e.id)
      });

      // Clear graph and rebuild from graphData
      graph.clear();

      // Build all nodes
      const allNodes = graphData.nodes.map((node) => {
        const existingNode = existingNodesMap.get(node.id);
        const x = existingNode?.style?.x ?? existingNode?.data?.x ?? node.x ?? 0;
        const y = existingNode?.style?.y ?? existingNode?.data?.y ?? node.y ?? 0;

        return {
          id: node.id,
          data: {
            label: typeof node.label === 'string' ? node.label : (node.label || node.id.slice(0, 10)),
            x: x,
            y: y,
            risk_score: node.risk_score,
            node_type: node.node_type,
            address: node.address,
            owner: node.owner,
            balance_crypto: node.balance_crypto,
            balance_usd: node.balance_usd,
            data_source: node.data_source,
            ellipticpp_class: node.ellipticpp_class,
            known_label: typeof node.label === 'number' ? node.label : node.known_label,
            model_risk: node.model_risk,
          },
          style: {
            stroke: '#fff',
            lineWidth: 2,
            x: x,
            y: y,
          },
        };
      });

      // Build all edges
      const allEdges = graphData.edges.map((edge) => ({
        id: edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`,
        source: edge.source,
        target: edge.target,
        data: {
          risk_score: edge.risk_score,
          amount: edge.amount,
          edge_id: edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`,
          tx_id: edge.tx_address || ((edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`).includes(':')
            ? (edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`).split(':')[0]
            : (edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`)),
          tx_address: edge.tx_address,
          data_source: edge.data_source,
          tx_class: edge.tx_class,
          tx_risk: edge.tx_risk ?? edge.tx_model?.prob ?? null,
        },
        style: {
          lineWidth: 2,
        },
      }));

      graph.addData({ nodes: allNodes, edges: allEdges });
      graph.stopLayout();
      graph.draw();
      return;
    }

    // Dedupe new nodes within batch
    const newNodesSeen = new Set();
    const newNodesToAdd = graphData.nodes.filter(node => {
      if (existingNodeIds.has(node.id) || newNodesSeen.has(node.id)) return false;
      newNodesSeen.add(node.id);
      return true;
    });

    // Helper function to check if position is too close to existing nodes
    const isTooCloseToExisting = (x, y, minDistance = 80) => {
      return existingNodesMap.values().some(node => {
        const nodeX = node.style?.x ?? node.data?.x;
        const nodeY = node.style?.y ?? node.data?.y;
        if (nodeX === undefined || nodeY === undefined) return false;
        const distance = Math.sqrt(Math.pow(x - nodeX, 2) + Math.pow(y - nodeY, 2));
        return distance < minDistance;
      });
    };

    // Helper function to find a valid position near a given point
    const findValidPosition = (preferredX, preferredY) => {
      // Try the preferred position first
      if (!isTooCloseToExisting(preferredX, preferredY)) {
        return { x: preferredX, y: preferredY };
      }

      // Try positions in a spiral pattern around the preferred position
      const maxAttempts = 50;
      const spiralStep = 30; // pixels between spiral rings

      for (let attempt = 1; attempt < maxAttempts; attempt++) {
        const angle = attempt * 0.5; // Spiral angle
        const radius = attempt * spiralStep;
        const x = preferredX + Math.cos(angle) * radius;
        const y = preferredY + Math.sin(angle) * radius;

        if (!isTooCloseToExisting(x, y)) {
          return { x, y };
        }
      }

      // Fallback: return a random position far from center
      return {
        x: preferredX + (Math.random() - 0.5) * 400,
        y: preferredY + (Math.random() - 0.5) * 400
      };
    };

    // Place new nodes: around the existing node they connect to (its real on-screen position),
    // otherwise in the middle of the canvas. Backend coordinates (0,0) are placeholders.
    const centerX = containerRef.current.offsetWidth / 2;
    const centerY = containerRef.current.offsetHeight / 2;
    const posOf = (id) => {
      const n = existingNodesMap.get(id);
      const px = n?.style?.x ?? n?.data?.x;
      const py = n?.style?.y ?? n?.data?.y;
      return px !== undefined && py !== undefined ? { x: px, y: py } : null;
    };
    const anchorOf = (id) => {
      for (const e of graphData.edges) {
        if (e.source === id && existingNodeIds.has(e.target)) return e.target;
        if (e.target === id && existingNodeIds.has(e.source)) return e.source;
      }
      return null;
    };
    const groups = new Map();
    newNodesToAdd.forEach(n => {
      const a = anchorOf(n.id);
      if (!groups.has(a)) groups.set(a, []);
      groups.get(a).push(n.id);
    });

    const newNodes = newNodesToAdd.map((node) => {
      const hasRealPos = node.x != null && node.y != null && !(node.x === 0 && node.y === 0);
      let x = hasRealPos ? node.x : undefined;
      let y = hasRealPos ? node.y : undefined;

      if (x === undefined || y === undefined) {
        const anchor = anchorOf(node.id);
        const base = (anchor && posOf(anchor)) || { x: centerX, y: centerY };
        const group = groups.get(anchor) || [node.id];
        const i = group.indexOf(node.id);
        if (!anchor && existingNodes.length === 0 && group.length === 1) {
          x = base.x; y = base.y;                       // the very first node: canvas center
        } else {
          const angle = (Math.PI * 2 * i) / group.length + Math.PI / 6;
          const radius = 160 + 20 * Math.floor(i / 12);
          x = base.x + Math.cos(angle) * radius;
          y = base.y + Math.sin(angle) * radius;
        }
      }

      // Проверяем, не слишком ли близко к существующим нодам
      const validPosition = findValidPosition(x, y);

      return {
        id: node.id,
        data: {
          label: typeof node.label === 'string' ? node.label : (node.label || node.id.slice(0, 10)),
          x: validPosition.x,
          y: validPosition.y,
          risk_score: node.risk_score,
          node_type: node.node_type,
          address: node.address,
          owner: node.owner,
          balance_crypto: node.balance_crypto,
          balance_usd: node.balance_usd,
          data_source: node.data_source,
          ellipticpp_class: node.ellipticpp_class,
          known_label: typeof node.label === 'number' ? node.label : node.known_label,
          model_risk: node.model_risk,
        },
        style: {
          stroke: '#fff',
          lineWidth: 2,
          x: validPosition.x,
          y: validPosition.y,
        },
      };
    });

    // Dedupe new edges within batch
    const newEdgesSeen = new Set();
    const newEdges = graphData.edges
      .filter(edge => {
        const edgeId = edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`;
        if (existingEdgeIds.has(edgeId) || newEdgesSeen.has(edgeId)) return false;
        newEdgesSeen.add(edgeId);
        return true;
      })
      .map((edge) => {
        const edgeId = edge.id ?? `${edge.tx_address}:${edge.source}:${edge.target}`;
        return {
          id: edgeId,
          source: edge.source,
          target: edge.target,
          data: {
            risk_score: edge.risk_score,
            amount: edge.amount,
            edge_id: edgeId,
            tx_id: edge.tx_address || (edgeId.includes(':') ? edgeId.split(':')[0] : edgeId),
            tx_address: edge.tx_address,
            data_source: edge.data_source,
            tx_class: edge.tx_class,
            tx_risk: edge.tx_risk ?? edge.tx_model?.prob ?? null,
          },
          style: {
            lineWidth: 2,
          },
        };
      });


    // Add new elements
    if (newNodes.length > 0 || newEdges.length > 0) {
      graph.addData({
        nodes: newNodes.length > 0 ? newNodes : undefined,
        edges: newEdges.length > 0 ? newEdges : undefined
      });

      // Восстанавливаем позиции существующих нод после layout
      if (existingNodesMap.size > 0) {
        // Stop the layout to prevent force simulation from moving nodes
        graph.stopLayout();

        // Используем setTimeout чтобы дождаться окончания layout
        setTimeout(() => {
          const updates = [];
          existingNodesMap.forEach((node, nodeId) => {
            // Сохраняем оригинальные позиции из node.style или node.data
            const x = node.style?.x ?? node.data?.x;
            const y = node.style?.y ?? node.data?.y;
            if (x !== undefined && y !== undefined) {
              updates.push({
                id: nodeId,
                style: {
                  x: x,
                  y: y,
                }
              });
            }
          });
          if (updates.length > 0) {
            graph.updateNodeData(updates);
            graph.draw();
          }

          // After layout completes, save the final positions back to parent
          const allCurrentNodes = graph.getNodeData();
          const finalNodes = allCurrentNodes.map(node => ({
            ...node,
            x: node.style?.x ?? node.data?.x,
            y: node.style?.y ?? node.data?.y,
          }));
          if (onGraphUpdate) {
            onGraphUpdate(finalNodes);
          }
        }, 0);
      } else {
        graph.render();
        // Stop layout after initial render for new graphs too
        setTimeout(() => graph.stopLayout(), 100);
      }
      // Keep everything visible: zoom out only if the graph no longer fits the viewport
      setTimeout(() => { graph.fitView({ when: 'overflow' }).catch(() => {}); }, 150);
    } else if (existingEdges.length > 0) {
      // Just redraw if only styles changed
      graph.draw();
    }
  }, [graphData, isInvestigationMode]);

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="w-full h-full bg-surface-1" />
      {/* Color mode toggle */}
      <div className="absolute top-4 left-4 flex bg-surface-2 border border-surface-3 rounded-lg overflow-hidden shadow-lg text-xs">
        <span className="px-3 py-1.5 text-gray-400">Color by</span>
        {[['label', 'Known label'], ['model', 'Model risk']].map(([mode, title]) => (
          <button
            key={mode}
            onClick={() => onColorModeChange?.(mode)}
            className={`px-3 py-1.5 font-medium transition-colors ${
              colorMode === mode ? 'bg-accent-primary text-white' : 'text-gray-300 hover:text-white'
            }`}
          >
            {title}
          </button>
        ))}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-surface-2 border border-surface-3 rounded-lg p-3 shadow-lg">
        <div className="space-y-2 text-sm">
          {(colorMode === 'model'
            ? [[RISK_COLORS.critical, '≥ 80%'], [RISK_COLORS.high, '50–80%'], [RISK_COLORS.medium, '20–50%'],
               [RISK_COLORS.low, '< 20%'], [RISK_COLORS.none, 'Not scored']]
            : [[LABEL_COLORS.illicit, 'Illicit'], [LABEL_COLORS.licit, 'Licit'], [LABEL_COLORS.unknown, 'Unknown']]
          ).map(([color, text]) => (
            <div key={text} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }}></div>
              <span className="text-gray-300">{text}</span>
            </div>
          ))}
          <div className="text-xs text-gray-500 mt-2 pt-2 border-t border-surface-3 max-w-[14rem]">
            {colorMode === 'model'
              ? 'Fraud probability: wallets — GraphSAGE, edges — transaction model'
              : 'Labels: Elliptic++ (historical)'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default GraphCanvas;
