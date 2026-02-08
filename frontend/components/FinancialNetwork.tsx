"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import ForceGraph2D, { ForceGraphMethods } from "react-force-graph-2d";
import { GraphNode, GraphLink } from "@/types/simulation";

interface FinancialNetworkProps {
    nodes: GraphNode[];
    links: GraphLink[];
    onNodeSelect?: (node: GraphNode | null) => void;
    selectedNodeId?: string | null;
    width?: number;
    height?: number;
}

/**
 * Get node color based on health (0.0 to 1.0) and type
 * Uses a professional color palette with better visual distinction
 */
function getHealthColor(health: number, alive: boolean, nodeType: string): string {
    if (!alive) return "#64748b"; // Slate grey for dead nodes

    // Different base colors for different node types
    if (nodeType === "clearing_house" || nodeType === "ccp") {
        return "#8b5cf6"; // Purple for CCPs
    }

    if (nodeType === "sector") {
        return "#06b6d4"; // Cyan for sectors
    }

    if (nodeType === "regulator") {
        return "#f59e0b"; // Amber for regulators
    }

    // For banks: gradient from red (unhealthy) to emerald (healthy)
    if (health < 0.3) {
        // Critical: Deep red to red
        const ratio = health / 0.3;
        return `rgb(${239}, ${Math.round(68 + ratio * 100)}, ${Math.round(68 + ratio * 50)})`; // #ef4444 to lighter
    } else if (health < 0.6) {
        // Warning: Orange to yellow
        const ratio = (health - 0.3) / 0.3;
        return `rgb(${Math.round(251 - ratio * 41)}, ${Math.round(146 + ratio * 45)}, ${Math.round(60 + ratio * 20)})`; // #fb923d to #eab308
    } else {
        // Healthy: Light green to emerald
        const ratio = (health - 0.6) / 0.4;
        return `rgb(${Math.round(134 - ratio * 84)}, ${Math.round(239 - ratio * 18)}, ${Math.round(172 - ratio * 72)})`; // #86efac to #10b981
    }
}

/**
 * Get node size based on type and capital
 */
function getNodeSize(node: GraphNode): number {
    const baseSize = node.type === "clearing_house" ? 8 : node.type === "bank" ? 6 : 5;
    const capitalFactor = Math.log10(Math.max(node.capital, 1)) * 0.5;
    return baseSize + capitalFactor;
}

export default function FinancialNetwork({
    nodes,
    links,
    onNodeSelect,
    selectedNodeId,
    width = 800,
    height = 600,
}: FinancialNetworkProps) {
    const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
    const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
    const nodePositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
    const [hasPositioned, setHasPositioned] = useState(false);

    // Memoize graph data to prevent rerenders and preserve positions
    const graphData = useMemo(() => {
        const processedNodes = nodes.map((node) => {
            const existing = nodePositionsRef.current.get(node.id);
            return {
                ...node,
                // Preserve existing positions if available and graph has been positioned
                ...(hasPositioned && existing ? { x: existing.x, y: existing.y, vx: 0, vy: 0 } : {}),
            };
        });

        const processedLinks = links.map((link) => ({
            ...link,
            source: typeof link.source === "string" ? link.source : link.source.id,
            target: typeof link.target === "string" ? link.target : link.target.id,
        }));

        return { nodes: processedNodes, links: processedLinks };
    }, [nodes, links, hasPositioned]);

    // Save node positions on engine tick
    useEffect(() => {
        const savePositions = () => {
            if (graphRef.current) {
                graphData.nodes.forEach((node: any) => {
                    if (node.x !== undefined && node.y !== undefined) {
                        nodePositionsRef.current.set(node.id, { x: node.x, y: node.y });
                    }
                });
            }
        };

        const interval = setInterval(savePositions, 100);
        return () => clearInterval(interval);
    }, [graphData]);

    // Center graph on mount and mark as positioned
    useEffect(() => {
        const timer = setTimeout(() => {
            if (graphRef.current) {
                graphRef.current.zoomToFit(400);
                setHasPositioned(true);
            }
        }, 500);

        return () => clearTimeout(timer);
    }, []);

    const handleNodeClick = (node: any) => {
        onNodeSelect?.(node as GraphNode);
    };

    return (
        <div className="relative bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
            <ForceGraph2D
                ref={graphRef}
                width={width}
                height={height}
                graphData={graphData}
                backgroundColor="#020617"
                nodeRelSize={6}
                nodeVal={(node: any) => getNodeSize(node as GraphNode)}
                nodeColor={(node: any) => {
                    const n = node as GraphNode;
                    // Highlight selected node
                    if (selectedNodeId === n.id) {
                        return "#3b82f6"; // Blue highlight
                    }
                    return getHealthColor(n.health, n.alive, n.type);
                }}
                nodeLabel={(node: any) => {
                    const n = node as GraphNode;
                    return `
            <div style="background: rgba(0,0,0,0.9); padding: 8px; border-radius: 4px; color: white; font-family: monospace;">
              <div><strong>${n.id}</strong></div>
              <div>Type: ${n.type}</div>
              <div>Health: ${(n.health * 100).toFixed(1)}%</div>
              <div>Capital: $${(n.capital / 1e6).toFixed(2)}M</div>
              <div>Alive: ${n.alive ? "✓" : "✗"}</div>
            </div>
          `;
                }}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNode(node as GraphNode | null)}
                linkColor={() => "#475569"}
                linkWidth={(link: any) => {
                    const l = link as GraphLink;
                    // Scale width based on weight (loan size)
                    return Math.max(0.5, Math.log10(l.weight + 1) * 0.5);
                }}
                linkDirectionalArrowLength={3}
                linkDirectionalArrowRelPos={1}
                linkDirectionalParticles={(link: any) => {
                    const l = link as GraphLink;
                    // More particles for larger loans
                    return Math.min(4, Math.max(1, Math.floor(l.weight / 1e6)));
                }}
                linkDirectionalParticleWidth={2}
                linkDirectionalParticleSpeed={0.003}
                linkDirectionalParticleColor={() => "#22d3ee"}
                enableNodeDrag={true}
                cooldownTicks={hasPositioned ? 0 : 100}
                warmupTicks={0}
                d3VelocityDecay={0.3}
                onEngineStop={() => {
                    if (!hasPositioned) {
                        setHasPositioned(true);
                        graphRef.current?.zoomToFit(400);
                    }
                }}
            />

            {/* Legend */}
            <div className="absolute top-4 right-4 bg-slate-900/90 backdrop-blur-sm p-4 rounded-lg border border-slate-700 text-xs text-slate-300 space-y-2">
                <div className="font-bold text-slate-100 mb-2">Legend</div>
                <div className="space-y-1.5">
                    <div className="font-semibold text-slate-200 text-[10px] uppercase tracking-wide">Banks</div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#10b981' }}></div>
                        <span>Healthy (60-100%)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#eab308' }}></div>
                        <span>Warning (30-60%)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }}></div>
                        <span>Critical (0-30%)</span>
                    </div>
                </div>
                <div className="border-t border-slate-700 pt-2 mt-2 space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#8b5cf6' }}></div>
                        <span>Clearing House</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#06b6d4' }}></div>
                        <span>Sectors</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f59e0b' }}></div>
                        <span>Regulator</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-slate-600"></div>
                        <span>Dead</span>
                    </div>
                </div>
                <div className="border-t border-slate-700 pt-2 mt-2">
                    <div className="flex items-center gap-2">
                        <div className="text-cyan-400">→</div>
                        <span>Liquidity Flow</span>
                    </div>
                </div>
            </div>

            {/* Stats overlay */}
            <div className="absolute bottom-4 left-4 bg-slate-900/90 backdrop-blur-sm p-3 rounded-lg border border-slate-700 text-xs text-slate-300 font-mono">
                <div>Nodes: {nodes.length}</div>
                <div>Edges: {links.length}</div>
                <div>
                    Alive: {nodes.filter((n) => n.alive).length}/{nodes.length}
                </div>
            </div>
        </div>
    );
}
