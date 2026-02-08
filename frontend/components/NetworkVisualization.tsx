"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import ForceGraph2D, { ForceGraphMethods, NodeObject } from "react-force-graph-2d";
import { GraphNode, GraphLink } from "@/types/simulation";

interface NetworkVisualizationProps {
    nodes: GraphNode[];
    links: GraphLink[];
    onNodeSelect?: (node: GraphNode | null) => void;
    selectedNodeId?: string | null;
    userNodeId: string;
    width?: number;
    height?: number;
}

/**
 * Get node color based on health (0.0 to 1.0) and type
 * Professional color palette
 */
function getHealthColor(health: number, alive: boolean, nodeType: string, isUser: boolean): string {
    if (!alive) return "#94a3b8"; // Slate grey for dead nodes

    // Different base colors for different node types
    if (nodeType === "clearing_house" || nodeType === "ccp") {
        return "#8b5cf6"; // Purple for CCPs
    }

    if (nodeType === "sector") {
        return "#0ea5e9"; // Sky blue for sectors
    }

    if (nodeType === "regulator") {
        return "#f59e0b"; // Amber for regulators
    }

    // For banks: gradient from red (unhealthy) to green (healthy)
    if (health < 0.3) {
        return "#ef4444"; // Red - critical
    } else if (health < 0.6) {
        return "#f59e0b"; // Amber - warning
    } else {
        return "#10b981"; // Green - healthy
    }
}

/**
 * Get node size based on type and capital
 */
function getNodeSize(node: GraphNode, isUser: boolean): number {
    const baseSize = node.type === "regulator" ? 10 :
        node.type === "clearing_house" || node.type === "ccp" ? 7 :
            node.type === "sector" ? 6 : 5;
    const capitalFactor = Math.log10(Math.max(node.capital, 1)) * 0.25;
    return baseSize + capitalFactor;
}

/**
 * Calculate multi-ring layout positions
 * Center: Regulators
 * Ring 1: Sectors (radius ~200)
 * Ring 2: CCPs (radius ~400)
 * Ring 3+: Banks (radius ~600+)
 */
function calculateRingPosition(node: GraphNode, index: number, nodeCounts: {
    regulators: number;
    sectors: number;
    ccps: number;
    banks: number;
    regulatorIndex: number;
    sectorIndex: number;
    ccpIndex: number;
    bankIndex: number;
}): { x: number; y: number } {
    if (node.type === "regulator") {
        // Center cluster - small radius
        const angle = (nodeCounts.regulatorIndex * 2 * Math.PI) / nodeCounts.regulators;
        const radius = 50; // Very close to center
        nodeCounts.regulatorIndex++;
        return {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
        };
    }

    if (node.type === "sector") {
        // First ring
        const angle = (nodeCounts.sectorIndex * 2 * Math.PI) / nodeCounts.sectors;
        const radius = 250;
        nodeCounts.sectorIndex++;
        return {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
        };
    }

    if (node.type === "clearing_house" || node.type === "ccp") {
        // Second ring
        const angle = (nodeCounts.ccpIndex * 2 * Math.PI) / nodeCounts.ccps;
        const radius = 450;
        nodeCounts.ccpIndex++;
        return {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
        };
    }

    // Banks in outer rings (multiple rings if needed)
    const banksPerRing = 35; // Distribute evenly
    const ringNumber = Math.floor(nodeCounts.bankIndex / banksPerRing);
    const positionInRing = nodeCounts.bankIndex % banksPerRing;
    const angle = (positionInRing * 2 * Math.PI) / banksPerRing;
    const radius = 650 + (ringNumber * 150); // Multiple rings for many banks
    nodeCounts.bankIndex++;
    return {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
    };
}

export default function NetworkVisualization({
    nodes,
    links,
    onNodeSelect,
    selectedNodeId,
    userNodeId,
    width = 800,
    height = 600,
}: NetworkVisualizationProps) {
    const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
    const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
    const [isInitialLayout, setIsInitialLayout] = useState(true);
    const nodePositionsRef = useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map());

    // Memoize graph data and use ring-based positioning
    const graphData = useMemo(() => {
        // Count nodes by type
        const nodeCounts = {
            regulators: nodes.filter(n => n.type === "regulator").length,
            sectors: nodes.filter(n => n.type === "sector").length,
            ccps: nodes.filter(n => n.type === "clearing_house" || n.type === "ccp").length,
            banks: nodes.filter(n => n.type === "bank").length,
            regulatorIndex: 0,
            sectorIndex: 0,
            ccpIndex: 0,
            bankIndex: 0,
        };

        const processedNodes = nodes.map((node) => {
            const existingPos = nodePositionsRef.current.get(node.id);
            const isUser = node.id === userNodeId;

            // Calculate ring position
            const ringPos = calculateRingPosition(node, 0, nodeCounts);

            // For initial layout, use ring positions
            if (!existingPos && isInitialLayout) {
                return {
                    ...node,
                    isUser,
                    x: ringPos.x,
                    y: ringPos.y,
                };
            }

            return {
                ...node,
                isUser,
                // Preserve positions if they exist and layout is complete
                ...(existingPos && !isInitialLayout ? existingPos : { x: ringPos.x, y: ringPos.y }),
            };
        });

        const processedLinks = links.map((link) => ({
            ...link,
            source: typeof link.source === "string" ? link.source : link.source.id,
            target: typeof link.target === "string" ? link.target : link.target.id,
        }));

        return { nodes: processedNodes, links: processedLinks };
    }, [nodes, links, userNodeId, isInitialLayout]);

    // Save node positions on every frame
    useEffect(() => {
        if (!graphRef.current || isInitialLayout) return;

        const saveInterval = setInterval(() => {
            graphData.nodes.forEach((node: any) => {
                if (node.x !== undefined && node.y !== undefined) {
                    nodePositionsRef.current.set(node.id, {
                        x: node.x,
                        y: node.y,
                        vx: node.vx || 0,
                        vy: node.vy || 0,
                    });
                }
            });
        }, 100);

        return () => clearInterval(saveInterval);
    }, [graphData, isInitialLayout]);

    // Center on graph center after initial layout
    useEffect(() => {
        if (!isInitialLayout) return;

        const timer = setTimeout(() => {
            if (graphRef.current) {
                // Configure D3 forces for multi-ring layout with adequate spacing
                const chargeForce = graphRef.current.d3Force('charge');
                if (chargeForce) {
                    (chargeForce as any).strength(-1200);
                }

                const linkForce = graphRef.current.d3Force('link');
                if (linkForce) {
                    (linkForce as any).distance(180).strength(0.1);
                }

                const centerForce = graphRef.current.d3Force('center');
                if (centerForce) {
                    (centerForce as any).strength(0.02);
                }

                graphRef.current.centerAt(0, 0, 1000);
                graphRef.current.zoom(0.8, 1000);
            }
        }, 500);

        return () => clearTimeout(timer);
    }, [graphData, isInitialLayout]);

    // Handle engine stop - mark layout as complete
    const handleEngineStop = () => {
        if (isInitialLayout) {
            setIsInitialLayout(false);
            // Save all node positions
            graphData.nodes.forEach((node: any) => {
                if (node.x !== undefined && node.y !== undefined) {
                    nodePositionsRef.current.set(node.id, {
                        x: node.x,
                        y: node.y,
                        vx: 0,
                        vy: 0,
                    });
                }
            });
        }
    };

    const handleNodeClick = (node: any) => {
        onNodeSelect?.(node as GraphNode);
    };

    // Custom node rendering with special user styling
    const drawNode = (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const n = node as GraphNode & { isUser?: boolean };
        const isUser = n.isUser || false;
        const isSelected = selectedNodeId === n.id;
        const size = getNodeSize(n, isUser);

        // Draw special halo for user node
        if (isUser) {
            // Outer glow
            const gradient = ctx.createRadialGradient(node.x, node.y, size, node.x, node.y, size + 12);
            gradient.addColorStop(0, "rgba(59, 130, 246, 0.4)");
            gradient.addColorStop(0.5, "rgba(59, 130, 246, 0.2)");
            gradient.addColorStop(1, "rgba(59, 130, 246, 0)");
            ctx.beginPath();
            ctx.arc(node.x, node.y, size + 12, 0, 2 * Math.PI);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Thick outer ring
            ctx.beginPath();
            ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
            ctx.strokeStyle = "#3b82f6";
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        // Draw selection ring
        if (isSelected && !isUser) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, size + 3, 0, 2 * Math.PI);
            ctx.strokeStyle = "#64748b";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw main node
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
        ctx.fillStyle = getHealthColor(n.health, n.alive, n.type, isUser);
        ctx.fill();

        // Draw border
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw label for user node
        if (isUser) {
            ctx.font = "bold 11px Inter, sans-serif";
            ctx.fillStyle = "#3b82f6";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("YOU", node.x, node.y + size + 18);
        }

        // Draw status indicator for dead nodes
        if (!n.alive) {
            ctx.font = "12px Arial";
            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("✕", node.x, node.y);
        }
    };

    return (
        <div className="relative bg-white rounded-xl border-2 border-slate-200 overflow-hidden shadow-lg">
            <ForceGraph2D
                ref={graphRef}
                width={width}
                height={height}
                graphData={graphData}
                backgroundColor="#ffffff"
                nodeRelSize={6}
                nodeVal={(node: any) => {
                    const n = node as GraphNode & { isUser?: boolean };
                    return getNodeSize(n, n.isUser || false);
                }}
                nodeCanvasObject={drawNode}
                nodeLabel={(node: any) => {
                    const n = node as GraphNode;
                    return `
            <div style="background: rgba(255,255,255,0.98); padding: 12px; border-radius: 8px; color: #0f172a; font-family: 'Inter', sans-serif; border: 2px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
              <div style="font-weight: bold; font-size: 14px; margin-bottom: 6px; color: #1e293b;">${n.id}</div>
              <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Type: <span style="color: #3b82f6; font-weight: 600;">${n.type}</span></div>
              <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Health: <span style="font-weight: 600; color: ${n.health > 0.6 ? '#10b981' : n.health > 0.3 ? '#f59e0b' : '#ef4444'};">${(n.health * 100).toFixed(1)}%</span></div>
              <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Capital: <span style="font-weight: 600;">$${(n.capital / 1e6).toFixed(2)}M</span></div>
              <div style="font-size: 12px; color: #64748b;">Status: <span style="font-weight: 600; color: ${n.alive ? '#10b981' : '#ef4444'};">${n.alive ? '✓ Active' : '✗ Defaulted'}</span></div>
            </div>
          `;
                }}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNode(node as GraphNode | null)}
                linkColor={() => "#1e293b"}
                linkWidth={() => 0.5}
                linkDirectionalArrowLength={0}
                linkDirectionalArrowRelPos={1}
                linkDirectionalParticles={0}
                enableNodeDrag={true}
                cooldownTicks={isInitialLayout ? 200 : 0}
                warmupTicks={0}
                d3VelocityDecay={0.3}
                d3AlphaDecay={isInitialLayout ? 0.01 : 0.1}
                d3AlphaMin={0.001}
                onEngineStop={handleEngineStop}
            />

            {/* Legend */}
            <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm p-4 rounded-xl border-2 border-slate-200 text-xs text-slate-700 space-y-2 shadow-lg">
                <div className="font-bold text-slate-900 mb-3 text-sm">Legend</div>
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-blue-600 border-2 border-blue-400 shadow-lg shadow-blue-500/50"></div>
                        <span className="font-semibold">Your Entity</span>
                    </div>
                    <div className="border-t border-slate-200 my-2"></div>
                    <div className="font-semibold text-slate-800 text-[10px] uppercase tracking-wide">Network Layers</div>
                    <div className="text-[10px] text-slate-600 space-y-1">
                        <div>• Center: Regulators (RBI, SEBI, etc.)</div>
                        <div>• Ring 1: Economic Sectors</div>
                        <div>• Ring 2: Clearing Houses (CCPs)</div>
                        <div>• Ring 3+: Banks</div>
                    </div>
                    <div className="border-t border-slate-200 my-2"></div>
                    <div className="font-semibold text-slate-800 text-[10px] uppercase tracking-wide">Health Status</div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                        <span>Healthy (60-100%)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                        <span>Warning (30-60%)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                        <span>Critical (&lt;30%)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-slate-400"></div>
                        <span>Defaulted</span>
                    </div>
                    <div className="border-t border-slate-200 my-2"></div>
                    <div className="font-semibold text-slate-800 text-[10px] uppercase tracking-wide">Entity Types</div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                        <span>Regulator</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-sky-500"></div>
                        <span>Sector</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                        <span>Clearing House</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                        <span>Bank</span>
                    </div>
                </div>
            </div>

            {/* Network Info */}
            <div className="absolute top-4 left-4 bg-gradient-to-r from-slate-900 to-slate-700 text-white px-4 py-2 rounded-lg font-semibold text-sm shadow-lg">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
                    Multi-Ring Network Topology
                </div>
            </div>
        </div>
    );
}
