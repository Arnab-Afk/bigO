'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Skeleton } from '@/components/ui/skeleton';

// Dynamically import ForceGraph3D to avoid SSR issues
const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), {
  ssr: false,
  loading: () => <Skeleton className="h-[600px] w-full" />,
});

interface GraphNode {
  id: string;
  name: string;
  default_probability: number;
  risk_level: string;
  pagerank: number;
  degree: number;
  capital?: number;
  // 3D position properties
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
}

interface Network3DVisualizationProps {
  nodes: GraphNode[];
  links: GraphLink[];
  onNodeClick?: (node: GraphNode | null) => void;
  selectedNode?: GraphNode | null;
  highlightNodes?: Set<string>;
  highlightLinks?: Set<string>;
  // Control props
  nodeSize?: number;
  showEdges?: boolean;
  cameraDistance?: number;
}

/**
 * Get node color based on risk level with green to red gradient
 */
const getNodeColor = (node: GraphNode, highlightNodes: Set<string>): string => {
  // Dim non-highlighted nodes
  if (highlightNodes.size > 0 && !highlightNodes.has(node.id)) {
    return 'rgba(100, 100, 100, 0.3)';
  }

  switch (node.risk_level) {
    case 'high':
      return '#ef4444'; // Red
    case 'medium':
      return '#f59e0b'; // Orange/Amber
    case 'low':
      return '#10b981'; // Green
    default:
      return '#6b7280'; // Gray
  }
};

/**
 * Calculate node size using logarithmic scale based on capital
 * Falls back to PageRank if capital is not available
 */
const getNodeSize = (node: GraphNode, sizeFactor: number = 1): number => {
  if (node.capital && node.capital > 0) {
    // Logarithmic scale: log10(capital) with minimum size
    const logScale = Math.log10(node.capital) - 3; // Normalize (assuming capital in thousands)
    return Math.max(2, logScale * 2) * sizeFactor;
  }
  // Fallback to PageRank-based sizing
  return (5 + node.pagerank * 50) * sizeFactor;
};

/**
 * Get link color based on weight with opacity
 */
const getLinkColor = (link: GraphLink, highlightLinks: Set<string>): string => {
  const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
  const targetId = typeof link.target === 'object' ? link.target.id : link.target;
  const linkId = `${sourceId}-${targetId}`;

  // Dim non-highlighted links
  if (highlightLinks.size > 0 && !highlightLinks.has(linkId)) {
    return 'rgba(150, 150, 150, 0.05)';
  }

  // Color intensity based on weight
  if (link.weight > 0.7) {
    return 'rgba(239, 68, 68, 0.6)'; // Red - high risk
  } else if (link.weight > 0.4) {
    return 'rgba(245, 158, 11, 0.5)'; // Orange - medium risk
  } else {
    return 'rgba(107, 114, 128, 0.4)'; // Gray - low risk
  }
};

/**
 * Get link width based on weight
 */
const getLinkWidth = (link: GraphLink): number => {
  return Math.max(0.5, link.weight * 3);
};

export default function Network3DVisualization({
  nodes,
  links,
  onNodeClick,
  selectedNode,
  highlightNodes = new Set(),
  highlightLinks = new Set(),
  nodeSize = 1,
  showEdges = true,
  cameraDistance = 1000,
}: Network3DVisualizationProps) {
  const fgRef = useRef<any>();
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [controlsEnabled, setControlsEnabled] = useState(true);

  // Process graph data
  const graphData = useMemo(() => {
    return {
      nodes: nodes.map((node) => ({ ...node })),
      links: links.map((link) => ({ ...link })),
    };
  }, [nodes, links]);

  // Initialize camera position and apply forces
  useEffect(() => {
    if (!fgRef.current) return;

    // Configure 3D forces for better visualization
    const fg = fgRef.current;

    // Charge force - repulsion between nodes
    fg.d3Force('charge')?.strength(-120);

    // Link force - controls edge length
    fg.d3Force('link')?.distance((link: GraphLink) => {
      // Longer distances for weaker connections
      return 30 + (1 - link.weight) * 50;
    });

    // Center force - gravity towards center
    fg.d3Force('center')?.strength(0.05);

    // Add collision force to prevent overlap
    fg.d3Force('collision', fg.d3.forceCollide((node: GraphNode) => {
      return getNodeSize(node, nodeSize) + 2;
    }));

    // Set initial camera position
    const distance = cameraDistance;
    fg.cameraPosition(
      { x: distance * 0.5, y: distance * 0.5, z: distance },
      { x: 0, y: 0, z: 0 },
      1000
    );
  }, [nodeSize, cameraDistance]);

  // Handle node click
  const handleNodeClick = useCallback(
    (node: any) => {
      if (onNodeClick) {
        onNodeClick(node as GraphNode);
      }

      // Animate camera to focus on clicked node
      if (fgRef.current && node) {
        const distance = 200;
        fgRef.current.cameraPosition(
          { x: node.x + distance * 0.5, y: node.y + distance * 0.5, z: node.z + distance },
          node,
          1000
        );
      }
    },
    [onNodeClick]
  );

  // Handle node hover
  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node as GraphNode | null);
    // Change cursor style
    if (fgRef.current) {
      fgRef.current.controls().enabled = node === null;
    }
  }, []);

  // Node label formatting
  const getNodeLabel = useCallback((node: any) => {
    const n = node as GraphNode;
    return `
      <div style="
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
      ">
        <div style="font-weight: 700; font-size: 14px; margin-bottom: 8px; color: #fff;">
          ${n.name}
        </div>
        <div style="font-size: 12px; opacity: 0.9; line-height: 1.6;">
          <div style="margin-bottom: 4px;">
            <span style="opacity: 0.7;">Risk Level:</span>
            <span style="color: ${n.risk_level === 'high' ? '#ef4444' : n.risk_level === 'medium' ? '#f59e0b' : '#10b981'}; font-weight: 600;">
              ${n.risk_level.toUpperCase()}
            </span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="opacity: 0.7;">Default Probability:</span>
            <span style="font-weight: 600;">${(n.default_probability * 100).toFixed(2)}%</span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="opacity: 0.7;">PageRank:</span>
            <span style="font-weight: 600;">${n.pagerank.toFixed(6)}</span>
          </div>
          ${n.capital ? `
          <div style="margin-bottom: 4px;">
            <span style="opacity: 0.7;">Capital:</span>
            <span style="font-weight: 600;">$${(n.capital / 1e6).toFixed(2)}M</span>
          </div>
          ` : ''}
          <div>
            <span style="opacity: 0.7;">Connections:</span>
            <span style="font-weight: 600;">${n.degree}</span>
          </div>
        </div>
      </div>
    `;
  }, []);

  // Custom node object for Three.js rendering
  const nodeThreeObject = useCallback(
    (node: any) => {
      const n = node as GraphNode;
      const isSelected = selectedNode?.id === n.id;
      const isHighlighted = highlightNodes.has(n.id);

      // Create sphere geometry
      const geometry = new (window as any).THREE.SphereGeometry(
        getNodeSize(n, nodeSize),
        16,
        16
      );

      // Material with emissive properties for glow effect
      const material = new (window as any).THREE.MeshLambertMaterial({
        color: getNodeColor(n, highlightNodes),
        emissive: isSelected || isHighlighted ? getNodeColor(n, new Set()) : '#000000',
        emissiveIntensity: isSelected ? 0.5 : isHighlighted ? 0.3 : 0,
        transparent: highlightNodes.size > 0 && !highlightNodes.has(n.id),
        opacity: highlightNodes.size > 0 && !highlightNodes.has(n.id) ? 0.3 : 1,
      });

      const mesh = new (window as any).THREE.Mesh(geometry, material);

      // Add ring for selected node
      if (isSelected) {
        const ringGeometry = new (window as any).THREE.RingGeometry(
          getNodeSize(n, nodeSize) + 2,
          getNodeSize(n, nodeSize) + 3,
          32
        );
        const ringMaterial = new (window as any).THREE.MeshBasicMaterial({
          color: '#ffffff',
          side: (window as any).THREE.DoubleSide,
        });
        const ring = new (window as any).THREE.Mesh(ringGeometry, ringMaterial);
        mesh.add(ring);
      }

      return mesh;
    },
    [selectedNode, highlightNodes, nodeSize]
  );

  return (
    <div className="relative h-full w-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 rounded-lg overflow-hidden">
      {typeof window !== 'undefined' && (
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          nodeLabel={getNodeLabel}
          nodeColor={(node: any) => getNodeColor(node as GraphNode, highlightNodes)}
          nodeVal={(node: any) => getNodeSize(node as GraphNode, nodeSize)}
          nodeThreeObject={nodeThreeObject}
          nodeOpacity={1}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          linkColor={(link: any) => getLinkColor(link as GraphLink, highlightLinks)}
          linkWidth={(link: any) => showEdges ? getLinkWidth(link as GraphLink) : 0}
          linkOpacity={showEdges ? 0.6 : 0}
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={(link: any) => (link as GraphLink).weight * 2}
          linkDirectionalParticleSpeed={0.005}
          enableNodeDrag={true}
          enableNavigationControls={controlsEnabled}
          showNavInfo={false}
          backgroundColor="rgba(0, 0, 0, 0)"
          controlType="orbit"
          warmupTicks={100}
          cooldownTicks={200}
          cooldownTime={5000}
        />
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm p-4 rounded-lg border border-white/10 text-white text-xs space-y-3 shadow-xl max-w-[200px]">
        <div className="font-bold text-sm mb-2">Risk Levels</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-red-500 shadow-lg shadow-red-500/50" />
            <span>High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-orange-500 shadow-lg shadow-orange-500/50" />
            <span>Medium Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-green-500 shadow-lg shadow-green-500/50" />
            <span>Low Risk</span>
          </div>
        </div>
        <div className="border-t border-white/10 pt-3 mt-3">
          <div className="font-semibold mb-1">Node Size</div>
          <p className="text-[10px] opacity-70 leading-relaxed">
            Proportional to capital (logarithmic scale) or PageRank
          </p>
        </div>
        <div className="border-t border-white/10 pt-3">
          <div className="font-semibold mb-1">Edge Intensity</div>
          <p className="text-[10px] opacity-70 leading-relaxed">
            Thicker/brighter edges indicate stronger dependencies
          </p>
        </div>
      </div>

      {/* Navigation Hint */}
      <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg text-white text-xs border border-white/10">
        <div className="flex items-center gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
            />
          </svg>
          <span>Drag to rotate • Scroll to zoom • Click nodes</span>
        </div>
      </div>
    </div>
  );
}
