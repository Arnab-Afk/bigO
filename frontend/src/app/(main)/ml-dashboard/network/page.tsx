'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import dynamic from 'next/dynamic';
import { ccpApi } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertCircle, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
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
}

interface GraphLink {
  source: string;
  target: string;
  weight: number;
}

export default function NetworkPage() {
  const fgRef = useRef<any>();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [weightThreshold, setWeightThreshold] = useState<number>(0.05);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set());

  const { data: networkData, isLoading, error } = useQuery({
    queryKey: ['ccp-network'],
    queryFn: () => ccpApi.getNetworkData(),
    refetchInterval: 30000,
  });

  // Filter nodes and links based on criteria
  const filteredData = useCallback(() => {
    if (!networkData) return { nodes: [], links: [] };

    const nodes = networkData.nodes.filter((node) => {
      if (riskFilter === 'all') return true;
      return node.risk_level === riskFilter;
    });

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = networkData.edges
      .filter((edge) => edge.weight >= weightThreshold)
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        weight: edge.weight,
      }));

    return { nodes, links };
  }, [networkData, riskFilter, weightThreshold]);

  const { nodes, links } = filteredData();

  // Handle node click
  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);

    // Highlight connected nodes
    const connectedNodes = new Set<string>();
    const connectedLinks = new Set<string>();

    links.forEach((link: any) => {
      if (link.source.id === node.id || link.source === node.id) {
        connectedNodes.add(typeof link.target === 'object' ? link.target.id : link.target);
        connectedLinks.add(`${link.source}-${link.target}`);
      }
      if (link.target.id === node.id || link.target === node.id) {
        connectedNodes.add(typeof link.source === 'object' ? link.source.id : link.source);
        connectedLinks.add(`${link.source}-${link.target}`);
      }
    });

    connectedNodes.add(node.id);
    setHighlightNodes(connectedNodes);
    setHighlightLinks(connectedLinks);
  }, [links]);

  // Node color based on risk level
  const getNodeColor = (node: GraphNode) => {
    if (highlightNodes.size > 0 && !highlightNodes.has(node.id)) {
      return 'rgba(100, 100, 100, 0.2)';
    }
    switch (node.risk_level) {
      case 'high':
        return '#ef4444';
      case 'medium':
        return '#f59e0b';
      case 'low':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  // Node size based on PageRank
  const getNodeSize = (node: GraphNode) => {
    return 5 + node.pagerank * 50;
  };

  // Link opacity based on weight
  const getLinkColor = (link: any) => {
    const linkId = `${typeof link.source === 'object' ? link.source.id : link.source}-${typeof link.target === 'object' ? link.target.id : link.target}`;
    if (highlightNodes.size > 0 && !highlightLinks.has(linkId)) {
      return 'rgba(150, 150, 150, 0.1)';
    }
    return link.weight > 0.5
      ? 'rgba(239, 68, 68, 0.4)'
      : link.weight > 0.3
      ? 'rgba(245, 158, 11, 0.4)'
      : 'rgba(107, 114, 128, 0.3)';
  };

  // Zoom controls
  const handleZoomIn = () => {
    if (fgRef.current) {
      fgRef.current.zoom(fgRef.current.zoom() * 1.2, 400);
    }
  };

  const handleZoomOut = () => {
    if (fgRef.current) {
      fgRef.current.zoom(fgRef.current.zoom() / 1.2, 400);
    }
  };

  const handleZoomFit = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 50);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Network Visualization</h1>
        <p className="text-muted-foreground">
          Interactive visualization of financial network dependencies
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load network: {(error as Error).message}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-4">
        {/* Graph Container */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Network Graph</CardTitle>
                  <CardDescription>
                    {nodes.length} nodes, {links.length} edges
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button size="icon" variant="outline" onClick={handleZoomIn}>
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="outline" onClick={handleZoomOut}>
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="outline" onClick={handleZoomFit}>
                    <Maximize2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-[600px] w-full" />
              ) : nodes.length > 0 ? (
                <div className="h-[600px] w-full border rounded-lg overflow-hidden bg-muted/20">
                  <ForceGraph2D
                    ref={fgRef}
                    graphData={{ nodes, links }}
                    nodeId="id"
                    nodeLabel={(node: any) => `${node.name}\nDefault Prob: ${(node.default_probability * 100).toFixed(2)}%\nPageRank: ${node.pagerank.toFixed(4)}`}
                    nodeColor={(node: any) => getNodeColor(node)}
                    nodeVal={(node: any) => getNodeSize(node)}
                    linkColor={(link: any) => getLinkColor(link)}
                    linkWidth={(link: any) => link.weight * 2}
                    linkDirectionalParticles={2}
                    linkDirectionalParticleWidth={(link: any) => link.weight * 3}
                    onNodeClick={handleNodeClick}
                    cooldownTicks={100}
                    onEngineStop={() => fgRef.current?.zoomToFit(400, 50)}
                  />
                </div>
              ) : (
                <div className="flex h-[600px] items-center justify-center text-muted-foreground">
                  No network data available. Run a simulation first.
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Controls and Info Panel */}
        <div className="space-y-4">
          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle>Filters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Risk Level Filter */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Risk Level</label>
                <Select value={riskFilter} onValueChange={setRiskFilter}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Levels</SelectItem>
                    <SelectItem value="high">High Risk</SelectItem>
                    <SelectItem value="medium">Medium Risk</SelectItem>
                    <SelectItem value="low">Low Risk</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Weight Threshold */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Edge Weight Threshold: {weightThreshold.toFixed(2)}
                </label>
                <Slider
                  value={[weightThreshold]}
                  onValueChange={([value]) => setWeightThreshold(value)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Hide edges below this weight
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Legend */}
          <Card>
            <CardHeader>
              <CardTitle>Legend</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <p className="text-sm font-medium">Risk Levels</p>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-red-500" />
                  <span className="text-sm">High Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-orange-500" />
                  <span className="text-sm">Medium Risk</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-green-500" />
                  <span className="text-sm">Low Risk</span>
                </div>
              </div>
              <div className="space-y-2 pt-2 border-t">
                <p className="text-sm font-medium">Node Size</p>
                <p className="text-xs text-muted-foreground">
                  Larger nodes have higher PageRank (systemic importance)
                </p>
              </div>
              <div className="space-y-2 pt-2 border-t">
                <p className="text-sm font-medium">Edge Color</p>
                <p className="text-xs text-muted-foreground">
                  Darker edges indicate stronger dependencies
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Selected Node Info */}
          {selectedNode && (
            <Card>
              <CardHeader>
                <CardTitle>Selected Node</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-sm font-medium">{selectedNode.name}</p>
                  <Badge
                    variant={
                      selectedNode.risk_level === 'high'
                        ? 'destructive'
                        : selectedNode.risk_level === 'medium'
                        ? 'default'
                        : 'secondary'
                    }
                    className="mt-1"
                  >
                    {selectedNode.risk_level.toUpperCase()}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Default Prob:</span>
                    <span className="font-medium">
                      {(selectedNode.default_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">PageRank:</span>
                    <span className="font-medium">
                      {selectedNode.pagerank.toFixed(6)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Degree:</span>
                    <span className="font-medium">{selectedNode.degree}</span>
                  </div>
                </div>
                <Button
                  className="w-full"
                  size="sm"
                  onClick={() => {
                    window.location.href = `/ml-dashboard/banks/${encodeURIComponent(selectedNode.name)}`;
                  }}
                >
                  View Details
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
