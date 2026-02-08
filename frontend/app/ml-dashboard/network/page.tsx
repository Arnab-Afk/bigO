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
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertCircle, ZoomIn, ZoomOut, Maximize2, Box, Grid3x3, Database } from 'lucide-react';
import Network3DVisualization from '@/components/Network3DVisualization';
import Network3DControls from '@/components/Network3DControls';

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
  capital?: number;
}

interface GraphLink {
  source: string;
  target: string;
  weight: number;
}

// Dashboard data structure
interface DashboardNode {
  id: string;
  group: string;
  health: number;
  color: string;
  alive: boolean;
  size: number;
  mode: string;
  type: string;
  capital?: number;
  crar?: number;
  liquidity?: number;
  npa_ratio?: number;
  risk_appetite?: number;
  economic_health?: number;
  debt_load?: number;
}

interface DashboardLink {
  source: string;
  target: string;
  value: number;
  type: string;
}

// Transform dashboard data to expected format
const transformDashboardData = (dashboardData: any): { nodes: GraphNode[], links: GraphLink[] } => {
  const nodes = dashboardData.d3_network.nodes.map((node: DashboardNode) => ({
    id: node.id,
    name: node.id,
    default_probability: 1 - node.health,
    risk_level: node.health > 0.6 ? 'low' : node.health > 0.3 ? 'medium' : 'high',
    pagerank: node.size / 100,
    degree: 10,
    capital: node.capital || 0,
  }));

  const links = dashboardData.d3_network.links.map((link: DashboardLink) => ({
    source: link.source,
    target: link.target,
    weight: link.value / 1000, // Normalize weight
  }));

  return { nodes, links };
};

export default function NetworkPage() {
  const fgRef = useRef<any>();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [weightThreshold, setWeightThreshold] = useState<number>(0.05);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set());
  const [dataSource, setDataSource] = useState<'ccp' | 'dashboard'>('dashboard');

  // 3D visualization state
  const [is3DMode, setIs3DMode] = useState<boolean>(false);
  const [nodeSize3D, setNodeSize3D] = useState<number>(1);
  const [showEdges3D, setShowEdges3D] = useState<boolean>(true);
  const [cameraDistance3D, setCameraDistance3D] = useState<number>(1000);

  // CCP Network data
  const { data: networkData, isLoading: isCCPLoading, error: ccpError } = useQuery({
    queryKey: ['ccp-network'],
    queryFn: () => ccpApi.getNetworkData(),
    refetchInterval: 30000,
    enabled: dataSource === 'ccp',
  });

  // Dashboard data
  const { data: dashboardData, isLoading: isDashboardLoading, error: dashboardError } = useQuery({
    queryKey: ['dashboard-network'],
    queryFn: async () => {
      const response = await fetch('http://localhost:17170/api/v1/abm/dashboard-data');
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      return response.json();
    },
    refetchInterval: 30000,
    enabled: dataSource === 'dashboard',
  });

  const isLoading = dataSource === 'ccp' ? isCCPLoading : isDashboardLoading;
  const error = dataSource === 'ccp' ? ccpError : dashboardError;

  // Filter nodes and links based on criteria
  const filteredData = useCallback(() => {
    let sourceData;
    
    if (dataSource === 'dashboard' && dashboardData) {
      sourceData = transformDashboardData(dashboardData);
    } else if (dataSource === 'ccp' && networkData) {
      sourceData = {
        nodes: networkData.nodes,
        links: networkData.edges.map((edge: any) => ({
          source: edge.source,
          target: edge.target,
          weight: edge.weight,
        })),
      };
    } else {
      return { nodes: [], links: [] };
    }

    const nodes = sourceData.nodes.filter((node) => {
      if (riskFilter === 'all') return true;
      return node.risk_level === riskFilter;
    });

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = sourceData.links
      .filter((link) => link.weight >= weightThreshold)
      .filter((link) => nodeIds.has(link.source) && nodeIds.has(link.target));

    return { nodes, links };
  }, [networkData, dashboardData, dataSource, riskFilter, weightThreshold]);

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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Network Visualization</h1>
          <p className="text-muted-foreground">
            Interactive visualization of financial network dependencies
          </p>
        </div>

        <div className="flex gap-3">
          {/* Data Source Toggle */}
          <div className="flex items-center gap-3 bg-muted/50 px-4 py-2 rounded-lg border">
            <div className="flex items-center gap-2">
              <Database className={`h-4 w-4 ${dataSource === 'ccp' ? 'text-primary' : 'text-muted-foreground'}`} />
              <Label htmlFor="data-source" className="text-sm font-medium cursor-pointer">
                CCP
              </Label>
            </div>
            <Switch
              id="data-source"
              checked={dataSource === 'dashboard'}
              onCheckedChange={(checked) => setDataSource(checked ? 'dashboard' : 'ccp')}
            />
            <div className="flex items-center gap-2">
              <Label htmlFor="data-source" className="text-sm font-medium cursor-pointer">
                ABM
              </Label>
              <Database className={`h-4 w-4 ${dataSource === 'dashboard' ? 'text-primary' : 'text-muted-foreground'}`} />
            </div>
          </div>

          {/* 2D/3D Toggle */}
          <div className="flex items-center gap-3 bg-muted/50 px-4 py-2 rounded-lg border">
            <div className="flex items-center gap-2">
              <Grid3x3 className={`h-4 w-4 ${!is3DMode ? 'text-primary' : 'text-muted-foreground'}`} />
              <Label htmlFor="visualization-mode" className="text-sm font-medium cursor-pointer">
                2D
              </Label>
            </div>
            <Switch
              id="visualization-mode"
              checked={is3DMode}
              onCheckedChange={setIs3DMode}
            />
            <div className="flex items-center gap-2">
              <Label htmlFor="visualization-mode" className="text-sm font-medium cursor-pointer">
                3D
              </Label>
              <Box className={`h-4 w-4 ${is3DMode ? 'text-primary' : 'text-muted-foreground'}`} />
            </div>
          </div>
        </div>
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
                  <CardTitle>Network Graph {is3DMode ? '(3D)' : '(2D)'}</CardTitle>
                  <CardDescription>
                    {nodes.length} nodes, {links.length} edges
                  </CardDescription>
                </div>
                {!is3DMode && (
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
                )}
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-[600px] w-full" />
              ) : nodes.length > 0 ? (
                <div className="h-[600px] w-full border rounded-lg overflow-hidden">
                  {is3DMode ? (
                    <Network3DVisualization
                      nodes={nodes}
                      links={links}
                      onNodeClick={handleNodeClick}
                      selectedNode={selectedNode}
                      highlightNodes={highlightNodes}
                      highlightLinks={highlightLinks}
                      nodeSize={nodeSize3D}
                      showEdges={showEdges3D}
                      cameraDistance={cameraDistance3D}
                    />
                  ) : (
                    <div className="h-full w-full bg-muted/20">
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
                  )}
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
          {/* 3D Controls (only in 3D mode) */}
          {is3DMode ? (
            <Network3DControls
              nodeSize={nodeSize3D}
              onNodeSizeChange={setNodeSize3D}
              showEdges={showEdges3D}
              onShowEdgesChange={setShowEdges3D}
              cameraDistance={cameraDistance3D}
              onCameraDistanceChange={setCameraDistance3D}
              nodeCount={nodes.length}
              edgeCount={links.length}
            />
          ) : (
            <>
              {/* Filters (2D mode only) */}
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
            </>
          )}

          {/* Legend (2D mode only) */}
          {!is3DMode && (
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
          )}

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
