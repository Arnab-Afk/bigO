'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/src/components/ui/card';
import { Slider } from '@/src/components/ui/slider';
import { Switch } from '@/src/components/ui/switch';
import { Label } from '@/src/components/ui/label';
import { Button } from '@/src/components/ui/button';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Eye,
  EyeOff,
  Expand,
  Shrink,
} from 'lucide-react';

interface Network3DControlsProps {
  // Node controls
  nodeSize: number;
  onNodeSizeChange: (size: number) => void;

  // Edge controls
  showEdges: boolean;
  onShowEdgesChange: (show: boolean) => void;

  // Camera controls
  cameraDistance: number;
  onCameraDistanceChange: (distance: number) => void;

  // Camera actions
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onZoomFit?: () => void;
  onResetCamera?: () => void;

  // Physics controls (optional)
  enablePhysics?: boolean;
  onEnablePhysicsChange?: (enable: boolean) => void;

  // Additional info
  nodeCount?: number;
  edgeCount?: number;
}

export default function Network3DControls({
  nodeSize,
  onNodeSizeChange,
  showEdges,
  onShowEdgesChange,
  cameraDistance,
  onCameraDistanceChange,
  onZoomIn,
  onZoomOut,
  onZoomFit,
  onResetCamera,
  enablePhysics,
  onEnablePhysicsChange,
  nodeCount = 0,
  edgeCount = 0,
}: Network3DControlsProps) {
  return (
    <div className="space-y-4">
      {/* Camera Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Camera Controls</CardTitle>
          <CardDescription className="text-xs">
            Adjust view and navigation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Quick Actions */}
          <div className="grid grid-cols-2 gap-2">
            {onZoomIn && (
              <Button
                size="sm"
                variant="outline"
                onClick={onZoomIn}
                className="flex items-center gap-2"
              >
                <ZoomIn className="h-4 w-4" />
                Zoom In
              </Button>
            )}
            {onZoomOut && (
              <Button
                size="sm"
                variant="outline"
                onClick={onZoomOut}
                className="flex items-center gap-2"
              >
                <ZoomOut className="h-4 w-4" />
                Zoom Out
              </Button>
            )}
            {onZoomFit && (
              <Button
                size="sm"
                variant="outline"
                onClick={onZoomFit}
                className="flex items-center gap-2 col-span-2"
              >
                <Maximize2 className="h-4 w-4" />
                Fit to View
              </Button>
            )}
            {onResetCamera && (
              <Button
                size="sm"
                variant="outline"
                onClick={onResetCamera}
                className="flex items-center gap-2 col-span-2"
              >
                <RotateCcw className="h-4 w-4" />
                Reset Camera
              </Button>
            )}
          </div>

          {/* Camera Distance Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Camera Distance</Label>
              <span className="text-xs text-muted-foreground">{cameraDistance}</span>
            </div>
            <Slider
              value={[cameraDistance]}
              onValueChange={([value]) => onCameraDistanceChange(value)}
              min={300}
              max={2000}
              step={50}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Initial camera distance from center
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Visual Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Visual Controls</CardTitle>
          <CardDescription className="text-xs">
            Customize visualization appearance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Node Size */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Node Size</Label>
              <span className="text-xs text-muted-foreground">{nodeSize.toFixed(1)}x</span>
            </div>
            <Slider
              value={[nodeSize]}
              onValueChange={([value]) => onNodeSizeChange(value)}
              min={0.5}
              max={3}
              step={0.1}
              className="w-full"
            />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Shrink className="h-3 w-3" />
                Smaller
              </span>
              <span className="flex items-center gap-1">
                Larger
                <Expand className="h-3 w-3" />
              </span>
            </div>
          </div>

          {/* Show Edges Toggle */}
          <div className="flex items-center justify-between py-2 px-3 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-2">
              {showEdges ? (
                <Eye className="h-4 w-4 text-muted-foreground" />
              ) : (
                <EyeOff className="h-4 w-4 text-muted-foreground" />
              )}
              <Label htmlFor="show-edges" className="text-sm font-medium cursor-pointer">
                Show Edges
              </Label>
            </div>
            <Switch
              id="show-edges"
              checked={showEdges}
              onCheckedChange={onShowEdgesChange}
            />
          </div>

          {/* Physics Toggle (if provided) */}
          {enablePhysics !== undefined && onEnablePhysicsChange && (
            <div className="flex items-center justify-between py-2 px-3 bg-muted/50 rounded-lg">
              <Label htmlFor="enable-physics" className="text-sm font-medium cursor-pointer">
                Enable Physics
              </Label>
              <Switch
                id="enable-physics"
                checked={enablePhysics}
                onCheckedChange={onEnablePhysicsChange}
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Network Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Network Statistics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Nodes</span>
            <span className="text-sm font-semibold">{nodeCount.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Edges</span>
            <span className="text-sm font-semibold">{edgeCount.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center pt-2 border-t">
            <span className="text-sm text-muted-foreground">Density</span>
            <span className="text-sm font-semibold">
              {nodeCount > 1
                ? ((edgeCount / (nodeCount * (nodeCount - 1))) * 100).toFixed(2)
                : '0.00'}
              %
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Avg Degree</span>
            <span className="text-sm font-semibold">
              {nodeCount > 0 ? ((edgeCount * 2) / nodeCount).toFixed(2) : '0.00'}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Instructions */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">Navigation Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs text-muted-foreground">
          <div className="flex items-start gap-2">
            <div className="mt-0.5">•</div>
            <div>
              <span className="font-medium text-foreground">Left Click + Drag:</span> Rotate view
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="mt-0.5">•</div>
            <div>
              <span className="font-medium text-foreground">Right Click + Drag:</span> Pan view
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="mt-0.5">•</div>
            <div>
              <span className="font-medium text-foreground">Scroll:</span> Zoom in/out
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="mt-0.5">•</div>
            <div>
              <span className="font-medium text-foreground">Click Node:</span> Select and focus
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="mt-0.5">•</div>
            <div>
              <span className="font-medium text-foreground">Hover Node:</span> View details
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Tip */}
      <div className="text-xs text-muted-foreground bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 mt-0.5 text-blue-500 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div>
            <p className="font-medium text-blue-700 dark:text-blue-400 mb-1">
              Performance Tip
            </p>
            <p className="text-blue-600/90 dark:text-blue-300/90">
              For large networks (&gt;1000 nodes), consider hiding edges or reducing node size for
              better performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
