'use client';

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ccpApi } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { PlayCircle, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

export default function SimulationPage() {
  const queryClient = useQueryClient();
  const [config, setConfig] = useState({
    year: null as number | null,
    sector_weight: 0.4,
    liquidity_weight: 0.4,
    market_weight: 0.2,
    edge_threshold: 0.05,
  });

  const runSimulation = useMutation({
    mutationFn: () => ccpApi.runSimulation(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ccp-status'] });
      queryClient.invalidateQueries({ queryKey: ['ccp-summary'] });
      queryClient.invalidateQueries({ queryKey: ['ccp-network'] });
      queryClient.invalidateQueries({ queryKey: ['ccp-banks'] });
    },
  });

  const totalWeight = config.sector_weight + config.liquidity_weight + config.market_weight;
  const isWeightValid = Math.abs(totalWeight - 1.0) < 0.01;

  const handleRunSimulation = () => {
    if (!isWeightValid) {
      alert('Channel weights must sum to 1.0');
      return;
    }
    runSimulation.mutate();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Simulation Runner</h1>
        <p className="text-muted-foreground">
          Configure and execute CCP risk simulation
        </p>
      </div>

      {/* Status Alert */}
      {runSimulation.isSuccess && (
        <Alert className="border-green-500 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Simulation completed successfully! Results are now available in all dashboards.
          </AlertDescription>
        </Alert>
      )}

      {runSimulation.isError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Simulation failed: {(runSimulation.error as Error).message}
          </AlertDescription>
        </Alert>
      )}

      {runSimulation.isPending && (
        <Alert>
          <RefreshCw className="h-4 w-4 animate-spin" />
          <AlertDescription>
            Running simulation... This may take 30-60 seconds.
          </AlertDescription>
          <Progress value={50} className="mt-2" />
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Simulation Configuration</CardTitle>
            <CardDescription>
              Adjust parameters for the CCP risk simulation
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Year Selection */}
            <div className="space-y-2">
              <Label>Target Year (Optional)</Label>
              <Select
                value={config.year?.toString() || 'latest'}
                onValueChange={(value) =>
                  setConfig({ ...config, year: value === 'latest' ? null : parseInt(value) })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select year" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="latest">Latest Available</SelectItem>
                  {[...Array(18)].map((_, i) => {
                    const year = 2025 - i;
                    return (
                      <SelectItem key={year} value={year.toString()}>
                        {year}
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Leave as "Latest" to use most recent data
              </p>
            </div>

            {/* Network Channel Weights */}
            <div className="space-y-4">
              <div>
                <Label className="text-base">Network Channel Weights</Label>
                <p className="text-sm text-muted-foreground">
                  Adjust the importance of each dependency channel
                </p>
              </div>

              {/* Sector Weight */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Sector Similarity</Label>
                  <span className="text-sm font-medium">{config.sector_weight.toFixed(2)}</span>
                </div>
                <Slider
                  value={[config.sector_weight]}
                  onValueChange={([value]) => setConfig({ ...config, sector_weight: value })}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="text-xs text-muted-foreground">
                  Weight for sector exposure similarity
                </p>
              </div>

              {/* Liquidity Weight */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Liquidity Similarity</Label>
                  <span className="text-sm font-medium">{config.liquidity_weight.toFixed(2)}</span>
                </div>
                <Slider
                  value={[config.liquidity_weight]}
                  onValueChange={([value]) => setConfig({ ...config, liquidity_weight: value })}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="text-xs text-muted-foreground">
                  Weight for maturity profile similarity
                </p>
              </div>

              {/* Market Weight */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Market Correlation</Label>
                  <span className="text-sm font-medium">{config.market_weight.toFixed(2)}</span>
                </div>
                <Slider
                  value={[config.market_weight]}
                  onValueChange={([value]) => setConfig({ ...config, market_weight: value })}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="text-xs text-muted-foreground">
                  Weight for stock return correlations
                </p>
              </div>

              {/* Weight Validation */}
              <div className="p-4 border rounded-lg bg-muted/50">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Total Weight:</span>
                  <span
                    className={`text-lg font-bold ${
                      isWeightValid ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {totalWeight.toFixed(2)}
                  </span>
                </div>
                {!isWeightValid && (
                  <p className="text-xs text-red-600 mt-1">
                    Weights must sum to 1.0 for valid simulation
                  </p>
                )}
              </div>
            </div>

            {/* Edge Threshold */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Edge Weight Threshold</Label>
                <span className="text-sm font-medium">{config.edge_threshold.toFixed(3)}</span>
              </div>
              <Slider
                value={[config.edge_threshold]}
                onValueChange={([value]) => setConfig({ ...config, edge_threshold: value })}
                min={0}
                max={0.5}
                step={0.01}
              />
              <p className="text-xs text-muted-foreground">
                Minimum edge weight to include in network (filters weak connections)
              </p>
            </div>

            {/* Run Button */}
            <Button
              onClick={handleRunSimulation}
              disabled={runSimulation.isPending || !isWeightValid}
              className="w-full"
              size="lg"
            >
              {runSimulation.isPending ? (
                <>
                  <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
                  Running Simulation...
                </>
              ) : (
                <>
                  <PlayCircle className="mr-2 h-5 w-5" />
                  Run Simulation
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Info Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>About the Simulation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">What This Does</h4>
                <p className="text-sm text-muted-foreground">
                  The simulation analyzes systemic risk in the banking network by:
                </p>
                <ul className="list-disc list-inside text-sm text-muted-foreground mt-2 space-y-1">
                  <li>Loading financial data for all banks</li>
                  <li>Building a multi-channel dependency network</li>
                  <li>Training ML models for default prediction</li>
                  <li>Computing spectral risk metrics</li>
                  <li>Calculating CCP margin requirements</li>
                  <li>Generating policy recommendations</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2">Network Channels</h4>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">Sector Channel</p>
                    <p className="text-xs text-muted-foreground">
                      Banks with similar sector exposures are more correlated
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Liquidity Channel</p>
                    <p className="text-xs text-muted-foreground">
                      Banks with similar maturity profiles face common liquidity risks
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Market Channel</p>
                    <p className="text-xs text-muted-foreground">
                      Stock price correlations indicate common market factors
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Expected Runtime</h4>
                <p className="text-sm text-muted-foreground">
                  Typical execution: 30-60 seconds
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Includes data loading, ML training, network analysis, and spectral computation
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recommended Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() =>
                  setConfig({
                    year: null,
                    sector_weight: 0.4,
                    liquidity_weight: 0.4,
                    market_weight: 0.2,
                    edge_threshold: 0.05,
                  })
                }
              >
                <span className="font-medium">Balanced (Default)</span>
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() =>
                  setConfig({
                    year: null,
                    sector_weight: 0.5,
                    liquidity_weight: 0.3,
                    market_weight: 0.2,
                    edge_threshold: 0.05,
                  })
                }
              >
                <span className="font-medium">Sector-Focused</span>
              </Button>
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() =>
                  setConfig({
                    year: null,
                    sector_weight: 0.2,
                    liquidity_weight: 0.5,
                    market_weight: 0.3,
                    edge_threshold: 0.05,
                  })
                }
              >
                <span className="font-medium">Liquidity-Focused</span>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
