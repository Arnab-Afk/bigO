/**
 * ML Dashboard Overview Page
 * Displays key metrics from the CCP ML API
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import {
  useCCPHealthCheck,
  useCCPStatus,
  useCCPNetwork,
  useCCPRiskScores,
  useCCPSpectralAnalysis,
  useCCPMargins,
  useCCPRunSimulation,
  useCCPStressTest,
  useCCPReinitialize,
} from '@/hooks/use-ccp-ml';
import {
  Play,
  RefreshCw,
  Building2,
  Network,
  TrendingUp,
  AlertTriangle,
  Zap,
} from 'lucide-react';

export default function MLDashboardPage() {
  // API hooks
  const { data: health } = useCCPHealthCheck();
  const { data: status, isLoading: statusLoading } = useCCPStatus();
  const { data: network } = useCCPNetwork();
  const { data: riskScores, isLoading: riskLoading } = useCCPRiskScores();
  const { data: spectral, isLoading: spectralLoading } = useCCPSpectralAnalysis();
  const { data: margins, isLoading: marginsLoading } = useCCPMargins();
  
  // Mutations
  const { mutate: runSimulation, isPending: isRunningSimulation } = useCCPRunSimulation();
  const { mutate: stressTest, isPending: isRunningStressTest } = useCCPStressTest();
  const { mutate: reinitialize, isPending: isReinitializing } = useCCPReinitialize();

  // Computed values
  const totalBanks = status?.num_banks || 0;
  const totalEdges = network?.num_edges || 0;
  const spectralRadius = spectral?.spectral_radius || 0;
  const fiedlerValue = spectral?.fiedler_value || 0;
  const contagionIndex = spectral?.contagion_index || 0;

  // Risk distribution
  const riskDistribution = riskScores
    ? {
        high: riskScores.filter((r) => r.stress_level > 0.7).length,
        medium: riskScores.filter((r) => r.stress_level > 0.3 && r.stress_level <= 0.7).length,
        low: riskScores.filter((r) => r.stress_level <= 0.3).length,
      }
    : null;

  return (
    <div className="flex-1 space-y-4 p-4 pt-6 md:p-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">CCP ML Dashboard</h2>
          <p className="text-muted-foreground">
            Network-Based CCP Risk Analysis - Real RBI Bank Data
          </p>
        </div>
        <div className="flex items-center gap-2">
          {health && (
            <Badge variant="outline" className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${health.initialized ? 'bg-green-500' : 'bg-yellow-500'} animate-pulse`} />
              {health.status}
            </Badge>
          )}
          <Button
            onClick={() => runSimulation()}
            disabled={isRunningSimulation}
          >
            <Play className="h-4 w-4 mr-2" />
            Run Simulation
          </Button>
          <Button
            variant="outline"
            onClick={() => reinitialize()}
            disabled={isReinitializing}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reload Data
          </Button>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Total Banks */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Banks</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{totalBanks}</div>
            )}
            <p className="text-xs text-muted-foreground">Indian banking system</p>
          </CardContent>
        </Card>

        {/* Network Connections */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network Edges</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{totalEdges}</div>
            )}
            <p className="text-xs text-muted-foreground">
              Interdependencies
            </p>
          </CardContent>
        </Card>

        {/* Spectral Radius */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Spectral Radius</CardTitle>
            <TrendingUp className={`h-4 w-4 ${spectralRadius > 1 ? 'text-red-500' : 'text-green-500'}`} />
          </CardHeader>
          <CardContent>
            {spectralLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className={`text-2xl font-bold ${spectralRadius > 1 ? 'text-red-600' : 'text-green-600'}`}>
                {spectralRadius.toFixed(3)}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              {spectral?.amplification_risk || 'Calculating...'}
            </p>
          </CardContent>
        </Card>

        {/* High Risk Banks */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">High Risk</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            {riskLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold text-red-600">
                {riskDistribution?.high || 0}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              Stress level &gt; 0.7
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
            <CardDescription>Banks by stress level</CardDescription>
          </CardHeader>
          <CardContent>
            {riskLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : riskDistribution ? (
              <div className="space-y-4">
                <div className="flex items-center">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium">High Risk</span>
                      <span className="text-sm text-muted-foreground">
                        {riskDistribution.high}
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-secondary">
                      <div
                        className="h-2 rounded-full bg-red-500"
                        style={{
                          width: `${(riskDistribution.high / totalBanks) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium">Medium Risk</span>
                      <span className="text-sm text-muted-foreground">
                        {riskDistribution.medium}
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-secondary">
                      <div
                        className="h-2 rounded-full bg-yellow-500"
                        style={{
                          width: `${(riskDistribution.medium / totalBanks) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium">Low Risk</span>
                      <span className="text-sm text-muted-foreground">
                        {riskDistribution.low}
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-secondary">
                      <div
                        className="h-2 rounded-full bg-green-500"
                        style={{
                          width: `${(riskDistribution.low / totalBanks) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <Alert>
                <AlertDescription>No risk data available. Run a simulation first.</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Spectral Analysis */}
        <Card>
          <CardHeader>
            <CardTitle>Spectral Analysis</CardTitle>
            <CardDescription>System-level risk indicators</CardDescription>
          </CardHeader>
          <CardContent>
            {spectralLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : spectral ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Spectral Radius (ρ)</span>
                  <span className={`text-lg font-bold ${spectralRadius > 1 ? 'text-red-600' : 'text-green-600'}`}>
                    {spectralRadius.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Fiedler Value (λ₂)</span>
                  <span className={`text-lg font-bold ${fiedlerValue < 0.1 ? 'text-red-600' : 'text-green-600'}`}>
                    {fiedlerValue.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Contagion Index</span>
                  <span className="text-lg font-bold">
                    {contagionIndex.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Spectral Gap</span>
                  <span className="text-lg font-bold">
                    {spectral.spectral_gap.toFixed(3)}
                  </span>
                </div>
                <div className="pt-2 border-t">
                  <div className="text-sm space-y-1">
                    <p><strong>Amplification:</strong> {spectral.amplification_risk}</p>
                    <p><strong>Fragmentation:</strong> {spectral.fragmentation_risk}</p>
                  </div>
                </div>
              </div>
            ) : (
              <Alert>
                <AlertDescription>No spectral data available. Run a simulation first.</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Stress Testing */}
      <Card>
        <CardHeader>
          <CardTitle>Stress Testing</CardTitle>
          <CardDescription>Apply shocks to the banking system</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 flex-wrap">
            <Button
              variant="destructive"
              onClick={() => {
                stressTest({
                  shock_type: 'capital',
                  shock_magnitude: 0.2,
                });
              }}
              disabled={isRunningStressTest}
            >
              <Zap className="h-4 w-4 mr-2" />
              Capital Shock (-20%)
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                stressTest({
                  shock_type: 'liquidity',
                  shock_magnitude: 0.3,
                });
              }}
              disabled={isRunningStressTest}
            >
              <Zap className="h-4 w-4 mr-2" />
              Liquidity Squeeze (-30%)
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                stressTest({
                  shock_type: 'market',
                  shock_magnitude: 0.4,
                });
              }}
              disabled={isRunningStressTest}
            >
              <Zap className="h-4 w-4 mr-2" />
              Market Shock (-40%)
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Top Banks by Margin Requirements */}
      <Card>
        <CardHeader>
          <CardTitle>Margin Requirements</CardTitle>
          <CardDescription>Top 10 banks by CCP margin requirements</CardDescription>
        </CardHeader>
        <CardContent>
          {marginsLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : margins && margins.length > 0 ? (
            <div className="space-y-2">
              {margins.slice(0, 10).map((margin, index) => (
                <div
                  key={margin.bank_name}
                  className="flex items-center justify-between p-3 rounded-lg border hover:bg-secondary/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-medium">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium">{margin.bank_name}</p>
                      <p className="text-xs text-muted-foreground">
                        {margin.explanation}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge variant="outline" className="font-mono">
                      {(margin.total_margin * 100).toFixed(2)}%
                    </Badge>
                    <p className="text-xs text-muted-foreground mt-1">
                      Base: {(margin.base_margin * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Alert>
              <AlertDescription>
                No margin data available. Run a simulation first.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
