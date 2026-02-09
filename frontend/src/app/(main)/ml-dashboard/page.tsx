'use client';

import { useEffect, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ccpApi, SimulationSummary } from '@/lib/api/ccp-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Building2,
  Network,
  PlayCircle,
  RefreshCw,
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const COLORS = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#10b981',
};

export default function MLDashboardPage() {
  const queryClient = useQueryClient();

  // Fetch status
  const { data: status } = useQuery({
    queryKey: ['ccp-status'],
    queryFn: () => ccpApi.getStatus(),
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Fetch summary (only if initialized)
  const { data: summary, isLoading, error } = useQuery({
    queryKey: ['ccp-summary'],
    queryFn: () => ccpApi.getSummary(),
    enabled: status?.initialized || false,
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Run simulation mutation
  const runSimulation = useMutation({
    mutationFn: () => ccpApi.runSimulation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ccp-status'] });
      queryClient.invalidateQueries({ queryKey: ['ccp-summary'] });
    },
  });

  // Prepare risk distribution data
  const riskDistribution = summary ? [
    { name: 'High Risk', value: summary.high_risk_count, color: COLORS.high },
    { name: 'Medium Risk', value: summary.medium_risk_count, color: COLORS.medium },
    { name: 'Low Risk', value: summary.low_risk_count, color: COLORS.low },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">ML Dashboard Overview</h1>
          <p className="text-muted-foreground">
            Central Counterparty Risk Analysis & Network Simulation
          </p>
        </div>
        <Button
          onClick={() => runSimulation.mutate()}
          disabled={runSimulation.isPending}
          size="lg"
        >
          {runSimulation.isPending ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <PlayCircle className="mr-2 h-4 w-4" />
              Run Simulation
            </>
          )}
        </Button>
      </div>

      {/* Status Alert */}
      {!status?.initialized && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            System not initialized. Click "Run Simulation" to start the analysis.
          </AlertDescription>
        </Alert>
      )}

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load data: {(error as Error).message}
          </AlertDescription>
        </Alert>
      )}

      {/* System Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Banks</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">{summary?.total_banks || 0}</div>
                <p className="text-xs text-muted-foreground">
                  Analyzed entities
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network Edges</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {summary?.network_metrics.total_edges || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Dependencies tracked
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">High Risk Banks</CardTitle>
            <TrendingUp className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold text-red-500">
                  {summary?.high_risk_count || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Requiring attention
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Update</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-32" />
            ) : (
              <>
                <div className="text-sm font-medium">
                  {summary?.last_run
                    ? new Date(summary.last_run).toLocaleTimeString()
                    : 'Never'}
                </div>
                <p className="text-xs text-muted-foreground">
                  Last simulation run
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        {/* Risk Distribution Chart */}
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
            <CardDescription>
              Distribution of banks by risk level
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-[300px]" />
            ) : summary ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name}: ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Spectral Metrics */}
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Spectral Metrics</CardTitle>
            <CardDescription>
              System-wide risk indicators
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {isLoading ? (
              <>
                <Skeleton className="h-16" />
                <Skeleton className="h-16" />
                <Skeleton className="h-16" />
              </>
            ) : summary ? (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Spectral Radius</p>
                    <p className="text-xs text-muted-foreground">
                      Amplification risk
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold">
                      {summary.spectral_metrics.spectral_radius.toFixed(3)}
                    </p>
                    <Badge
                      variant={
                        summary.spectral_metrics.spectral_radius > 0.9
                          ? 'destructive'
                          : summary.spectral_metrics.spectral_radius > 0.7
                          ? 'default'
                          : 'secondary'
                      }
                    >
                      {summary.spectral_metrics.spectral_radius > 0.9
                        ? 'HIGH'
                        : summary.spectral_metrics.spectral_radius > 0.7
                        ? 'MEDIUM'
                        : 'LOW'}
                    </Badge>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Fiedler Value</p>
                    <p className="text-xs text-muted-foreground">
                      Network fragmentation
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold">
                      {summary.spectral_metrics.fiedler_value.toFixed(3)}
                    </p>
                    <Badge
                      variant={
                        summary.spectral_metrics.fiedler_value < 0.1
                          ? 'destructive'
                          : summary.spectral_metrics.fiedler_value < 0.3
                          ? 'default'
                          : 'secondary'
                      }
                    >
                      {summary.spectral_metrics.fiedler_value < 0.1
                        ? 'HIGH'
                        : summary.spectral_metrics.fiedler_value < 0.3
                        ? 'MEDIUM'
                        : 'LOW'}
                    </Badge>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Contagion Index</p>
                    <p className="text-xs text-muted-foreground">
                      Spread potential
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold">
                      {summary.spectral_metrics.contagion_index.toFixed(3)}
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex h-full items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* CCP Metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Total Margin Required</CardTitle>
            <CardDescription>Aggregate margin requirement</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-10 w-32" />
            ) : summary ? (
              <div className="text-3xl font-bold">
                ${(summary.ccp_metrics.total_margin_requirement / 1e6).toFixed(2)}M
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Default Fund Size</CardTitle>
            <CardDescription>Required default fund</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-10 w-32" />
            ) : summary ? (
              <div className="text-3xl font-bold">
                ${(summary.ccp_metrics.default_fund_size / 1e6).toFixed(2)}M
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cover-N Standard</CardTitle>
            <CardDescription>Coverage standard met</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-10 w-20" />
            ) : summary ? (
              <div>
                <div className="text-3xl font-bold">
                  Cover-{summary.ccp_metrics.cover_n_standard}
                </div>
                <p className="text-sm text-muted-foreground">
                  Top {summary.ccp_metrics.cover_n_standard} exposures covered
                </p>
              </div>
            ) : (
              <div className="text-muted-foreground">No data</div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
