'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, Zap, RotateCcw, Loader2, Activity, AlertTriangle, Users, Building2, Network as NetworkIcon } from 'lucide-react';
import dynamic from 'next/dynamic';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import * as api from '@/lib/api';
import { SimulationSnapshot, GraphNode, RiskDecision } from '@/types/simulation';
import { UserEntity } from '@/types/user';
import EntityOnboarding from '@/components/EntityOnboarding';
import PolicyControlPanel from '@/components/PolicyControlPanel';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Custom tooltip for time series charts
const CustomTooltip = (props: any) => {
  const { active, payload, label, valueFormatter } = props;
  if (active && payload && payload.length) {
    const value = payload[0].value;
    const formattedValue = valueFormatter ? valueFormatter(value) : value;
    return (
      <div className="bg-white border-2 border-slate-200 rounded-lg p-3 shadow-lg">
        <div className="text-lg font-bold text-slate-900 mb-1">
          {formattedValue}
        </div>
        <div className="text-xs text-slate-500">
          Timestep {label}
        </div>
      </div>
    );
  }
  return null;
};

// Dynamically import NetworkVisualization with SSR disabled
const NetworkVisualization = dynamic(() => import('@/components/NetworkVisualization'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[600px]">
      <Loader2 className="w-8 h-8 animate-spin text-primary" />
    </div>
  ),
});

export default function MLDashboardPage() {
  // User entity state
  const [userEntity, setUserEntity] = useState<UserEntity | null>(null);

  // Simulation state
  const [simId, setSimId] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<SimulationSnapshot[]>([]);
  const [currentSnapshot, setCurrentSnapshot] = useState<SimulationSnapshot | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingDecision, setPendingDecision] = useState<RiskDecision | null>(null);

  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const initializeSimulation = async () => {
    if (!userEntity) return;

    setIsLoading(true);
    setError(null);
    try {
      const randomSeed = Math.floor(Math.random() * 1000000);
      const response = await api.initialize({
        name: `${userEntity.name}'s Simulation`,
        max_timesteps: 100,
        enable_shocks: true,
        random_seed: randomSeed,
        shock_probability: 0.12,
        use_real_data: false,
        user_entity: {
          id: userEntity.id,
          type: userEntity.type,
          name: userEntity.name,
          policies: userEntity.policies,
        },
      });

      setSimId(response.simulation_id);
      const state = await api.getState(response.simulation_id);
      setCurrentSnapshot(state.snapshot);
      setHistory([state.snapshot]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize simulation');
    } finally {
      setIsLoading(false);
    }
  };

  // Initialize simulation after user entity is set
  useEffect(() => {
    if (userEntity && !simId) {
      initializeSimulation();
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [userEntity]);

  // Auto-play loop
  useEffect(() => {
    if (isPlaying && simId) {
      playIntervalRef.current = setInterval(() => {
        stepSimulation(1);
      }, 1000);
    } else if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, simId]);

  const stepSimulation = async (steps: number = 1) => {
    if (!simId) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await api.step(simId, steps);

      if (response.snapshots && response.snapshots.length > 0) {
        const newSnapshots = response.snapshots;
        setHistory((prev) => [...prev, ...newSnapshots]);
        setCurrentSnapshot(newSnapshots[newSnapshots.length - 1]);

        const latestSnapshot = newSnapshots[newSnapshots.length - 1];
        const userNode = latestSnapshot.agents.find((a: GraphNode) => a.id === userEntity?.id);
        if (userNode && !userNode.alive) {
          setIsPlaying(false);
        }
      }

      if (response.pending_decision) {
        setPendingDecision(response.pending_decision);
        setIsPlaying(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to step simulation');
      setIsPlaying(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDecisionResponse = async (approved: boolean, customParams?: Record<string, any>) => {
    if (!simId || !pendingDecision) return;

    setIsLoading(true);
    setError(null);
    try {
      await api.respondToDecision(simId, {
        decision_id: pendingDecision.decision_id,
        approved,
        custom_params: customParams,
      });

      setPendingDecision(null);
      const state = await api.getState(simId);
      setCurrentSnapshot(state.snapshot);
      setHistory((prev) => [...prev, state.snapshot]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to respond to decision');
    } finally {
      setIsLoading(false);
    }
  };

  const handleShockRealEstate = async () => {
    if (!simId) return;

    setIsLoading(true);
    setError(null);
    try {
      await api.shockRealEstate(simId, -0.3);
      const state = await api.getState(simId);
      setCurrentSnapshot(state.snapshot);
      setHistory((prev) => [...prev, state.snapshot]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply shock');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    setIsPlaying(false);
    setIsLoading(true);
    setError(null);
    try {
      await initializeSimulation();
      setSelectedNode(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset simulation');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEntityCreated = (entity: UserEntity) => {
    setUserEntity(entity);
  };

  const handlePolicyChange = async (policies: any) => {
    if (!userEntity) return;
    setUserEntity((prev) => (prev ? { ...prev, policies } : null));
  };

  const togglePlayPause = () => {
    setIsPlaying((prev) => !prev);
  };

  // Show onboarding if no user entity
  if (!userEntity) {
    return <EntityOnboarding onComplete={handleEntityCreated} />;
  }

  // Prepare chart data
  const chartData = history.map((snapshot) => ({
    timestep: snapshot.timestep,
    liquidity: snapshot.global_metrics.total_liquidity / 1e9,
    npa: snapshot.global_metrics.total_npa / 1e6,
    system_health: snapshot.global_metrics.system_health * 100,
    alive_agents: snapshot.global_metrics.alive_agents,
  }));

  // Get user node health and alive status
  const userNode = currentSnapshot?.agents.find((a: GraphNode) => a.id === userEntity.id);
  const userHealth = userNode?.health ?? 1.0;
  const isUserAlive = userNode?.alive ?? true;

  // Sync live agent policies with userEntity
  const liveUserEntity = (() => {
    if (!userNode || userEntity.type !== 'bank') return userEntity;

    const bankPolicies = userEntity.policies as any;
    const updatedPolicies = { ...bankPolicies };

    if (userNode.risk_appetite !== undefined) {
      updatedPolicies.riskAppetite = userNode.risk_appetite * 100;
    }

    return {
      ...userEntity,
      policies: updatedPolicies,
    };
  })();

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
