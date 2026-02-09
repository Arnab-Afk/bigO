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
import { ChatBot, ChatBotToggle } from '@/components/ChatBot';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:17170';

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

  // ChatBot state
  const [showChat, setShowChat] = useState(false);
  const [narrations, setNarrations] = useState<Array<{eventType: string, narration: string, timestamp: string}>>([]);

  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Simple polling for demo reliability (no WebSocket complexity)
  const [isPolling, setIsPolling] = useState(false);
  const wsConnected = false; // Simplified for demo

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

        // Get LLM narration for demo (simple polling, no WebSocket)
        try {
          const narrationResponse = await fetch(`${API_BASE}/api/v1/llm/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              simulation_state: {
                step: latestSnapshot.step,
                defaults: latestSnapshot.defaults || [],
                stability: latestSnapshot.stability_metrics
              },
              query: "Explain what happened in this simulation step in 1-2 sentences"
            })
          });

          if (narrationResponse.ok) {
            const narrationData = await narrationResponse.json();
            setNarrations((prev) => [
              ...prev,
              {
                eventType: 'simulation_step',
                narration: narrationData.explanation,
                timestamp: new Date().toISOString()
              }
            ]);
          }
        } catch (err) {
          console.warn('LLM narration unavailable:', err);
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
      setHistory((prev: SimulationSnapshot[]) => [...prev, state.snapshot]);
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
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Network Simulation</h1>
          <p className="text-muted-foreground">
            {currentSnapshot
              ? `Timestep ${currentSnapshot.timestep} • ${currentSnapshot.global_metrics.alive_agents}/${currentSnapshot.global_metrics.total_agents} Active • Health: ${(currentSnapshot.global_metrics.system_health * 100).toFixed(0)}%`
              : 'Initializing simulation...'}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={togglePlayPause}
            disabled={!simId || isLoading || !isUserAlive}
            size="lg"
          >
            {isPlaying ? (
              <>
                <Pause className="mr-2 h-4 w-4" />
                Pause
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Play
              </>
            )}
          </Button>

          <Button
            onClick={() => stepSimulation(1)}
            disabled={!simId || isLoading || isPlaying || !isUserAlive}
            variant="secondary"
          >
            <SkipForward className="h-4 w-4" />
          </Button>

          <Button
            onClick={handleShockRealEstate}
            disabled={!simId || isLoading || !isUserAlive}
            variant="destructive"
          >
            <Zap className="mr-2 h-4 w-4" />
            Apply Shock
          </Button>

          <Button
            onClick={handleReset}
            disabled={!simId || isLoading}
            variant="outline"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* User Death Notice */}
      {!isUserAlive && (
        <Alert className="border-destructive bg-destructive/10">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          <AlertDescription className="text-destructive font-semibold">
            💀 Your Entity Has Defaulted - Controls disabled. Click Reset to start over.
          </AlertDescription>
        </Alert>
      )}

      {/* System Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentSnapshot ? `${(currentSnapshot.global_metrics.system_health * 100).toFixed(0)}%` : '—'}
            </div>
            <p className="text-xs text-muted-foreground">Overall system stability</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Entities</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentSnapshot ? `${currentSnapshot.global_metrics.alive_agents}/${currentSnapshot.global_metrics.total_agents}` : '—'}
            </div>
            <p className="text-xs text-muted-foreground">Operational entities</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Liquidity</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentSnapshot ? `$${(currentSnapshot.global_metrics.total_liquidity / 1e9).toFixed(2)}B` : '—'}
            </div>
            <p className="text-xs text-muted-foreground">System-wide liquidity</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Your Health</CardTitle>
            <Activity className={`h-4 w-4 ${!isUserAlive ? 'text-destructive' : userHealth > 0.7 ? 'text-green-500' : userHealth > 0.4 ? 'text-amber-500' : 'text-red-500'}`} />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${!isUserAlive ? 'text-destructive' : userHealth > 0.7 ? 'text-green-500' : userHealth > 0.4 ? 'text-amber-500' : 'text-red-500'}`}>
              {(userHealth * 100).toFixed(0)}%
            </div>
            <p className="text-xs text-muted-foreground">{isUserAlive ? 'Active' : 'Defaulted'}</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-4">
        {/* Left: Policy Controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="relative">
            {!isUserAlive && (
              <div className="absolute inset-0 z-10 bg-background/80 backdrop-blur-sm rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="text-4xl mb-2">💀</div>
                  <div className="font-bold">Controls Disabled</div>
                  <div className="text-sm text-muted-foreground mt-1">Entity defaulted</div>
                </div>
              </div>
            )}
            <PolicyControlPanel
              userEntity={liveUserEntity}
              onPolicyChange={handlePolicyChange}
              isSimulationRunning={isPlaying}
              currentHealth={userHealth}
            />
          </div>
        </div>

        {/* Center: Network Visualization */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Network Topology</CardTitle>
            <CardDescription>Real-time financial network visualization</CardDescription>
          </CardHeader>
          <CardContent className="p-2">
            {currentSnapshot ? (
              <NetworkVisualization
                nodes={currentSnapshot.agents as GraphNode[]}
                links={currentSnapshot.links}
                onNodeSelect={setSelectedNode}
                selectedNodeId={selectedNode?.id ?? null}
                userNodeId={userEntity.id}
                width={typeof window !== 'undefined' ? Math.min(window.innerWidth * 0.5, 900) : 700}
                height={600}
              />
            ) : (
              <div className="flex items-center justify-center h-[600px]">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right: Inspector & Metrics */}
        <div className="lg:col-span-1 space-y-6">
          {/* Entity Inspector */}
          <Card>
            <CardHeader>
              <CardTitle>Entity Inspector</CardTitle>
              <CardDescription>Selected entity details</CardDescription>
            </CardHeader>
            <CardContent>
              {selectedNode ? (
                <div className="space-y-4">
                  <div>
                    <div className="text-sm font-semibold text-muted-foreground mb-1">Entity ID</div>
                    <div className="font-mono text-sm">{selectedNode.id}</div>
                  </div>

                  <div>
                    <div className="text-sm font-semibold text-muted-foreground mb-1">Type</div>
                    <Badge variant="secondary">{selectedNode.type.replace('_', ' ').toUpperCase()}</Badge>
                  </div>

                  <div>
                    <div className="text-sm font-semibold text-muted-foreground mb-2">Health</div>
                    <div className={`text-2xl font-bold ${selectedNode.health > 0.7 ? 'text-green-600' : selectedNode.health > 0.4 ? 'text-amber-600' : 'text-red-600'}`}>
                      {(selectedNode.health * 100).toFixed(1)}%
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2 mt-2">
                      <div
                        className={`h-2 rounded-full transition-all ${selectedNode.health > 0.7 ? 'bg-green-500' : selectedNode.health > 0.4 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{ width: `${selectedNode.health * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="text-sm font-semibold text-muted-foreground mb-1">Status</div>
                    <Badge variant={selectedNode.alive ? 'default' : 'destructive'}>
                      {selectedNode.alive ? '✓ Active' : '✗ Defaulted'}
                    </Badge>
                  </div>

                  <div className="pt-3 border-t space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Capital</span>
                      <span className="font-mono font-semibold">${(selectedNode.capital / 1e6).toFixed(2)}M</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Liquidity</span>
                      <span className="font-mono font-semibold">${(selectedNode.liquidity / 1e6).toFixed(2)}M</span>
                    </div>
                    {selectedNode.npa !== undefined && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">NPA</span>
                        <span className="font-mono font-semibold text-destructive">${(selectedNode.npa / 1e6).toFixed(2)}M</span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground text-sm">
                  Click on an entity to inspect
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Time Series Charts */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>System Health Over Time</CardTitle>
            <CardDescription>Historical system stability trend</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="timestep" className="text-xs" />
                <YAxis domain={[0, 100]} className="text-xs" />
                <Tooltip content={<CustomTooltip valueFormatter={(v: any) => `${v.toFixed(1)}%`} />} />
                <Line type="monotone" dataKey="system_health" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Entities</CardTitle>
            <CardDescription>Operational entities over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="entitiesGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="timestep" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip content={<CustomTooltip valueFormatter={(v: any) => `${v} entities`} />} />
                <Area type="monotone" dataKey="alive_agents" stroke="hsl(var(--primary))" strokeWidth={2} fillOpacity={1} fill="url(#entitiesGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Risk Decision Modal */}
      {pendingDecision && isUserAlive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4">
          <Card className="max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <CardHeader className={`${pendingDecision.risk_level === 'critical' ? 'bg-destructive/10 border-b-destructive' :
              pendingDecision.risk_level === 'high' ? 'bg-orange-50 border-b-orange-200' :
                pendingDecision.risk_level === 'medium' ? 'bg-yellow-50 border-b-yellow-200' :
                  'bg-blue-50 border-b-blue-200'
              } border-b-2`}>
              <CardTitle className="text-2xl">{pendingDecision.title}</CardTitle>
              <CardDescription className="text-base">{pendingDecision.description}</CardDescription>
              <Badge variant="secondary" className="mt-2 w-fit">
                Timestep {pendingDecision.timestep}
              </Badge>
            </CardHeader>

            <CardContent className="pt-6">
              {/* Current Metrics */}
              <div className="mb-6 p-4 bg-secondary/50 rounded-lg">
                <h3 className="text-sm font-bold uppercase tracking-wide mb-3">Current Metrics</h3>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(pendingDecision.current_metrics).map(([key, value]) => (
                    <div key={key} className="bg-background p-3 rounded-lg border">
                      <div className="text-xs text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</div>
                      <div className="text-lg font-bold">
                        {typeof value === 'number' ? (value * 100).toFixed(1) + (key.includes('ratio') || key.includes('health') ? '%' : '') : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="space-y-4">
                {/* Recommended Action */}
                <div className="border-2 border-green-200 rounded-lg p-4 bg-green-50">
                  <div className="flex items-start gap-3">
                    <div className="mt-1 w-10 h-10 rounded-full bg-green-600 flex items-center justify-center flex-shrink-0">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="flex-1 space-y-3">
                      <Badge variant="default" className="bg-green-600">Recommended</Badge>
                      <h4 className="font-bold text-lg">{pendingDecision.recommended_action.description}</h4>
                      <p className="text-sm text-muted-foreground">{pendingDecision.recommended_action.impact}</p>
                      <Button
                        onClick={() => handleDecisionResponse(true)}
                        disabled={isLoading}
                        className="w-full bg-green-600 hover:bg-green-700"
                      >
                        {isLoading ? 'Applying...' : 'Apply Recommended Action'}
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Alternative Action */}
                <div className="border-2 border-border rounded-lg p-4 bg-secondary/30">
                  <div className="flex items-start gap-3">
                    <div className="mt-1 w-10 h-10 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
                    <div className="flex-1 space-y-3">
                      <Badge variant="secondary">Alternative</Badge>
                      <h4 className="font-bold text-lg">{pendingDecision.alternative_action.description}</h4>
                      <p className="text-sm text-muted-foreground">{pendingDecision.alternative_action.impact}</p>
                      <Button
                        onClick={() => handleDecisionResponse(false)}
                        disabled={isLoading}
                        variant="secondary"
                        className="w-full"
                      >
                        {isLoading ? 'Processing...' : 'Reject & Maintain Strategy'}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ChatBot - Always show toggle when simulation is active */}
      {simId && !showChat && (
        <ChatBotToggle onClick={() => setShowChat(true)} />
      )}
      {simId && showChat && (
        <ChatBot
          simulationId={simId}
          narrations={narrations}
          onClose={() => setShowChat(false)}
        />
      )}
    </div>
  );
}
