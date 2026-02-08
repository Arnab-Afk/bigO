"use client";

import { useState, useEffect, useRef } from "react";
import { Play, Pause, SkipForward, Zap, RotateCcw, Loader2, Activity, AlertTriangle, Users } from "lucide-react";
import dynamic from "next/dynamic";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import * as api from "@/lib/api";
import { SimulationSnapshot, GraphNode, RiskDecision } from "@/types/simulation";
import { UserEntity } from "@/types/user";
import EntityOnboarding from "@/components/EntityOnboarding";
import PolicyControlPanel from "@/components/PolicyControlPanel";

// Custom tooltip for timeseries charts (value bigger, timestamp smaller)
const CustomTooltip = (props: any) => {
    const { active, payload, label, valueFormatter } = props;
    if (active && payload && payload.length) {
        const value = payload[0].value;
        const formattedValue = valueFormatter ? valueFormatter(value) : value;

        return (
            <div style={{
                backgroundColor: "#ffffff",
                border: "2px solid #e2e8f0",
                borderRadius: "8px",
                padding: "8px 12px",
            }}>
                <div style={{ fontSize: "16px", fontWeight: "bold", color: "#1e293b", marginBottom: "4px" }}>
                    {formattedValue}
                </div>
                <div style={{ fontSize: "10px", color: "#64748b" }}>
                    Timestep {label}
                </div>
            </div>
        );
    }
    return null;
};

// Dynamically import NetworkVisualization with SSR disabled
const NetworkVisualization = dynamic(() => import("@/components/NetworkVisualization"), {
    ssr: false,
    loading: () => (
        <div className="flex items-center justify-center h-[600px] bg-white rounded-xl border-2 border-slate-200">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
    ),
});

export default function DashboardPage() {
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
            console.log(`🎲 Initializing simulation with seed: ${randomSeed}`);
            console.log(`👤 User: ${userEntity.name} (${userEntity.type})`);

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
            console.log(`✅ Simulation created: ${response.simulation_id}`);

            const state = await api.getState(response.simulation_id);
            setCurrentSnapshot(state.snapshot);
            setHistory([state.snapshot]);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to initialize simulation");
            console.error("Initialization error:", err);
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

                // Check if user died in this step
                const latestSnapshot = newSnapshots[newSnapshots.length - 1];
                const userNode = latestSnapshot.agents.find((a: GraphNode) => a.id === userEntity?.id);
                if (userNode && !userNode.alive) {
                    setIsPlaying(false); // Stop simulation
                    console.log("💀 User entity died - stopping simulation");
                }
            }

            // Check for pending decision - pause if present
            if (response.pending_decision) {
                setPendingDecision(response.pending_decision);
                setIsPlaying(false); // Pause simulation
                console.log("⏸️ Simulation paused - user decision required");
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to step simulation");
            console.error("Step error:", err);
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
                custom_params: customParams
            });

            setPendingDecision(null); // Clear the decision
            console.log(`✅ Decision ${approved ? "approved" : "rejected"}`);

            // Fetch updated state to reflect policy changes
            const state = await api.getState(simId);
            setCurrentSnapshot(state.snapshot);
            setHistory((prev) => [...prev, state.snapshot]);
            console.log("📊 Updated state after decision applied");

            // Optionally resume playing if it was playing before
            // setIsPlaying(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to respond to decision");
            console.error("Decision response error:", err);
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
            setError(err instanceof Error ? err.message : "Failed to apply shock");
            console.error("Shock error:", err);
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
            setError(err instanceof Error ? err.message : "Failed to reset simulation");
            console.error("Reset error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleEntityCreated = (entity: UserEntity) => {
        setUserEntity(entity);
        console.log("User entity created:", entity);
    };

    const handlePolicyChange = async (policies: any) => {
        if (!userEntity) return;

        setUserEntity((prev) => prev ? { ...prev, policies } : null);
        console.log("Policies updated:", policies);
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

    // Get user node health, alive status, and live policies
    const userNode = currentSnapshot?.agents.find((a: GraphNode) => a.id === userEntity.id);
    const userHealth = userNode?.health ?? 1.0;
    const isUserAlive = userNode?.alive ?? true;

    // Sync live agent policies with userEntity for display (for banks only)
    const liveUserEntity = (() => {
        if (!userNode || userEntity.type !== 'bank') return userEntity;

        const bankPolicies = userEntity.policies as any;
        const updatedPolicies = { ...bankPolicies };

        if (userNode.risk_appetite !== undefined) {
            updatedPolicies.riskAppetite = userNode.risk_appetite * 100; // Convert 0-1 to 0-100
        }

        return {
            ...userEntity,
            policies: updatedPolicies
        };
    })();

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
            {/* Header */}
            <header className="border-b-2 border-slate-200 bg-white/80 backdrop-blur-sm shadow-sm">
                <div className="container mx-auto px-4 py-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                                Financial Network Simulator
                            </h1>
                            <p className="text-xs text-slate-600 font-medium mt-1">
                                {currentSnapshot
                                    ? `Timestep ${currentSnapshot.timestep} • ${currentSnapshot.global_metrics.alive_agents}/${currentSnapshot.global_metrics.total_agents} Active • Health: ${(currentSnapshot.global_metrics.system_health * 100).toFixed(0)}%`
                                    : "Initializing simulation..."}
                            </p>
                        </div>

                        <div className="flex items-center gap-2">
                            <button
                                onClick={togglePlayPause}
                                disabled={!simId || isLoading || !isUserAlive}
                                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:text-slate-500 text-white rounded-lg transition-all font-semibold shadow-sm"
                            >
                                {isPlaying ? (
                                    <>
                                        <Pause className="w-4 h-4" />
                                        Pause
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4" />
                                        Play
                                    </>
                                )}
                            </button>

                            <button
                                onClick={() => stepSimulation(1)}
                                disabled={!simId || isLoading || isPlaying || !isUserAlive}
                                className="flex items-center gap-2 px-3 py-2 bg-slate-200 hover:bg-slate-300 disabled:bg-slate-100 disabled:text-slate-400 text-slate-700 rounded-lg transition-all"
                            >
                                <SkipForward className="w-4 h-4" />
                            </button>

                            <button
                                onClick={handleShockRealEstate}
                                disabled={!simId || isLoading || !isUserAlive}
                                className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-slate-300 disabled:text-slate-500 text-white rounded-lg transition-all font-semibold shadow-sm"
                            >
                                <Zap className="w-4 h-4" />
                                Apply Shock
                            </button>

                            <button
                                onClick={handleReset}
                                disabled={!simId || isLoading}
                                className="flex items-center gap-2 px-3 py-2 bg-slate-200 hover:bg-slate-300 disabled:bg-slate-100 disabled:text-slate-400 text-slate-700 rounded-lg transition-all"
                            >
                                <RotateCcw className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {error && (
                        <div className="mt-3 px-4 py-2 bg-red-50 border-2 border-red-200 rounded-lg text-red-700 text-sm font-medium flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" />
                            {error}
                        </div>
                    )}

                    {/* User Death Notice */}
                    {!isUserAlive && (
                        <div className="mt-3 px-6 py-4 bg-gradient-to-r from-red-900 to-red-700 border-2 border-red-800 rounded-xl text-white font-bold flex items-center gap-3 shadow-lg">
                            <div className="flex-shrink-0 w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </div>
                            <div className="flex-1">
                                <div className="text-lg font-extrabold mb-1">💀 Your Entity Has Defaulted</div>
                                <div className="text-sm text-red-100 font-medium">Controls disabled. Simulation continues for observation. Click Reset to start over.</div>
                            </div>
                        </div>
                    )}
                </div>
            </header>

            {/* Main Dashboard */}
            <main className="container mx-auto px-4 py-6">
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Left: User Controls (1/4) */}
                    <div className="space-y-6">
                        {/* Overlay when user is dead */}
                        <div className="relative">
                            {!isUserAlive && (
                                <div className="absolute inset-0 z-10 bg-slate-900/70 backdrop-blur-sm rounded-xl flex items-center justify-center">
                                    <div className="text-center text-white">
                                        <div className="text-4xl mb-2">💀</div>
                                        <div className="font-bold text-lg">Controls Disabled</div>
                                        <div className="text-sm text-slate-300 mt-1">Your entity has defaulted</div>
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

                        {/* Quick Stats */}
                        <div className="bg-white rounded-xl shadow-lg border-2 border-slate-200 p-4 space-y-3">
                            <h3 className="font-bold text-slate-900 text-sm uppercase tracking-wide">Quick Stats</h3>
                            <div className="space-y-2">
                                <StatCard
                                    icon={<Activity className="w-4 h-4" />}
                                    label="System Health"
                                    value={currentSnapshot ? `${(currentSnapshot.global_metrics.system_health * 100).toFixed(0)}%` : "—"}
                                    color="blue"
                                />
                                <StatCard
                                    icon={<Users className="w-4 h-4" />}
                                    label="Active Entities"
                                    value={currentSnapshot ? `${currentSnapshot.global_metrics.alive_agents}/${currentSnapshot.global_metrics.total_agents}` : "—"}
                                    color="green"
                                />
                                <StatCard
                                    icon={<AlertTriangle className="w-4 h-4" />}
                                    label="Defaults"
                                    value={currentSnapshot ? `${currentSnapshot.global_metrics.total_agents - currentSnapshot.global_metrics.alive_agents}` : "—"}
                                    color="red"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Center: Network Visualization (2/4) */}
                    <div className="lg:col-span-2">
                        {currentSnapshot ? (
                            <NetworkVisualization
                                nodes={currentSnapshot.agents as GraphNode[]}
                                links={currentSnapshot.links}
                                onNodeSelect={setSelectedNode}
                                selectedNodeId={selectedNode?.id ?? null}
                                userNodeId={userEntity.id}
                                width={typeof window !== "undefined" ? Math.min(window.innerWidth * 0.55, 1100) : 900}
                                height={800}
                            />
                        ) : (
                            <div className="bg-white rounded-xl border-2 border-slate-200 flex items-center justify-center h-[800px]">
                                <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                            </div>
                        )}
                    </div>

                    {/* Right: Metrics & Inspector (1/4) */}
                    <div className="space-y-6">
                        {/* Metrics Panel */}
                        <div className="bg-white rounded-xl shadow-lg border-2 border-slate-200 p-4">
                            <h2 className="text-sm font-bold mb-4 text-slate-900 uppercase tracking-wide">System Metrics</h2>

                            {/* Charts */}
                            <div className="space-y-4">
                                <div>
                                    <div className="text-xs font-semibold text-slate-600 mb-2">System Health Over Time</div>
                                    <ResponsiveContainer width="100%" height={120}>
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                            <XAxis dataKey="timestep" stroke="#94a3b8" style={{ fontSize: "10px" }} />
                                            <YAxis stroke="#94a3b8" style={{ fontSize: "10px" }} domain={[0, 100]} />
                                            <Tooltip content={<CustomTooltip valueFormatter={(v: any) => `${v.toFixed(1)}%`} />} />
                                            <Line type="monotone" dataKey="system_health" stroke="#3b82f6" strokeWidth={2} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>

                                <div>
                                    <div className="text-xs font-semibold text-slate-600 mb-2">Active Entities</div>
                                    <ResponsiveContainer width="100%" height={120}>
                                        <AreaChart data={chartData}>
                                            <defs>
                                                <linearGradient id="entitiesGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                            <XAxis dataKey="timestep" stroke="#94a3b8" style={{ fontSize: "10px" }} />
                                            <YAxis stroke="#94a3b8" style={{ fontSize: "10px" }} />
                                            <Tooltip content={<CustomTooltip valueFormatter={(v: any) => `${v} entities`} />} />
                                            <Area type="monotone" dataKey="alive_agents" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#entitiesGradient)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        {/* Inspector Panel */}
                        <div className="bg-white rounded-xl shadow-lg border-2 border-slate-200 p-4">
                            <h2 className="text-sm font-bold mb-4 text-slate-900 uppercase tracking-wide">Entity Inspector</h2>

                            {selectedNode ? (
                                <div className="space-y-3 text-sm">
                                    <div className="pb-3 border-b-2 border-slate-100">
                                        <div className="text-xs font-semibold text-slate-500 mb-1">Entity ID</div>
                                        <div className="font-bold text-slate-900">{selectedNode.id}</div>
                                    </div>

                                    <div>
                                        <div className="text-xs font-semibold text-slate-500 mb-1">Type</div>
                                        <div className="uppercase text-blue-600 font-semibold">{selectedNode.type.replace("_", " ")}</div>
                                    </div>

                                    <div>
                                        <div className="text-xs font-semibold text-slate-500 mb-1">Health</div>
                                        <div className={`text-lg font-bold ${selectedNode.health > 0.7 ? "text-green-600" : selectedNode.health > 0.4 ? "text-amber-600" : "text-red-600"}`}>
                                            {(selectedNode.health * 100).toFixed(1)}%
                                        </div>
                                        <div className="w-full bg-slate-200 rounded-full h-2 mt-2">
                                            <div
                                                className={`h-2 rounded-full transition-all ${selectedNode.health > 0.7 ? "bg-green-500" : selectedNode.health > 0.4 ? "bg-amber-500" : "bg-red-500"}`}
                                                style={{ width: `${selectedNode.health * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <div className="text-xs font-semibold text-slate-500 mb-1">Status</div>
                                        <div className={`font-semibold ${selectedNode.alive ? "text-green-600" : "text-red-600"}`}>
                                            {selectedNode.alive ? "✓ Active" : "✗ Defaulted"}
                                        </div>
                                    </div>

                                    <div className="pt-3 border-t-2 border-slate-100 space-y-2">
                                        <div className="flex justify-between">
                                            <span className="text-xs font-semibold text-slate-500">Capital</span>
                                            <span className="font-mono font-semibold text-slate-900">${(selectedNode.capital / 1e6).toFixed(2)}M</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-xs font-semibold text-slate-500">Liquidity</span>
                                            <span className="font-mono font-semibold text-slate-900">${(selectedNode.liquidity / 1e6).toFixed(2)}M</span>
                                        </div>
                                        {selectedNode.npa !== undefined && (
                                            <div className="flex justify-between">
                                                <span className="text-xs font-semibold text-slate-500">NPA</span>
                                                <span className="font-mono font-semibold text-red-600">${(selectedNode.npa / 1e6).toFixed(2)}M</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-slate-400 text-sm text-center py-12">
                                    Click on an entity to inspect details
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            {/* Risk Decision Modal - Only show if user is alive */}
            {pendingDecision && isUserAlive && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                    <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        {/* Header */}
                        <div className={`p-6 border-b-2 ${pendingDecision.risk_level === 'critical' ? 'bg-red-50 border-red-200' :
                            pendingDecision.risk_level === 'high' ? 'bg-orange-50 border-orange-200' :
                                pendingDecision.risk_level === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                                    'bg-blue-50 border-blue-200'
                            }`}>
                            <h2 className="text-2xl font-bold text-slate-900 mb-2">
                                {pendingDecision.title}
                            </h2>
                            <p className="text-slate-700">
                                {pendingDecision.description}
                            </p>
                            <div className="mt-3 inline-block px-3 py-1 text-xs font-bold uppercase tracking-wide rounded-full bg-white/70 text-slate-900">
                                Timestep {pendingDecision.timestep}
                            </div>
                        </div>

                        {/* Current Metrics */}
                        <div className="p-6 border-b border-slate-200 bg-slate-50">
                            <h3 className="text-sm font-bold text-slate-600 uppercase tracking-wide mb-3">Current Metrics</h3>
                            <div className="grid grid-cols-2 gap-3">
                                {Object.entries(pendingDecision.current_metrics).map(([key, value]) => (
                                    <div key={key} className="bg-white p-3 rounded-lg border border-slate-200">
                                        <div className="text-xs text-slate-500 capitalize">{key.replace(/_/g, ' ')}</div>
                                        <div className="text-lg font-bold text-slate-900">
                                            {typeof value === 'number' ? (value * 100).toFixed(1) + (key.includes('ratio') || key.includes('health') ? '%' : '') : value}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Options */}
                        <div className="p-6 space-y-4">
                            {/* Recommended Action */}
                            <div className="border-2 border-green-200 rounded-xl p-4 bg-green-50">
                                <div className="flex items-start gap-3">
                                    <div className="mt-1">
                                        <div className="w-10 h-10 rounded-full bg-green-600 flex items-center justify-center">
                                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                            </svg>
                                        </div>
                                    </div>
                                    <div className="flex-1">
                                        <div className="text-xs font-bold text-green-800 uppercase tracking-wide mb-1">Recommended</div>
                                        <h4 className="font-bold text-slate-900 mb-2">{pendingDecision.recommended_action.description}</h4>
                                        <p className="text-sm text-slate-600 mb-3">{pendingDecision.recommended_action.impact}</p>
                                        <button
                                            onClick={() => handleDecisionResponse(true)}
                                            disabled={isLoading}
                                            className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-300 text-white font-semibold rounded-lg transition-all shadow-sm"
                                        >
                                            {isLoading ? "Applying..." : "Apply Recommended Action"}
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Alternative Action */}
                            <div className="border-2 border-slate-200 rounded-xl p-4 bg-slate-50">
                                <div className="flex items-start gap-3">
                                    <div className="mt-1">
                                        <div className="w-10 h-10 rounded-full bg-slate-400 flex items-center justify-center">
                                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                            </svg>
                                        </div>
                                    </div>
                                    <div className="flex-1">
                                        <div className="text-xs font-bold text-slate-600 uppercase tracking-wide mb-1">Alternative</div>
                                        <h4 className="font-bold text-slate-900 mb-2">{pendingDecision.alternative_action.description}</h4>
                                        <p className="text-sm text-slate-600 mb-3">{pendingDecision.alternative_action.impact}</p>
                                        <button
                                            onClick={() => handleDecisionResponse(false)}
                                            disabled={isLoading}
                                            className="w-full px-4 py-2 bg-slate-500 hover:bg-slate-600 disabled:bg-slate-300 text-white font-semibold rounded-lg transition-all shadow-sm"
                                        >
                                            {isLoading ? "Processing..." : "Reject & Maintain Current Strategy"}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// StatCard Component
interface StatCardProps {
    icon: React.ReactNode;
    label: string;
    value: string;
    color: "blue" | "green" | "red" | "amber";
}

function StatCard({ icon, label, value, color }: StatCardProps) {
    const colorClasses = {
        blue: "bg-blue-50 text-blue-600",
        green: "bg-green-50 text-green-600",
        red: "bg-red-50 text-red-600",
        amber: "bg-amber-50 text-amber-600",
    };

    return (
        <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
            <div className="flex-1">
                <div className="text-xs font-medium text-slate-600">{label}</div>
                <div className="text-lg font-bold text-slate-900">{value}</div>
            </div>
        </div>
    );
}
