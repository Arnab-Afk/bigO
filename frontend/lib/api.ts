import {
    InitializeParams,
    InitializeResponse,
    StepResponse,
    ShockParams,
    ShockResponse,
    SimulationState,
    UserDecisionRequest,
} from "@/types/simulation";

const BASE_URL = "http://localhost:17170/api/v1/abm";

class APIError extends Error {
    constructor(public status: number, message: string) {
        super(message);
        this.name = "APIError";
    }
}

async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
        const errorText = await response.text().catch(() => "Unknown error");
        throw new APIError(
            response.status,
            `API Error ${response.status}: ${errorText}`
        );
    }
    return response.json();
}

/**
 * Initialize a new simulation
 * POST /initialize
 */
export async function initialize(
    params: InitializeParams
): Promise<InitializeResponse> {
    const response = await fetch(`${BASE_URL}/initialize`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
    });

    return handleResponse<InitializeResponse>(response);
}

/**
 * Step the simulation forward by N steps
 * POST /{simId}/step
 */
export async function step(
    simId: string,
    steps: number = 1
): Promise<StepResponse> {
    const response = await fetch(`${BASE_URL}/${simId}/step`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ num_steps: steps }),
    });

    const rawResponse = await handleResponse<any>(response);

    // Transform snapshots to frontend format
    return {
        current_timestep: rawResponse.current_timestep,
        pending_decision: rawResponse.pending_decision || undefined,
        snapshots: rawResponse.snapshots.map((snap: any) => {
            // Calculate system-wide metrics
            const agents = snap.network_state.nodes.map((node: any) => ({
                id: node.id,
                type: node.type,
                health: node.health || 0,
                alive: node.alive,
                capital: snap.agent_states[node.id]?.capital || 0,
                liquidity: snap.agent_states[node.id]?.liquidity || 0,
                npa: snap.agent_states[node.id]?.npa_ratio || 0,
                crar: snap.agent_states[node.id]?.crar || 0,
                risk_appetite: snap.agent_states[node.id]?.risk_appetite,
                credit_supply_limit: snap.agent_states[node.id]?.credit_supply_limit,
                interbank_limit: snap.agent_states[node.id]?.interbank_limit,
            }));

            // Calculate actual total capital and system health
            const totalCapital = agents.reduce((sum: number, a: any) => sum + (a.capital || 0), 0);
            const aliveAgents = agents.filter((a: any) => a.alive);
            const avgHealth = aliveAgents.length > 0
                ? aliveAgents.reduce((sum: number, a: any) => sum + (a.health || 0), 0) / aliveAgents.length
                : 0;

            return {
                timestep: snap.timestep,
                agents: agents,
                links: snap.network_state.edges.map((edge: any) => ({
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight,
                    type: edge.type,
                })),
                global_metrics: {
                    total_liquidity: snap.global_metrics.system_liquidity || 0,
                    total_capital: totalCapital,
                    total_npa: snap.global_metrics.system_npa || 0,
                    system_health: avgHealth,
                    alive_agents: aliveAgents.length,
                    total_agents: agents.length,
                    timestep: snap.timestep,
                    survival_rate: snap.global_metrics.survival_rate || 0,
                    avg_crar: snap.global_metrics.avg_crar || 0,
                },
                events: snap.events,
            };
        }),
    };
}

/**
 * Respond to a user decision alert
 * POST /{simId}/decision
 */
export async function respondToDecision(
    simId: string,
    decision: UserDecisionRequest
): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${BASE_URL}/${simId}/decision`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(decision),
    });

    return handleResponse<{ success: boolean; message: string }>(response);
}

/**
 * Apply a generic shock with sector and severity selection
 * POST /{simId}/shock
 */
export async function applyShock(
    simId: string,
    params: {
        shock_type: string;
        severity: string;
        magnitude?: number;
        target?: string | null;
    }
): Promise<ShockResponse> {
    const response = await fetch(`${BASE_URL}/${simId}/shock`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            shock_type: params.shock_type,
            severity: params.severity,
            magnitude: params.magnitude ?? -0.3,
            target: params.target ?? null
        }),
    });

    return handleResponse<ShockResponse>(response);
}

/**
 * Apply a shock to a specific target (legacy function)
 * POST /{simId}/shock
 */
export async function shock(
    simId: string,
    params: ShockParams
): Promise<ShockResponse> {
    const response = await fetch(`${BASE_URL}/${simId}/shock`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
    });

    return handleResponse<ShockResponse>(response);
}

/**
 * Get current simulation state
 * GET /{simId}/state
 */
export async function getState(simId: string): Promise<SimulationState> {
    const response = await fetch(`${BASE_URL}/${simId}/state`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    });

    const rawState = await handleResponse<any>(response);

    // Transform backend response to frontend format
    const agents = rawState.network_state.nodes.map((node: any) => ({
        id: node.id,
        type: node.type,
        health: node.health || 0,
        alive: node.alive,
        capital: rawState.agent_states[node.id]?.capital || 0,
        liquidity: rawState.agent_states[node.id]?.liquidity || 0,
        npa: rawState.agent_states[node.id]?.npa_ratio || 0,
        crar: rawState.agent_states[node.id]?.crar || 0,
        risk_appetite: rawState.agent_states[node.id]?.risk_appetite,
        credit_supply_limit: rawState.agent_states[node.id]?.credit_supply_limit,
        interbank_limit: rawState.agent_states[node.id]?.interbank_limit,
    }));

    // Calculate actual metrics
    const totalCapital = agents.reduce((sum: number, a: any) => sum + (a.capital || 0), 0);
    const aliveAgents = agents.filter((a: any) => a.alive);
    const avgHealth = aliveAgents.length > 0
        ? aliveAgents.reduce((sum: number, a: any) => sum + (a.health || 0), 0) / aliveAgents.length
        : 0;

    return {
        simulation_id: rawState.simulation_id,
        status: "running" as const,
        current_timestep: rawState.timestep,
        max_timesteps: 100, // This should come from config
        snapshot: {
            timestep: rawState.timestep,
            agents: agents,
            links: rawState.network_state.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                weight: edge.weight,
                type: edge.type,
            })),
            global_metrics: {
                total_liquidity: rawState.global_metrics.system_liquidity || 0,
                total_capital: totalCapital,
                total_npa: rawState.global_metrics.system_npa || 0,
                system_health: avgHealth,
                alive_agents: aliveAgents.length,
                total_agents: agents.length,
                timestep: rawState.timestep,
                survival_rate: rawState.global_metrics.survival_rate || 0,
                avg_crar: rawState.global_metrics.avg_crar || 0,
            },
        },
    };
}

/**
 * Update agent policies during simulation
 * POST /{simId}/agent-policy
 */
export async function updateAgentPolicy(
    simId: string,
    agentId: string,
    policies: Record<string, any>
): Promise<{ simulation_id: string; agent_id: string; updated_policies: any; status: string }> {
    const response = await fetch(`${BASE_URL}/${simId}/agent-policy`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            agent_id: agentId,
            policies: policies,
        }),
    });

    return handleResponse<{ simulation_id: string; agent_id: string; updated_policies: any; status: string }>(response);
}

/**
 * Reset the simulation to initial state
 * POST /{simId}/reset
 */
export async function reset(simId: string): Promise<{ message: string }> {
    const response = await fetch(`${BASE_URL}/${simId}/reset`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
    });

    return handleResponse<{ message: string }>(response);
}

/**
 * Helper to shock Real Estate sector
 */
export async function shockRealEstate(
    simId: string,
    magnitude: number = -0.3
): Promise<ShockResponse> {
    return shock(simId, {
        shock_type: "sector_crisis",
        target: "real_estate",
        magnitude,
    });
}

export { APIError };
