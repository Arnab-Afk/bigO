// Core simulation types for the ABM system

export interface Agent {
    id: string;
    type: "bank" | "sector" | "clearing_house" | "ccp" | "regulator";
    health: number; // 0.0 to 1.0
    alive: boolean;
    capital: number;
    liquidity: number;
    npa: number;
    risk_appetite?: number;
    credit_supply_limit?: number;
    interbank_limit?: number;
    strategy?: string; // AI strategy mode
    sector?: string;
    position?: {
        x: number;
        y: number;
        z?: number;
    };
}

export interface Link {
    source: string;
    target: string;
    weight: number; // Loan size/strength
    type?: "loan" | "clearing" | "interbank";
}

export interface GlobalMetrics {
    total_liquidity: number;
    total_capital: number;
    total_npa: number;
    system_health: number;
    alive_agents: number;
    total_agents: number;
    timestep: number;
    survival_rate?: number;
    avg_crar?: number;
}

export interface SimulationSnapshot {
    timestep: number;
    agents: Agent[];
    links: Link[];
    global_metrics: GlobalMetrics;
    events?: string[];
}

export interface RiskDecision {
    decision_id: string;
    agent_id: string;
    timestep: number;
    title: string;
    description: string;
    risk_level: "low" | "medium" | "high" | "critical";
    current_metrics: Record<string, number>;
    recommended_action: {
        type: string;
        description: string;
        impact: string;
    };
    alternative_action: {
        type: string;
        description: string;
        impact: string;
    };
}

export interface SimulationState {
    simulation_id: string;
    status: "initialized" | "running" | "paused" | "completed";
    current_timestep: number;
    max_timesteps: number;
    snapshot: SimulationSnapshot;
}

export interface InitializeParams {
    name: string;
    max_timesteps: number;
    enable_shocks: boolean;
    random_seed?: number;
    shock_probability?: number;
    use_real_data?: boolean;
    user_entity?: {
        id: string;
        type: string;
        name: string;
        policies: any;
    };
}

export interface InitializeResponse {
    simulation_id: string;
    name: string;
    config: {
        max_timesteps: number;
        enable_shocks: boolean;
        shock_probability: number;
        random_seed: number | null;
    };
    network_stats: any;
    initial_state: any;
}

export interface StepResponse {
    snapshots: SimulationSnapshot[];
    current_timestep: number;
    pending_decision?: RiskDecision;
}

export interface UserDecisionRequest {
    decision_id: string;
    approved: boolean;
    custom_params?: Record<string, any>;
}

export interface ShockParams {
    shock_type: string;
    target?: string;
    magnitude: number;
}

export interface ShockResponse {
    simulation_id: string;
    shock_applied: boolean;
    shock_event: any;
}

export interface GraphNode extends Agent {
    // Extended properties for force graph
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
}

export interface GraphLink extends Omit<Link, "source" | "target"> {
    // Force graph compatible
    source: string | GraphNode;
    target: string | GraphNode;
}
