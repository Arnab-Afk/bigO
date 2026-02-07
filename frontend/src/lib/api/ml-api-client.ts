/**
 * ML Dashboard API Client
 * Integration with api.rudranet.xyz
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.rudranet.xyz';

// Types
export interface Institution {
  id: string;
  external_id: string;
  name: string;
  short_name?: string;
  type: string;
  tier: string;
  jurisdiction?: string;
  region?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  current_state?: InstitutionState;
  total_outbound_exposure?: string;
  total_inbound_exposure?: string;
}

export interface InstitutionState {
  timestamp: string;
  capital_ratio: string;
  liquidity_buffer: string;
  total_credit_exposure: string;
  default_probability: string;
  stress_level: string;
  risk_score?: string;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface NetworkNode {
  id: string;
  label: string;
  type: string;
  tier: string;
  metrics?: {
    degree?: number;
    betweenness?: number;
    pagerank?: number;
    eigenvector?: number;
  };
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  exposure_type?: string;
}

export interface NetworkMetrics {
  total_nodes: number;
  total_edges: number;
  density: number;
  average_degree: number;
  clustering_coefficient?: number;
  diameter?: number;
  spectral_radius?: number;
  fiedler_value?: number;
}

export interface AdvancedNetworkAnalysis {
  centrality_metrics: Record<string, {
    degree: number;
    betweenness: number;
    eigenvector: number;
    pagerank: number;
  }>;
  systemic_risk_indicators: {
    spectral_radius: number;
    fiedler_value: number;
    contagion_index: number;
  };
  community_detection?: {
    modularity: number;
    communities: Record<string, string[]>;
  };
}

export interface Simulation {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_timestep: number;
  total_timesteps: number;
  progress: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface SimulationResult {
  id: string;
  simulation_id: string;
  result_type: string;
  total_defaults: number;
  max_cascade_depth: number;
  survival_rate: string;
  final_systemic_stress: string;
  total_system_loss: string;
  time_to_first_default?: number;
  metrics_data: Record<string, any>;
  timeline_data?: Record<string, any>;
  cascade_data?: Record<string, any>;
}

export interface ABMSimulation {
  simulation_id: string;
  name: string;
  config: {
    max_timesteps: number;
    enable_shocks: boolean;
    shock_probability: number;
    use_real_data: boolean;
  };
  network_stats: {
    num_nodes: number;
    num_edges: number;
    density: number;
  };
  initial_state: Record<string, any>;
}

export interface ABMState {
  simulation_id: string;
  timestep: number;
  global_metrics: Record<string, number>;
  agent_states: Record<string, any>;
  network_state: Record<string, any>;
}

export interface Scenario {
  id: string;
  name: string;
  description?: string;
  category: string;
  is_template: boolean;
  num_timesteps: number;
  shocks: Shock[];
  created_at: string;
}

export interface Shock {
  id: string;
  name: string;
  description?: string;
  shock_type: string;
  magnitude: string;
  duration: number;
  trigger_timestep: number;
}

// API Client Class
class MLApiClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Health Endpoints
  async healthCheck() {
    return this.fetch<{ status: string; version: string; timestamp: string }>('/api/v1/health');
  }

  // Institution Endpoints
  async listInstitutions(params?: {
    page?: number;
    limit?: number;
    type?: string;
    tier?: string;
    is_active?: boolean;
    search?: string;
  }) {
    const query = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) query.append(key, String(value));
      });
    }
    return this.fetch<{
      items: Institution[];
      total: number;
      page: number;
      limit: number;
      pages: number;
    }>(`/api/v1/institutions?${query}`);
  }

  async getInstitution(id: string) {
    return this.fetch<Institution>(`/api/v1/institutions/${id}`);
  }

  async getInstitutionStates(id: string, limit: number = 100) {
    return this.fetch<InstitutionState[]>(`/api/v1/institutions/${id}/states?limit=${limit}`);
  }

  // Network Endpoints
  async getNetworkGraph(params?: {
    format?: string;
    include_weights?: boolean;
    min_exposure?: number;
  }) {
    const query = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) query.append(key, String(value));
      });
    }
    return this.fetch<NetworkGraph>(`/api/v1/network/graph?${query}`);
  }

  async getNetworkMetrics() {
    return this.fetch<NetworkMetrics>('/api/v1/network/metrics');
  }

  async getInstitutionCentrality(institutionId: string) {
    return this.fetch<{
      institution_id: string;
      degree_centrality: number;
      in_degree: number;
      out_degree: number;
    }>(`/api/v1/network/centrality/${institutionId}`);
  }

  async getSystemicImportance(limit: number = 20) {
    return this.fetch<{
      institutions: Array<{
        id: string;
        name: string;
        score: number;
        tier: string;
        total_exposure: string;
      }>;
    }>(`/api/v1/network/systemic-importance?limit=${limit}`);
  }

  async analyzeNetwork(minExposure?: number) {
    const query = minExposure ? `?min_exposure=${minExposure}` : '';
    return this.fetch<AdvancedNetworkAnalysis>(`/api/v1/network/analyze${query}`, {
      method: 'POST',
    });
  }

  async findContagionPaths(sourceId: string, threshold: number = 0.3, maxLength: number = 5) {
    return this.fetch<{
      source_id: string;
      paths: Array<{
        path: string[];
        total_exposure: number;
        contagion_probability: number;
      }>;
    }>(`/api/v1/network/contagion-paths?source_id=${sourceId}&threshold=${threshold}&max_length=${maxLength}`, {
      method: 'POST',
    });
  }

  async simulateCascade(shockedInstitutions: string[], maxRounds: number = 10) {
    return this.fetch<{
      initial_shock: string[];
      rounds: number;
      affected_institutions: string[];
      cascade_tree: Record<string, any>;
      total_loss: string;
    }>(`/api/v1/network/cascade-simulation?max_rounds=${maxRounds}`, {
      method: 'POST',
      body: JSON.stringify(shockedInstitutions),
    });
  }

  // Simulation Endpoints
  async listSimulations(params?: {
    page?: number;
    limit?: number;
    status?: string;
  }) {
    const query = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) query.append(key, String(value));
      });
    }
    return this.fetch<{
      items: Simulation[];
      total: number;
      page: number;
      limit: number;
      pages: number;
    }>(`/api/v1/simulations?${query}`);
  }

  async createSimulation(data: {
    name: string;
    description?: string;
    total_timesteps?: number;
    scenario_id?: string;
    parameters?: Record<string, any>;
  }) {
    return this.fetch<Simulation>('/api/v1/simulations', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getSimulation(id: string) {
    return this.fetch<Simulation>(`/api/v1/simulations/${id}`);
  }

  async startSimulation(id: string) {
    return this.fetch<Simulation>(`/api/v1/simulations/${id}/start`, {
      method: 'POST',
    });
  }

  async cancelSimulation(id: string) {
    return this.fetch<Simulation>(`/api/v1/simulations/${id}/cancel`, {
      method: 'POST',
    });
  }

  async getSimulationResults(id: string) {
    return this.fetch<SimulationResult[]>(`/api/v1/simulations/${id}/results`);
  }

  // Agent-Based Model Endpoints
  async initializeABM(data: {
    name: string;
    max_timesteps?: number;
    enable_shocks?: boolean;
    shock_probability?: number;
    random_seed?: number;
    use_real_data?: boolean;
    data_source?: string;
  }) {
    return this.fetch<ABMSimulation>('/api/v1/abm/initialize', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async runABM(simId: string, steps: number = 10) {
    return this.fetch<{
      simulation_id: string;
      current_timestep: number;
      snapshots: any[];
    }>(`/api/v1/abm/${simId}/run?steps=${steps}`, {
      method: 'POST',
    });
  }

  async getABMState(simId: string) {
    return this.fetch<ABMState>(`/api/v1/abm/${simId}/state`);
  }

  async applyShock(simId: string, data: {
    shock_type: 'sector_crisis' | 'liquidity_squeeze' | 'interest_rate_shock' | 'asset_price_crash';
    target?: string;
    magnitude?: number;
  }) {
    return this.fetch<Record<string, any>>(`/api/v1/abm/${simId}/shock`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateCCPPolicy(simId: string, data: {
    ccp_id?: string;
    rule_name: string;
    condition: string;
    action: string;
  }) {
    return this.fetch<Record<string, any>>(`/api/v1/abm/${simId}/policy`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getABMHistory(simId: string, limit?: number) {
    const query = limit ? `?limit=${limit}` : '';
    return this.fetch<{
      simulation_id: string;
      total_timesteps: number;
      snapshots: any[];
    }>(`/api/v1/abm/${simId}/history${query}`);
  }

  async resetABM(simId: string) {
    return this.fetch<Record<string, any>>(`/api/v1/abm/${simId}/reset`, {
      method: 'POST',
    });
  }

  async deleteABM(simId: string) {
    return this.fetch<Record<string, any>>(`/api/v1/abm/${simId}`, {
      method: 'DELETE',
    });
  }

  async listABMSimulations() {
    return this.fetch<{
      simulations: Record<string, any>;
    }>('/api/v1/abm/list');
  }

  async exportNetwork(simId: string, format: 'gexf' | 'graphml' | 'json' = 'json') {
    return this.fetch<Record<string, any>>(`/api/v1/abm/${simId}/export?format=${format}`, {
      method: 'POST',
    });
  }

  // Scenario Endpoints
  async listScenarios(params?: {
    page?: number;
    limit?: number;
    category?: string;
    is_template?: boolean;
  }) {
    const query = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) query.append(key, String(value));
      });
    }
    return this.fetch<{
      items: Scenario[];
      total: number;
      page: number;
      limit: number;
      pages: number;
    }>(`/api/v1/scenarios?${query}`);
  }

  async getScenarioTemplates() {
    return this.fetch<Scenario[]>('/api/v1/scenarios/templates');
  }

  async getScenario(id: string) {
    return this.fetch<Scenario>(`/api/v1/scenarios/${id}`);
  }

  async createScenario(data: {
    name: string;
    description?: string;
    category?: string;
    num_timesteps?: number;
    shocks: Array<{
      name: string;
      shock_type: string;
      magnitude?: number;
      duration?: number;
      trigger_timestep?: number;
    }>;
  }) {
    return this.fetch<Scenario>('/api/v1/scenarios', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

// Export singleton instance
export const mlApi = new MLApiClient();
export default mlApi;
