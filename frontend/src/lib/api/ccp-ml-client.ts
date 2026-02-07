/**
 * CCP ML API Client
 * Integration with local backend/ccp_ml API
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
export interface CCPStatus {
  status: string;
  service?: string;
  initialized?: boolean;
  last_run?: string;
  n_banks?: number;
  n_features?: number;
  n_edges?: number;
}

export interface SimulationConfig {
  year?: number;
  sector_weight?: number;
  liquidity_weight?: number;
  market_weight?: number;
  edge_threshold?: number;
}

export interface SimulationResult {
  status: string;
  timestamp: string;
  config: SimulationConfig;
  network: {
    n_nodes: number;
    n_edges: number;
  };
  spectral: {
    spectral_radius: number;
    fiedler_value: number;
    amplification_risk: string;
    fragmentation_risk: string;
    contagion_index: number;
  };
  ccp: {
    n_participants: number;
    risk_distribution: Record<string, number>;
    margin_summary: Record<string, any>;
    default_fund: Record<string, any>;
  };
  policies: any[];
}

export interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metrics?: any[];
}

export interface NetworkNode {
  bank_name: string;
  pagerank?: number;
  degree_centrality?: number;
  betweenness_centrality?: number;
  eigenvector_centrality?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  sector_similarity?: number;
  liquidity_similarity?: number;
  market_correlation?: number;
}

export interface RiskScore {
  bank_name: string;
  default_probability?: number;
  stress_level?: number;
  capital_ratio?: number;
}

export interface BankRiskDetail {
  bank_name: string;
  features: Record<string, any>;
  network_position: Record<string, any> | null;
}

export interface SpectralAnalysis {
  spectral_radius: number;
  fiedler_value: number;
  spectral_gap: number;
  effective_rank: number;
  eigenvalue_entropy: number;
  amplification_risk: string;
  fragmentation_risk: string;
  contagion_index: number;
}

export interface StressTestConfig {
  shock_magnitude: number;
  target_banks?: string[];
  shock_type: 'capital' | 'liquidity' | 'market';
}

export interface StressTestResult {
  status: string;
  shock_config: StressTestConfig;
  baseline: {
    risk_distribution: Record<string, number> | null;
    default_fund: number;
  };
  stressed: {
    risk_distribution: Record<string, number>;
    default_fund: number;
  };
  impact: {
    fund_increase_pct: number;
    new_high_risk_count: number;
  };
}

export interface MarginRequirement {
  bank_name: string;
  base_margin: number;
  network_addon: number;
  total_margin: number;
  explanation: string;
}

export interface DefaultFund {
  total_fund: number;
  allocation: Record<string, any>;
}

// API Client Class
class CCPMLApiClient {
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

  // Health & Status
  async healthCheck() {
    return this.fetch<CCPStatus>('/');
  }

  async getStatus() {
    return this.fetch<CCPStatus>('/api/status');
  }

  // Simulation
  async runSimulation(config?: SimulationConfig) {
    return this.fetch<SimulationResult>('/api/simulate', {
      method: 'POST',
      body: JSON.stringify(config || {}),
    });
  }

  async reinitialize() {
    return this.fetch<{ status: string; message: string }>('/api/reinitialize', {
      method: 'POST',
    });
  }

  // Network
  async getNetwork() {
    return this.fetch<NetworkData>('/api/network');
  }

  async getNetworkNodes() {
    return this.fetch<NetworkNode[]>('/api/network/nodes');
  }

  async getNetworkEdges() {
    return this.fetch<NetworkEdge[]>('/api/network/edges');
  }

  // Risk
  async getRiskScores() {
    return this.fetch<RiskScore[]>('/api/risk/scores');
  }

  async getBankRisk(bankName: string) {
    return this.fetch<BankRiskDetail>('/api/risk/bank', {
      method: 'POST',
      body: JSON.stringify({ bank_name: bankName }),
    });
  }

  // Spectral Analysis
  async getSpectralAnalysis() {
    return this.fetch<SpectralAnalysis>('/api/spectral');
  }

  // Stress Testing
  async runStressTest(config: StressTestConfig) {
    return this.fetch<StressTestResult>('/api/stress-test', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // Margins & Default Fund
  async getMargins() {
    return this.fetch<MarginRequirement[]>('/api/margins');
  }

  async getDefaultFund() {
    return this.fetch<DefaultFund>('/api/default-fund');
  }
}

// Export singleton instance
export const ccpMLApi = new CCPMLApiClient();
export default ccpMLApi;
