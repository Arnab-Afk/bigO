/**
 * CCP Risk Analysis API Client
 * 
 * Client for communicating with the CCP risk analysis backend endpoints
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:17170/api/v1';

export interface SimulationConfig {
  year?: number;
  sector_weight?: number;
  liquidity_weight?: number;
  market_weight?: number;
  edge_threshold?: number;
}

export interface StressTestConfig {
  shock_magnitude: number;
  target_banks?: string[];
  shock_type: string;
}

export interface BankRiskScore {
  bank_name: string;
  default_probability: number;
  risk_tier: string;
  capital_ratio: number;
  stress_level: number;
  pagerank: number;
  degree_centrality: number;
  betweenness_centrality: number;
  eigenvector_centrality: number;
}

export interface NetworkNode {
  id: string;
  name: string;
  default_probability: number;
  risk_level: 'high' | 'medium' | 'low';
  pagerank: number;
  degree_centrality: number;
  degree: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

export interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metrics: {
    total_nodes: number;
    total_edges: number;
    density: number;
  };
}

export interface NetworkMetrics {
  total_nodes: number;
  total_edges: number;
  avg_degree: number;
  density: number;
  clustering_coefficient: number;
}

export interface SpectralMetrics {
  spectral_radius: number;
  fiedler_value: number;
  contagion_index: number;
  eigenvalue_entropy: number;
  fragility_score: number;
  risk_level: string;
}

export interface CCPMetrics {
  total_margin_requirement: number;
  default_fund_size: number;
  cover_n_standard: number;
  total_exposure: number;
  largest_counterparty_exposure: number;
}

export interface SimulationSummary {
  total_banks: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  last_run: string | null;
  network_metrics: NetworkMetrics;
  spectral_metrics: SpectralMetrics;
  ccp_metrics: CCPMetrics;
}

export interface PolicyRecommendation {
  priority: string;
  category: string;
  title: string;
  description: string;
  affected_banks: string[];
  impact: string;
  implementation: string;
}

export interface BankDetail {
  bank_name: string;
  current_metrics: {
    default_probability: number;
    capital_ratio: number;
    stress_level: number;
    liquidity_buffer: number;
    leverage: number;
  };
  network_position: {
    neighbors: string[];
    degree: number;
    pagerank: number;
    betweenness: number;
  };
  margin_requirement: {
    base_margin: number;
    network_addon: number;
    total_margin: number;
    explanation: string;
  } | null;
  historical_trend: Array<{
    year: number;
    capital_ratio: number;
    stress_level: number;
    liquidity_buffer: number;
  }>;
}

export interface MarginRequirement {
  bank_name: string;
  base_margin: number;
  network_addon: number;
  stressed_margin: number;
  total_margin: number;
  confidence_level: number;
  explanation: string;
}

export interface DefaultFund {
  total_fund_size: number;
  cover_n: number;
  confidence_level: number;
  contributions: Array<{
    bank_name: string;
    base_contribution: number;
    systemic_addon: number;
    total_contribution: number;
  }>;
  explanation: string;
}

class CCPApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `HTTP error! status: ${response.status}`,
      }));
      throw new Error(error.detail || 'API request failed');
    }

    return response.json();
  }

  // Simulation
  async runSimulation(config: SimulationConfig = {}): Promise<any> {
    return this.request('/ccp/simulate', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getStatus(): Promise<{
    initialized: boolean;
    last_run: string | null;
    available: boolean;
    total_banks: number;
  }> {
    return this.request('/ccp/status');
  }

  async getSummary(): Promise<SimulationSummary> {
    return this.request('/ccp/summary');
  }

  // Network
  async getNetworkData(): Promise<NetworkData> {
    return this.request('/ccp/network');
  }

  // Banks
  async getAllBanks(): Promise<BankRiskScore[]> {
    return this.request('/ccp/banks');
  }

  async getBankDetail(bankName: string): Promise<BankDetail> {
    return this.request(`/ccp/banks/${encodeURIComponent(bankName)}`);
  }

  // Spectral Analysis
  async getSpectralMetrics(): Promise<any> {
    return this.request('/ccp/spectral');
  }

  // Stress Testing
  async runStressTest(config: StressTestConfig): Promise<any> {
    return this.request('/ccp/stress-test', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // Policies
  async getPolicyRecommendations(): Promise<PolicyRecommendation[]> {
    return this.request('/ccp/policies');
  }

  // Margins
  async getMarginRequirements(): Promise<MarginRequirement[]> {
    return this.request('/ccp/margins');
  }

  // Default Fund
  async getDefaultFund(): Promise<DefaultFund> {
    return this.request('/ccp/default-fund');
  }
}

// Export singleton instance
export const ccpApi = new CCPApiClient();
