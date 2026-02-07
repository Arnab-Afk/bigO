/**
 * React Query hooks for ML Dashboard data fetching
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { mlApi } from '@/lib/api/ml-api-client';
import type {
  Institution,
  NetworkGraph,
  NetworkMetrics,
  AdvancedNetworkAnalysis,
  Simulation,
  SimulationResult,
  ABMSimulation,
  ABMState,
  Scenario,
} from '@/lib/api/ml-api-client';

// Query Keys
export const mlQueryKeys = {
  all: ['ml'] as const,
  health: () => [...mlQueryKeys.all, 'health'] as const,
  institutions: () => [...mlQueryKeys.all, 'institutions'] as const,
  institutionsList: (filters?: any) => [...mlQueryKeys.institutions(), 'list', filters] as const,
  institution: (id: string) => [...mlQueryKeys.institutions(), id] as const,
  institutionStates: (id: string) => [...mlQueryKeys.institution(id), 'states'] as const,
  network: () => [...mlQueryKeys.all, 'network'] as const,
  networkGraph: (filters?: any) => [...mlQueryKeys.network(), 'graph', filters] as const,
  networkMetrics: () => [...mlQueryKeys.network(), 'metrics'] as const,
  networkAnalysis: () => [...mlQueryKeys.network(), 'analysis'] as const,
  systemicImportance: () => [...mlQueryKeys.network(), 'systemic-importance'] as const,
  simulations: () => [...mlQueryKeys.all, 'simulations'] as const,
  simulationsList: (filters?: any) => [...mlQueryKeys.simulations(), 'list', filters] as const,
  simulation: (id: string) => [...mlQueryKeys.simulations(), id] as const,
  simulationResults: (id: string) => [...mlQueryKeys.simulation(id), 'results'] as const,
  abm: () => [...mlQueryKeys.all, 'abm'] as const,
  abmList: () => [...mlQueryKeys.abm(), 'list'] as const,
  abmState: (simId: string) => [...mlQueryKeys.abm(), simId, 'state'] as const,
  abmHistory: (simId: string) => [...mlQueryKeys.abm(), simId, 'history'] as const,
  scenarios: () => [...mlQueryKeys.all, 'scenarios'] as const,
  scenariosList: (filters?: any) => [...mlQueryKeys.scenarios(), 'list', filters] as const,
  scenarioTemplates: () => [...mlQueryKeys.scenarios(), 'templates'] as const,
  scenario: (id: string) => [...mlQueryKeys.scenarios(), id] as const,
};

// Health Check
export function useHealthCheck() {
  return useQuery({
    queryKey: mlQueryKeys.health(),
    queryFn: () => mlApi.healthCheck(),
    staleTime: 30000, // 30 seconds
  });
}

// Institutions
export function useInstitutions(params?: {
  page?: number;
  limit?: number;
  type?: string;
  tier?: string;
  is_active?: boolean;
  search?: string;
}) {
  return useQuery({
    queryKey: mlQueryKeys.institutionsList(params),
    queryFn: () => mlApi.listInstitutions(params),
    staleTime: 60000, // 1 minute
  });
}

export function useInstitution(id: string, enabled = true) {
  return useQuery({
    queryKey: mlQueryKeys.institution(id),
    queryFn: () => mlApi.getInstitution(id),
    enabled: enabled && !!id,
    staleTime: 60000,
  });
}

export function useInstitutionStates(id: string, limit = 100) {
  return useQuery({
    queryKey: mlQueryKeys.institutionStates(id),
    queryFn: () => mlApi.getInstitutionStates(id, limit),
    enabled: !!id,
    staleTime: 120000, // 2 minutes
  });
}

// Network
export function useNetworkGraph(params?: {
  format?: string;
  include_weights?: boolean;
  min_exposure?: number;
}) {
  return useQuery({
    queryKey: mlQueryKeys.networkGraph(params),
    queryFn: () => mlApi.getNetworkGraph(params),
    staleTime: 120000, // 2 minutes
  });
}

export function useNetworkMetrics() {
  return useQuery({
    queryKey: mlQueryKeys.networkMetrics(),
    queryFn: () => mlApi.getNetworkMetrics(),
    staleTime: 120000,
  });
}

export function useNetworkAnalysis(minExposure?: number) {
  return useQuery({
    queryKey: mlQueryKeys.networkAnalysis(),
    queryFn: () => mlApi.analyzeNetwork(minExposure),
    staleTime: 300000, // 5 minutes - expensive operation
  });
}

export function useSystemicImportance(limit = 20) {
  return useQuery({
    queryKey: mlQueryKeys.systemicImportance(),
    queryFn: () => mlApi.getSystemicImportance(limit),
    staleTime: 120000,
  });
}

// Network Mutations
export function useContagionPaths() {
  return useMutation({
    mutationFn: ({
      sourceId,
      threshold,
      maxLength,
    }: {
      sourceId: string;
      threshold?: number;
      maxLength?: number;
    }) => mlApi.findContagionPaths(sourceId, threshold, maxLength),
  });
}

export function useCascadeSimulation() {
  return useMutation({
    mutationFn: ({
      shockedInstitutions,
      maxRounds,
    }: {
      shockedInstitutions: string[];
      maxRounds?: number;
    }) => mlApi.simulateCascade(shockedInstitutions, maxRounds),
  });
}

// Simulations
export function useSimulations(params?: {
  page?: number;
  limit?: number;
  status?: string;
}) {
  return useQuery({
    queryKey: mlQueryKeys.simulationsList(params),
    queryFn: () => mlApi.listSimulations(params),
    staleTime: 30000, // 30 seconds
  });
}

export function useSimulation(id: string, enabled = true) {
  return useQuery({
    queryKey: mlQueryKeys.simulation(id),
    queryFn: () => mlApi.getSimulation(id),
    enabled: enabled && !!id,
    refetchInterval: (data) => {
      // Auto-refresh if simulation is running
      return data?.status === 'running' || data?.status === 'queued' ? 5000 : false;
    },
  });
}

export function useSimulationResults(id: string, enabled = true) {
  return useQuery({
    queryKey: mlQueryKeys.simulationResults(id),
    queryFn: () => mlApi.getSimulationResults(id),
    enabled: enabled && !!id,
    staleTime: 300000, // 5 minutes - results don't change
  });
}

export function useCreateSimulation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      description?: string;
      total_timesteps?: number;
      scenario_id?: string;
      parameters?: Record<string, any>;
    }) => mlApi.createSimulation(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.simulationsList() });
    },
  });
}

export function useStartSimulation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => mlApi.startSimulation(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.simulation(id) });
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.simulationsList() });
    },
  });
}

export function useCancelSimulation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => mlApi.cancelSimulation(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.simulation(id) });
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.simulationsList() });
    },
  });
}

// Agent-Based Model
export function useABMSimulations() {
  return useQuery({
    queryKey: mlQueryKeys.abmList(),
    queryFn: () => mlApi.listABMSimulations(),
    staleTime: 30000,
  });
}

export function useABMState(simId: string, enabled = true) {
  return useQuery({
    queryKey: mlQueryKeys.abmState(simId),
    queryFn: () => mlApi.getABMState(simId),
    enabled: enabled && !!simId,
    staleTime: 5000, // 5 seconds - frequently changing
  });
}

export function useABMHistory(simId: string, limit?: number) {
  return useQuery({
    queryKey: mlQueryKeys.abmHistory(simId),
    queryFn: () => mlApi.getABMHistory(simId, limit),
    enabled: !!simId,
    staleTime: 30000,
  });
}

export function useInitializeABM() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      max_timesteps?: number;
      enable_shocks?: boolean;
      shock_probability?: number;
      random_seed?: number;
      use_real_data?: boolean;
      data_source?: string;
    }) => mlApi.initializeABM(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmList() });
    },
  });
}

export function useRunABM() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ simId, steps }: { simId: string; steps?: number }) =>
      mlApi.runABM(simId, steps),
    onSuccess: (_, { simId }) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmState(simId) });
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmHistory(simId) });
    },
  });
}

export function useApplyShock() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      simId,
      data,
    }: {
      simId: string;
      data: {
        shock_type: 'sector_crisis' | 'liquidity_squeeze' | 'interest_rate_shock' | 'asset_price_crash';
        target?: string;
        magnitude?: number;
      };
    }) => mlApi.applyShock(simId, data),
    onSuccess: (_, { simId }) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmState(simId) });
    },
  });
}

export function useUpdateCCPPolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      simId,
      data,
    }: {
      simId: string;
      data: {
        ccp_id?: string;
        rule_name: string;
        condition: string;
        action: string;
      };
    }) => mlApi.updateCCPPolicy(simId, data),
    onSuccess: (_, { simId }) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmState(simId) });
    },
  });
}

export function useResetABM() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (simId: string) => mlApi.resetABM(simId),
    onSuccess: (_, simId) => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmState(simId) });
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmHistory(simId) });
    },
  });
}

export function useDeleteABM() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (simId: string) => mlApi.deleteABM(simId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.abmList() });
    },
  });
}

export function useExportNetwork() {
  return useMutation({
    mutationFn: ({ simId, format }: { simId: string; format?: 'gexf' | 'graphml' | 'json' }) =>
      mlApi.exportNetwork(simId, format),
  });
}

// Scenarios
export function useScenarios(params?: {
  page?: number;
  limit?: number;
  category?: string;
  is_template?: boolean;
}) {
  return useQuery({
    queryKey: mlQueryKeys.scenariosList(params),
    queryFn: () => mlApi.listScenarios(params),
    staleTime: 60000,
  });
}

export function useScenarioTemplates() {
  return useQuery({
    queryKey: mlQueryKeys.scenarioTemplates(),
    queryFn: () => mlApi.getScenarioTemplates(),
    staleTime: 300000, // 5 minutes - templates rarely change
  });
}

export function useScenario(id: string, enabled = true) {
  return useQuery({
    queryKey: mlQueryKeys.scenario(id),
    queryFn: () => mlApi.getScenario(id),
    enabled: enabled && !!id,
    staleTime: 60000,
  });
}

export function useCreateScenario() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
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
    }) => mlApi.createScenario(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mlQueryKeys.scenariosList() });
    },
  });
}
