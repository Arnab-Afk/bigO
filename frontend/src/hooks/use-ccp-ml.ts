/**
 * React Query hooks for CCP ML API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ccpMLApi } from '@/lib/api/ccp-ml-client';
import type {
  SimulationConfig,
  StressTestConfig,
} from '@/lib/api/ccp-ml-client';

// Query Keys
export const ccpMLQueryKeys = {
  all: ['ccp-ml'] as const,
  health: () => [...ccpMLQueryKeys.all, 'health'] as const,
  status: () => [...ccpMLQueryKeys.all, 'status'] as const,
  network: () => [...ccpMLQueryKeys.all, 'network'] as const,
  networkNodes: () => [...ccpMLQueryKeys.network(), 'nodes'] as const,
  networkEdges: () => [...ccpMLQueryKeys.network(), 'edges'] as const,
  risk: () => [...ccpMLQueryKeys.all, 'risk'] as const,
  riskScores: () => [...ccpMLQueryKeys.risk(), 'scores'] as const,
  bankRisk: (bankName: string) => [...ccpMLQueryKeys.risk(), 'bank', bankName] as const,
  spectral: () => [...ccpMLQueryKeys.all, 'spectral'] as const,
  margins: () => [...ccpMLQueryKeys.all, 'margins'] as const,
  defaultFund: () => [...ccpMLQueryKeys.all, 'default-fund'] as const,
};

// Health & Status
export function useCCPHealthCheck() {
  return useQuery({
    queryKey: ccpMLQueryKeys.health(),
    queryFn: () => ccpMLApi.healthCheck(),
    staleTime: 30000, // 30 seconds
  });
}

export function useCCPStatus() {
  return useQuery({
    queryKey: ccpMLQueryKeys.status(),
    queryFn: () => ccpMLApi.getStatus(),
    staleTime: 10000, // 10 seconds
    refetchInterval: 10000, // Auto-refresh every 10s
  });
}

// Network
export function useCCPNetwork() {
  return useQuery({
    queryKey: ccpMLQueryKeys.network(),
    queryFn: () => ccpMLApi.getNetwork(),
    staleTime: 60000, // 1 minute
  });
}

export function useCCPNetworkNodes() {
  return useQuery({
    queryKey: ccpMLQueryKeys.networkNodes(),
    queryFn: () => ccpMLApi.getNetworkNodes(),
    staleTime: 60000,
  });
}

export function useCCPNetworkEdges() {
  return useQuery({
    queryKey: ccpMLQueryKeys.networkEdges(),
    queryFn: () => ccpMLApi.getNetworkEdges(),
    staleTime: 60000,
  });
}

// Risk
export function useCCPRiskScores() {
  return useQuery({
    queryKey: ccpMLQueryKeys.riskScores(),
    queryFn: () => ccpMLApi.getRiskScores(),
    staleTime: 60000,
  });
}

export function useCCPBankRisk(bankName: string, enabled = true) {
  return useQuery({
    queryKey: ccpMLQueryKeys.bankRisk(bankName),
    queryFn: () => ccpMLApi.getBankRisk(bankName),
    enabled: enabled && !!bankName,
    staleTime: 60000,
  });
}

// Spectral Analysis
export function useCCPSpectralAnalysis() {
  return useQuery({
    queryKey: ccpMLQueryKeys.spectral(),
    queryFn: () => ccpMLApi.getSpectralAnalysis(),
    staleTime: 120000, // 2 minutes
  });
}

// Margins & Default Fund
export function useCCPMargins() {
  return useQuery({
    queryKey: ccpMLQueryKeys.margins(),
    queryFn: () => ccpMLApi.getMargins(),
    staleTime: 60000,
  });
}

export function useCCPDefaultFund() {
  return useQuery({
    queryKey: ccpMLQueryKeys.defaultFund(),
    queryFn: () => ccpMLApi.getDefaultFund(),
    staleTime: 60000,
  });
}

// Mutations
export function useCCPRunSimulation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (config?: SimulationConfig) => ccpMLApi.runSimulation(config),
    onSuccess: () => {
      // Invalidate all queries to refresh data
      queryClient.invalidateQueries({ queryKey: ccpMLQueryKeys.all });
    },
  });
}

export function useCCPStressTest() {
  return useMutation({
    mutationFn: (config: StressTestConfig) => ccpMLApi.runStressTest(config),
  });
}

export function useCCPReinitialize() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => ccpMLApi.reinitialize(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ccpMLQueryKeys.all });
    },
  });
}
