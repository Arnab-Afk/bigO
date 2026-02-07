# ML Dashboard API Integration Guide

## Overview

The RUDRA API at **api.rudranet.xyz** is fully integrated with **live Agent-Based Model (ABM) simulations**! This guide explains how to use the endpoints in your frontend application.

## 🎯 Current Integration Status

✅ **Health Check** - Working  
✅ **ABM Simulations** - Fully functional with real data  
✅ **Real-time State** - Live metrics updating  
✅ **Stress Testing** - All shock types working  
⚠️ **Static Institutions** - Empty (needs data seeding)  
⚠️ **Network Metrics** - Empty (needs data seeding)

**Dashboard Focus**: Agent-Based Model simulations with live data

## 🚀 Quick Demo

The dashboard at `/ml-dashboard` now shows:

1. **Real-Time Metrics**:
   - Total agents (banks, CCPs, sectors, regulators)
   - Network density and connections
   - System liquidity percentage
   - Total defaults and survival rate

2. **Live Simulation Control**:
   - Switch between active simulations
   - Run simulations step-by-step or in batches
   - Real-time metric updates

3. **System Health Monitoring**:
   - System NPA (Non-Performing Assets)
   - Average CRAR (Capital Risk-Adequacy Ratio)
   - Market volatility
   - Base repo rate
   - Average NPA across banks

4. **Stress Testing**:
   - Liquidity squeeze (-30%)
   - Interest rate shocks (+50%)
   - Asset price crashes (-40%)
   - Sector-specific crises (Real Estate -50%)

5. **Simulation Management**:
   - Create new simulations
   - View all active simulations
   - Monitor timesteps and network stats

All data is **live** from the API!

## Quick Start

### 1. Environment Setup

Add to your `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://api.rudranet.xyz
```

### 2. Using the API Client

```typescript
import { mlApi } from '@/lib/api/ml-api-client';

// Get all institutions
const institutions = await mlApi.listInstitutions({ limit: 50 });

// Get network graph
const networkGraph = await mlApi.getNetworkGraph();

// Run a simulation
const sim = await mlApi.createSimulation({
  name: 'Stress Test 2026',
  total_timesteps: 100,
});
```

### 3. Using React Query Hooks (Recommended)

```typescript
import { useInstitutions, useNetworkMetrics } from '@/hooks/use-ml-data';

function MyComponent() {
  const { data: institutions, isLoading } = useInstitutions({ limit: 50 });
  const { data: metrics } = useNetworkMetrics();

  if (isLoading) return <div>Loading...</div>;

  return <div>{institutions?.total} institutions</div>;
}
```

## Available Endpoints

### 📊 Institutions (Banks)

#### List all institutions
```typescript
const { data } = useInstitutions({
  page: 1,
  limit: 20,
  type: 'bank',           // bank, ccp, exchange, etc.
  tier: 'tier_1',         // g_sib, d_sib, tier_1, tier_2, tier_3
  is_active: true,
  search: 'HDFC'          // Search by name or external_id
});
```

**Response:**
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "HDFC Bank",
      "type": "bank",
      "tier": "tier_1",
      "current_state": {
        "capital_ratio": "15.5",
        "liquidity_buffer": "0.8",
        "stress_level": "0.3",
        "default_probability": "0.05"
      }
    }
  ],
  "total": 72,
  "page": 1,
  "pages": 4
}
```

#### Get single institution
```typescript
const { data: bank } = useInstitution('institution-uuid');
```

#### Get historical states
```typescript
const { data: history } = useInstitutionStates('institution-uuid', 100);
```

---

### 🕸️ Network Analysis

#### Get network graph
```typescript
const { data: graph } = useNetworkGraph({
  format: 'adjacency',
  include_weights: true,
  min_exposure: 1000000  // Filter small exposures
});
```

**Response:**
```json
{
  "nodes": [
    {
      "id": "uuid",
      "label": "HDFC Bank",
      "type": "bank",
      "tier": "tier_1"
    }
  ],
  "edges": [
    {
      "source": "uuid1",
      "target": "uuid2",
      "weight": 1500000,
      "exposure_type": "interbank_lending"
    }
  ]
}
```

#### Get network metrics
```typescript
const { data: metrics } = useNetworkMetrics();
```

**Response:**
```json
{
  "total_nodes": 72,
  "total_edges": 2156,
  "density": 0.423,
  "average_degree": 29.9,
  "clustering_coefficient": 0.65,
  "spectral_radius": 1.23,    // Amplification risk
  "fiedler_value": 0.08       // Fragmentation risk
}
```

#### Advanced network analysis
```typescript
const { data: analysis } = useNetworkAnalysis();
```

**Response includes:**
- **Centrality metrics**: PageRank, betweenness, eigenvector, degree
- **Systemic risk indicators**: spectral radius, Fiedler value, contagion index
- **Community detection**: modularity, community assignments

#### Systemic importance ranking
```typescript
const { data: rankings } = useSystemicImportance(20);
```

**Response:**
```json
{
  "institutions": [
    {
      "id": "uuid",
      "name": "State Bank of India",
      "score": 0.95,
      "tier": "g_sib",
      "total_exposure": "5000000000"
    }
  ]
}
```

#### Find contagion paths (Mutation)
```typescript
const { mutate: findPaths } = useContagionPaths();

findPaths({
  sourceId: 'bank-uuid',
  threshold: 0.3,     // Minimum contagion probability
  maxLength: 5        // Max path length
});
```

#### Simulate cascade (Mutation)
```typescript
const { mutate: simulateCascade } = useCascadeSimulation();

simulateCascade({
  shockedInstitutions: ['uuid1', 'uuid2'],
  maxRounds: 10
});
```

---

### 🎮 Simulations

#### List simulations
```typescript
const { data: sims } = useSimulations({
  page: 1,
  limit: 10,
  status: 'completed'  // pending, queued, running, completed, failed
});
```

#### Create simulation
```typescript
const { mutate: createSim } = useCreateSimulation();

createSim({
  name: 'Q1 2026 Stress Test',
  description: 'Testing liquidity shock',
  total_timesteps: 100,
  scenario_id: 'scenario-uuid',  // Optional
  parameters: {
    shock_magnitude: 0.5
  }
});
```

#### Start simulation
```typescript
const { mutate: startSim } = useStartSimulation();
startSim('simulation-uuid');
```

#### Get simulation (Auto-refreshes when running)
```typescript
const { data: sim } = useSimulation('simulation-uuid');
// Auto-refreshes every 5s if status is 'running' or 'queued'
```

#### Get simulation results
```typescript
const { data: results } = useSimulationResults('simulation-uuid');
```

**Response:**
```json
{
  "total_defaults": 5,
  "max_cascade_depth": 3,
  "survival_rate": "0.93",
  "final_systemic_stress": "0.67",
  "total_system_loss": "2500000000",
  "time_to_first_default": 12,
  "metrics_data": { ... },
  "timeline_data": { ... }
}
```

---

### 🤖 Agent-Based Model (Real-time)

#### Initialize ABM simulation
```typescript
const { mutate: initABM } = useInitializeABM();

initABM({
  name: 'Real-time ABM Test',
  max_timesteps: 100,
  enable_shocks: true,
  shock_probability: 0.1,
  use_real_data: true,
  random_seed: 42
});
```

**Response:**
```json
{
  "simulation_id": "sim_12345",
  "name": "Real-time ABM Test",
  "config": { ... },
  "network_stats": {
    "num_nodes": 72,
    "num_edges": 2156,
    "density": 0.423
  },
  "initial_state": { ... }
}
```

#### Run ABM simulation
```typescript
const { mutate: runABM } = useRunABM();

runABM({
  simId: 'sim_12345',
  steps: 10  // Number of timesteps to run
});
```

#### Get ABM state (Auto-refreshes)
```typescript
const { data: state } = useABMState('sim_12345');
```

**Response:**
```json
{
  "simulation_id": "sim_12345",
  "timestep": 45,
  "global_metrics": {
    "system_stress": 0.65,
    "total_defaults": 3,
    "liquidity_ratio": 0.72
  },
  "agent_states": {
    "bank_1": {
      "capital_ratio": 12.5,
      "is_defaulted": false,
      "stress_level": 0.4
    }
  },
  "network_state": { ... }
}
```

#### Apply shock (Stress Testing)
```typescript
const { mutate: applyShock } = useApplyShock();

applyShock({
  simId: 'sim_12345',
  data: {
    shock_type: 'liquidity_squeeze',  // or sector_crisis, interest_rate_shock, asset_price_crash
    magnitude: -0.3,  // -30% shock
    target: 'bank_uuid'  // Optional for sector_crisis
  }
});
```

**Shock Types:**
- `sector_crisis` - Crash a specific sector's economic health
- `liquidity_squeeze` - Reduce global liquidity
- `interest_rate_shock` - Spike interest rates
- `asset_price_crash` - Global asset price decline

#### Update CCP policy
```typescript
const { mutate: updatePolicy } = useUpdateCCPPolicy();

updatePolicy({
  simId: 'sim_12345',
  data: {
    ccp_id: 'CCP_MAIN',
    rule_name: 'Emergency Haircut',
    condition: 'system_npa > 8.0',  // Python expression
    action: 'self.haircut_rate += 0.05'  // Python action
  }
});
```

⚠️ **Warning**: Policy rules use `eval()` and are for demo only. Not production-safe!

#### Get ABM history
```typescript
const { data: history } = useABMHistory('sim_12345', 100);
```

#### Reset ABM
```typescript
const { mutate: resetABM } = useResetABM();
resetABM('sim_12345');
```

#### Export network
```typescript
const { mutate: exportNetwork } = useExportNetwork();

exportNetwork({
  simId: 'sim_12345',
  format: 'json'  // or 'gexf', 'graphml'
});
```

---

### 📋 Scenarios

#### List scenarios
```typescript
const { data: scenarios } = useScenarios({
  category: 'stress_test',
  is_template: false
});
```

#### Get template scenarios
```typescript
const { data: templates } = useScenarioTemplates();
```

#### Create scenario
```typescript
const { mutate: createScenario } = useCreateScenario();

createScenario({
  name: 'Liquidity Crisis 2026',
  description: 'Multi-bank liquidity shock',
  category: 'stress_test',
  num_timesteps: 100,
  shocks: [
    {
      name: 'Initial Liquidity Shock',
      shock_type: 'liquidity_freeze',
      magnitude: 0.7,
      duration: 5,
      trigger_timestep: 10
    }
  ]
});
```

**Shock Types:**
- `institution_default`
- `liquidity_freeze`
- `market_volatility`
- `margin_call`
- `credit_downgrade`
- `operational_failure`
- `regulatory_intervention`
- `interest_rate_shock`
- `fx_shock`
- `cyber_attack`

---

## Complete Example: Building a Dashboard

```typescript
'use client';

import {
  useInstitutions,
  useNetworkMetrics,
  useSystemicImportance,
  useSimulations,
} from '@/hooks/use-ml-data';

export default function Dashboard() {
  // Fetch data
  const { data: institutions } = useInstitutions({ is_active: true });
  const { data: metrics } = useNetworkMetrics();
  const { data: rankings } = useSystemicImportance(10);
  const { data: sims } = useSimulations({ limit: 5 });

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total Banks"
          value={institutions?.total}
          icon={<Building2 />}
        />
        <MetricCard
          title="Network Density"
          value={metrics?.density.toFixed(3)}
          icon={<Network />}
        />
        <MetricCard
          title="High Risk Banks"
          value={calculateHighRisk(institutions?.items)}
          icon={<AlertTriangle />}
        />
        <MetricCard
          title="Active Sims"
          value={sims?.items.filter(s => s.status === 'running').length}
          icon={<Activity />}
        />
      </div>

      {/* Systemic Importance */}
      <Card>
        <CardHeader>
          <CardTitle>Systemically Important Institutions</CardTitle>
        </CardHeader>
        <CardContent>
          {rankings?.institutions.map((inst, idx) => (
            <div key={inst.id} className="flex justify-between py-2">
              <span>#{idx + 1}. {inst.name}</span>
              <Badge>{inst.score.toFixed(2)}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
```

---

## Error Handling

All hooks include error handling:

```typescript
const { data, error, isLoading, isError } = useInstitutions();

if (isError) {
  return <Alert>Error: {error.message}</Alert>;
}

if (isLoading) {
  return <Skeleton />;
}

return <div>{data?.items.map(...)}</div>;
```

---

## Caching Strategy

React Query automatically caches data with these stale times:

| Data Type | Stale Time | Auto-Refresh |
|-----------|------------|--------------|
| Health Check | 30s | No |
| Institutions | 1m | No |
| Network Graph | 2m | No |
| Network Metrics | 2m | No |
| Simulations | 30s | If running |
| ABM State | 5s | No |
| Results | 5m | No |

---

## Best Practices

1. **Use React Query hooks** instead of direct API calls
2. **Enable queries conditionally** when data depends on other data
3. **Use mutations** for write operations (create, update, delete)
4. **Handle loading states** with skeleton components
5. **Show error states** with alert components
6. **Invalidate queries** after mutations to refresh data

---

## API Rate Limits

The API doesn't specify rate limits in the docs. Monitor response headers for rate limit info.

---

## Next Steps

1. ✅ API client created: `lib/api/ml-api-client.ts`
2. ✅ React Query hooks: `hooks/use-ml-data.ts`
3. ✅ Example dashboard: `app/(main)/ml-dashboard/page.tsx`

**To implement:**
- Network visualization component (D3.js or Cytoscape)
- Bank detail page with time-series charts
- Simulation runner with real-time progress
- Stress testing interface
- Policy recommendations display

---

## Support

- API Docs: https://api.rudranet.xyz/docs
- OpenAPI Schema: https://api.rudranet.xyz/openapi.json
