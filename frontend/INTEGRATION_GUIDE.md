# CCP ML Dashboard Integration

Complete integration between Next.js frontend and FastAPI backend for Central Counterparty (CCP) risk analysis and network simulation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │ ML Dashboard │──│ React Query │──│ CCP ML API Client    │   │
│  │    Page      │  │   Hooks     │  │ (ccp-ml-client.ts)   │   │
│  └─────────────┘  └─────────────┘  └──────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ HTTPS
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Backend API (FastAPI)                          │
│                    api.rudranet.xyz                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Network    │  │   Spectral   │  │  Real-time Sim      │  │
│  │   Builder    │  │   Analyzer   │  │  Engine             │  │
│  ├──────────────┤  ├──────────────┤  ├─────────────────────┤  │
│  │ CCP Risk     │  │   Graph      │  │  WebSocket          │  │
│  │ Model        │  │   Generator  │  │  Broadcasting       │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                                  │
│  Data: 72 RBI Banks (Real Reserve Bank of India Data)          │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### ✅ Implemented

#### Core Analytics
- **Network Analysis**: Banking interdependency network with centrality metrics
- **Spectral Analysis**: Systemic risk metrics (spectral radius, Fiedler value, contagion index)
- **Risk Scoring**: Individual bank risk assessments with stress levels
- **CCP Margins**: Margin requirements based on network importance
- **Stress Testing**: Capital, liquidity, and market shocks

#### Real-time Simulation (NEW!)
- **Progressive Execution**: Timestep-based simulation with live state tracking
- **Contagion Propagation**: Network-based stress transmission
- **Shock Application**: Apply shocks during simulation runtime
- **Live Metrics**: Real-time defaults, stress levels, capital ratios
- **History Tracking**: Full simulation history with replay capability

#### Visualization
- **Interactive Graphs**: Plotly-based network, risk, and time series visualizations
- **Static Exports**: Matplotlib PNG exports for reports
- **Real-time Updates**: WebSocket-based live data streaming

### 🎯 Dashboard Components

#### 1. Header Bar
- API connection status (api.rudranet.xyz)
- Real-time simulation status badge
- Quick actions (Run Simulation, Reload Data)

#### 2. Key Metrics Cards
- **Total Banks**: 72 RBI banks loaded
- **Network Edges**: Interdependency connections
- **Spectral Radius**: System amplification indicator (>1 = unstable)
- **High Risk Banks**: Count of banks with stress > 0.7

#### 3. Risk Distribution
- Visual histogram showing Low/Medium/High risk banks
- Color-coded (Green/Yellow/Red)
- Percentage breakdown

#### 4. Spectral Analysis
- Spectral Radius (ρ): System stability measure
- Fiedler Value (λ₂): Network connectivity
- Contagion Index: Systemic risk propagation
- Spectral Gap: Separation between eigenvalues

#### 5. Stress Testing Panel
- **Capital Shock**: -20% capital reduction
- **Liquidity Squeeze**: -30% liquidity buffer
- **Market Shock**: -40% market value crash

#### 6. Real-time Simulation Engine (NEW!)
- **Initialize**: Set up 100-timestep simulation
- **Run 10 Steps**: Execute multiple timesteps
- **Single Step**: Step-by-step execution
- **Apply Shock**: Inject shocks during runtime (capital/liquidity/stress)
- **Live Metrics**: Defaults, avg stress, avg capital, progress bar
- **Reset**: Reinitialize simulation

#### 7. Margin Requirements Table
- Top 10 banks by margin requirements
- Base margin + network addon
- Systemic importance explanations

## API Endpoints

### Core Endpoints
```
GET  /                      - Health check
GET  /api/status            - Simulation status
GET  /api/network           - Network data with metrics
GET  /api/risk/scores       - Risk scores for all banks
GET  /api/spectral          - Spectral analysis results
GET  /api/margins           - Margin requirements
POST /api/simulate          - Run full simulation
POST /api/stress-test       - Run stress test
POST /api/reinitialize      - Reload data
```

### Real-time Simulation (NEW!)
```
POST /api/realtime/init     - Initialize simulation
POST /api/realtime/step     - Execute timesteps
GET  /api/realtime/status   - Get simulation status
GET  /api/realtime/history  - Get full history
POST /api/realtime/stop     - Stop simulation
```

### Graph Generation (NEW!)
```
POST /api/graphs/generate   - Generate visualization
GET  /api/graphs/available  - List available graphs
```

### WebSocket (NEW!)
```
WS   /ws/simulation         - Live updates stream
```

## Configuration

### Environment Variables

**Frontend** (`frontend/.env.local`):
```env
NEXT_PUBLIC_API_URL=https://api.rudranet.xyz
```

**Backend** (`backend/ccp_ml/api.py`):
- Host: 0.0.0.0
- Port: 8000
- CORS: Enabled for all origins

## Data Flow

### 1. Static Simulation
```
User clicks "Run Simulation"
    ↓
Frontend: runSimulation()
    ↓
Backend: POST /api/simulate
    ↓
- Network Builder creates graph
- Spectral Analyzer computes metrics
- CCP Engine calculates margins
    ↓
Frontend: Updates all cards with results
```

### 2. Real-time Simulation
```
User clicks "Initialize Simulation"
    ↓
Frontend: initRealtime({ max_timesteps: 100 })
    ↓
Backend: POST /api/realtime/init
    ↓
Backend: Realtime simulation engine initialized
    ↓
User clicks "Run 10 Steps"
    ↓
Frontend: stepRealtime({ n_steps: 10 })
    ↓
Backend: Execute 10 timesteps
    ├─ Apply shocks (if provided)
    ├─ Propagate contagion through network
    ├─ Update bank states
    └─ Capture state snapshots
    ↓
Backend: Broadcast updates via WebSocket (optional)
    ↓
Frontend: Display live metrics and progress bar
```

### 3. Stress Testing
```
User clicks "Liquidity Squeeze (-30%)"
    ↓
Frontend: stressTest({ shock_type: 'liquidity', shock_magnitude: 0.3 })
    ↓
Backend: POST /api/stress-test
    ↓
- Apply shock to features
- Re-run CCP analysis
- Compute impact (fund increase, new high-risk banks)
    ↓
Frontend: Display impact results
```

## Tech Stack

### Frontend
- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **State Management**: @tanstack/react-query v5.90.20
- **UI Components**: Shadcn UI
- **Styling**: TailwindCSS v4
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.10+
- **Data**: Pandas, NumPy
- **Network**: NetworkX
- **Visualization**: Plotly, Matplotlib, Seaborn
- **ML**: Scikit-learn
- **WebSocket**: FastAPI WebSockets

### Data Sources
- **RBI Banks**: 72 banks from Reserve Bank of India
- **CRAR Data**: Capital Adequacy Ratios
- **NPA Data**: Non-Performing Assets
- **Sector Exposure**: Sensitive sector data
- **Maturity Profiles**: Liquidity profiles
- **Repo Rates**: Interest rate data

## React Query Configuration

### Query Keys Structure
```typescript
ccpMLQueryKeys = {
  all: ['ccp-ml'],
  health: ['ccp-ml', 'health'],
  status: ['ccp-ml', 'status'],
  network: ['ccp-ml', 'network'],
  risk: ['ccp-ml', 'risk'],
  riskScores: ['ccp-ml', 'risk', 'scores'],
  spectral: ['ccp-ml', 'spectral'],
  margins: ['ccp-ml', 'margins'],
  realtime: ['ccp-ml', 'realtime'],
  realtimeStatus: ['ccp-ml', 'realtime', 'status'],
  realtimeHistory: ['ccp-ml', 'realtime', 'history'],
  graphs: ['ccp-ml', 'graphs'],
}
```

### Refetch Intervals
- **Status**: 10 seconds (auto-refresh)
- **Realtime Status**: 5 seconds (live tracking)
- **Network/Risk Data**: 60 seconds
- **Spectral Analysis**: 120 seconds

### Cache Strategy
- **Stale Time**: Data considered fresh for specified duration
- **Invalidation**: Mutations invalidate related queries
- **Retry**: 1 retry on failure (default)

## Type Definitions

### Key Types

```typescript
// Simulation Step (Real-time)
interface SimulationStep {
  timestep: number;
  timestamp: string;
  bank_states: Array<{
    bank_name: string;
    capital_ratio: number;
    stress_level: number;
    defaulted: number;
  }>;
  network_metrics: {
    num_nodes: number;
    num_edges: number;
    density: number;
  };
  spectral_metrics: {
    spectral_radius: number;
    fiedler_value: number;
    contagion_index: number;
  };
  risk_distribution: {
    low: number;
    medium: number;
    high: number;
  };
  default_count: number;
  total_stress: number;
  average_capital_ratio: number;
}

// Network Data
interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metrics?: any[];
}

// Risk Score
interface RiskScore {
  bank_name: string;
  default_probability?: number;
  stress_level?: number;
  capital_ratio?: number;
}

// Spectral Analysis
interface SpectralAnalysis {
  spectral_radius: number;
  fiedler_value: number;
  spectral_gap: number;
  amplification_risk: string;
  fragmentation_risk: string;
  contagion_index: number;
}
```

## Usage Examples

### 1. Fetching Status
```typescript
const { data: status } = useCCPStatus();
console.log(`Loaded ${status?.n_banks} banks`);
```

### 2. Running Simulation
```typescript
const { mutate: runSimulation } = useCCPRunSimulation();
runSimulation(undefined); // Use default config
```

### 3. Real-time Simulation
```typescript
const { mutate: initRealtime } = useCCPRealtimeInit();
const { mutate: stepRealtime } = useCCPRealtimeStep();

// Initialize
initRealtime({ max_timesteps: 100 });

// Execute steps
stepRealtime({ n_steps: 10 });

// With shock
stepRealtime({
  n_steps: 5,
  shock_config: {
    type: 'liquidity',
    magnitude: 0.3
  }
});
```

### 4. Stress Testing
```typescript
const { mutate: stressTest } = useCCPStressTest();

stressTest({
  shock_type: 'capital',
  shock_magnitude: 0.2
});
```

## Performance Metrics

- **Initial Load**: ~2-3 seconds (loads 72 banks with features)
- **Network Build**: ~500ms-1s
- **Spectral Analysis**: ~300-500ms
- **Single Simulation Step**: ~100ms
- **10 Simulation Steps**: ~1 second
- **Stress Test**: ~800ms-1.5s
- **Graph Generation**: 1-3 seconds (depends on type)

## Development

### Running Locally

**Frontend**:
```bash
cd frontend
npm install
npm run dev
# Access: http://localhost:3000/ml-dashboard
```

**Backend** (if testing locally):
```bash
cd backend/ccp_ml
pip install -r requirements.txt
uvicorn api:app --reload
# Access: http://localhost:8000
```

### Building for Production

**Frontend**:
```bash
npm run build
npm run start
```

## Deployment

### Frontend
- Platform: Vercel/Netlify
- Environment: Set `NEXT_PUBLIC_API_URL=https://api.rudranet.xyz`
- Build: `npm run build`

### Backend
- Platform: api.rudranet.xyz
- Framework: FastAPI with Uvicorn
- Workers: 4 (production)
- CORS: Enabled for frontend domain

## Troubleshooting

### API Connection Issues
1. Check `.env.local` has correct API URL
2. Verify api.rudranet.xyz is accessible
3. Check browser console for CORS errors
4. Inspect Network tab for failed requests

### Real-time Simulation Not Working
1. Ensure backend has realtime module installed
2. Check if simulation is initialized
3. Verify timestep limits not exceeded
4. Check browser console for errors

### Data Not Loading
1. Check API status at `/api/status`
2. Verify backend has loaded data files
3. Clear browser cache and reload
4. Check if API server is responding

### WebSocket Issues
1. Ensure backend supports WebSocket protocol
2. Check firewall/proxy settings
3. Verify WSS protocol for HTTPS sites
4. Check browser WebSocket support

## Future Enhancements

- [ ] Network visualization D3.js component
- [ ] Time series charts for simulation history
- [ ] Bank detail pages with historical data
- [ ] Export simulation results to CSV/Excel
- [ ] Comparison view for multiple stress scenarios
- [ ] Custom shock designer
- [ ] Real-time graph streaming during simulation
- [ ] Animated network contagion visualization
- [ ] PDF report generation
- [ ] Multi-simulation comparison dashboard

## Documentation

- **Backend API**: https://api.rudranet.xyz/docs (Swagger UI)
- **Backend README**: `backend/ccp_ml/README.md`
- **Frontend Components**: `frontend/src/components/`
- **API Client**: `frontend/src/lib/api/ccp-ml-client.ts`
- **Hooks**: `frontend/src/hooks/use-ccp-ml.ts`

## Support

For issues or questions:
1. Check API documentation at https://api.rudranet.xyz/docs
2. Review backend logs for detailed errors
3. Test endpoints with `backend/ccp_ml/test_api.py`
4. Check browser console for frontend errors

## Version

**Frontend**: v1.0.0 - Real-time simulation integration
**Backend**: v1.0.0 - Real-time simulation and graph generation

## License

See main project LICENSE file.
