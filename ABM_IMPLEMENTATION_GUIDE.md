# Agent-Based Model (ABM) Implementation Guide

## Overview

This implementation transforms RUDRA from a static risk analysis system into a **dynamic Agent-Based Model** that simulates strategic interactions in financial networks over time.

### Key Concepts

**Conway's Game of Life for Finance:**
- **Survival:** Bank.Capital > Regulatory_Min
- **Death:** Bank.Capital ≤ 0
- **Neighbor Interaction:** Weighted edges transmit losses
- **Feedback Loops:** Defensive actions create cascading liquidity crunches

---

## Architecture

### 1. Core Components

```
backend/app/engine/
├── agents.py                    # Agent class hierarchy
├── simulation_engine.py         # Main simulation loop
├── initial_state_loader.py      # CSV data calibration
└── visualization.py             # Dashboard utilities
```

### 2. Agent Types

#### **BankAgent** (The Players)

**Policy Variables** (Adjustable):
- `credit_supply_limit`: Maximum lending capacity
- `lending_spread`: Interest rate premium
- `interbank_limit`: Max lending to other banks
- `risk_appetite`: Willingness to take risk [0, 1]

**State Variables** (Computed):
- `capital`: Tier-1 Capital (equity)
- `crar`: Capital Adequacy Ratio (%)
- `npa_ratio`: Non-Performing Assets (%)
- `liquidity`: Available cash

**Game Theory Strategy:**
1. **Panic Rule:** If `neighbor_defaults > 0` OR `perceived_stress > 0.5`
   - Switch to DEFENSIVE mode
   - Cut lending by 50%
   - Hoard liquidity
   - ⚠️ This is the **Prisoner's Dilemma**: Individually rational but collectively destructive

2. **Greed Rule:** If `perceived_stress < 0.2` AND `crar > min_crar + 3%`
   - Switch to AGGRESSIVE mode
   - Increase lending
   - Maximize yield

3. **Survival Rule:** If `crar < min_crar`
   - EMERGENCY DELEVERAGING
   - Recall loans
   - Likely triggers contagion

#### **SectorAgent** (The Borrowers)

Represents economic sectors (Real Estate, Commodities, etc.)

**State Variables:**
- `economic_health`: [0, 1] health index
- `debt_load`: Total debt to banks

**Behavior:**
- When `health < 0.7`, banks with exposure take proportional losses
- This increases bank NPAs and reduces CRAR
- Creates the channel for **exogenous shocks → endogenous contagion**

#### **CCPAgent** (The User/Regulator)

Central Counterparty managing the system

**Policy Variables:**
- `initial_margin_requirement`: Collateral percentage
- `haircut_rate`: Loss absorption rate

**Custom Rules:**
- User can inject policy rules via API
- Example: `IF system_npa > 8% THEN increase_haircuts_by 5%`

#### **RegulatorAgent** (Central Bank)

**Policy Variables:**
- `base_repo_rate`: Cost of money
- `min_crar_requirement`: Regulatory floor

**Behavior:**
- Counter-cyclical policy
- Lower rates during stress
- Raise rates during growth

---

## Simulation Loop

### Order of Operations (Per Timestep)

```
1. SHOCK (Optional)
   └─ Exogenous event (sector crisis, rate shock, etc.)

2. PERCEPTION (Parallel)
   └─ All agents observe neighbors and global state

3. DECISION (Parallel)
   └─ Agents execute game theory strategies

4. ACTION (Sequential)
   └─ Banks → Sectors → CCP → Regulator
   └─ Order matters for causal consistency

5. CONTAGION
   └─ Distribute losses from defaults
   └─ If Bank A fails:
      • All creditors take haircut (50% recovery)
      • Creditors' capital drops
      • May trigger secondary failures (CASCADE)

6. METRICS
   └─ Update global state (survival_rate, avg_crar, etc.)

7. RECORD
   └─ Save snapshot to history
```

---

## Data Integration

### CSV Files → Initial State

| CSV File | Usage |
|----------|-------|
| `bank_crar.csv` | Initial capital, CRAR for each bank |
| `bank_npa.csv` | Initial NPA ratios |
| `bank_sensitive_sector.csv` | **Creates edges:** Bank → Sector with exposure weights |
| `bank_maturity_profile.csv` | Calculates liquidity from short-term assets |
| `reverse_repo.csv` | Sets global `system_liquidity` |

### Loading Process

```python
from app.engine.initial_state_loader import load_ecosystem_from_data

ecosystem = load_ecosystem_from_data(
    "backend/ccp_ml/data",
    max_timesteps=100,
    enable_shocks=True,
    random_seed=42
)
```

**What Happens:**
1. Parse latest year from each CSV
2. Create `BankAgent` for each unique bank
3. Create `SectorAgent` for each sector column
4. Create edges: `(bank_id, sector_id, exposure_amount)`
5. Add interbank network (random preferential attachment)
6. Add CCP, Regulator

---

## API Endpoints

Base URL: `/abm`

### 1. Initialize Simulation

```http
POST /abm/initialize
```

**Request:**
```json
{
  "name": "Test Simulation",
  "max_timesteps": 100,
  "enable_shocks": true,
  "shock_probability": 0.1,
  "random_seed": 42,
  "use_real_data": true,
  "data_source": "backend/ccp_ml/data"
}
```

**Response:**
```json
{
  "simulation_id": "uuid",
  "network_stats": {
    "num_banks": 10,
    "num_sectors": 5,
    "num_edges": 67
  },
  "initial_state": { ... }
}
```

### 2. Step Simulation

```http
POST /abm/{sim_id}/step
```

**Request:**
```json
{
  "num_steps": 1
}
```

Advances the simulation by N timesteps.

### 3. Apply Shock

```http
POST /abm/{sim_id}/shock
```

**Request:**
```json
{
  "shock_type": "sector_crisis",
  "target": "SECTOR_REAL_ESTATE",
  "magnitude": -0.3
}
```

**Shock Types:**
- `sector_crisis`: Crash a sector
- `liquidity_squeeze`: Reduce global liquidity
- `interest_rate_shock`: Spike rates
- `asset_price_crash`: All banks lose capital

### 4. Set CCP Policy

```http
POST /abm/{sim_id}/policy
```

**Request:**
```json
{
  "ccp_id": "CCP_MAIN",
  "rule_name": "Emergency Haircut",
  "condition": "system_npa > 8.0",
  "action": "self.haircut_rate += 0.05"
}
```

⚠️ **WARNING:** Uses `eval()` - unsafe in production. Use a DSL or rules engine.

### 5. Get State

```http
GET /abm/{sim_id}/state
```

Returns current state without advancing time.

### 6. Get History

```http
GET /abm/{sim_id}/history?limit=50
```

Returns all (or last N) snapshots.

---

## Game Theory Mechanics

### The Prisoner's Dilemma in Banking

| Bank A \ Bank B | Lend Aggressively | Hoard Liquidity |
|----------------|-------------------|-----------------|
| **Lend Aggressively** | (Good, Good) | (Fail, Survive) |
| **Hoard Liquidity** | (Survive, Fail) | (Fair, Fair) |

**Nash Equilibrium:** Both hoard → Systemic liquidity crunch

**Implementation:**
```python
# In BankAgent.decide()
if self.perceived_systemic_stress > 0.5:
    self.mode = AgentMode.DEFENSIVE
    self.credit_supply_limit *= 0.7  # Cut lending
    self.interbank_limit *= 0.5      # Reduce interbank
```

### Contagion Formula

When Bank A defaults:
```python
for creditor in network.predecessors(bank_a):
    exposure = edge_weight(creditor, bank_a)
    loss = exposure * haircut_rate  # e.g., 0.5 (50% recovery)
    creditor.capital -= loss
    
    if creditor.capital <= 0:
        creditor.default()  # CASCADE
```

---

## Integration with Existing ML Models

### Using XGBoost for "Smart" Banks

Don't throw away your trained models. Inject them as agent "brains":

```python
# Load your trained XGBoost model
import joblib
model = joblib.load("ml_models/default_predictor/model.pkl")

# Inject into BankAgent
for bank in ecosystem.agents.values():
    if isinstance(bank, BankAgent):
        bank.set_predictor(model)

# Inside BankAgent.decide():
if self.default_predictor:
    features = extract_features(sector)
    pd_estimate = self.default_predictor.predict_proba(features)[0][1]
    
    if pd_estimate > 0.3:
        self.risk_appetite *= 0.5  # Reduce exposure
```

---

## Visualization

### D3.js Format

```python
from app.engine.visualization import NetworkVisualizer

snapshot = ecosystem.history[-1].to_dict()
d3_data = NetworkVisualizer.convert_to_d3(snapshot)

# Returns:
{
  "nodes": [
    {"id": "BANK_1", "group": "bank", "health": 0.8, "color": "#2ecc71"},
    ...
  ],
  "links": [
    {"source": "BANK_1", "target": "SECTOR_REAL_ESTATE", "value": 2500},
    ...
  ]
}
```

### Dashboard Data

```python
from app.engine.visualization import prepare_dashboard_data

history_dicts = [s.to_dict() for s in ecosystem.history]
dashboard = prepare_dashboard_data(history_dicts)

# Contains:
# - d3_network: Current network
# - survival_rate_ts: Time series
# - health_heatmap: Matrix for heatmap viz
# - critical_nodes: Alerts
# - cascade_tree: Contagion analysis
```

---

## Running Examples

### 1. Basic Simulation

```bash
cd backend
python examples/abm_example.py
```

This runs 6 complete examples:
1. Basic simulation with synthetic data
2. Load from real CSV data
3. Manual shock application
4. CCP policy intervention
5. Dashboard data generation
6. Time series analysis

### 2. Start API Server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Then access:
- Swagger UI: `http://localhost:8000/docs`
- API endpoints: `http://localhost:8000/abm/...`

---

## Frontend Integration

### React Component Example

```typescript
// Initialize simulation
const response = await fetch('/abm/initialize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    name: 'User Simulation',
    max_timesteps: 100,
    use_real_data: true
  })
});

const { simulation_id } = await response.json();

// Step simulation
const stepResponse = await fetch(`/abm/${simulation_id}/step`, {
  method: 'POST',
  body: JSON.stringify({ num_steps: 1 })
});

const { snapshots } = await stepResponse.json();
const latest = snapshots[0];

// Visualize with D3/Cytoscape
renderNetwork(latest.network_state);
```

### Recommended Libraries

**Network Visualization:**
- `react-force-graph` - 3D force-directed graphs
- `cytoscape` - Advanced graph layouts
- `d3.js` - Custom visualizations

**Time Series:**
- `recharts` - Simple React charts
- `plotly.js` - Interactive charts

---

## Performance Considerations

### Scalability

**Current Implementation:**
- In-memory agent store (OK for < 1000 agents)
- Single-threaded simulation loop

**For Production:**
- Use Redis for `ACTIVE_SIMULATIONS`
- Implement async step execution with Celery
- Add persistent storage (PostgreSQL) for history
- Use graph databases (Neo4j) for network queries

### Optimization Tips

```python
# Batch agent perception (currently parallel-safe)
# Vectorize calculations with NumPy
# Use NetworkX efficiently (avoid repeated traversals)
```

---

## Extending the Model

### Add New Agent Type

```python
from app.engine.agents import Agent, AgentType

class HedgeFundAgent(Agent):
    def __init__(self, agent_id: str, ...):
        super().__init__(agent_id, AgentType.HEDGE_FUND)
        # Add specific attributes
    
    def perceive(self, network, global_state):
        # Custom perception
        pass
    
    # Implement abstract methods...
```

### Add New Shock Type

```python
# In simulation_engine.py
class ShockType(Enum):
    CYBER_ATTACK = "cyber_attack"

# In FinancialEcosystem.apply_shock()
elif shock_type == ShockType.CYBER_ATTACK:
    # Implementation
    pass
```

---

## References

### Academic Foundation

This implementation is based on:
- **Agent-Based Computational Economics** (Tesfatsion & Judd)
- **Systemic Risk in Banking Networks** (Acemoglu et al., 2015)
- **Fire Sales and Contagion** (Cont & Wagalath, 2016)

### Related Work

- Bank of England's RAMSI (Risk Assessment Model for Systemic Institutions)
- European Systemic Risk Board ABM models
- IMF's Network-Based Stress Testing

---

## Troubleshooting

### Issue: "Simulation not found"

```python
# Check active simulations
GET /abm/list
```

Simulations are stored in-memory. They are lost on server restart.

### Issue: All banks default immediately

- Check initial CRAR values (should be > 9%)
- Reduce shock probability
- Disable shocks for testing

### Issue: No contagion observed

- Increase interbank exposures
- Lower bank capital buffers
- Apply stronger shocks

---

## Next Steps

1. **Frontend Dashboard:**
   - Real-time network visualization
   - Interactive policy controls
   - Time series dashboards

2. **Advanced Features:**
   - Multi-asset classes (derivatives, bonds)
   - Central bank interventions (TARP-style bailouts)
   - Market making agents
   - Reinforcement Learning for agent strategies

3. **Production Deployment:**
   - Redis for session management
   - Database persistence
   - Authentication/authorization
   - Rate limiting

---

**Questions?** Check `examples/abm_example.py` for complete working code.

**API Documentation:** Start server and visit `/docs`
