# 🎮 RUDRA Agent-Based Model - Quick Start

## What You Just Got

A complete **Agent-Based Model (ABM)** system for simulating financial network dynamics with game-theoretic agents that make strategic decisions in discrete time steps.

### Conway's Game of Life → Finance Edition

```
IF Bank.Capital > Regulatory_Min: SURVIVE
IF Bank.Capital ≤ 0: DEFAULT (trigger cascade)

Neighbors: Connected banks/sectors via exposure edges
Strategy: Nash Equilibrium → Defensive hoarding (Prisoner's Dilemma)
```

---

## 🚀 Immediate Testing

### Option 1: Run Example Scripts

```bash
cd backend
python examples/abm_example.py
```

This runs 6 complete demonstrations:
1. ✅ Basic synthetic simulation
2. ✅ Load from your RBI CSV data
3. ✅ Manual shock + contagion
4. ✅ CCP policy intervention
5. ✅ Dashboard data generation
6. ✅ Time series extraction

**Output:** JSON files ready for frontend visualization

### Option 2: Use the API

```bash
# Start server
cd backend
uvicorn app.main:app --reload --port 8000

# Visit Swagger UI
open http://localhost:8000/docs
```

Try these endpoints under **Agent-Based Model**:
- `POST /abm/initialize` - Create simulation
- `POST /abm/{sim_id}/step` - Advance time
- `POST /abm/{sim_id}/shock` - Apply crisis
- `GET /abm/{sim_id}/state` - View network

---

## 📂 New File Structure

```
backend/app/engine/
├── agents.py                 # BankAgent, CCPAgent, SectorAgent, RegulatorAgent
├── simulation_engine.py      # FinancialEcosystem (main simulation loop)
├── initial_state_loader.py   # Loads RBI CSV data → Agents
└── visualization.py          # D3/Cytoscape formatters

backend/app/api/v1/
└── abm_simulation.py         # FastAPI endpoints

backend/examples/
└── abm_example.py            # Complete working examples

ABM_IMPLEMENTATION_GUIDE.md   # 📖 Full documentation (READ THIS)
```

---

## 💡 Core Concepts

### Agents Have:

**Policy Variables** (You/AI control):
```python
bank.credit_supply_limit = 10000  # Max lending
bank.risk_appetite = 0.5          # [0, 1] risk tolerance
```

**State Variables** (Computed):
```python
bank.crar  # Capital Adequacy Ratio
bank.npa_ratio  # Non-Performing Assets
bank.liquidity  # Available cash
```

### The Game Theory

**Prisoner's Dilemma in Banking:**

| Your Bank → | Lend | Hoard |
|-------------|------|-------|
| **Other Banks Lend** | 💚 Growth | ⚠️ You survive, they fail |
| **Other Banks Hoard** | 💀 You fail | 🟡 Recession but stable |

**Nash Equilibrium:** Everyone hoards → Liquidity crisis

**Implementation:**
```python
if neighbor_defaults > 0:
    self.mode = "DEFENSIVE"
    self.credit_supply *= 0.5  # Cut lending 50%
```

### Contagion Cascade

```
1. Shock: Real Estate sector crashes (-30% health)
   ↓
2. Banks with RE exposure take losses
   ↓
3. Bank A's capital drops → CRAR < 9% → DEFAULT
   ↓
4. Bank B (creditor to Bank A) takes 50% haircut
   ↓
5. Bank B defaults → Cascade continues
```

---

## 🎯 Quick Use Cases

### 1. Test a Shock Scenario

```bash
curl -X POST http://localhost:8000/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{"name": "Crisis Test", "use_real_data": true}'

# Returns: {"simulation_id": "abc-123", ...}

curl -X POST http://localhost:8000/abm/abc-123/shock \
  -d '{"shock_type": "sector_crisis", "target": "SECTOR_REAL_ESTATE", "magnitude": -0.4}'

curl -X POST http://localhost:8000/abm/abc-123/step \
  -d '{"num_steps": 10}'

curl http://localhost:8000/abm/abc-123/state
```

### 2. Set CCP Policy Rule

```bash
curl -X POST http://localhost:8000/abm/abc-123/policy \
  -d '{
    "ccp_id": "CCP_MAIN",
    "rule_name": "Emergency Haircut",
    "condition": "system_npa > 8.0",
    "action": "self.haircut_rate += 0.05"
  }'
```

⚠️ **Security Note:** This uses `eval()` - FOR DEMO ONLY. Use a DSL in production.

---

## 📊 Visualization Integration

### For D3.js / React Force Graph

```javascript
const response = await fetch(`/abm/${simId}/state`);
const snapshot = await response.json();

// Convert to D3 format
const viz = await fetch(`/abm/${simId}/visualize?format=d3`);
const {nodes, links} = await viz.json();

// Render with react-force-graph
<ForceGraph2D graphData={{nodes, links}} />
```

### For Cytoscape.js

```javascript
const response = await fetch(`/abm/${simId}/state`);
const snapshot = await response.json();

// In backend: NetworkVisualizer.convert_to_cytoscape(snapshot)
const elements = convertToCytoscape(snapshot);

cy.add(elements);
```

---

## 🔧 Customization

### Add Your Own Agent Type

```python
# backend/app/engine/agents.py

class HedgeFundAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.HEDGE_FUND)
        self.leverage = 5.0
        self.volatility_target = 0.15
    
    def perceive(self, network, global_state):
        market_vol = global_state['market_volatility']
        if market_vol > self.volatility_target:
            self.mode = AgentMode.DEFENSIVE
    
    def decide(self):
        if self.mode == AgentMode.DEFENSIVE:
            return {'action': 'DELEVERAGE'}
        return {'action': 'INCREASE_POSITIONS'}
    
    def act(self, network):
        # Your custom logic
        return []
    
    def compute_health(self):
        return 1.0 - (self.leverage / 10.0)
```

### Add New Shock Type

```python
# backend/app/engine/simulation_engine.py

class ShockType(Enum):
    CYBER_ATTACK = "cyber_attack"

# In FinancialEcosystem.apply_shock():
elif shock_type == ShockType.CYBER_ATTACK:
    # Randomly disable a bank's IT systems
    random_bank = random.choice(list(self.agents.values()))
    random_bank.liquidity *= 0.5  # Freeze half their liquidity
```

---

## 🧪 Data Integration

### Your CSV Files → Simulation

The system uses your existing data:

| CSV | What It Does |
|-----|--------------|
| `bank_crar.csv` | Sets initial Bank capital, CRAR |
| `bank_npa.csv` | Sets initial NPA ratio |
| `bank_sensitive_sector.csv` | **Creates network edges:** Bank → Sector exposure |
| `bank_maturity_profile.csv` | Calculates liquidity from short-term assets |
| `reverse_repo.csv` | Sets global liquidity level |

**Location:** `backend/ccp_ml/data/`

If files are missing, system creates synthetic data automatically.

---

## 📚 Next Steps

1. **Read the Full Guide:**
   ```bash
   cat ABM_IMPLEMENTATION_GUIDE.md
   ```

2. **Build a Frontend:**
   - Use `/abm/{sim_id}/state` for real-time updates
   - Use WebSockets for live streaming (extend API)
   - Visualize with D3/Cytoscape

3. **Extend the Model:**
   - Add derivatives agents
   - Implement central bank bailouts (TARP)
   - Add reinforcement learning for agent strategies

4. **Production Deploy:**
   - Move `ACTIVE_SIMULATIONS` to Redis
   - Add database persistence for history
   - Implement authentication

---

## 🐛 Troubleshooting

**Problem:** ImportError on agents.py
```bash
cd backend
pip install networkx numpy pandas
```

**Problem:** "Data directory not found"
```bash
# System falls back to synthetic data automatically
# To use real data, ensure: backend/ccp_ml/data/*.csv exists
```

**Problem:** All banks default instantly
```bash
# In API request, set:
"enable_shocks": false
# Or increase initial_crar in bank_crar.csv
```

---

## 📞 Support

- **Documentation:** `ABM_IMPLEMENTATION_GUIDE.md`
- **Examples:** `backend/examples/abm_example.py`
- **API Docs:** `http://localhost:8000/docs` (after starting server)

---

## 🎓 Academic Reference

This implementation is based on:
- Acemoglu et al. (2015) - Systemic Risk in Banking Networks
- Cont & Wagalath (2016) - Fire Sales and Contagion
- Bank of England's RAMSI model architecture

---

**Happy Simulating! 🚀**

For questions, check the implementation guide or inspect `examples/abm_example.py` for working code.
