# Migration Guide: Static Analysis → Agent-Based Model

## Overview

This guide explains how to transition from the existing static risk analysis system to the new dynamic Agent-Based Model (ABM).

---

## What Changed?

### Before (Static Analysis)

```python
# Load data
X_train, y_train = load_training_data()

# Train model
model = XGBoostClassifier()
model.fit(X_train, y_train)

# Predict
X_test = extract_features(bank_data)
default_probability = model.predict_proba(X_test)

# Static risk score
```

**Limitations:**
- ❌ No time dimension
- ❌ No strategic interactions
- ❌ No contagion dynamics
- ❌ Snapshot analysis only

### After (Agent-Based Model)

```python
# Load ecosystem
ecosystem = load_ecosystem_from_data("data/", max_timesteps=100)

# Run simulation
for t in range(100):
    snapshot = ecosystem.step()
    
    # Each bank makes strategic decisions
    # Contagion propagates through network
    # Shocks can occur
    # System evolves over time

# Dynamic risk trajectory
```

**Advantages:**
- ✅ Time-stepped evolution
- ✅ Strategic agent behaviors (Game Theory)
- ✅ Network contagion & cascades
- ✅ Policy intervention testing
- ✅ What-if scenarios

---

## System Architecture Comparison

### Old Architecture

```
Frontend → FastAPI → XGBoost Model → Static Predictions
                  ↓
            PostgreSQL (Historical data)
```

### New Architecture

```
Frontend → FastAPI → ABM Simulation Engine
                         ↓
                    ┌────┴────┐
                    │         │
              Agent Network  CSV Data Loader
                    │         │
                    └────┬────┘
                         ↓
                   Time-stepped Evolution
                         ↓
                   Visualization Layer
```

---

## Feature Mapping

| Old Feature | New Equivalent | Enhancement |
|-------------|----------------|-------------|
| `train_ml_model.py` | Still valid (becomes agent "brain") | XGBoost → `BankAgent.default_predictor` |
| `ccp_engine.py` | → `simulation_engine.py` | Static → Dynamic time-stepping |
| `network_builder.py` | → `initial_state_loader.py` | One-time → Evolving network |
| `risk_model.py` | → `agents.py` (Policy logic) | Fixed rules → Strategic adaptation |
| `/ml/predict` endpoint | → `/abm/{sim_id}/state` | Snapshot → Live state |

---

## API Migration

### Old Endpoint: Static Prediction

```http
POST /ml/predict
{
  "bank_id": "HDFC",
  "features": {...}
}

Response:
{
  "default_probability": 0.12,
  "risk_score": 0.34
}
```

### New Endpoint: Dynamic Simulation

```http
POST /abm/initialize
{
  "name": "HDFC Analysis",
  "use_real_data": true
}

Response:
{
  "simulation_id": "abc-123",
  "network_stats": {...}
}

---

POST /abm/abc-123/step
{
  "num_steps": 10
}

Response:
{
  "snapshots": [
    {
      "timestep": 1,
      "agent_states": {
        "HDFC_BANK": {
          "crar": 13.5,
          "health": 0.85,
          "mode": "normal"
        }
      }
    },
    ...
  ]
}
```

**Key Difference:** Instead of single prediction, you get time series of states.

---

## Code Migration Examples

### Example 1: Risk Assessment

**Before:**
```python
from app.ml.inference import DefaultPredictor

predictor = DefaultPredictor.load("models/default_predictor")
features = extract_features(bank_data)
prob = predictor.predict(features)

print(f"Default probability: {prob:.2%}")
```

**After:**
```python
from app.engine import load_ecosystem_from_data

ecosystem = load_ecosystem_from_data("data/")

# Get specific bank
bank = ecosystem.get_agent("HDFC_BANK")
initial_health = bank.compute_health()

# Simulate future
ecosystem.run(steps=20)

final_health = bank.compute_health()
print(f"Health trajectory: {initial_health:.2f} → {final_health:.2f}")
```

### Example 2: Network Analysis

**Before:**
```python
from ccp_ml.network_builder import InterBankNetwork

network = InterBankNetwork(data)
centrality = network.compute_centrality()
systemic_importance = network.identify_sifis()

print(f"Systemically important banks: {systemic_importance}")
```

**After:**
```python
from app.engine import load_ecosystem_from_data, NetworkVisualizer

ecosystem = load_ecosystem_from_data("data/")
snapshot = ecosystem._create_snapshot([])

metrics = NetworkVisualizer.compute_network_metrics(snapshot.to_dict())
print(f"Network density: {metrics['density']:.3f}")
print(f"Avg betweenness: {metrics['avg_betweenness_centrality']:.3f}")

# Run simulation to see evolution
ecosystem.run(steps=30)
```

### Example 3: Stress Testing

**Before:**
```python
from ccp_ml.simulate import stress_test

results = stress_test(
    banks=bank_data,
    shock_scenario="real_estate_crash"
)

print(f"Banks failed: {results['failures']}")
```

**After:**
```python
from app.engine import load_ecosystem_from_data, ShockType

ecosystem = load_ecosystem_from_data("data/")

# Run normal scenario
ecosystem.run(steps=10)
baseline_survival = ecosystem.global_state['survival_rate']

# Apply stress
ecosystem.apply_shock(
    ShockType.SECTOR_CRISIS,
    target="SECTOR_REAL_ESTATE",
    magnitude=-0.4
)

# Observe cascade
ecosystem.run(steps=20)
stressed_survival = ecosystem.global_state['survival_rate']

print(f"Baseline survival: {baseline_survival:.1%}")
print(f"Stressed survival: {stressed_survival:.1%}")
print(f"Impact: {(baseline_survival - stressed_survival):.1%} failure increase")
```

---

## Data Migration

### CSV Files: No Change Required

Your existing CSV files work as-is:
- `bank_crar.csv`
- `bank_npa.csv`
- `bank_sensitive_sector.csv`
- `bank_maturity_profile.csv`
- `reverse_repo.csv`

**New Usage:**
```python
from app.engine.initial_state_loader import InitialStateLoader

loader = InitialStateLoader("backend/ccp_ml/data")
loader.load_all_data()

# Automatically parses and creates agents
banks = loader.create_bank_agents()  # BankAgent instances
sectors = loader.create_sector_agents()  # SectorAgent instances
exposures = loader.create_bank_sector_exposures(banks, sectors)
```

### ML Model Integration

**Keep your trained XGBoost models!**

```python
import joblib
from app.engine import load_ecosystem_from_data

# Load your trained model
model = joblib.load("ml_models/default_predictor/model.pkl")

# Build ecosystem
ecosystem = load_ecosystem_from_data("data/")

# Inject model into banks
for agent in ecosystem.agents.values():
    if hasattr(agent, 'set_predictor'):
        agent.set_predictor(model)

# Now banks use ML for lending decisions
ecosystem.run(steps=50)
```

---

## Frontend Integration

### Old Approach

```typescript
// Single API call for prediction
const response = await fetch('/ml/predict', {
  method: 'POST',
  body: JSON.stringify(bankData)
});

const prediction = await response.json();
displayRiskScore(prediction.risk_score);
```

### New Approach

```typescript
// 1. Initialize simulation
const initResponse = await fetch('/abm/initialize', {
  method: 'POST',
  body: JSON.stringify({
    name: 'User Simulation',
    use_real_data: true
  })
});

const { simulation_id } = await initResponse.json();

// 2. Real-time stepping
async function stepSimulation() {
  const stepResponse = await fetch(`/abm/${simulation_id}/step`, {
    method: 'POST',
    body: JSON.stringify({ num_steps: 1 })
  });
  
  const { snapshots } = await stepResponse.json();
  const latest = snapshots[0];
  
  // Update network visualization
  updateNetworkGraph(latest.network_state);
  
  // Update metrics dashboard
  updateMetrics(latest.global_metrics);
}

// 3. Interactive controls
async function applyShock() {
  await fetch(`/abm/${simulation_id}/shock`, {
    method: 'POST',
    body: JSON.stringify({
      shock_type: 'sector_crisis',
      target: 'SECTOR_REAL_ESTATE',
      magnitude: -0.3
    })
  });
  
  stepSimulation();  // See impact
}
```

---

## Testing Strategy

### Phase 1: Parallel Running (Recommended)

Keep both systems running:
```python
# Old system
old_prediction = ml_model.predict(features)

# New system
ecosystem = load_ecosystem_from_data("data/")
ecosystem.run(steps=20)
new_risk = ecosystem.get_agent("BANK_1").compute_health()

# Compare
print(f"Old risk score: {old_prediction}")
print(f"New health trajectory: {new_risk}")
```

### Phase 2: Validation

Compare outputs for known scenarios:
```python
def validate_migration():
    # Known stress scenario from 2008
    ecosystem = load_ecosystem_from_data("data/2008_crisis")
    ecosystem.apply_shock(ShockType.ASSET_PRICE_CRASH, magnitude=-0.3)
    ecosystem.run(steps=50)
    
    # Validate against historical data
    survival_rate = ecosystem.global_state['survival_rate']
    assert survival_rate < 0.7, "Should show significant stress"
```

### Phase 3: Full Cutover

Once validated, switch frontend to ABM endpoints.

---

## Performance Considerations

### Old System
- **Latency:** ~100ms (single inference)
- **Throughput:** 100 requests/sec
- **Memory:** Low (stateless)

### New System
- **Latency:** ~50ms per step (for 100 agents)
- **Throughput:** 20 simulations/sec
- **Memory:** Higher (state maintenance)

**Optimization:**
```python
# For production, use Redis for simulation storage
import redis

REDIS_CLIENT = redis.Redis()

# Store simulation state
REDIS_CLIENT.set(
    f"sim:{sim_id}",
    pickle.dumps(ecosystem),
    ex=3600  # 1 hour TTL
)
```

---

## Backward Compatibility

The ABM system **does not break** existing functionality:

✅ Old ML endpoints still work  
✅ Static analysis scripts still valid  
✅ Database models unchanged  
✅ Frontend can gradually adopt ABM  

**Both systems coexist:**
```python
# Old route still exists
@router.post("/ml/predict")
async def predict_default(request: PredictionRequest):
    # Legacy static prediction
    pass

# New route added
@router.post("/abm/initialize")
async def initialize_abm(request: SimulationInitRequest):
    # Dynamic ABM simulation
    pass
```

---

## Recommended Migration Path

### Week 1: Setup & Testing
1. Run `examples/abm_example.py` to verify installation
2. Test with synthetic data
3. Load your CSV data and validate agents are created correctly

### Week 2: Integration
1. Add ABM endpoints to frontend (in parallel with old endpoints)
2. Create a "Simulation Mode" toggle in UI
3. Test both paths

### Week 3: Enhancement
1. Add custom visualizations for ABM network
2. Implement real-time stepping controls
3. Add policy intervention UI

### Week 4: Production
1. Move to Redis for simulation storage
2. Add monitoring and logging
3. Performance testing
4. Gradual rollout

---

## Common Issues & Solutions

### Issue: Import errors

```python
# Problem
from app.engine.agents import BankAgent
ImportError: No module named 'app.engine.agents'

# Solution
cd backend
pip install -e .  # Install in editable mode
```

### Issue: CSV parsing errors

```python
# Problem
KeyError: 'Bank' in bank_crar.csv

# Solution - Check column names
loader = InitialStateLoader("data/")
loader.load_all_data()

# Inspect DataFrame
print(loader.bank_crar_df.columns)

# Adjust column mapping in initial_state_loader.py
```

### Issue: Simulation runs too fast/slow

```python
# Too fast - agents not making meaningful decisions
config = SimulationConfig(
    max_timesteps=200,  # Increase duration
    shock_probability=0.15  # More shocks
)

# Too slow - reduce agent count
# In loader, filter to top N banks:
banks = loader.create_bank_agents()
top_banks = sorted(banks.values(), key=lambda b: b.capital, reverse=True)[:20]
```

---

## Support & Resources

**Documentation:**
- [ABM_IMPLEMENTATION_GUIDE.md](ABM_IMPLEMENTATION_GUIDE.md) - Full technical documentation
- [ABM_QUICKSTART.md](ABM_QUICKSTART.md) - Quick start guide
- `backend/examples/abm_example.py` - Working code examples

**API:**
- Swagger UI: `http://localhost:8000/docs` (after starting server)
- Endpoints: `/abm/*` for Agent-Based Model

**Code:**
- Agent definitions: `backend/app/engine/agents.py`
- Simulation loop: `backend/app/engine/simulation_engine.py`
- API routes: `backend/app/api/v1/abm_simulation.py`

---

## Next Steps

1. ✅ Complete this migration guide
2. ✅ Test ABM with your data
3. ⏳ Build frontend visualization
4. ⏳ Add custom agent types
5. ⏳ Deploy to production

**Questions?** Check the implementation guide or inspect example scripts.
