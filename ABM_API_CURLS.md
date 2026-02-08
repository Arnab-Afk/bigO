# RUDRA ABM API - cURL Examples

Base URL: `https://api.rudranet.xyz/api/v1`

## 1. Initialize a New Simulation

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Simulation",
    "max_timesteps": 100,
    "enable_shocks": true,
    "shock_probability": 0.1,
    "random_seed": 42,
    "use_real_data": true,
    "data_source": "backend/ccp_ml/data"
  }'
```

**Save the `simulation_id` from the response for subsequent calls!**

Example response:
```json
{
  "simulation_id": "abc-123-def-456",
  "name": "My First Simulation",
  "network_stats": {
    "num_banks": 10,
    "num_sectors": 5,
    "num_edges": 67
  }
}
```

---

## 2. List All Active Simulations

```bash
curl -X GET https://api.rudranet.xyz/api/v1/abm/list
```

---

## 3. Step the Simulation (Single Step)

**Replace `{simulation_id}` with your actual ID from step 1**

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/step \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 1
  }'
```

**Example with actual ID:**
```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/abc-123-def-456/step \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 1
  }'
```

---

## 4. Run Multiple Steps

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/step \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 10
  }'
```

**Alternative: Using query parameter**
```bash
curl -X POST "https://api.rudranet.xyz/api/v1/abm/{simulation_id}/run?steps=10"
```

---

## 5. Get Current State (Without Stepping)

```bash
curl -X GET https://api.rudranet.xyz/api/v1/abm/{simulation_id}/state
```

**Pretty print with jq:**
```bash
curl -s https://api.rudranet.xyz/api/v1/abm/{simulation_id}/state | jq .
```

---

## 6. Apply Shock: Sector Crisis

**Crash the Real Estate sector by 40%:**

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/shock \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "sector_crisis",
    "target": "SECTOR_REAL_ESTATE",
    "magnitude": -0.4
  }'
```

**Available shock types:**
- `sector_crisis` - Crash a specific sector
- `liquidity_squeeze` - Reduce global liquidity
- `interest_rate_shock` - Spike interest rates
- `asset_price_crash` - Global asset price decline

---

## 7. Apply Shock: Liquidity Squeeze

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/shock \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "liquidity_squeeze",
    "magnitude": -0.3
  }'
```

---

## 8. Apply Shock: Interest Rate Spike

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/shock \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "interest_rate_shock",
    "magnitude": -0.2
  }'
```

---

## 9. Apply Shock: Asset Price Crash

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/shock \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "asset_price_crash",
    "magnitude": -0.25
  }'
```

---

## 10. Set CCP Policy Rule

**Example: If system NPA > 8%, increase haircuts**

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/policy \
  -H "Content-Type: application/json" \
  -d '{
    "ccp_id": "CCP_MAIN",
    "rule_name": "Emergency Haircut Rule",
    "condition": "system_npa > 8.0",
    "action": "self.haircut_rate += 0.05"
  }'
```

**Example: If liquidity drops below 0.5, increase margins**

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/policy \
  -H "Content-Type: application/json" \
  -d '{
    "ccp_id": "CCP_MAIN",
    "rule_name": "Liquidity Margin Rule",
    "condition": "system_liquidity < 0.5",
    "action": "self.initial_margin_requirement += 2.0"
  }'
```

---

## 11. Get Simulation History

**Get all snapshots:**
```bash
curl -X GET https://api.rudranet.xyz/api/v1/abm/{simulation_id}/history
```

**Get last 20 snapshots:**
```bash
curl -X GET "https://api.rudranet.xyz/api/v1/abm/{simulation_id}/history?limit=20"
```

---

## 12. Reset Simulation

**Reset to time t=0 with initial conditions:**

```bash
curl -X POST https://api.rudranet.xyz/api/v1/abm/{simulation_id}/reset
```

---

## 13. Export Network for Visualization

**Export as JSON:**
```bash
curl -X POST "https://api.rudranet.xyz/api/v1/abm/{simulation_id}/export?format=json"
```

**Export as GEXF (for Gephi):**
```bash
curl -X POST "https://api.rudranet.xyz/api/v1/abm/{simulation_id}/export?format=gexf"
```

**Export as GraphML (for Cytoscape):**
```bash
curl -X POST "https://api.rudranet.xyz/api/v1/abm/{simulation_id}/export?format=graphml"
```

---

## 14. Delete Simulation

```bash
curl -X DELETE https://api.rudranet.xyz/api/v1/abm/{simulation_id}
```

---

## Complete Workflow Example

**Step-by-step walkthrough:**

### Step 1: Initialize
```bash
# Initialize and capture the simulation_id
RESPONSE=$(curl -s -X POST https://api.rudranet.xyz/api/v1/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Simulation",
    "max_timesteps": 100,
    "enable_shocks": true,
    "use_real_data": true
  }')

# Extract simulation_id (requires jq)
SIM_ID=$(echo $RESPONSE | jq -r '.simulation_id')
echo "Simulation ID: $SIM_ID"
```

### Step 2: Run Normal Scenario
```bash
# Run 10 normal steps
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10}'

# Check current state
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'
```

### Step 3: Apply Stress
```bash
# Apply Real Estate sector crisis
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/shock \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "sector_crisis",
    "target": "SECTOR_REAL_ESTATE",
    "magnitude": -0.4
  }'

# Step forward to observe contagion
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 15}'
```

### Step 4: Check Results
```bash
# Get final state
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '{
  timestep: .timestep,
  survival_rate: .global_metrics.survival_rate,
  avg_crar: .global_metrics.avg_crar,
  total_defaults: .global_metrics.total_defaults
}'
```

### Step 5: Get Time Series
```bash
# Get complete history
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/history?limit=50 | \
  jq '.snapshots[] | {t: .timestep, survival: .global_metrics.survival_rate}'
```

---

## Testing Scenarios

### Scenario 1: Mild Stress Test

```bash
# Initialize
SIM_ID=$(curl -s -X POST https://api.rudranet.xyz/api/v1/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{"name": "Mild Stress", "enable_shocks": false}' | jq -r '.simulation_id')

# Baseline
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps": 5}'

# Apply mild shock
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/shock \
  -H "Content-Type: application/json" \
  -d '{"shock_type": "sector_crisis", "target": "SECTOR_COMMODITIES", "magnitude": -0.15}'

# Observe
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps": 10}'
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'
```

### Scenario 2: Severe Crisis

```bash
# Initialize
SIM_ID=$(curl -s -X POST https://api.rudranet.xyz/api/v1/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{"name": "Severe Crisis", "enable_shocks": false}' | jq -r '.simulation_id')

# Baseline
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps": 5}'

# Apply severe shock
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/shock \
  -H "Content-Type: application/json" \
  -d '{"shock_type": "asset_price_crash", "magnitude": -0.5}'

# Observe cascade
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps": 20}'
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'
```

### Scenario 3: Policy Intervention

```bash
# Initialize
SIM_ID=$(curl -s -X POST https://api.rudranet.xyz/api/v1/abm/initialize \
  -H "Content-Type: application/json" \
  -d '{"name": "Policy Test", "enable_shocks": true}' | jq -r '.simulation_id')

# Set proactive policy
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/policy \
  -H "Content-Type: application/json" \
  -d '{
    "ccp_id": "CCP_MAIN",
    "rule_name": "Proactive Haircut",
    "condition": "system_npa > 5.0",
    "action": "self.haircut_rate += 0.05"
  }'

# Run with policy active
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/run?steps=30

# Check outcome
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'
```

---

## Quick Reference

| Action | Endpoint | Method |
|--------|----------|--------|
| Initialize | `/abm/initialize` | POST |
| List all | `/abm/list` | GET |
| Step | `/abm/{id}/step` | POST |
| Run | `/abm/{id}/run?steps=N` | POST |
| Get state | `/abm/{id}/state` | GET |
| Apply shock | `/abm/{id}/shock` | POST |
| Set policy | `/abm/{id}/policy` | POST |
| Get history | `/abm/{id}/history` | GET |
| Reset | `/abm/{id}/reset` | POST |
| Export | `/abm/{id}/export?format=json` | POST |
| Delete | `/abm/{id}` | DELETE |

---

## Tips

**1. Use jq for pretty output:**
```bash
curl -s https://api.rudranet.xyz/api/v1/abm/{simulation_id}/state | jq .
```

**2. Save simulation ID:**
```bash
SIM_ID="abc-123-def-456"  # Your actual ID
```

**3. Extract specific metrics:**
```bash
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | \
  jq '{survival: .global_metrics.survival_rate, crar: .global_metrics.avg_crar}'
```

**4. Watch simulation in real-time:**
```bash
# Linux/Mac
watch -n 2 "curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'"
```

**5. Chain operations:**
```bash
# Initialize → Step → Shock → Step → Get Results (all in one line)
SIM_ID=$(curl -s -X POST https://api.rudranet.xyz/api/v1/abm/initialize -H "Content-Type: application/json" -d '{"name":"Quick Test"}' | jq -r '.simulation_id') && \
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps":5}' && \
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/shock -H "Content-Type: application/json" -d '{"shock_type":"sector_crisis","target":"SECTOR_REAL_ESTATE","magnitude":-0.3}' && \
curl -X POST https://api.rudranet.xyz/api/v1/abm/$SIM_ID/step -H "Content-Type: application/json" -d '{"num_steps":10}' && \
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.global_metrics'
```

---

## Troubleshooting

**Error: "Simulation not found"**
```bash
# Check if simulation exists
curl https://api.rudranet.xyz/api/v1/abm/list | jq '.simulations[] | .simulation_id'
```

**Error: "Invalid shock type"**
Valid types: `sector_crisis`, `liquidity_squeeze`, `interest_rate_shock`, `asset_price_crash`

**Error: "Target not found"**
```bash
# Get list of available sectors/agents from state
curl -s https://api.rudranet.xyz/api/v1/abm/$SIM_ID/state | jq '.agent_states | keys'
```

---

**Happy Testing! 🚀**

For more details, see: [ABM_IMPLEMENTATION_GUIDE.md](ABM_IMPLEMENTATION_GUIDE.md)
