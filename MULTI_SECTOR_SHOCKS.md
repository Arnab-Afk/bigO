# Multi-Sector Shock Framework

## Overview

The multi-sector shock framework extends RUDRA beyond generic "real estate" shocks to model **10 distinct economic sectors**, each with sector-specific health indicators, shock scenarios, and spillover effects.

## Sectors Modeled

| Sector | Code | Description | Key Characteristics |
|--------|------|-------------|---------------------|
| **Real Estate** | `real_estate` | Property & construction | High leverage, long cycles, systemic risk |
| **Infrastructure** | `infrastructure` | Roads, power, telecom | Government-dependent, long gestation |
| **Manufacturing** | `manufacturing` | Industrial production | Supply chain sensitive, export linkages |
| **Agriculture** | `agriculture` | Crops, farming, agri-business | Weather-dependent, seasonal, high rural linkage |
| **Energy** | `energy` | Oil, gas, power | High spillover to all sectors, price volatility |
| **Export-Oriented** | `export_oriented` | Export manufacturing & services | Global demand dependent, FX exposure | Energy** | `energy` | Oil, gas, power, renewables | Price sensitivity, broad spillover |
| **Services** | `services` | IT, consulting, hospitality | Discretionary spending dependent |
| **MSME** | `msme` | Micro/Small/Medium Enterprises | High default risk, credit constrained |
| **Technology** | `technology` | Software, hardware, startups | Volatile funding, innovation cycles |
| **Retail Trade** | `retail_trade` | Consumer goods, e-commerce | Consumer sentiment driven |

---

## Quantifiable Health Indicators

Each sector is monitored through **11 measurable metrics**:

### 1. Financial Metrics (40% weight)
- **Debt Service Coverage**: Cash flow / Debt payments
  - Healthy: >1.5x
  - Stressed: <1.0x
- **Leverage Ratio**: Debt / Assets
  - Healthy: <40%
  - Stressed: >60%
- **Profitability Margin**: Operating income / Revenue
  - Healthy: >15%
  - Stressed: <5% or negative

### 2. Market Conditions (30% weight)
- **Demand Index**: Current demand vs baseline [0-1]
- **Capacity Utilization**: % of capacity in use
  - Healthy: >70%
  - Recession: <50%
- **Price Index**: Sector price level (1.0 = baseline)
  - Inflation: >1.2
  - Deflation: <0.9

### 3. Credit Quality (20% weight)
- **Default Rate**: % of sector loans in default
  - Healthy: <3%
  - Crisis: >10%
- **Restructuring Rate**: % of loans restructured
  - Normal: <5%
  - Stressed: >15%

### 4. Sentiment (10% weight)
- **Business Confidence**: Sentiment index [0-1]
  - Optimistic: >0.7
  - Pessimistic: <0.4
- **Investment Growth**: YoY capital expenditure growth
  - Expansion: >10%
  - Contraction: <-5%

**Overall Health Score**: Weighted combination of all metrics, normalized to [0-1].

---

## Shock Severity Levels

| Severity | Health Impact | Description | Example |
|----------|---------------|-------------|---------|
| **Mild** | -10% to -20% | Minor correction | Interest rate hike, regulatory tightening |
| **Moderate** | -20% to -40% | Significant stress | Demand slump, credit crunch |
| **Severe** | -40% to -60% | Crisis conditions | Sector-wide failures, cascade defaults |
| **Crisis** | -60% to -80% | Systemic collapse | 2008-style meltdown, complete loss of confidence |

---

## Pre-Defined Shock Scenarios

### Real Estate Shocks

#### MODERATE: Property Bubble Burst
- **Trigger**: Property bubble burst + credit crunch
- **Impact**:
  - Demand: -30%
  - Prices: -25%
  - Leverage: +10%
  - Default rate: +5%
  - Confidence: -25%
- **Spillover**: Infrastructure (40%), Manufacturing (40%), Retail (40%)
- **Recovery**: 16 timesteps

#### SEVERE: Real Estate Crisis
- **Trigger**: Developer defaults cascade
- **Impact**:
  - Demand: -50%
  - Prices: -40%
  - Default rate: +10%
- **Spillover**: Infrastructure, Manufacturing, Retail, MSMEs (60%)
- **Recovery**: 24 timesteps

---

### Manufacturing Shocks

#### MODERATE: Supply Chain Disruption
- **Trigger**: Supply chain disruption + input cost spike
- **Impact**:
  - Demand: -20%
  - Prices: +15% (cost-push inflation)
  - Default rate: +4%
- **Spillover**: Export-oriented, MSMEs, Retail (45%)
- **Recovery**: 12 timesteps

#### SEVERE: Industrial Recession
- **Trigger**: Industrial recession + demand collapse
- **Impact**:
  - Demand: -45%
  - Prices: -15% (deflation from excess capacity)
  - Default rate: +9%
  - Confidence: -50%
- **Spillover**: Export, MSME, Retail, Services (65%)
- **Recovery**: 20 timesteps

---

### Agriculture Shocks

#### MODERATE: Poor Monsoon
- **Trigger**: Drought conditions / poor monsoon
- **Impact**:
  - Crop prices: -20%
  - Default rate: +8% (high baseline default in agriculture)
  - Leverage: +10%
- **Spillover**: MSMEs, Retail (30%)
- **Recovery**: 4 timesteps (seasonal)

#### SEVERE: Multi-Year Drought
- **Trigger**: Multi-year drought + rural income collapse
- **Impact**:
  - Prices: -40%
  - Default rate: +15%
  - Confidence: -45%
- **Spillover**: MSME, Retail, Services (50%)
- **Recovery**: 8 timesteps

---

### Energy Shocks

#### MODERATE: Oil Price Shock
- **Trigger**: Oil price shock (supply disruption)
- **Impact**:
  - Energy prices: +40%
  - Demand: -10%
  - **High spillover intensity**: 55% (affects all sectors)
- **Spillover**: Manufacturing, Export, Retail
- **Recovery**: 8 timesteps

#### SEVERE: Energy Crisis
- **Trigger**: Energy crisis + power shortages
- **Impact**:
  - Prices: +70%
  - Demand: -30%
  - Default rate: +7%
- **Spillover**: Manufacturing, Export, Retail, Infrastructure (75%)
- **Recovery**: 16 timesteps

---

### MSME Shocks

#### MODERATE: Credit Squeeze
- **Trigger**: Credit squeeze + working capital crisis
- **Impact**:
  - Leverage: +15% (MSMEs highly leveraged)
  - Default rate: +10% (MSME default rates are baseline high)
  - Prices: +10% (input costs rise, can't pass through)
  - Confidence: -35%
- **Spillover**: Retail, Services (25%)
- **Recovery**: 12 timesteps

#### SEVERE: MSME Crisis
- **Trigger**: Mass bankruptcies
- **Impact**:
  - Demand: -45%
  - Default rate: +18%
  - Leverage: +22%
  - Confidence: -60%
- **Spillover**: Retail, Services, Manufacturing (40%)
- **Recovery**: 18 timesteps

---

## Spillover Mechanism

When a sector is shocked, **correlated sectors** automatically receive spillover impacts:

```python
spillover_health_impact = primary_shock_intensity × spillover_intensity × correlation_factor
```

### Correlation Matrix (Examples)

| Primary Shock | Correlated Sectors | Spillover Intensity |
|---------------|-------------------|---------------------|
| Real Estate Crisis | Infrastructure, Manufacturing, Retail, MSME | 60% |
| Energy Crisis | Manufacturing, Export, Retail, Infrastructure | 75% |
| Manufacturing Recession | Export, MSME, Retail, Services | 65% |
| Agriculture Drought | MSME, Retail | 30% |

---

## Bank Loss Calculation

Banks suffer losses based on their **exposure to shocked sectors**:

```python
loss_rate = sector_default_rate × health_deterioration_factor
bank_loss = exposure_to_sector × loss_rate

# Where health_deterioration_factor = (0.7 - sector_health) / 0.7
# Amplified 2× if health < 0.3
```

**Example**:
- Bank exposure to Real Estate: $5B
- Real Estate health drops to 0.35 (from 0.80)
- Default rate increases to 8%
- Loss rate: 8% × (0.7-0.35)/0.7 = 4%
- Bank loss: $5B × 4% = **$200M**

---

## Natural Recovery

Sectors gradually recover over time if no new shocks occur:

```python
recovery_rate = 5% per timestep

demand_index += recovery_rate × 0.5
default_rate -= recovery_rate × 0.3
confidence += recovery_rate × 0.8
leverage -= recovery_rate × 0.2
```

Recovery periods vary by shock severity:
- Mild: 10 timesteps
- Moderate: 12-16 timesteps
- Severe: 18-24 timesteps
- Crisis: 30-36 timesteps

---

## API Usage

### Apply Sector-Specific Shock

```bash
curl -X POST "http://localhost:17170/api/v1/abm/{sim_id}/shock" \
  -H "Content-Type: application/json" \
  -d '{
    "shock_type": "real_estate_shock",
    "severity": "severe",
    "magnitude": 0,
    "target": null
  }'
```

### Available Shock Types

```json
{
  "legacy_shocks": [
    "sector_crisis",
    "liquidity_squeeze",
    "interest_rate_shock",
    "asset_price_crash"
  ],
  "sector_specific_shocks": [
    "real_estate_shock",
    "infrastructure_shock",
    "manufacturing_shock",
    "agriculture_shock",
    "energy_shock",
    "export_shock",
    "services_shock",
    "msme_shock",
    "technology_shock",
    "retail_shock"
  ],
  "severity_levels": ["mild", "moderate", "severe", "crisis"]
}
```

---

## Integration with Existing Systems

### With Geopolitical Factors
- Energy shocks amplified by geopolitical tensions
- Export shocks triggered by trade wars
- Currency depreciation affects export-oriented sectors

### With Eigenvector Loss Mutualization
- Sector loan losses distributed to banks via eigenvector centrality
- CCP maintains zero risk while mutualizing systemic shocks
- Banks' interconnectedness determines loss allocation

### With Regulator Agent
- Regulator adjusts CRAR requirements based on sector concentrations
- Countercyclical buffers increased when multiple sectors stressed
- Sectoral exposure limits enforced automatically

---

## Testing

Run the demonstration script:

```bash
cd backend
source .venv/bin/activate
python test_multi_sector_shocks.py
```

**Output includes**:
- Overview of all sectors and shock scenarios
- Before/after health indicators
- Bank loss computations
- Spillover effects visualization
- Natural recovery trajectories

---

## Implementation Files

| File | Purpose |
|------|---------|
| `app/engine/sector_shocks.py` | Core framework (700+ lines) |
| `app/engine/simulation_engine.py` | Integration with ABM |
| `app/api/v1/abm_simulation.py` | API endpoints |
| `test_multi_sector_shocks.py` | Demonstration script |

---

## Key Advantages

✅ **Realistic**: Based on actual economic sector characteristics  
✅ **Quantifiable**: 11 measurable indicators per sector  
✅ **Granular**: 4 severity levels with pre-defined scenarios  
✅ **Dynamic**: Automatic spillovers and recovery  
✅ **Integrated**: Works with geopolitical factors & regulators  
✅ **Extensible**: Easy to add new sectors/scenarios  

---

## Example Simulation Scenario

**T=0**: Healthy economy, all sectors at 75-85% health  
**T=5**: Apply SEVERE real estate shock  
  - Real estate health: 85% → 40%  
  - Banks with 10% RE exposure take 3-5% capital losses  
  - Spillover: Infrastructure 75% → 60%, Manufacturing 80% → 65%  
**T=10**: MSME sector shows stress (delayed effect)  
  - Credit tightens as banks reduce lending  
  - MSME health: 75% → 55%  
**T=15**: Regulator intervenes  
  - CRAR requirement: 9% → 12% (geopolitical surcharge)  
  - Repo rate cut: 6.5% → 6.0%  
**T=20**: Natural recovery begins  
  - Real estate: 40% → 52%  
  - Banks rebuild capital buffers  
**T=36**: Full recovery to 70-75% health across sectors

---

## Future Enhancements

- **Sector-specific exposures**: Track each bank's portfolio composition
- **Dynamic correlations**: Adjust spillover based on economic cycles
- **Macro feedback loops**: GDP growth affects all sectors
- **Policy interventions**: Sector-specific bailouts, stimulus
- **Machine learning**: Predict sector stress from leading indicators
