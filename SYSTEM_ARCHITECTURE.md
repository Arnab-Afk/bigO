# RUDRA System Architecture

**Version:** 1.0  
**Date:** February 8, 2026  
**Project:** Resilient Unified Decision & Risk Analytics

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Layers](#architecture-layers)
4. [Component Details](#component-details)
5. [Data Flow Architecture](#data-flow-architecture)
6. [Agent-Based Model (ABM)](#agent-based-model-abm)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [API Architecture](#api-architecture)
9. [Database Schema](#database-schema)
10. [Deployment Architecture](#deployment-architecture)
11. [Performance Optimizations](#performance-optimizations)
12. [Security Considerations](#security-considerations)

---

## Executive Summary

RUDRA is a **network-based game-theoretic platform** for modeling systemic risk in financial infrastructure. It simulates how strategic decisions by financial institutions (banks, CCPs, exchanges) interact through credit exposures and liquidity dependencies, producing macro-level outcomes like stability or cascading failures.

**Key Capabilities:**
- Real-time agent-based simulation with 15-25 financial institutions
- Game-theoretic Nash equilibrium computation for strategic decision-making
- Multi-mechanism cascade modeling (credit, liquidity, margin)
- ML-powered default prediction and risk assessment
- Interactive dashboard for policy experimentation
- Geopolitical and macro-prudential factor integration

**Technology Stack:**
- **Backend:** Python 3.11+, FastAPI, SQLAlchemy, NetworkX
- **Frontend:** Next.js 14, React, TypeScript, Recharts
- **Databases:** PostgreSQL, Neo4j, Redis, TimescaleDB
- **ML:** XGBoost, scikit-learn, pandas, yfinance
- **Infrastructure:** Docker Compose, Uvicorn, Node.js

---

## System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RUDRA Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Frontend   │────│    Backend   │────│  Databases   │    │
│  │  (Next.js)   │    │   (FastAPI)  │    │ (Postgres/   │    │
│  │  Port 3000   │    │  Port 17170  │    │  Neo4j/      │    │
│  └──────────────┘    └──────────────┘    │  Redis)      │    │
│         │                    │            └──────────────┘    │
│         │                    │                    │            │
│         └────────────────────┴────────────────────┘            │
│                          HTTP/REST                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Agent-Based Simulation Engine               │  │
│  │  • BankAgent (15-25 institutions)                       │  │
│  │  • CCPAgent (Central Counterparty)                      │  │
│  │  • RegulatorAgent (Central Bank + Geopolitical State)   │  │
│  │  • SectorAgent (Economic sectors)                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   ML Risk Pipeline                       │  │
│  │  • Default Prediction (XGBoost)                         │  │
│  │  • Spectral Network Analysis                            │  │
│  │  • Feature Engineering (RBI Data)                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Repository Structure

```
rudra-datathon/
├── backend/                    # Python FastAPI Application
│   ├── app/                    # Main application code
│   │   ├── api/v1/            # REST API endpoints
│   │   ├── engine/            # Simulation & game theory
│   │   ├── ml/                # ML models & training
│   │   ├── models/            # SQLAlchemy ORM models
│   │   ├── schemas/           # Pydantic validation schemas
│   │   ├── core/              # Config, logging, security
│   │   └── db/                # Database connections
│   ├── ccp_ml/                # CCP-centric ML pipeline
│   ├── scripts/               # Data processing & training
│   ├── tests/                 # Unit & integration tests
│   └── alembic/               # Database migrations
├── frontend/                   # Next.js Dashboard
│   ├── app/                   # Pages & routes
│   ├── components/            # React components
│   ├── lib/                   # API client & utilities
│   └── types/                 # TypeScript definitions
├── web/                       # Marketing website
├── docs/                      # Technical documentation
└── docker-compose.yml         # Infrastructure setup
```

---

## Architecture Layers

### 2.1 Presentation Layer (Frontend)

**Technology:** Next.js 14, React 18, TypeScript, Tailwind CSS

**Components:**
- **Dashboard (`app/dashboard/page.tsx`)**: Main simulation interface
  - Real-time network visualization (D3.js force-directed graph)
  - Time-series charts for system metrics
  - Policy control panel for user-controlled entity
  - Shock application interface
- **Entity Onboarding (`components/EntityOnboarding.tsx`)**: User entity creation
- **Network Visualization (`components/NetworkVisualization.tsx`)**: Interactive graph
- **Policy Controls (`components/PolicyControlPanel.tsx`)**: Dynamic policy sliders

**Key Features:**
- Real-time state synchronization with backend
- Auto-play simulation with pause/resume
- Risk decision alerts with user prompts
- Responsive UI with health indicators

### 2.2 Application Layer (Backend)

**Technology:** FastAPI, Python 3.11+, Uvicorn

**Port:** 17170 (not 8000 - critical convention)

**Core Modules:**

#### A. API Layer (`app/api/v1/`)
- **`abm_simulation.py`**: Stateful ABM simulation endpoints
  - `POST /initialize`: Create new simulation
  - `POST /{sim_id}/step`: Advance timesteps
  - `POST /{sim_id}/shock`: Apply exogenous shocks
  - `POST /{sim_id}/agent-policy`: Update user agent policies
  - `GET /{sim_id}/state`: Retrieve current state
  - `POST /{sim_id}/decision`: User decision responses
- **`institutions.py`**: CRUD for financial institutions
- **`network.py`**: Graph analysis & contagion paths
- **`ml.py`**: ML prediction endpoints
- **`exposures.py`**: Interbank exposure management

#### B. Simulation Engine (`app/engine/`)
- **`simulation_engine.py`**: Core orchestrator
  - Manages FinancialEcosystem lifecycle
  - Coordinates agent perceive-decide-act loop
  - Handles shock propagation
  - Tracks global state metrics
- **`agents.py`**: Agent implementations
  - `BankAgent`: Balance sheet, CRAR, NPA, lending policies
  - `CCPAgent`: Margin calls, default fund, eigenvector mutualization
  - `RegulatorAgent`: Monetary policy, geopolitical awareness
  - `SectorAgent`: Economic health, industry dynamics
- **`game_theory.py`**: Nash equilibrium solver
  - Strategic decision optimization
  - Payoff matrix computation
  - Best response dynamics
- **`contagion.py`**: Cascade mechanisms
  - Credit contagion (counterparty defaults)
  - Liquidity contagion (fire sales, margin calls)
  - Market contagion (correlation channels)
- **`geopolitical_state.py`**: External risk factors
  - Forex reserves & currency pressure
  - US Treasury holdings
  - Geopolitical tension levels
- **`bayesian.py`**: Incomplete information handling
- **`initial_state_loader.py`**: RBI CSV data ingestion

#### C. ML Pipeline (`ccp_ml/`)
- **`ccp_engine.py`**: CCP-centric risk modeling
- **`feature_engineering.py`**: Network + financial features
- **`spectral_analyzer.py`**: Graph eigenvalue analysis
- **`network_builder.py`**: Correlation matrix construction (Yahoo Finance)
- **`risk_model.py`**: XGBoost default predictor
- **`simulate.py`**: Monte Carlo stress testing

### 2.3 Data Layer

#### A. PostgreSQL (Port 5432)
**Purpose:** Relational data storage

**Schema:**
- `institutions`: Banks, CCPs, regulators
- `exposures`: Interbank credit relationships
- `simulations`: Simulation configurations
- `simulation_results`: Timestep snapshots
- `scenarios`: Predefined shock scenarios
- `shocks`: Applied shock events

#### B. Neo4j (Ports 7687/7474)
**Purpose:** Graph network queries

**Use Cases:**
- Complex pathfinding (contagion paths)
- Centrality calculations
- Community detection
- Systemic importance scoring

#### C. Redis (Port 6379)
**Purpose:** Caching & Celery backend

**Data:**
- Simulation state cache
- API rate limiting
- Task queue management

#### D. TimescaleDB (Port 5433) [Optional]
**Purpose:** Time-series simulation data

**Use Cases:**
- Historical metric tracking
- Performance analytics
- Audit trails

---

## Component Details

### 3.1 In-Memory Simulation State

**Critical Design Decision:** Simulations stored in-memory in `ACTIVE_SIMULATIONS` dict.

```python
# backend/app/api/v1/abm_simulation.py
ACTIVE_SIMULATIONS: Dict[str, FinancialEcosystem] = {}

# Each simulation has UUID key
sim_id = str(uuid4())
ACTIVE_SIMULATIONS[sim_id] = FinancialEcosystem(...)
```

**Implications:**
- ✅ Fast access (no DB roundtrips)
- ✅ Simple synchronous execution
- ❌ Not persistent (restarts lose state)
- ❌ Single-server limitation
- **Production TODO:** Use Redis for distributed state

### 3.2 Agent Lifecycle

**Three-Phase Loop:**

```python
# 1. PERCEIVE: Observe environment
def perceive(self, network: nx.DiGraph, global_state: Dict):
    # Update neighbor health, system stress
    # Store references for ML processing
    
# 2. DECIDE: Strategic optimization
def decide(self) -> Dict[str, Any]:
    # Compute best response policies
    # Use ML risk advisor if enabled
    # Apply game-theoretic constraints
    
# 3. ACT: Execute decisions
def act(self, actions: Dict[str, Any]):
    # Update balance sheet
    # Adjust lending limits
    # Change risk appetite
```

**Agent Types & Attributes:**

| Agent Type | Key Attributes | Policy Variables |
|------------|---------------|------------------|
| BankAgent | `capital`, `risk_weighted_assets`, `liquidity`, `npa_ratio`, `crar` | `risk_appetite`, `credit_supply_limit`, `interbank_limit`, `regulatory_min_crar` |
| CCPAgent | `default_fund_size`, `margin_buffer`, `active_exposures` | `initial_margin_requirement`, `haircut_rate`, `variation_margin_threshold` |
| RegulatorAgent | `base_repo_rate`, `min_crar_requirement`, `geopolitical_state` | `countercyclical_buffer`, `sectoral_concentration_limit`, `leverage_ratio_min` |
| SectorAgent | `economic_health`, `debt_load`, `output_level` | `growth_rate`, `volatility`, `credit_demand` |

### 3.3 User-as-Agent Model

**Frontend treats user as the center node (blue highlight).**

**Flow:**
1. User creates entity via `EntityOnboarding.tsx` (bank/CCP/regulator/sector)
2. Entity initialized with policy controls
3. Policies adjustable via sliders in `PolicyControlPanel.tsx`
4. Changes sent to `POST /{sim_id}/agent-policy` endpoint
5. Backend updates agent attributes immediately
6. Next timestep reflects new policies

**Policy Mapping (Frontend → Backend):**

```typescript
// Frontend (camelCase) → Backend (snake_case)
{
  'riskAppetite': 'risk_appetite',              // 0-1 range
  'minCapitalRatio': 'regulatory_min_crar',     // Percentage → Decimal
  'liquidityBuffer': 'liquidity',               // Updates liquidity directly
  'maxExposurePerCounterparty': 'interbank_limit', // Percentage of capital
  'initialMargin': 'initial_margin_requirement',
  'haircut': 'haircut_rate',
  'baseRepoRate': 'base_repo_rate',
  'minimumCRAR': 'min_crar_requirement'
}
```

---

## Data Flow Architecture

### 4.1 Simulation Initialization Flow

```
┌─────────────┐
│  Frontend   │
│  Dashboard  │
└──────┬──────┘
       │ POST /api/v1/abm/initialize
       │ {
       │   name: "My Simulation",
       │   user_entity: {id, type, name, policies}
       │ }
       ▼
┌──────────────────────┐
│   Backend FastAPI    │
│  abm_simulation.py   │
└──────┬───────────────┘
       │ 1. Load RBI CSV data
       │ 2. Create agents (15-25 banks)
       │ 3. Build network (NetworkX graph)
       │ 4. Initialize ML models
       │ 5. Store in ACTIVE_SIMULATIONS
       ▼
┌──────────────────────┐
│  FinancialEcosystem  │
│  simulation_engine   │
├──────────────────────┤
│ agents: List[Agent]  │
│ network: nx.DiGraph  │
│ global_state: Dict   │
│ timestep: int        │
└──────┬───────────────┘
       │ Return simulation_id + initial_state
       ▼
┌─────────────┐
│  Frontend   │
│  Renders    │
│  Network    │
└─────────────┘
```

### 4.2 Timestep Execution Flow

```
POST /{sim_id}/step?num_steps=1
            │
            ▼
┌───────────────────────────────┐
│  FOR EACH TIMESTEP            │
├───────────────────────────────┤
│  1. Agent.perceive()          │ ◄── All agents observe environment
│     - Update neighbor health  │
│     - Calculate systemic risk │
│                               │
│  2. Agent.decide()            │ ◄── Game-theoretic optimization
│     - ML risk advisor?        │
│     - Nash equilibrium        │
│     - Best response policies  │
│                               │
│  3. Agent.act()               │ ◄── Execute decisions
│     - Update balance sheets   │
│     - Adjust credit limits    │
│                               │
│  4. Check defaults            │ ◄── Solvency checks
│     - CRAR < regulatory_min?  │
│     - Liquidity < 0?          │
│                               │
│  5. Risk decisions?           │ ◄── Alert user if needed
│     - High distress detected  │
│     - Suggested action        │
│                               │
│  6. Capture snapshot          │ ◄── Save state
│     - Network state           │
│     - Agent states            │
│     - Global metrics          │
│                               │
│  7. Update global_state       │ ◄── Aggregate metrics
│     - system_health           │
│     - total_liquidity         │
│     - avg_crar                │
└───────────────────────────────┘
            │
            ▼
Return {snapshots[], pending_decision?}
```

### 4.3 Shock Propagation Flow

```
POST /{sim_id}/shock
{
  shock_type: "real_estate_shock",
  severity: "severe",
  target: "real_estate"
}
            │
            ▼
┌───────────────────────────────┐
│  Apply Shock (contagion.py)  │
├───────────────────────────────┤
│  1. Sector shock              │
│     - Real estate NPA +30%    │
│     - Reduce sector health    │
│                               │
│  2. Direct impact             │
│     - Banks with real estate  │
│       exposure affected       │
│     - Capital erosion         │
│                               │
│  3. Credit contagion          │ ◄── Cascade Phase 1
│     - Counterparty defaults   │
│     - Exposure losses         │
│                               │
│  4. Liquidity contagion       │ ◄── Cascade Phase 2
│     - Fire sales              │
│     - Margin calls            │
│     - Interbank freeze        │
│                               │
│  5. CCP loss mutualization    │ ◄── Eigenvector-based
│     - Zero risk for CCP       │
│     - Losses to banks         │
│       by centrality           │
│                               │
│  6. Regulator intervention?   │ ◄── Policy response
│     - Liquidity injection     │
│     - CRAR relief             │
└───────────────────────────────┘
            │
            ▼
Return {affected_agents, total_losses}
```

### 4.4 Policy Update Flow

```
Frontend Slider Change (Risk Appetite: 0.5 → 0.7)
            │
            ▼
POST /{sim_id}/agent-policy
{
  agent_id: "user_bank_123",
  policies: {
    riskAppetite: 0.7
  }
}
            │
            ▼
┌───────────────────────────────┐
│  Backend Policy Handler       │
├───────────────────────────────┤
│  1. Get agent from ecosystem  │
│  2. Map frontend → backend    │
│     riskAppetite → risk_appetite
│  3. Validate range (0-1)      │
│  4. setattr(agent, 'risk_appetite', 0.7)
│  5. Return success            │
└───────────────────────────────┘
            │
            ▼
Next timestep uses new risk_appetite
(affects credit_supply_limit, lending decisions)
```

---

## Agent-Based Model (ABM)

### 5.1 BankAgent Deep Dive

**Balance Sheet Model:**

```python
# Capital Adequacy
CRAR = capital / risk_weighted_assets
if CRAR < regulatory_min_crar:
    # Restrict lending, raise capital

# Liquidity Management
liquidity_ratio = liquidity / (debt_obligations + margin_calls)
if liquidity_ratio < 0.1:
    # Fire sell assets, borrow from interbank

# Non-Performing Assets
npa_ratio = bad_loans / total_loans
if npa_ratio > 0.08:
    # Reduce credit supply, increase provisions
```

**Decision Variables:**
- `credit_supply_limit`: Max total lending
- `interbank_limit`: Max lending to single counterparty
- `risk_appetite`: Willingness to lend to risky sectors (0-1)
- `lending_spread`: Interest margin over base rate (bps)

**Strategic Behavior:**
1. **Conservative mode** (distress): Reduce risk_appetite, hoard liquidity
2. **Aggressive mode** (healthy): Increase lending, pursue higher returns
3. **Nash equilibrium**: Balance profit vs. systemic risk

**Game Theory Integration:**
```python
def decide(self):
    if self.use_ml and self.ml_risk_advisor:
        # ML suggests optimal policies
        optimized_policies = self.ml_risk_advisor.recommend_policies(...)
        # Blend with current state
        self.risk_appetite = blend(current, optimized, factor=0.3)
```

### 5.2 CCPAgent Deep Dive

**Eigenvector-Based Loss Mutualization:**

```python
# Zero risk for CCP: Losses redistributed to members
if member_defaults:
    total_losses = calculate_default_losses()
    
    # Compute eigenvector centrality
    centrality_scores = nx.eigenvector_centrality(network)
    
    # Allocate losses proportionally
    for bank in surviving_banks:
        loss_share = centrality_scores[bank.id] * total_losses
        bank.capital -= loss_share
        bank.npa_ratio += loss_share / bank.capital
```

**Margin Management:**
- **Initial Margin**: Collateral required upfront (10-30%)
- **Variation Margin**: Mark-to-market adjustments
- **Haircut Rate**: Loss-sharing percentage
- **Default Fund**: Pooled resources for systemic shocks

### 5.3 RegulatorAgent Deep Dive

**Geopolitical State Integration:**

```python
class GeopoliticalState:
    tension_level: GeopoliticalTension  # low/moderate/high/crisis
    currency_pressure: CurrencyPressure # stable/depreciation
    forex_reserves: ForexReserves       # USD reserves, gold
    us_treasury_holdings: float         # Billions USD
    
    def to_dict(self) -> Dict:
        # Convert enums to numeric scores for API
        tension_scores = {
            GeopoliticalTension.LOW: 0.0,
            GeopoliticalTension.MODERATE: 0.33,
            GeopoliticalTension.HIGH: 0.66,
            GeopoliticalTension.CRISIS: 1.0
        }
        return {
            'tension_level': tension_scores[self.tension_level],
            'currency_pressure': ...,
            'forex_adequacy': self.forex_reserves.compute_adequacy_score()
        }
```

**Monetary Policy Levers:**
- **Base Repo Rate**: Cost of borrowing (affects credit supply)
- **Countercyclical Buffer**: Dynamic capital requirement (0-2.5%)
- **Liquidity Injection**: Emergency lending to banks
- **CRAR Relief**: Temporary regulatory forbearance

**Intervention Logic:**
```python
if system_npa > 0.08 or avg_crar < 10.5:
    # Crisis mode
    self.base_repo_rate -= 0.5  # Cut rates
    self.inject_liquidity(amount=100_000_000)
    self.countercyclical_buffer = 0  # Release capital buffer
```

---

## Machine Learning Pipeline

### 6.1 CCP ML Architecture

**Purpose:** CCP-centric risk modeling for default prediction and network analysis.

**Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Data Ingestion                                    │
├─────────────────────────────────────────────────────────────┤
│  • RBI CSV Data (7 files):                                  │
│    - banks.csv: Capital, assets, CRAR                       │
│    - capital_ratios.csv: Tier-1/Tier-2 breakdown            │
│    - maturity_profiles.csv: Asset-liability mismatch        │
│    - sector_exposures.csv: Concentration risk               │
│    - regional_distribution.csv: Geographic diversity        │
│    - liquidity_metrics.csv: LCR, NSFR                       │
│    - npa_trends.csv: Asset quality                          │
│  • Yahoo Finance API: Real-time stock correlations          │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Feature Engineering (feature_engineering.py)      │
├─────────────────────────────────────────────────────────────┤
│  • Financial Ratios:                                        │
│    - CRAR, Tier-1 ratio, Leverage ratio                     │
│    - NPA ratio, Provision coverage ratio                    │
│    - ROA, ROE, NIM                                          │
│  • Network Features:                                        │
│    - Degree centrality (in/out)                             │
│    - Betweenness centrality                                 │
│    - Eigenvector centrality                                 │
│    - PageRank score                                         │
│    - Clustering coefficient                                 │
│  • Systemic Features:                                       │
│    - CoVaR (Conditional Value at Risk)                      │
│    - SRISK (Systemic Risk)                                  │
│    - Marginal Expected Shortfall (MES)                      │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Network Construction (network_builder.py)         │
├─────────────────────────────────────────────────────────────┤
│  • Correlation Matrix:                                      │
│    - Stock return correlations (yfinance)                   │
│    - Interbank exposure matrix                              │
│  • Graph Construction:                                      │
│    - Nodes: Financial institutions                          │
│    - Edges: Weighted by exposure + correlation              │
│    - Directed graph (creditor → debtor)                     │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Spectral Analysis (spectral_analyzer.py)          │
├─────────────────────────────────────────────────────────────┤
│  • Eigenvalue Decomposition:                                │
│    - Laplacian matrix                                       │
│    - Largest eigenvalue (stability metric)                  │
│    - Spectral gap (robustness)                              │
│  • Community Detection:                                     │
│    - Louvain algorithm                                      │
│    - Identify bank clusters                                 │
│  • Centrality Distribution:                                 │
│    - Power law analysis                                     │
│    - Systemic importance ranking                            │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Default Prediction (risk_model.py)                │
├─────────────────────────────────────────────────────────────┤
│  • XGBoost Classifier:                                      │
│    - Features: 50+ financial + network metrics              │
│    - Target: Binary default indicator                       │
│    - Hyperparameters: max_depth=6, n_estimators=200         │
│  • Training:                                                │
│    - Synthetic default scenarios                            │
│    - Cross-validation (5-fold)                              │
│    - Class imbalance handling (SMOTE)                       │
│  • Outputs:                                                 │
│    - Default probability [0, 1]                             │
│    - Feature importance scores                              │
│    - SHAP values (explainability)                           │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Policy Recommendation (ml_risk_advisor)           │
├─────────────────────────────────────────────────────────────┤
│  • Risk-Based Policy Optimization:                          │
│    IF default_prob > 0.7:                                   │
│      - Reduce risk_appetite to 0.3                          │
│      - Increase liquidity buffer to 20%                     │
│      - Cut credit_supply_limit by 40%                       │
│    ELIF default_prob > 0.5:                                 │
│      - Moderate risk reduction                              │
│    ELSE:                                                    │
│      - Allow aggressive lending                             │
└─────────────────────────────────────────────────────────────┘
```

**Model Artifacts:**
- **Location:** `backend/ml_models/default_predictor/`
- **Files:**
  - `xgboost_model.pkl`: Trained classifier
  - `scaler.pkl`: Feature normalization
  - `feature_names.json`: Column mapping
  - `metadata.json`: Training info

**Training Script:**
```bash
cd backend
python scripts/ccp_pipeline.py --full-pipeline --save-report
# Runtime: 3-7 minutes
# Output: ccp_risk_report.txt
```

### 6.2 ML Integration in Simulation

**Agents use ML for decision-making:**

```python
class BankAgent:
    def decide(self):
        if self.default_predictor and not self.is_user_controlled:
            # ML-based risk assessment
            features = extract_features(self, network, global_state)
            default_prob = self.default_predictor.predict_proba(features)[0][1]
            
            # Skip ML if agent is healthy (performance optimization)
            if self.crar > 12.0 and self.npa_ratio < 0.05:
                return self.conservative_policy()
            
            # Adjust policies based on risk
            if default_prob > 0.6:
                self.risk_appetite *= 0.5  # Reduce aggressiveness
                self.credit_supply_limit *= 0.7
```

**Performance Optimization:** ML disabled by default for healthy agents to reduce CPU usage.

---

## API Architecture

### 7.1 Endpoint Catalog

**Base URL:** `http://localhost:17170/api/v1/abm`

| Endpoint | Method | Purpose | Request Body | Response |
|----------|--------|---------|--------------|----------|
| `/initialize` | POST | Create simulation | `{name, user_entity, max_timesteps}` | `{simulation_id, initial_state}` |
| `/{sim_id}/step` | POST | Advance timesteps | `{num_steps: 1-100}` | `{snapshots[], pending_decision?}` |
| `/{sim_id}/state` | GET | Get current state | - | `{timestep, agents, network, metrics}` |
| `/{sim_id}/shock` | POST | Apply shock | `{shock_type, severity, target}` | `{affected_agents, losses}` |
| `/{sim_id}/agent-policy` | POST | Update policies | `{agent_id, policies}` | `{updated_policies}` |
| `/{sim_id}/decision` | POST | User decision | `{decision_id, approved}` | `{success, message}` |
| `/{sim_id}/reset` | POST | Reset to initial | - | `{message}` |
| `/{sim_id}/export` | POST | Export data | - | `{json_data}` |

**Additional APIs:**
- **`/api/v1/institutions`**: CRUD for institutions
- **`/api/v1/network`**: Graph analysis endpoints
- **`/api/v1/ml`**: ML prediction & training
- **`/api/v1/exposures`**: Exposure management

### 7.2 Request/Response Examples

**Initialize Simulation:**
```json
POST /api/v1/abm/initialize
{
  "name": "Real Estate Stress Test",
  "max_timesteps": 100,
  "enable_shocks": true,
  "shock_probability": 0.12,
  "use_real_data": false,
  "user_entity": {
    "id": "user_bank_001",
    "type": "bank",
    "name": "State Bank of User",
    "policies": {
      "riskAppetite": 0.6,
      "minCapitalRatio": 11.5,
      "liquidityBuffer": 15,
      "maxExposurePerCounterparty": 25
    }
  }
}

Response:
{
  "simulation_id": "3f8a4f6-8036-f4b7d736a40d",
  "name": "Real Estate Stress Test",
  "network_stats": {
    "total_agents": 20,
    "total_banks": 17,
    "total_edges": 142
  },
  "initial_state": {
    "timestep": 0,
    "system_health": 0.85,
    "total_capital": 45000000000
  }
}
```

**Update Agent Policy:**
```json
POST /api/v1/abm/3f8a4f6-8036-f4b7d736a40d/agent-policy
{
  "agent_id": "user_bank_001",
  "policies": {
    "riskAppetite": 0.4,
    "liquidityBuffer": 20
  }
}

Response:
{
  "simulation_id": "3f8a4f6-8036-f4b7d736a40d",
  "agent_id": "user_bank_001",
  "updated_policies": {
    "riskAppetite": 0.4,
    "liquidityBuffer": 0.2
  },
  "status": "success"
}
```

**Apply Shock:**
```json
POST /api/v1/abm/3f8a4f6-8036-f4b7d736a40d/shock
{
  "shock_type": "real_estate_shock",
  "severity": "severe",
  "magnitude": -0.35
}

Response:
{
  "shock_applied": "real_estate_shock",
  "severity": "severe",
  "affected_agents": 12,
  "total_losses": 4500000000,
  "cascading_defaults": 2,
  "message": "Real Estate sector shock (severe): 12 banks affected"
}
```

### 7.3 WebSocket Support [Future]

**Not yet implemented, but planned for:**
- Real-time streaming updates
- Live metric dashboards
- Multi-user collaboration

---

## Database Schema

### 8.1 PostgreSQL Tables

**institutions:**
```sql
CREATE TABLE institutions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type InstitutionType NOT NULL, -- BANK, CCP, REGULATOR, SECTOR
    capital NUMERIC(20, 2),
    liquidity NUMERIC(20, 2),
    crar NUMERIC(5, 2),
    npa_ratio NUMERIC(5, 4),
    systemic_tier SystemicTier, -- TIER_1, TIER_2, TIER_3
    created_at TIMESTAMP DEFAULT NOW()
);
```

**exposures:**
```sql
CREATE TABLE exposures (
    id UUID PRIMARY KEY,
    creditor_id UUID REFERENCES institutions(id),
    debtor_id UUID REFERENCES institutions(id),
    amount NUMERIC(20, 2) NOT NULL,
    exposure_type ExposureType, -- CREDIT, DERIVATIVE, REPO
    maturity_days INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(creditor_id, debtor_id, exposure_type)
);
```

**simulations:**
```sql
CREATE TABLE simulations (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status SimulationStatus, -- RUNNING, COMPLETED, FAILED
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
```

**simulation_results:**
```sql
CREATE TABLE simulation_results (
    id UUID PRIMARY KEY,
    simulation_id UUID REFERENCES simulations(id),
    timestep INTEGER NOT NULL,
    global_state JSONB,
    network_state JSONB,
    agent_states JSONB,
    events JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 8.2 Neo4j Graph Schema

**Nodes:**
- `Institution`: Financial entities
  - Properties: `id`, `name`, `type`, `capital`, `health`
- `Sector`: Economic sectors
  - Properties: `id`, `name`, `economic_health`

**Relationships:**
- `(Bank)-[:LENDS_TO {amount, maturity}]->(Bank)`
- `(Bank)-[:CLEARS_THROUGH]->(CCP)`
- `(Bank)-[:REGULATED_BY]->(Regulator)`
- `(Bank)-[:EXPOSED_TO {weight}]->(Sector)`

**Cypher Query Example:**
```cypher
// Find contagion paths from defaulted bank
MATCH path = (failed:Institution {alive: false})-[:LENDS_TO*1..3]->(victim:Institution)
WHERE victim.alive = true
RETURN path, 
       reduce(loss = 0, rel IN relationships(path) | loss + rel.amount) AS total_exposure
ORDER BY total_exposure DESC
LIMIT 10
```

---

## Deployment Architecture

### 9.1 Local Development Setup

**Prerequisites:**
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

**Start Infrastructure:**
```bash
# Terminal 1: Databases
docker-compose up -d postgres redis neo4j

# Terminal 2: Backend
cd backend
source .venv/bin/activate  # or venv\Scripts\activate on Windows
alembic upgrade head
python -m uvicorn app.main:app --host 0.0.0.0 --port 17170 --reload

# Terminal 3: Frontend
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000
```

**Environment Variables (`.env`):**
```bash
# Backend
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rudra
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=WARNING
DEBUG=False
ALLOWED_ORIGINS=["*"]

# Frontend (automatically uses http://localhost:17170)
NEXT_PUBLIC_API_URL=http://localhost:17170/api/v1
```

### 9.2 Docker Compose Services

```yaml
services:
  postgres:
    image: postgres:15
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: rudra
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.13
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes:
      - redis_data:/data

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports: ["5433:5432"]
    environment:
      POSTGRES_DB: rudra_timeseries
```

### 9.3 Production Deployment [Recommended]

**Cloud Provider:** AWS/GCP/Azure

**Architecture:**
```
┌─────────────────────────────────────────────┐
│  Load Balancer (ALB/Cloud Load Balancer)    │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐
│  Frontend   │  │  Frontend   │  (Next.js on Vercel/Netlify)
│  Instance 1 │  │  Instance 2 │
└─────────────┘  └─────────────┘
       │                │
       └───────┬────────┘
               │
┌──────────────▼──────────────────────────────┐
│  API Gateway / Service Mesh                 │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐
│  Backend    │  │  Backend    │  (FastAPI on ECS/Cloud Run)
│  Instance 1 │  │  Instance 2 │
└──────┬──────┘  └──────┬──────┘
       │                │
       └───────┬────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐
│ RDS   │  │ Neo4j │  │ Redis │
│(Postgres)│(Aura)│ │(ElastiCache)
└───────┘  └───────┘  └───────┘
```

**Key Considerations:**
- **Stateful Simulations:** Migrate from in-memory to Redis/DynamoDB
- **Database:** Use managed services (RDS, Neo4j Aura, ElastiCache)
- **Container Orchestration:** Kubernetes or ECS/Cloud Run
- **Secrets Management:** AWS Secrets Manager, HashiCorp Vault
- **Monitoring:** Prometheus + Grafana, DataDog, New Relic
- **CDN:** CloudFront for frontend assets

---

## Performance Optimizations

### 10.1 Recent Optimizations (Feb 2026)

**1. Bank Count Reduction:**
- **Before:** 75-126 banks per simulation
- **After:** 15-25 banks
- **Impact:** 5-8x faster timesteps (4-5s → <1s)

**2. ML Inference Optimization:**
- **Before:** ML prediction every timestep for all agents
- **After:** Skip ML for healthy agents (CRAR > 12%, NPA < 5%)
- **Impact:** 60% CPU reduction

**3. Neighbor Caching:**
```python
# Before: Re-query neighbors every timestep
neighbors = list(network.predecessors(self.id))

# After: Cache neighbors (topology rarely changes)
if not hasattr(self, '_cached_neighbors'):
    self._cached_neighbors = list(network.predecessors(self.id))
neighbors = self._cached_neighbors
```

**4. UUID Generation Caching:**
```python
# Before: str(uuid.uuid4()) in every to_dict() call
def to_dict(self):
    return {'id': str(uuid.uuid4()), ...}

# After: Cache UUID once
def __init__(self):
    self._cached_uuid = str(uuid.uuid4())
def to_dict(self):
    return {'id': self._cached_uuid, ...}
```

**5. Logging Minimization:**
- **Before:** LOG_LEVEL=INFO (25+ logs per timestep)
- **After:** LOG_LEVEL=WARNING (only errors/warnings)
- **Impact:** Reduced console I/O overhead

### 10.2 Further Optimization Opportunities

**1. Parallel Agent Processing:**
- Use `asyncio.gather()` for independent agent decisions
- GPU acceleration for ML inference (CUDA)

**2. Network Analysis Caching:**
- Pre-compute centrality metrics
- Update only on network topology changes

**3. Database Query Optimization:**
- Use connection pooling (already implemented)
- Add indexes on frequently queried columns
- Batch insert for simulation_results

**4. Frontend Optimization:**
- Virtual scrolling for large agent lists
- WebGL for network rendering (react-force-graph)
- Service worker caching

---

## Security Considerations

### 11.1 Current Security Posture

**Authentication:** ❌ Not implemented (single-user demo)

**Authorization:** ❌ No role-based access control

**API Security:**
- ✅ CORS enabled (configurable origins)
- ❌ No rate limiting
- ❌ No API keys/tokens
- ⚠️ `eval()` used in policy rules (UNSAFE - demo only)

**Data Validation:**
- ✅ Pydantic schemas for all inputs
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ Type checking (TypeScript + mypy)

**Environment Security:**
- ✅ `.env` files for secrets (not committed)
- ⚠️ DEBUG=True in development
- ❌ No secret rotation

### 11.2 Production Security Checklist

**Must Implement:**
- [ ] **Authentication:** OAuth2/JWT tokens
- [ ] **Rate Limiting:** Redis-backed (10 req/sec per IP)
- [ ] **Input Sanitization:** Remove `eval()` in policy rules
- [ ] **HTTPS:** TLS 1.3 certificates
- [ ] **Database Encryption:** At rest + in transit
- [ ] **Audit Logging:** Track all API calls
- [ ] **Dependency Scanning:** Snyk/Dependabot
- [ ] **Secret Management:** AWS Secrets Manager

**Recommended:**
- [ ] **Web Application Firewall (WAF)**
- [ ] **DDoS Protection:** Cloudflare/AWS Shield
- [ ] **Penetration Testing**
- [ ] **SOC 2 Compliance** (if handling real financial data)

---

## Appendix

### A. Known Limitations

1. **Simulation State:** Not persistent (in-memory only)
2. **Concurrency:** Single-threaded timestep execution
3. **Scalability:** Limited to 25-30 agents before performance degrades
4. **Real Data:** Requires manual CSV updates (no live data feeds)
5. **Multi-User:** No support for concurrent users

### B. Roadmap

**Q1 2026:**
- [ ] WebSocket support for live updates
- [ ] Redis-based simulation persistence
- [ ] Multi-agent reinforcement learning (MARL)

**Q2 2026:**
- [ ] Authentication & multi-user support
- [ ] Advanced network visualizations (3D graphs)
- [ ] Scenario library (pre-configured stress tests)

**Q3 2026:**
- [ ] Integration with real market data APIs
- [ ] Regulatory reporting module (BCBS, FSB formats)
- [ ] Mobile-responsive dashboard

### C. References

**Academic:**
- Basel III Capital Framework (BCBS)
- Systemic Risk Measures (Acharya et al., 2017)
- Network Theory in Finance (Battiston et al., 2016)

**Technical:**
- FastAPI Documentation: https://fastapi.tiangolo.com
- NetworkX Documentation: https://networkx.org
- XGBoost Paper: https://arxiv.org/abs/1603.02754

**Data Sources:**
- RBI Statistical Tables: https://rbi.org.in
- Yahoo Finance API: https://pypi.org/project/yfinance

---

## Document Version Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-08 | RUDRA Team | Initial architecture documentation |

**Last Updated:** February 8, 2026  
**Status:** Living Document (updated as system evolves)

---

**For questions or contributions, contact:** project-rudra@example.com
