# RUDRA Financial Network Simulator - System Summary

## Overview
**RUDRA** (Resilient Unified Decision & Risk Analytics) is a real-time, interactive financial network simulation platform that models systemic risk through agent-based modeling and game theory. Users control a financial entity (bank, regulator, CCP, or sector) and make strategic decisions while AI-powered risk management provides intelligent recommendations.

---

## Backend Model Architecture

### Core Engine (engine)
- **Agent-Based Model (ABM)**: Time-stepped simulation where agents perceive, decide, and act
- **Network Graph**: NetworkX DiGraph representing financial exposures and relationships
- **Contagion Mechanics**: Cascading defaults through credit/liquidity/margin channels

### Agent Types
1. **BankAgent** - Balance sheet management, CRAR compliance, lending decisions
2. **SectorAgent** - Economic health propagation, debt sustainability
3. **CCPAgent** - Central clearing, default fund management, margin requirements
4. **RegulatorAgent** - Monetary policy, capital requirement enforcement

### ML Risk Mitigation System
- **MLRiskMitigationAdvisor**: Real-time risk assessment using default prediction models
- **Policy Optimization**: Scipy-based optimization for risk-adjusted strategies
- **Proactive Actions**: Liquidity building, credit adjustment, collateral management
- **Risk Levels**: Low → Medium → High → Critical (with escalating interventions)

### Dynamic Network Features
- **Node Creation** (3% probability/tick): New banks join the network
- **Edge Formation** (5% per bank/tick): New interbank loans and exposures
- **Auto-Connection**: RBI and CCP automatically link to all banks
- **Network Evolution**: Realistic growth patterns with randomized parameters

### Health Calculation
- **Banks**: CRAR (50%) + Liquidity (30%) + NPA Health (20%), clamped to [0,1]
- **System Health**: Capital-weighted average accounting for dead nodes
- **Contagion Impact**: 50% haircut on exposures to defaulted institutions

---

## Frontend Interface Architecture

### Tech Stack
- **Framework**: Next.js 16 with React Server Components
- **Visualization**: ForceGraph2D for dynamic network rendering
- **Charts**: Recharts for time-series metrics (system health, active entities)
- **UI**: Tailwind CSS + shadcn/ui components

### Main Dashboard (page.tsx)

**Layout (3-Column Grid)**:
1. **Left Panel**: Policy controls + quick stats (system health, active entities, defaults)
2. **Center Panel**: Interactive force-directed network graph with node selection
3. **Right Panel**: Time-series charts + entity inspector

**User Controls**:
- ▶️ Play/Pause: Auto-advance simulation at 1 tick/second
- ⏭️ Step Forward: Manual single-step progression
- ⚡ Real Estate Shock: Apply -30% sector crisis
- 🔄 Reset: Restart with new random seed

### Interactive Network Visualization
- **Node Sizing**: Logarithmic scaling by capital + type-specific base sizes
- **Color Coding**: 
  - Blue (user) / Green (banks) / Orange (sectors) / Purple (CCP) / Red (regulator)
  - Dead nodes: Gray with red X
- **Tooltips**: Real-time capital, health, CRAR, status on hover
- **Selection**: Click to inspect detailed metrics in right panel

### Time-Series Charts
- **System Health Over Time**: 0-100% scale, blue line chart
- **Active Entities**: Green area chart with gradient fill
- **Custom Tooltips**: Large value (16px bold) + small timestamp (10px)

---

## User Decision System

### Risk Alert Modal
**Triggers**:
- 🚨 **Critical** (CRAR < regulatory minimum): Mandatory deleverage
- ⚠️ **High** (health < 30% OR 2+ neighbor defaults): Defensive posture
- 📉 **Medium** (system stress > 70% + moderate health): Risk reduction

**Interface**:
- **Risk-coded header**: Red/orange/yellow/blue based on severity
- **Current metrics**: Live CRAR, health, liquidity, NPA ratios
- **Recommended action** (green): AI-optimized safe choice
- **Alternative action** (gray): Maintain current strategy
- **Cooldown**: 5-timestep pause after action to prevent alert spam

**Actions Applied**:
- **Deleverage**: Reduces RWA to achieve target CRAR, cuts lending 60%, boosts liquidity to 30% of capital
- **Defensive**: Reduces interbank 20%, builds liquidity buffer to 25%
- **Reduce Risk**: Lowers risk appetite to 30%, cuts credit 10%

---

## Data Flow

```
User Action → Frontend (React state) → API Client (api.ts)
                ↓
Backend (FastAPI) → Simulation Engine → Agents (perceive/decide/act)
                ↓
Network Update → Contagion Propagation → Global Metrics
                ↓
Snapshot Creation → Response Transform → Frontend Update
                ↓
Chart Rerender + Network Reposition + Decision Check
```

### API Endpoints
- `POST /api/v1/abm/initialize`: Create simulation with user entity
- `POST /api/v1/abm/{sim_id}/step`: Advance N timesteps, returns snapshots + pending decisions
- `POST /api/v1/abm/{sim_id}/decision`: Respond to risk alert, apply action
- `POST /api/v1/abm/{sim_id}/shock`: Apply exogenous shock (sector crisis, rate hike, etc.)
- `GET /api/v1/abm/{sim_id}/state`: Get current state without advancing

---

## Key Features Summary

### ✅ Implemented
1. **ML-Enhanced Risk Management**: Agents use default prediction models for proactive decision-making
2. **Interactive User Control**: Play as bank/regulator with real-time policy adjustments
3. **Dynamic Network**: Nodes and edges form/break during simulation
4. **Risk Decision Alerts**: AI pauses simulation for critical user decisions with recommendations
5. **Real-Time Visualization**: 60fps force-directed graph with live metric updates
6. **Contagion Modeling**: Multi-channel cascade (credit, liquidity, margin shocks)
7. **Health Tracking**: Capital-weighted system health accounting for defaults
8. **Time-Series Analysis**: Historical charts for system health, entity survival

### 🚀 Performance
- **Backend**: Python 3.12 + FastAPI with async/await, in-memory state storage
- **Frontend**: Next.js Turbopack for instant HMR, dynamic imports for graph rendering
- **Simulation**: ~50-100 agents, 100+ timesteps, sub-100ms per tick

### 🎯 Use Cases
- **Systemic Risk Research**: Model cascade effects and vulnerabilities
- **Policy Testing**: Experiment with capital requirements, interest rates, liquidity rules
- **Financial Education**: Interactive learning about network externalities
- **Stress Testing**: Apply shocks and observe resilience/fragility patterns

---

## Architecture Highlights

**Separation of Concerns**:
- Backend: Pure simulation engine, no UI dependencies
- Frontend: Presentation layer, API client abstraction
- ML Module: Standalone risk assessment, pluggable into any agent

**Scalability**:
- Stateless API design (simulation stored by ID)
- Async database operations (PostgreSQL, Neo4j, TimescaleDB support)
- Configurable simulation parameters (timesteps, shock probability, ML enable/disable)

**Extensibility**:
- New agent types: Inherit from `Agent` base class
- New risk actions: Add to `MLRiskMitigationAdvisor.generate_mitigation_actions()`
- New visualizations: Add chart components with Recharts
- New shock types: Extend `ShockType` enum and `apply_shock()` method