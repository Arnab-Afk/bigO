# COMPREHENSIVE ML DASHBOARD INTEGRATION ANALYSIS
**Date**: February 8, 2026  
**Project**: RUDRA - Financial Infrastructure Risk Platform  
**Objective**: Integrate CCP ML Simulation Data into Next.js Shadcn Admin Dashboard

---

## 📋 EXECUTIVE SUMMARY

### Problem Statement (PS.md)
**Network-Based Game-Theoretic Modeling of Financial Infrastructure**

The system models strategic interactions among financial institutions (banks, exchanges, clearing houses) within shared infrastructure using:
- Game theory to analyze strategic behavior
- Network analysis for credit exposures and settlement obligations
- Machine learning for risk prediction
- Spectral analysis for systemic fragility

**Business Impact**: Help regulators and financial institutions understand how micro-level decisions create macro-level financial risks, identify fragile structures, and support better regulatory policies.

---

## 🏗️ CURRENT SYSTEM ARCHITECTURE

### 1. ML Implementation (`ccp_ml`)
Located in: `C:\Users\arnab\Downloads\ccp_ml`

**Complete Pipeline (6 Layers)**:
1. **Data Loading** - `data_loader.py`
   - Loads 7 RBI datasets (CRAR, NPAs, Maturity Profiles, etc.)
   - 72 banks, 18 years (2008-2025)
   - ML-ready dataset with 47 features

2. **Feature Engineering** - `feature_engineering.py`
   - Creates 47 features from raw data
   - Includes capital ratios, leverage, stress levels, network metrics

3. **Network Construction** - `network_builder.py`
   - Builds interdependence network using:
     - Sector similarity (40%)
     - Liquidity similarity (40%)
     - Market correlation (20%)
   - Generates network metrics: PageRank, degree centrality, betweenness, eigenvector centrality

4. **Spectral Analysis** - `spectral_analyzer.py`
   - Spectral radius (ρ) - amplification risk
   - Fiedler value (λ2) - fragmentation risk
   - Contagion index
   - Eigenvalue entropy

5. **Risk Modeling** - `risk_model.py`
   - ML model (Gradient Boosting) for default probability prediction
   - Feature importance analysis
   - Risk classification (low/medium/high)

6. **CCP Engine** - `ccp_engine.py`
   - Margin requirement calculation
   - Default fund allocation
   - Policy recommendations
   - Cover-N standards

### 2. API Layer
**FastAPI Backend** (`api.py`):
- `/api/simulate` - Run full CCP simulation
- `/api/network` - Get network graph data
- `/api/network/nodes` - Node metrics
- `/api/network/edges` - Edge relationships
- `/api/risk/scores` - Risk scores for all banks
- `/api/risk/bank` - Detailed bank analysis
- `/api/spectral` - Spectral metrics
- `/api/stress-test` - Stress test simulations
- `/api/margins` - Margin requirements
- `/api/default-fund` - Default fund allocations

### 3. Simulation Outputs
Located in: `C:\Users\arnab\Downloads\ccp_ml\simulation_output`

**4 Key Files**:
1. `simulation_summary.json` - High-level metrics
2. `network_graph.json` - Network visualization data (nodes + edges)
3. `risk_scores.json` - Bank-level risk probabilities
4. `ccp_results.json` - Full CCP analysis results

---

## 📊 DATA STRUCTURE

### Available Datasets
**Location**: `C:\Users\arnab\Downloads\data`

1. **Capital Adequacy** (CRAR) - 72 banks
2. **Non-Performing Assets** (NPAs) - Movement and trends
3. **Maturity Profiles** - Liabilities and assets
4. **Sector Exposures** - Sensitive sectors
5. **Select Ratios** - Bank group-wise metrics
6. **Repo/Reverse Repo** - Market auction data
7. **ML Ready Dataset** - `rbi_banks_ml_ready.csv`

### Key Metrics Available
```json
{
  "Bank Level": {
    "capital_ratio": "Capital adequacy ratio",
    "liquidity_buffer": "Liquidity coverage",
    "leverage": "Debt to equity leverage",
    "stress_level": "Financial stress indicator (0-1)",
    "default_probability": "ML predicted default risk",
    "credit_exposure": "Credit risk exposure",
    "gross_npa": "Gross non-performing assets",
    "net_npa": "Net non-performing assets"
  },
  "Network Metrics": {
    "degree_centrality": "Connection importance",
    "betweenness_centrality": "Bridge importance",
    "eigenvector_centrality": "Influence in network",
    "pagerank": "Systemic importance score"
  },
  "Systemic Metrics": {
    "spectral_radius": "System amplification risk",
    "fiedler_value": "Network fragmentation risk",
    "contagion_index": "Contagion spread potential"
  },
  "CCP Metrics": {
    "margin_requirements": "Per-bank margin needs",
    "default_fund_size": "Total required fund",
    "cover_n": "Cover-N standard coverage"
  }
}
```

---

## 🎨 DASHBOARD INTEGRATION PLAN

### Phase 1: Backend Integration
**Objective**: Move ML system into backend API

#### Tasks:
1. **Copy ML Module to Backend**
   ```
   c:\Users\arnab\bigO\backend\app\ml\
   ├── ccp_ml/
   │   ├── __init__.py
   │   ├── data_loader.py
   │   ├── feature_engineering.py
   │   ├── network_builder.py
   │   ├── spectral_analyzer.py
   │   ├── risk_model.py
   │   ├── ccp_engine.py
   │   └── data/  (copy CSV files)
   ```

2. **Create Backend API Endpoints**
   - Integrate with existing FastAPI backend
   - Add routes under `/api/v1/ml/`
   - Endpoints:
     - `GET /api/v1/ml/status` - System status
     - `POST /api/v1/ml/simulate` - Run simulation
     - `GET /api/v1/ml/network` - Network graph
     - `GET /api/v1/ml/banks` - All banks with metrics
     - `GET /api/v1/ml/banks/{bank_id}` - Single bank detail
     - `GET /api/v1/ml/spectral` - Spectral analysis
     - `POST /api/v1/ml/stress-test` - Stress testing
     - `GET /api/v1/ml/margins` - Margin requirements
     - `GET /api/v1/ml/policies` - Policy recommendations

3. **Background Task Processing**
   - Use Celery/Redis for long-running simulations
   - WebSocket for real-time updates

### Phase 2: Frontend Dashboard Pages
**Framework**: Next.js + Shadcn Admin Template

#### Page Structure:
```
src/app/(main)/
├── ml-dashboard/           # Main ML Dashboard
│   ├── page.tsx           # Overview + Key Metrics
│   ├── network/           # Network Visualization
│   │   └── page.tsx
│   ├── banks/             # Bank Analysis
│   │   ├── page.tsx       # List view
│   │   └── [id]/
│   │       └── page.tsx   # Detail view
│   ├── simulation/        # Run Simulations
│   │   └── page.tsx
│   ├── stress-test/       # Stress Testing
│   │   └── page.tsx
│   └── reports/           # Policy Reports
│       └── page.tsx
```

### Phase 3: Dashboard Components

#### 1. **Overview Dashboard** (`/ml-dashboard`)
**Purpose**: High-level system snapshot

**Components**:
- **System Status Card**
  - Total banks analyzed
  - Last simulation time
  - Network edges count
  - Simulation status

- **Risk Distribution Chart** (Donut/Pie Chart)
  - High risk banks (count & %)
  - Medium risk banks
  - Low risk banks

- **Spectral Metrics Panel**
  - Spectral radius with risk indicator
  - Fiedler value with fragmentation status
  - Contagion index gauge

- **CCP Metrics Cards**
  - Total margin requirement
  - Default fund size
  - Cover-N standard

- **Recent Policy Recommendations** (List)
  - Priority level (HIGH/MEDIUM/LOW)
  - Category (capital, liquidity, market)
  - Affected banks count

**Visualizations**:
- Line chart: Risk trends over time
- Bar chart: Top 10 systemically important banks (by PageRank)
- Heatmap: Risk distribution by year

#### 2. **Network Visualization** (`/ml-dashboard/network`)
**Purpose**: Interactive network graph

**Features**:
- **Force-directed graph** using D3.js or Cytoscape.js
  - Nodes: Banks (sized by PageRank)
  - Edges: Dependencies (weighted by connection strength)
  - Color coding by risk level
  
- **Interactive Controls**:
  - Zoom/Pan
  - Filter by:
    - Edge threshold
    - Risk level
    - Bank group
  - Highlight specific bank and neighbors

- **Network Metrics Panel** (Side panel):
  - Selected node details
  - Centrality measures
  - Direct connections

**Visualizations**:
- Network graph (main)
- Adjacency matrix (alternative view)
- Community detection clusters

#### 3. **Bank Analysis** (`/ml-dashboard/banks`)
**Purpose**: Detailed bank-level analysis

**List View**:
- **Sortable/Filterable Table**:
  - Bank name
  - Capital ratio
  - Stress level
  - Default probability
  - PageRank
  - Risk classification
  - Actions (View Detail)

- **Filters**:
  - Risk level
  - Bank group
  - Capital ratio range
  - Year selection

**Detail View** (`/ml-dashboard/banks/[id]`):
- **Bank Profile Card**
  - Name, ID, Bank group
  - Latest metrics

- **Risk Assessment**:
  - Default probability gauge
  - Risk factors breakdown
  - Feature importance for this bank

- **Network Position**:
  - Centrality metrics visualization
  - Direct connections list
  - Ego network graph

- **Time Series Charts**:
  - Capital ratio trends
  - NPA trends
  - Stress level over time

- **Margin Requirements**:
  - Base margin
  - Network add-on
  - Total margin
  - Explanation

#### 4. **Simulation Runner** (`/ml-dashboard/simulation`)
**Purpose**: Run custom simulations

**Configuration Panel**:
- **Network Parameters**:
  - Sector weight (slider)
  - Liquidity weight (slider)
  - Market weight (slider)
  - Edge threshold (slider)

- **Target Selection**:
  - Year selector
  - Bank subset (multi-select)

- **Actions**:
  - Run Simulation button
  - Export Results
  - Compare with Previous

**Results Display**:
- Real-time progress bar
- Results summary cards
- Side-by-side comparison view
- Download JSON/CSV

#### 5. **Stress Testing** (`/ml-dashboard/stress-test`)
**Purpose**: Scenario analysis

**Test Configuration**:
- **Shock Type** (Select):
  - Capital shock
  - Liquidity shock
  - Market shock

- **Shock Magnitude** (Slider: 0-100%)

- **Target Banks** (Multi-select or "All")

**Results Visualization**:
- **Before/After Comparison**:
  - Risk distribution change
  - Default fund requirement change
  - Affected banks list

- **Impact Analysis**:
  - Banks pushed to higher risk category
  - Contagion spread visualization
  - Recovery recommendations

#### 6. **Policy Reports** (`/ml-dashboard/reports`)
**Purpose**: Regulatory recommendations

**Recommendations List**:
- **Grouped by Priority**:
  - HIGH (red badge)
  - MEDIUM (yellow badge)
  - LOW (green badge)

- **Recommendation Cards**:
  - Category (capital/liquidity/market)
  - Affected banks (expandable list)
  - Rationale
  - Suggested actions (checklist)
  - Export PDF button

---

## 🎨 UI/UX DESIGN SPECIFICATIONS

### Color Scheme
```css
/* Risk Colors */
--risk-high: #EF4444;      /* Red */
--risk-medium: #F59E0B;    /* Amber */
--risk-low: #10B981;       /* Green */

/* Spectral Analysis */
--spectral-critical: #DC2626;
--spectral-warning: #FBBF24;
--spectral-safe: #059669;

/* Network */
--node-default: #6366F1;   /* Indigo */
--node-central: #8B5CF6;   /* Purple */
--edge-strong: #4B5563;    /* Gray 600 */
--edge-weak: #D1D5DB;      /* Gray 300 */
```

### Charts & Visualizations
**Libraries**:
1. **Recharts** - For standard charts (line, bar, pie)
2. **D3.js** - For custom network graph
3. **React-Force-Graph** - Alternative for network
4. **Tremor** - Business dashboards components

### Responsive Design
- Desktop: Full 3-column layout
- Tablet: 2-column layout
- Mobile: Single column, collapsible panels

---

## 🔧 TECHNICAL IMPLEMENTATION

### Frontend Stack
```json
{
  "framework": "Next.js 16 (App Router)",
  "ui": "Shadcn UI components",
  "styling": "TailwindCSS v4",
  "charts": "Recharts + D3.js",
  "state": "Zustand + React Query",
  "forms": "React Hook Form + Zod",
  "tables": "TanStack Table"
}
```

### Backend Stack
```json
{
  "framework": "FastAPI",
  "ml": "scikit-learn, pandas, numpy",
  "network": "NetworkX",
  "async": "Celery + Redis (optional)",
  "database": "PostgreSQL (for caching results)"
}
```

### Data Flow
```
User Action (Frontend)
  ↓
Next.js API Route (/api/ml/*)
  ↓
Backend FastAPI (/api/v1/ml/*)
  ↓
CCP ML Pipeline
  ↓
Data Loader → Feature Engineering → Network Builder
  ↓
Spectral Analyzer + Risk Model
  ↓
CCP Engine (Margins, Policies)
  ↓
JSON Response
  ↓
Frontend React Query Cache
  ↓
UI Components (Charts, Tables, Graphs)
```

---

## 📈 KEY FEATURES TO IMPLEMENT

### Must-Have (MVP)
1. ✅ Overview dashboard with key metrics
2. ✅ Bank list with search/filter
3. ✅ Bank detail page with risk analysis
4. ✅ Simple network visualization
5. ✅ Policy recommendations display
6. ✅ Export functionality (CSV/JSON)

### Should-Have (v1.5)
7. Interactive network graph with controls
8. Stress testing interface
9. Time-series analysis
10. Simulation runner with custom parameters
11. Real-time simulation progress
12. Comparative analysis (year-over-year)

### Nice-to-Have (v2.0)
13. Advanced filtering and querying
14. Custom report generation
15. Email alerts for high-risk scenarios
16. Multi-currency support
17. Integration with external data sources
18. Automated scheduled simulations

---

## 📁 FILE STRUCTURE

### Backend
```
backend/app/
├── ml/
│   ├── __init__.py
│   ├── ccp_ml/          # ML module (copied from Downloads)
│   ├── routes/
│   │   └── ml_routes.py # API endpoints
│   ├── services/
│   │   └── ml_service.py # Business logic
│   └── data/            # CSV datasets
├── models/
│   └── ml_models.py     # Database models for caching
└── schemas/
    └── ml_schemas.py    # Pydantic schemas
```

### Frontend
```
frontend/src/
├── app/(main)/ml-dashboard/
│   ├── layout.tsx
│   ├── page.tsx         # Overview
│   ├── network/
│   ├── banks/
│   ├── simulation/
│   ├── stress-test/
│   └── reports/
├── components/ml/
│   ├── charts/
│   │   ├── RiskDistributionChart.tsx
│   │   ├── SpectralMetricsPanel.tsx
│   │   ├── NetworkGraph.tsx
│   │   └── TrendChart.tsx
│   ├── tables/
│   │   └── BankTable.tsx
│   └── cards/
│       ├── SystemStatusCard.tsx
│       ├── CCPMetricsCard.tsx
│       └── PolicyCard.tsx
├── lib/api/
│   └── ml-api.ts        # API client functions
└── hooks/
    └── useMLData.ts     # React Query hooks
```

---

## 🚀 IMPLEMENTATION ROADMAP

### Week 1: Backend Setup
- [ ] Copy ML module to backend
- [ ] Create API endpoints
- [ ] Test all endpoints with Postman
- [ ] Set up database caching (optional)

### Week 2: Frontend Foundation
- [ ] Create dashboard layout
- [ ] Implement overview page
- [ ] Add bank list page
- [ ] Build basic charts (Recharts)

### Week 3: Advanced Features
- [ ] Network visualization (D3.js)
- [ ] Bank detail page
- [ ] Simulation runner
- [ ] Stress testing interface

### Week 4: Polish & Deploy
- [ ] UI/UX refinements
- [ ] Performance optimization
- [ ] Testing & bug fixes
- [ ] Documentation
- [ ] Deployment

---

## 📊 SUCCESS METRICS

### Performance
- Page load time < 2s
- Simulation execution < 30s
- Network graph rendering < 5s

### Usability
- Intuitive navigation (< 3 clicks to any feature)
- Mobile responsive
- Accessible (WCAG 2.1 AA)

### Functionality
- All 72 banks displayed correctly
- Network graph with 2000+ edges
- Real-time updates for simulations
- Export in 3 formats (JSON, CSV, PDF)

---

## 🎯 NEXT STEPS

1. **Review this analysis** with stakeholders
2. **Prioritize features** (MVP vs v2.0)
3. **Set up backend** infrastructure
4. **Design mockups** for key pages
5. **Start implementation** following the roadmap

---

**End of Analysis**
