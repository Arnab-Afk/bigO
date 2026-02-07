# ML DASHBOARD IMPLEMENTATION PLAN
**Project**: RUDRA Financial Infrastructure Risk Platform  
**Sprint Duration**: 4 Weeks  
**Team**: Full Stack Development

---

## 🎯 PROJECT GOALS

### Primary Objective
Integrate the CCP ML simulation system into a production-ready Next.js dashboard that provides:
1. Real-time financial risk monitoring for 72 banks
2. Interactive network visualization of systemic dependencies
3. Stress testing capabilities for regulatory compliance
4. Policy recommendations based on ML predictions

### Success Criteria
- ✅ All 72 banks displayed with real-time metrics
- ✅ Interactive network graph with 2000+ connections
- ✅ Simulation execution time < 30 seconds
- ✅ Mobile-responsive design
- ✅ Export capabilities (CSV, JSON, PDF)

---

## 📋 SPRINT BREAKDOWN

### **SPRINT 1: Backend Foundation** (Week 1)
**Objective**: Set up ML backend infrastructure

#### Tasks:
1. **Day 1-2: Copy ML Module**
   - [x] Copy `ccp_ml` folder to `backend/app/ml/`
   - [x] Copy data files to `backend/app/ml/data/`
   - [ ] Update import paths in ML modules
   - [ ] Test ML pipeline independently

2. **Day 3-4: Create API Endpoints**
   - [ ] Create `backend/app/api/routes/ml_routes.py`
   - [ ] Implement 8 core endpoints:
     ```python
     GET  /api/v1/ml/status
     POST /api/v1/ml/simulate
     GET  /api/v1/ml/network
     GET  /api/v1/ml/banks
     GET  /api/v1/ml/banks/{bank_id}
     GET  /api/v1/ml/spectral
     POST /api/v1/ml/stress-test
     GET  /api/v1/ml/policies
     ```
   - [ ] Add CORS configuration
   - [ ] Create Pydantic schemas

3. **Day 5: Testing & Documentation**
   - [ ] Test all endpoints with Postman
   - [ ] Write API documentation (OpenAPI/Swagger)
   - [ ] Create sample requests/responses
   - [ ] Performance benchmarking

**Deliverables**:
- ✅ Working FastAPI backend with 8 ML endpoints
- ✅ API documentation
- ✅ Test suite for endpoints

---

### **SPRINT 2: Frontend Foundation** (Week 2)
**Objective**: Build core dashboard pages

#### Tasks:
1. **Day 1: Project Setup**
   - [ ] Verify Next.js + Shadcn setup
   - [ ] Install dependencies:
     ```bash
     npm install @tanstack/react-query recharts zustand
     npm install lucide-react date-fns
     ```
   - [ ] Create folder structure
   - [ ] Set up API client (`lib/api/ml-api.ts`)

2. **Day 2: Layout & Navigation**
   - [ ] Create main layout: `app/(main)/ml-dashboard/layout.tsx`
   - [ ] Add sidebar navigation items:
     - Overview
     - Network
     - Banks
     - Simulation
     - Stress Tests
     - Reports
   - [ ] Implement breadcrumbs

3. **Day 3-4: Overview Dashboard**
   - [ ] Create `app/(main)/ml-dashboard/page.tsx`
   - [ ] Build components:
     - `SystemStatusCard` - Total banks, last run, status
     - `RiskDistributionChart` - Donut chart (high/medium/low)
     - `SpectralMetricsPanel` - Gauges for spectral metrics
     - `CCPMetricsCards` - Margin, default fund, cover-N
     - `PolicyList` - Recent recommendations
   - [ ] Connect to API with React Query
   - [ ] Add loading states

4. **Day 5: Bank List Page**
   - [ ] Create `app/(main)/ml-dashboard/banks/page.tsx`
   - [ ] Build `BankTable` component with TanStack Table:
     - Sortable columns
     - Search functionality
     - Risk level filter
     - Pagination
   - [ ] Add export to CSV button

**Deliverables**:
- ✅ Overview dashboard with key metrics
- ✅ Bank list page with search/filter
- ✅ Responsive layout

---

### **SPRINT 3: Advanced Features** (Week 3)
**Objective**: Implement network visualization and interactive features

#### Tasks:
1. **Day 1-2: Network Visualization**
   - [ ] Install D3.js or React-Force-Graph:
     ```bash
     npm install d3 @types/d3
     # OR
     npm install react-force-graph-2d
     ```
   - [ ] Create `app/(main)/ml-dashboard/network/page.tsx`
   - [ ] Build `NetworkGraph` component:
     - Force-directed layout
     - Node sizing by PageRank
     - Color coding by risk level
     - Interactive hover tooltips
   - [ ] Add controls:
     - Zoom/Pan
     - Filter by threshold
     - Highlight bank and neighbors

2. **Day 3: Bank Detail Page**
   - [ ] Create `app/(main)/ml-dashboard/banks/[id]/page.tsx`
   - [ ] Build components:
     - `BankProfileCard` - Basic info
     - `RiskAssessmentPanel` - Default probability gauge
     - `NetworkPositionCard` - Centrality metrics
     - `TrendCharts` - Time series (capital ratio, NPAs)
     - `MarginRequirementsCard` - Breakdown
   - [ ] Implement ego network graph

3. **Day 4: Simulation Runner**
   - [ ] Create `app/(main)/ml-dashboard/simulation/page.tsx`
   - [ ] Build configuration form:
     - Sliders for weights (sector, liquidity, market)
     - Year selector
     - Bank subset multi-select
   - [ ] Add run simulation button
   - [ ] Display results with charts
   - [ ] Export functionality

4. **Day 5: Stress Testing**
   - [ ] Create `app/(main)/ml-dashboard/stress-test/page.tsx`
   - [ ] Build test configuration:
     - Shock type selector
     - Magnitude slider
     - Target banks multi-select
   - [ ] Display before/after comparison
   - [ ] Show impact analysis

**Deliverables**:
- ✅ Interactive network graph
- ✅ Bank detail pages with full analysis
- ✅ Simulation runner
- ✅ Stress testing interface

---

### **SPRINT 4: Polish & Deploy** (Week 4)
**Objective**: Refinement, testing, and production deployment

#### Tasks:
1. **Day 1: UI/UX Refinement**
   - [ ] Review all pages for consistency
   - [ ] Add animations and transitions
   - [ ] Improve error handling
   - [ ] Add empty states
   - [ ] Implement toast notifications

2. **Day 2: Performance Optimization**
   - [ ] Code splitting
   - [ ] Image optimization
   - [ ] Lazy loading for heavy components
   - [ ] Memoization for expensive calculations
   - [ ] Bundle size analysis

3. **Day 3: Testing**
   - [ ] Unit tests for utility functions
   - [ ] Integration tests for API calls
   - [ ] E2E tests for critical paths (Playwright)
   - [ ] Accessibility audit (Lighthouse)
   - [ ] Cross-browser testing

4. **Day 4: Documentation**
   - [ ] User guide
   - [ ] Developer documentation
   - [ ] Deployment guide
   - [ ] API reference
   - [ ] Troubleshooting guide

5. **Day 5: Deployment**
   - [ ] Setup production environment
   - [ ] Configure environment variables
   - [ ] Deploy backend (e.g., AWS, GCP, Railway)
   - [ ] Deploy frontend (Vercel, Netlify)
   - [ ] Setup monitoring (Sentry, LogRocket)
   - [ ] Final testing in production

**Deliverables**:
- ✅ Production-ready application
- ✅ Complete documentation
- ✅ Deployed and monitored

---

## 🔧 TECHNICAL TASKS CHECKLIST

### Backend Setup
- [ ] Copy ML codebase to backend
- [ ] Install Python dependencies:
  ```bash
  pip install fastapi uvicorn pandas numpy scikit-learn networkx
  ```
- [ ] Create API routes
- [ ] Add authentication (JWT)
- [ ] Setup CORS
- [ ] Create database models (for caching)
- [ ] Write unit tests

### Frontend Setup
- [ ] Initialize Next.js project (if needed)
- [ ] Install UI dependencies:
  ```bash
  npm install @shadcn/ui lucide-react
  npm install recharts d3 @tanstack/react-query zustand
  npm install @tanstack/react-table date-fns
  ```
- [ ] Setup Tailwind configuration
- [ ] Create API client
- [ ] Setup React Query provider
- [ ] Create reusable components

### Data Integration
- [ ] Copy all 7 CSV files to backend
- [ ] Verify data integrity
- [ ] Create data validation scripts
- [ ] Setup automated data refresh (if needed)

### Testing
- [ ] Backend unit tests (pytest)
- [ ] Frontend unit tests (Vitest/Jest)
- [ ] Integration tests
- [ ] E2E tests (Playwright)
- [ ] Load testing (for simulations)

---

## 📊 COMPONENT LIST

### Cards
- [ ] `SystemStatusCard` - Shows banks count, last run, status
- [ ] `RiskDistributionCard` - Pie chart with risk breakdown
- [ ] `SpectralMetricsCard` - Gauges for ρ, λ2, contagion
- [ ] `CCPMetricsCard` - Margin, fund, cover-N
- [ ] `BankProfileCard` - Basic bank information
- [ ] `NetworkPositionCard` - Centrality metrics
- [ ] `PolicyCard` - Single policy recommendation

### Charts
- [ ] `RiskDistributionChart` - Donut/Pie chart
- [ ] `TrendChart` - Line chart for time series
- [ ] `BarChart` - Top banks by PageRank
- [ ] `HeatMap` - Risk by year
- [ ] `GaugeChart` - For spectral metrics
- [ ] `NetworkGraph` - Interactive D3 visualization

### Tables
- [ ] `BankTable` - Sortable, filterable bank list
- [ ] `PolicyTable` - Recommendations list
- [ ] `MarginTable` - Margin requirements

### Forms
- [ ] `SimulationConfigForm` - Weights, year, banks
- [ ] `StressTestForm` - Shock type, magnitude, targets
- [ ] `BankSearchForm` - Search and filters

### Modals
- [ ] `ExportModal` - Choose format (CSV/JSON/PDF)
- [ ] `BankDetailModal` - Quick view (optional)
- [ ] `PolicyDetailModal` - Expanded policy view

---

## 🎨 DESIGN SYSTEM

### Color Palette
```css
/* Risk Levels */
--risk-critical: #DC2626;  /* Red 600 */
--risk-high: #EF4444;      /* Red 500 */
--risk-medium: #F59E0B;    /* Amber 500 */
--risk-low: #10B981;       /* Green 500 */
--risk-safe: #059669;      /* Green 600 */

/* Brand */
--primary: #6366F1;        /* Indigo 500 */
--secondary: #8B5CF6;      /* Purple 500 */
--accent: #EC4899;         /* Pink 500 */

/* Neutrals */
--background: #FFFFFF;
--foreground: #0F172A;     /* Slate 900 */
--muted: #F1F5F9;          /* Slate 100 */
--border: #E2E8F0;         /* Slate 200 */
```

### Typography
```css
--font-heading: "Inter", sans-serif;
--font-body: "Inter", sans-serif;
--font-mono: "JetBrains Mono", monospace;

--text-xs: 0.75rem;
--text-sm: 0.875rem;
--text-base: 1rem;
--text-lg: 1.125rem;
--text-xl: 1.25rem;
--text-2xl: 1.5rem;
--text-3xl: 1.875rem;
```

### Spacing
```css
--spacing-1: 0.25rem;   /* 4px */
--spacing-2: 0.5rem;    /* 8px */
--spacing-4: 1rem;      /* 16px */
--spacing-6: 1.5rem;    /* 24px */
--spacing-8: 2rem;      /* 32px */
```

---

## 📈 PERFORMANCE TARGETS

### Backend
- API Response Time: < 500ms (for cached data)
- Simulation Execution: < 30s (for full run)
- Concurrent Users: 50+

### Frontend
- First Contentful Paint: < 1.5s
- Time to Interactive: < 3s
- Lighthouse Score: > 90

### Network Graph
- Render Time (2000 nodes): < 5s
- Interaction FPS: 60fps
- Smooth zoom/pan

---

## 🚨 RISK MITIGATION

### Technical Risks
1. **Large Dataset Performance**
   - *Solution*: Implement pagination, virtual scrolling
   - *Fallback*: Server-side filtering/sorting

2. **Network Graph Rendering**
   - *Solution*: WebGL rendering, node clustering
   - *Fallback*: Limit visible nodes to top 100

3. **Simulation Time**
   - *Solution*: Background jobs with WebSocket updates
   - *Fallback*: Caching previous results

### UX Risks
1. **Complex Data Overwhelming Users**
   - *Solution*: Progressive disclosure, tooltips
   - *Fallback*: Guided tours, help documentation

2. **Mobile Experience**
   - *Solution*: Responsive design, simplified mobile views
   - *Fallback*: Desktop-only complex features

---

## 📝 NOTES & RECOMMENDATIONS

### Development Best Practices
1. Use TypeScript for type safety
2. Implement error boundaries
3. Add loading skeletons
4. Use React Query for caching
5. Optimize re-renders with useMemo/useCallback
6. Follow accessibility guidelines

### Data Handling
1. Cache simulation results in database
2. Implement incremental data updates
3. Validate API responses
4. Handle edge cases (no data, errors)

### User Experience
1. Add tooltips for technical terms
2. Provide contextual help
3. Show examples/templates
4. Implement keyboard shortcuts
5. Add dark mode support

---

## ✅ DEFINITION OF DONE

A feature is complete when:
- [ ] Code is written and reviewed
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] UI matches design mockups
- [ ] Responsive across devices
- [ ] Accessible (keyboard navigation, screen readers)
- [ ] Documentation updated
- [ ] Performance benchmarks met
- [ ] Deployed to staging
- [ ] Stakeholder approval received

---

## 🎯 IMMEDIATE NEXT STEPS

### Today (Feb 8, 2026)
1. **Review Analysis** - Go through comprehensive analysis doc
2. **Prioritize Features** - Decide MVP vs v2.0 features
3. **Setup Environment** - Ensure backend and frontend are running

### Tomorrow (Feb 9, 2026)
1. **Start Sprint 1** - Copy ML module to backend
2. **Create API Endpoints** - Implement core endpoints
3. **Test Backend** - Verify all endpoints work

### This Week
1. Complete Sprint 1 (Backend Foundation)
2. Begin Sprint 2 (Frontend Foundation)
3. Daily standup to track progress

---

**Ready to start implementation? 🚀**

Let me know if you want to:
1. Start copying the ML module to backend now
2. Review specific features first
3. Create mockups before coding
4. Adjust the timeline

