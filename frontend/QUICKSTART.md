# CCP ML Dashboard - Quick Start

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation & Setup

1. **Install Dependencies**
```bash
cd frontend
npm install
```

2. **Verify Environment Configuration**
Check `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=https://api.rudranet.xyz
```

3. **Start Development Server**
```bash
npm run dev
```

4. **Access Dashboard**
Open browser to: **http://localhost:3000/ml-dashboard**

## 📊 Dashboard Features

### 1. Overview Metrics
- **72 RBI Banks** loaded from Reserve Bank of India data
- **Network edges** showing interdependencies
- **Spectral radius** indicating system stability
- **High risk banks** with stress > 0.7

### 2. Basic Operations

#### Run Static Simulation
1. Click **"Run Simulation"** button in header
2. Wait 2-3 seconds for results
3. View updated metrics across all cards

#### Apply Stress Test
1. Scroll to **"Stress Testing"** section
2. Click one of three shock types:
   - Capital Shock (-20%)
   - Liquidity Squeeze (-30%)
   - Market Shock (-40%)
3. View impact in alerts/notifications

#### Reload Data
1. Click **"Reload Data"** button in header
2. System reinitializes with fresh data from API

### 3. Real-time Simulation (NEW!)

#### Initialize
1. Scroll to **"Real-time Simulation Engine"** section
2. Click **"Initialize Simulation"** button
3. Wait for initialization (creates 100-timestep simulation)

#### Run Steps
- **Run 10 Steps**: Execute 10 timesteps at once
- **Single Step**: Step through one timestep at a time
- **Stop**: Halt execution (if running continuously)
- **Reset**: Clear history and reinitialize

#### Apply Shocks During Simulation
While simulation is running:
1. Click **Capital Shock**, **Liquidity Shock**, or **Stress Shock**
2. System executes 5 steps with shock applied at start
3. Watch live metrics update:
   - Defaults count
   - Average stress level
   - Average capital ratio
   - Progress percentage

#### Monitor Live Metrics
Real-time display shows:
- **Defaults**: Number of banks that defaulted (red if > 0)
- **Avg Stress**: System stress level (red if > 70%)
- **Avg Capital**: Capital adequacy (red if < 9%)
- **Progress**: Simulation completion percentage

## 🎯 Common Use Cases

### Scenario 1: Analyze Current System State
```
1. Open dashboard → View initial metrics
2. Check spectral radius (>1 = unstable)
3. Review risk distribution (Low/Medium/High)
4. Examine top 10 margin requirements
```

### Scenario 2: Test Capital Shock Impact
```
1. Click "Capital Shock (-20%)"
2. Wait for results
3. Compare baseline vs stressed default fund
4. Check fund increase percentage
```

### Scenario 3: Simulate Contagion Propagation
```
1. Click "Initialize Simulation"
2. Click "Single Step" multiple times
3. Watch defaults and stress levels evolve
4. Apply "Liquidity Shock" mid-simulation
5. Observe cascade effects
```

### Scenario 4: Compare Multiple Shocks
```
1. Note baseline metrics
2. Apply Capital Shock → record impact
3. Click "Reload Data" to reset
4. Apply Liquidity Shock → record impact
5. Click "Reload Data" to reset
6. Apply Market Shock → record impact
7. Compare results
```

## 🔧 Troubleshooting

### Dashboard Shows "Not Initialized"
- **Solution**: Backend is loading data, wait 5-10 seconds and refresh

### Real-time Simulation Controls Disabled
- **Solution**: Click "Initialize Simulation" first

### "API Error" in Console
- **Solution**: 
  1. Check internet connection
  2. Verify api.rudranet.xyz is accessible
  3. Check browser console for CORS errors
  4. Try hard refresh (Ctrl+Shift+R)

### Simulation Stuck at 100/100
- **Solution**: Click "Reset" to start new simulation

### Metrics Not Updating
- **Solution**:
  1. Check network tab for failed requests
  2. Click "Reload Data" to force refresh
  3. Check API status badge (should be green)

## 📱 UI Components Guide

### Header Bar
- **Status Badge**: Green = Online, Yellow = Initializing
- **Sim Badge**: Shows current timestep (appears during real-time sim)
- **Run Simulation**: Execute full analysis
- **Reload Data**: Reinitialize with fresh data

### Metric Cards
- **Green values**: Healthy metrics
- **Red values**: Risk indicators
- **Yellow values**: Warning thresholds

### Risk Distribution
- **Green bar**: Low risk banks (stress ≤ 30%)
- **Yellow bar**: Medium risk (30% < stress ≤ 70%)
- **Red bar**: High risk (stress > 70%)

### Spectral Analysis
- **Spectral Radius**: >1 = System amplifies shocks
- **Fiedler Value**: <0.1 = Fragmented network
- **Contagion Index**: Higher = More systemic risk

### Progress Bar (Real-time Sim)
- Blue bar shows completion (0-100%)
- Updates live during simulation

## 🎨 Color Legend

| Color | Meaning |
|-------|---------|
| 🟢 Green | Healthy / Low Risk |
| 🟡 Yellow | Warning / Medium Risk |
| 🔴 Red | Critical / High Risk |
| 🔵 Blue | Information / Progress |
| ⚪ Gray | Disabled / Unavailable |

## 📊 Understanding Metrics

### Spectral Radius (ρ)
- **< 1**: System stable, shocks dampen
- **= 1**: Critical point, shocks persist
- **> 1**: System unstable, shocks amplify

### Fiedler Value (λ₂)
- **> 0.1**: Well-connected network
- **0.05-0.1**: Moderately connected
- **< 0.05**: Risk of fragmentation

### Stress Level
- **0-0.3**: Low stress (safe)
- **0.3-0.7**: Medium stress (monitor)
- **0.7-1.0**: High stress (critical)

### Capital Ratio
- **> 12%**: Well-capitalized
- **9-12%**: Adequately capitalized
- **< 9%**: Under-capitalized (regulatory minimum)

## 🚀 Performance Tips

1. **Initial Load**: Wait 2-3 seconds for data
2. **Simulation Steps**: Run 10 steps at once for speed
3. **Browser**: Use Chrome/Edge for best performance
4. **Cache**: Clear cache if data seems stale
5. **Network**: Stable internet required for real-time features

## 📖 Next Steps

1. ✅ Explore basic features (run simulation, view metrics)
2. ✅ Try stress testing (apply different shocks)
3. ✅ Run real-time simulation (initialize & step through)
4. ✅ Monitor margin requirements (top 10 banks)
5. 📚 Read full documentation: `INTEGRATION_GUIDE.md`
6. 🔧 Review API docs: https://api.rudranet.xyz/docs

## 💡 Tips & Tricks

- **Auto-refresh**: Status updates every 10 seconds automatically
- **Real-time status**: Updates every 5 seconds during simulation
- **Keyboard**: Tab through buttons for quick navigation
- **Mobile**: Dashboard is responsive but best on desktop
- **Multiple tabs**: Each tab maintains separate query cache

## 🐛 Known Limitations

- Maximum 100 timesteps per simulation
- WebSocket support required for live updates
- Large simulations (>500 steps) not recommended
- Graph generation takes 1-3 seconds
- No simulation pause/resume (use stop & reset)

## 📞 Support

- Backend API Docs: https://api.rudranet.xyz/docs
- Integration Guide: `INTEGRATION_GUIDE.md`
- Backend README: `backend/ccp_ml/README.md`
- Test API: `python backend/ccp_ml/test_api.py`

---

**Version**: 1.0.0
**Last Updated**: February 8, 2026
**API**: api.rudranet.xyz
