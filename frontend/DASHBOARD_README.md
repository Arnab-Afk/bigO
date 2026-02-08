# Systemic Risk Simulation Dashboard

A Next.js 14 dashboard for visualizing financial network risk simulations with real-time agent-based modeling.

## Features

- **Interactive Network Graph**: Force-directed graph visualization showing banks, sectors, and clearing houses
- **Real-time Simulation**: Play/pause controls with auto-stepping every 1 second
- **Health Monitoring**: Color-coded nodes (red/yellow/green) based on agent health
- **Liquidity Flow**: Animated particles showing money flow between institutions
- **System Metrics**: Time-series charts for liquidity, NPA, and system health
- **Agent Inspector**: Detailed view of individual agent properties
- **Shock Testing**: Apply economic shocks (e.g., Real Estate sector collapse)
- **Dark Mode**: Bloomberg Terminal aesthetic with cyan/slate color scheme

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Visualization**:
  - `react-force-graph-2d` - Network graph
  - `recharts` - Time-series charts
- **Icons**: `lucide-react`
- **API**: FastAPI backend at `https://api.rudranet.xyz/api/v1/abm`

## Project Structure

```
frontend/
├── app/
│   ├── dashboard/
│   │   └── page.tsx          # Main dashboard page
│   ├── layout.tsx
│   └── globals.css
├── components/
│   └── FinancialNetwork.tsx  # Force graph component
├── lib/
│   ├── api.ts                # API wrapper functions
│   └── utils.ts
├── types/
│   └── simulation.ts         # TypeScript interfaces
└── package.json
```

## Setup & Installation

### Prerequisites

- Node.js 18+
- npm or yarn

### Install Dependencies

```bash
cd frontend
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000/dashboard](http://localhost:3000/dashboard)

## Usage

### Dashboard Controls

- **Play/Pause**: Start/stop the automatic simulation loop (steps every 1 second)
- **Next Step**: Manually advance the simulation by 1 timestep
- **Shock Real Estate**: Apply a -30% shock to the real estate sector
- **Reset**: Return the simulation to its initial state

### Network Visualization

- **Node Colors**:
  - 🟢 Green: Healthy (health 80-100%)
  - 🟡 Yellow: Medium (health 40-80%)
  - 🔴 Red: Unhealthy (health 0-40%)
  - ⚫ Grey: Dead/Defaulted
  - 🔵 Blue: Selected node

- **Node Sizes**: Larger nodes = higher capital
- **Link Width**: Thicker links = larger loan amounts
- **Animated Particles**: Cyan dots show liquidity flow direction

### Inspector Panel

Click any node to view:
- Agent ID and Type
- Health percentage with visual bar
- Capital, Liquidity, NPA amounts
- AI Strategy (if applicable)
- Sector affiliation

## API Integration

The dashboard communicates with the backend via these endpoints:

```typescript
// Initialize simulation
POST /initialize
Body: { name: string, max_timesteps: number, enable_shocks: boolean }

// Step forward
POST /{simId}/step
Body: { steps: number }

// Apply shock
POST /{simId}/shock
Body: { target: string, magnitude: number }

// Get current state
GET /{simId}/state

// Reset
POST /{simId}/reset
```

All API calls are typed and wrapped in `lib/api.ts`.

## Customization

### Change API URL

Edit `lib/api.ts`:

```typescript
const BASE_URL = "https://your-api-url.com/api/v1/abm";
```

### Adjust Simulation Speed

In `app/dashboard/page.tsx`, change the interval:

```typescript
playIntervalRef.current = setInterval(() => {
  stepSimulation(1);
}, 1000); // Change 1000ms to your preferred speed
```

### Modify Color Scheme

The dashboard uses Tailwind CSS classes. Key colors:
- Primary: `cyan-400` to `blue-500`
- Background: `slate-950`
- Panels: `slate-900`
- Borders: `slate-800`

### Graph Layout

Adjust force simulation in `components/FinancialNetwork.tsx`:

```typescript
d3VelocityDecay={0.3}  // Higher = faster settling
cooldownTicks={100}     // Higher = more iterations
nodeRelSize={6}         // Base node size multiplier
```

## Troubleshooting

### Network Graph Not Rendering

Ensure `react-force-graph-2d` is installed:
```bash
npm install react-force-graph-2d
```

### CORS Issues

If you see CORS errors, your backend needs to allow your frontend origin:
```python
# FastAPI backend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Type Errors

Ensure all types are properly imported:
```typescript
import { SimulationSnapshot, GraphNode } from "@/types/simulation";
```

## Build for Production

```bash
npm run build
npm start
```

Or deploy to Vercel:
```bash
vercel deploy
```

## License

MIT
