# CCP ML Backend API

Real-time Central Counterparty Risk Analysis and Network Simulation API.

## Features

### Core Capabilities
- **Network Analysis**: Build and analyze banking interdependency networks
- **Spectral Analysis**: Compute systemic risk metrics using spectral graph theory
- **CCP Risk Modeling**: Calculate margin requirements and default fund sizes
- **Stress Testing**: Apply various shocks to the banking system
- **Real-time Simulation**: Progressive timestep-based simulation with live updates
- **Graph Generation**: Create interactive and static visualizations

### Data Sources
- **RBI Bank Data**: Real Reserve Bank of India banking statistics
  - Capital Adequacy Ratios (CRAR)
  - Non-Performing Assets (NPAs)
  - Sensitive Sector Exposures
  - Maturity Profiles
  - Repo/Reverse Repo Rates

## Quick Start

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt
```

### Running the Server

**Option 1: Using the startup script (Windows)**
```powershell
.\start_server.ps1
```

**Option 2: Using uvicorn (recommended)**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Option 3: Using Python directly**
```bash
python api.py
```

### Access Points
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Health & Status
- `GET /` - Health check
- `GET /api/status` - Get simulation status

### Network Analysis
- `GET /api/network` - Get full network data with metrics
- `GET /api/network/nodes` - Get all nodes with centrality metrics
- `GET /api/network/edges` - Get all edges with weights

### Risk Analysis
- `GET /api/risk/scores` - Get risk scores for all banks
- `POST /api/risk/bank` - Get detailed risk analysis for specific bank

### Spectral Analysis
- `GET /api/spectral` - Get spectral analysis results
  - Spectral Radius (system amplification)
  - Fiedler Value (network connectivity)
  - Contagion Index (systemic risk)

### Simulation
- `POST /api/simulate` - Run full CCP simulation
- `POST /api/stress-test` - Run stress test scenario
- `GET /api/margins` - Get margin requirements for all banks
- `GET /api/default-fund` - Get default fund allocation
- `POST /api/reinitialize` - Reload data and reinitialize

### Real-time Simulation (NEW!)
- `POST /api/realtime/init` - Initialize real-time simulation
- `POST /api/realtime/step` - Execute simulation steps
- `GET /api/realtime/status` - Get current simulation status
- `GET /api/realtime/history` - Get full simulation history
- `POST /api/realtime/stop` - Stop running simulation

### Graph Generation (NEW!)
- `POST /api/graphs/generate` - Generate visualization graphs
- `GET /api/graphs/available` - Get list of available graph types

### WebSocket
- `WS /ws/simulation` - Real-time simulation updates

## Real-time Simulation

The new real-time simulation engine allows progressive execution of timestep-based simulations with live updates.

### Example Flow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize simulation
response = requests.post(f"{BASE_URL}/api/realtime/init", json={
    "max_timesteps": 100
})

# 2. Run simulation steps
response = requests.post(f"{BASE_URL}/api/realtime/step", json={
    "n_steps": 10,
    "shock_config": {
        "type": "liquidity",
        "magnitude": 0.3
    }
})

# 3. Get history
response = requests.get(f"{BASE_URL}/api/realtime/history")
history = response.json()["history"]

# 4. Analyze results
for step in history:
    print(f"Timestep {step['timestep']}: {step['default_count']} defaults")
```

### Shock Types
- **capital**: Reduce capital ratios
- **liquidity**: Reduce liquidity buffers
- **stress**: Increase stress levels directly

## Graph Generation

Generate various visualizations in Plotly (interactive) or Matplotlib (static) formats.

### Available Graph Types

1. **Network Graph**: Banking network topology with centrality metrics
2. **Risk Distribution**: Histograms and pie charts of risk levels
3. **Time Series**: Evolution of defaults, stress, and capital over time
4. **Spectral Analysis**: Eigenvalue spectrum and key metrics

### Example Usage

```python
# Generate interactive network graph
response = requests.post(f"{BASE_URL}/api/graphs/generate", json={
    "graph_type": "network",
    "format": "plotly",
    "highlight_nodes": ["State Bank of India", "HDFC Bank"]
})

graph_data = response.json()["data"]
```

### Graph Formats
- **plotly**: Interactive graphs (returns JSON)
- **matplotlib**: Static images (returns base64-encoded PNG)

## WebSocket Real-time Updates

Connect to the WebSocket endpoint for live simulation updates.

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation');

ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'subscribe' }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'simulation_update') {
        console.log('New simulation step:', data.data);
    }
};
```

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will test all endpoints including:
- Health check
- Network analysis
- Risk scores
- Spectral analysis
- Margins
- Stress testing
- Real-time simulation
- Graph generation

## Data Structure

### Bank Features
- `capital_ratio`: Capital adequacy ratio
- `liquidity_buffer`: Liquidity buffer size
- `stress_level`: Current stress level (0-1)
- `leverage`: Leverage ratio
- `credit_exposure`: Credit exposure amount
- `default_probability`: Probability of default
- Network metrics (degree, betweenness, eigenvector centrality, pagerank)

### Network Edge
- `source`: Source bank
- `target`: Target bank
- `weight`: Composite edge weight
- `sector_similarity`: Sector exposure similarity
- `liquidity_similarity`: Liquidity profile similarity
- `market_correlation`: Market correlation

### Simulation Step
- `timestep`: Current timestep
- `timestamp`: Execution time
- `bank_states`: State of all banks
- `network_metrics`: Network statistics
- `spectral_metrics`: Spectral analysis results
- `risk_distribution`: Risk level distribution
- `default_count`: Number of defaults
- `total_stress`: Average system stress
- `average_capital_ratio`: Average capital ratio

## Configuration

### Network Builder Parameters
- `sector_weight`: Weight for sector similarity (default: 0.4)
- `liquidity_weight`: Weight for liquidity similarity (default: 0.4)
- `market_weight`: Weight for market correlation (default: 0.2)
- `edge_threshold`: Minimum edge weight to include (default: 0.05)

### Simulation Parameters
- `max_timesteps`: Maximum simulation length (default: 100)
- `shock_magnitude`: Shock intensity 0-1 (default: 0.2)
- `target_banks`: Specific banks to shock (default: all)

## Architecture

```
ccp_ml/
├── api.py                    # Main FastAPI application
├── realtime_simulation.py    # Real-time simulation engine
├── graph_generator.py        # Visualization generation
├── data_loader.py            # Data loading utilities
├── feature_engineering.py    # Feature creation
├── network_builder.py        # Network construction
├── spectral_analyzer.py      # Spectral analysis
├── risk_model.py             # Risk modeling
├── ccp_engine.py             # CCP simulation engine
├── requirements.txt          # Python dependencies
├── start_server.ps1          # PowerShell startup script
├── start_server.bat          # Batch startup script
└── test_api.py               # API test suite
```

## Performance

- **Initialization**: ~2-5 seconds (loads 72 RBI banks)
- **Network Build**: ~1 second
- **Spectral Analysis**: ~0.5 seconds
- **Simulation Step**: ~0.1 seconds
- **Graph Generation**: ~1-3 seconds (depends on type/format)

## Deployment

### Local Development
```bash
uvicorn api:app --reload
```

### Production
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Future)
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Import Errors
```bash
# Ensure you're in the right directory
cd backend/ccp_ml

# Install dependencies
pip install -r requirements.txt
```

### Data Not Found
Ensure data files exist in `backend/ccp_ml/data/`:
- rbi_banks_ml_ready.csv
- 3.Bank-wise Capital Adequacy Ratios (CRAR) of Scheduled Commercial Banks.csv
- 6.Movement of Non Performing Assets (NPAs) of Scheduled Commercial Banks.csv
- etc.

## API Reference

Full interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Run the test suite: `python test_api.py`
3. Check server logs for detailed error messages

## Version

**v1.0.0** - Real-time simulation and graph generation release

## License

See main project LICENSE file.
