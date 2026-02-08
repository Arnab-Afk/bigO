# RUDRA - AI Agent Instructions

## Project Overview
**RUDRA** (Resilient Unified Decision & Risk Analytics) is a network-based game-theoretic platform for modeling systemic risk in financial infrastructure. It simulates how strategic decisions by financial institutions (banks, CCPs, exchanges) interact through credit exposures and liquidity dependencies, producing macro-level outcomes like stability or cascading failures.

## Architecture: Multi-Component Monorepo

### Core Components
```
backend/          Python FastAPI - Game-theoretic simulation engine + ML risk models
├── app/          Main API application (runs on port 17170)
├── ccp_ml/       ML pipeline for CCP risk analysis (standalone module)
├── scripts/      Data training/seeding utilities
└── examples/     Usage demonstrations

frontend/         Next.js dashboard for interactive network simulations (port 3000)
web/              Next.js landing page/marketing site
```

### Key Backend Services
- **`app/engine/simulation_engine.py`** - Agent-Based Model (ABM) orchestrator with game-theoretic agents
- **`app/engine/game_theory.py`** - Nash equilibrium computation for strategic decision-making
- **`app/engine/contagion.py`** - Multi-mechanism cascade modeling (credit, liquidity, margin)
- **`app/engine/bayesian.py`** - Incomplete information handling via Bayesian belief networks
- **`app/api/v1/abm_simulation.py`** - Stateful simulation API with in-memory storage (`ACTIVE_SIMULATIONS` dict)
- **`ccp_ml/main.py`** - CCP-centric ML pipeline (spectral analysis + default prediction)

### Database Stack (Docker Compose)
- **PostgreSQL** (5432): Relational data (institutions, exposures, scenarios)
- **Neo4j** (7687/7474): Graph network for complex pathfinding queries
- **Redis** (6379): Caching and Celery backend
- **TimescaleDB** (5433): Time-series simulation data (optional)

## Critical Developer Workflows

### Starting the Full Stack
```bash
# Terminal 1: Start infrastructure
docker-compose up -d postgres redis neo4j

# Terminal 2: Backend (ensures migrations run first)
cd backend
source .venv/bin/activate  # or `venv\Scripts\activate` on Windows
alembic upgrade head
python -m uvicorn app.main:app --host 0.0.0.0 --port 17170 --reload

# Terminal 3: Frontend dashboard
cd frontend
npm run dev  # Runs on http://localhost:3000
```

### Running ML Pipeline
```bash
cd backend
# Full training + network analysis + risk report
python scripts/ccp_pipeline.py --full-pipeline --save-report

# Uses Yahoo Finance API for real-time market data (yfinance package)
# Outputs: ccp_risk_report.txt, ml_models/ccp_default_predictor/
# Runtime: 3-7 minutes
```

### Database Migrations
```bash
cd backend
# Create new migration after model changes
alembic revision --autogenerate -m "description"
# Apply migrations
alembic upgrade head
```

## Project-Specific Patterns

### 1. API Port Convention
**Always use port 17170** for backend (not 8000). Frontend's `lib/api.ts` hardcodes `http://localhost:17170/api/v1`.

### 2. Simulation State Management
Simulations are stored **in-memory** in `ACTIVE_SIMULATIONS` dict (not database). Each simulation has UUID key. Production should use Redis.
```python
# Pattern in abm_simulation.py
sim_id = str(uuid4())
ACTIVE_SIMULATIONS[sim_id] = FinancialEcosystem(...)
```

### 3. User-as-Agent Model
Frontend dashboard treats user as the center node (blue highlight). See `frontend/components/EntityOnboarding.tsx` for how user entity is initialized with policy controls.

### 4. Data Loading Strategy
- **RBI CSV data**: 7 files in `backend/ccp_ml/data/` (banks, capital ratios, maturity profiles, etc.)
- **Loading function**: `app/engine/initial_state_loader.py::load_ecosystem_from_data()`
- Always use `use_real_data=True` for production simulations

### 5. Async-First Backend
All database operations use `async/await` with SQLAlchemy asyncpg. FastAPI lifespan handles DB initialization in `app/main.py`.

### 6. Structured Logging
Use the configured logger from `app/core/logging.py`:
```python
from app.core.logging import logger
logger.info("Message", extra_field=value, correlation_id=request_id)
```

### 7. Dual Frontend Setup
- **`frontend/`**: Main application dashboard with `/dashboard` route
- **`web/`**: Separate marketing site (different Next.js instance)
Both use shadcn/ui components with custom theme.

## File Navigation Landmarks

### When Adding New Simulation Features
1. Define Pydantic schema in `backend/app/schemas/`
2. Create SQLAlchemy model in `backend/app/models/`
3. Add API route in `backend/app/api/v1/`
4. Extend engine logic in `backend/app/engine/`
5. Update TypeScript types in `frontend/types/`
6. Add UI components in `frontend/components/`

### When Working with ML Models
- Training scripts: `backend/scripts/train_*.py`
- Model artifacts: `backend/ml_models/default_predictor/`
- Feature engineering: `backend/ccp_ml/feature_engineering.py`
- Network construction: `backend/ccp_ml/network_builder.py`
- Spectral analysis: `backend/ccp_ml/spectral_analyzer.py`

## Integration Points

### Frontend ↔ Backend
- Base URL: `http://localhost:17170/api/v1/abm`
- Key endpoints: `/initialize`, `/{sim_id}/step`, `/{sim_id}/state`, `/{sim_id}/shock`
- Response pattern: Returns snapshots for time-series visualization
- Error handling: `APIError` class wraps HTTP errors in `frontend/lib/api.ts`

### Backend ↔ Databases
- PostgreSQL: SQLAlchemy ORM models in `app/models/`, async session in `app/db/session.py`
- Neo4j: Direct driver in `app/graph/` (for complex network queries)
- Redis: Celery task queue (not heavily used in current version)

### ML Pipeline ↔ Yahoo Finance
- Module: `yfinance` package in `requirements.txt`
- Usage: `ccp_ml/network_builder.py` fetches real-time stock prices for market correlation channel
- Fallback: Synthetic data if API fails

## Testing & Documentation

### Running Tests
```bash
cd backend
pytest tests/ -v
# Key test: tests/test_institutions.py for API validation
```

### API Documentation
- Swagger UI: http://localhost:17170/docs (when DEBUG=True)
- ReDoc: http://localhost:17170/redoc
- Organized by tags: Health, Institutions, Network, Simulations, ABM

### Reference Documentation
- **Big picture**: `readme.md` - System architecture and objectives
- **ML workflow**: `ML_Flow.md` - CCP-centric risk modeling process
- **ABM usage**: `ABM_QUICKSTART.md` - Agent-based simulation guide
- **Run commands**: `RUN_INSTRUCTIONS.md` - Complete pipeline execution
- **Technical deep-dive**: `docs/TECHNICAL_DOCUMENTATION.md` (1579 lines)

## Common Gotchas

1. **Port conflicts**: Backend uses 17170 (not 8000). Check `frontend/lib/api.ts` if connection fails.
2. **Virtual environment**: Backend requires `.venv` activation before running. Install with `pip install -r requirements.txt`.
3. **Data directory**: ML scripts expect `backend/ccp_ml/data/` with 7 CSV files. Check with `ls -la app/ml/data/*.csv`.
4. **Simulation cleanup**: In-memory sims persist until reset. Use `DELETE /{sim_id}/reset` endpoint.
5. **Alembic conflicts**: Always run `alembic upgrade head` before starting server if models changed.
6. **CORS**: `ALLOWED_ORIGINS=["*"]` in development. Restrict in production via `.env`.

## When to Use Each Component

- **`ccp_ml/` pipeline**: Offline risk analysis, model training, research experiments
- **`app/engine/` + ABM API**: Real-time interactive simulation with frontend
- **`scripts/`**: Data preprocessing, one-time training, database seeding
- **`examples/`**: Quick testing without API server

## Configuration Management
Centralized in `backend/app/core/config.py` using Pydantic Settings:
- Loads from `.env` file (create from `.env.example`)
- Override with environment variables
- Access via `from app.core.config import settings`
