# RUDRA Backend API

## Resilient Unified Decision & Risk Analytics

Network-Based Game-Theoretic Modeling of Financial Infrastructure

### 🎯 Key Features

- **Game-Theoretic Modeling** - Nash equilibrium computation for strategic agents
- **Advanced Network Analysis** - Centrality metrics, path finding, bottleneck detection  
- **Multi-Mechanism Contagion** - Credit, liquidity, margin, and information cascades
- **Bayesian Beliefs** - Incomplete information and signal processing
- **Async Simulation Engine** - Celery-based background task execution
- **Graph Database Integration** - Neo4j for complex network queries

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

### Development Setup

1. **Clone and navigate to backend**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Start databases with Docker**:
   ```bash
   docker-compose up -d postgres redis neo4j
   ```

6. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

7. **Start Celery worker** (in separate terminal):
   ```bash
   celery -A app.tasks worker --loglevel=info
   ```

7. **Seed sample data** (optional):
   ```bash
   python -m scripts.seed_data
   ```

8. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

9. **Open API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

---

## 📁 Project Structure

```
backend/
├── alembic/                 # Database migrations
│   ├── versions/            # Migration scripts
│   └── env.py               # Alembic environment
├── app/
│   ├── api/                 # API routes
│   │   └── v1/              # Version 1 endpoints
│   │       ├── health.py    # Health checks
│   │       ├── institutions.py
│   │       ├── exposures.py
│   │       ├── network.py
│   │       ├── simulations.py
│   │       └── scenarios.py
│   ├── core/                # Core configuration
│   │   ├── config.py        # Settings
│   │   └── logging.py       # Structured logging
│   ├── db/                  # Database layer
│   │   ├── base.py          # SQLAlchemy base
│   │   └── session.py       # Session management
│   ├── models/              # SQLAlchemy models
│   │   ├── institution.py
│   │   ├── institution_state.py
│   │   ├── exposure.py
│   │   ├── simulation.py
│   │   └── scenario.py
│   ├── schemas/             # Pydantic schemas
│   │   ├── common.py
│   │   ├── institution.py
│   │   ├── exposure.py
│   │   ├── simulation.py
│   │   └── scenario.py
│   ├── engine/              # 🆕 Simulation Engine
│   │   ├── game_theory.py   # Nash equilibrium & utility functions
│   │   ├── network.py       # Centrality & path analysis
│   │   ├── contagion.py     # Cascade propagation
│   │   ├── bayesian.py      # Belief updates & signals
│   │   └── simulation.py    # Orchestrator
│   ├── tasks/               # 🆕 Celery background tasks
│   │   └── simulation_tasks.py
│   └── main.py              # FastAPI application
├── scripts/
│   └── seed_data.py         # Data seeding
├── tests/                   # Test suite
├── .env.example             # Environment template
├── alembic.ini              # Alembic configuration
├── Dockerfile               # Container image
├── pyproject.toml           # Python project config
└── requirements.txt         # Dependencies
```

---

## 🔧 API Endpoints

### Health & Status
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Service health check |
| GET | `/api/v1/ready` | Kubernetes readiness |
| GET | `/api/v1/live` | Kubernetes liveness |

### Institutions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/institutions` | List institutions |
| GET | `/api/v1/institutions/{id}` | Get institution details |
| POST | `/api/v1/institutions` | Create institution |
| PUT | `/api/v1/institutions/{id}` | Update institution |
| DELETE | `/api/v1/institutions/{id}` | Delete institution |
| GET | `/api/v1/institutions/{id}/states` | Get state history |

### Exposures
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/exposures` | List exposures |
| GET | `/api/v1/exposures/{id}` | Get exposure details |
| POST | `/api/v1/exposures` | Create exposure |
| PUT | `/api/v1/exposures/{id}` | Update exposure |
| DELETE | `/api/v1/exposures/{id}` | Delete exposure |
| GET | `/api/v1/exposures/matrix/summary` | Get exposure matrix |

### Network Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/network/graph` | Get network graph |
| GET | `/api/v1/network/metrics` | Network-level metrics |
| GET | `/api/v1/network/centrality/{id}` | Institution centrality |
| GET | `/api/v1/network/systemic-importance` | Systemic importance ranking |
| POST | `/api/v1/network/analyze` | 🆕 Advanced analysis (all centralities) |
| POST | `/api/v1/network/contagion-paths` | 🆕 Find contagion paths |
| POST | `/api/v1/network/cascade-simulation` | 🆕 Simulate cascade |

### Simulations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/simulations` | List simulations |
| GET | `/api/v1/simulations/{id}` | Get simulation details |
| POST | `/api/v1/simulations` | Create simulation |
| POST | `/api/v1/simulations/{id}/start` | Start simulation |
| POST | `/api/v1/simulations/{id}/cancel` | Cancel simulation |
| GET | `/api/v1/simulations/{id}/results` | Get results |
| DELETE | `/api/v1/simulations/{id}` | Delete simulation |

### Scenarios
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/scenarios` | List scenarios |
| GET | `/api/v1/scenarios/templates` | List templates |
| GET | `/api/v1/scenarios/{id}` | Get scenario details |
| POST | `/api/v1/scenarios` | Create scenario |
| PUT | `/api/v1/scenarios/{id}` | Update scenario |
| DELETE | `/api/v1/scenarios/{id}` | Delete scenario |
| POST | `/api/v1/scenarios/{id}/duplicate` | Duplicate scenario |

---

## 🗄️ Database Schema

### Core Entities

**Institution** - Financial institutions in the network
- `id`, `external_id`, `name`, `type`, `tier`, `jurisdiction`

**InstitutionState** - Time-varying state snapshots
- Capital ratios, liquidity metrics, exposure, risk scores

**Exposure** - Directed edges in the financial network
- Source/target institutions, amounts, collateral, risk parameters

**Simulation** - Simulation run records
- Scenario reference, status, progress, results

**Scenario** - Simulation scenario definitions
- Parameters, shocks, timing

**Shock** - Exogenous shock events
- Type, magnitude, duration, trigger timing

---

## � Simulation Engine

### Game-Theoretic Modeling
The engine implements strategic agent behavior using:
- **Utility Functions**: `U_i = Revenue - ρ·Risk - λ·Liquidity - γ·Regulatory`
- **Nash Equilibrium**: Best response iteration for agent decisions
- **Action Space**: Credit adjustments, margin changes, liquidity decisions

### Network Analysis Capabilities
- **Centrality Metrics**: Degree, betweenness, eigenvector, PageRank, Katz, closeness
- **Path Finding**: Critical contagion paths with probability thresholds
- **Bottleneck Detection**: Nodes whose failure most disrupts network flow
- **Systemic Risk Indices**: HHI concentration, interconnectedness, complexity

### Contagion Mechanisms
1. **Credit Contagion**: `Loss_j = Σ LGD × Exposure × Default_Indicator`
2. **Liquidity Spirals**: `Price_{t+1} = Price_t × (1 - α × FireSales_t)`
3. **Margin Spirals**: `Margin_t = VaR × (1 + β × Volatility_t)`
4. **Information Contagion**: `P(default_j | default_i) > P(default_j)`

### Bayesian Belief System
- **Signal Processing**: Noisy observations with reliability scores
- **Belief Updates**: `P(θ|signal) ∝ P(signal|θ) × P(θ)`
- **Entropy Tracking**: Measure uncertainty in agent beliefs
- **Belief Cascades**: Detect coordinated belief shifts

### Async Execution
- **Celery Tasks**: Background simulation execution
- **Real-time Updates**: Progress tracking via database
- **Monte Carlo**: Parameter variation across multiple runs

---

## �🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_institutions.py -v
```

---

## 🐳 Docker

### Build image
```bash
docker build -t rudra-api .
```

### Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres redis neo4j

# View logs
docker-compose logs -f api

# Stop all
docker-compose down
```

---

## 📊 Development Commands

```bash
# Format code
black app tests
isort app tests

# Type checking
mypy app

# Lint
ruff check app

# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_USER` | PostgreSQL user | `rudra` |
| `POSTGRES_PASSWORD` | PostgreSQL password | - |
| `POSTGRES_DB` | PostgreSQL database | `rudra` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | - |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `SECRET_KEY` | JWT secret key | - |

---

## 📝 License

MIT License - See LICENSE file for details.
