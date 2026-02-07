# Machine Learning Integration - Quick Start

This guide helps you quickly set up and use the ML capabilities in RUDRA.

## Installation

1. **Install ML dependencies**:
```bash
cd backend
pip install torch torchvision scikit-learn optuna mlflow torch-geometric
```

Or update from requirements.txt:
```bash
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

## Quick Training

Train a default predictor model:

```bash
cd backend

# Basic training (100 simulations, 100 epochs)
python scripts/train_ml_model.py

# With hyperparameter search
python scripts/train_ml_model.py \
    --simulations 200 \
    --epochs 100 \
    --hyperparameter-search \
    --register

# Custom parameters
python scripts/train_ml_model.py \
    --simulations 150 \
    --epochs 80 \
    --batch-size 64 \
    --learning-rate 0.001
```

## Using ML in Simulations

```python
from app.engine.simulation import SimulationEngine
import networkx as nx

# Create network
network = nx.DiGraph()
# ... add nodes and edges ...

# Create simulation engine with ML enabled
engine = SimulationEngine(
    network=network,
    max_timesteps=100,
    enable_ml=True,  # Enable ML predictions
)

# Run simulation
sim_state = engine.run_simulation(
    simulation_id="sim_with_ml",
    initial_states=initial_states,
    shocks=shocks,
    shock_timing=shock_timing,
)

# Access ML predictions
for timestep in sim_state.timesteps:
    for inst_id, agent_state in timestep.agent_states.items():
        print(f"Institution {inst_id}:")
        print(f"  Default prob: {agent_state.default_probability:.4f}")
        print(f"  ML confidence: {agent_state.ml_prediction_confidence:.4f}")
        print(f"  Model version: {agent_state.ml_model_version}")
```

## API Usage

### Start the server

```bash
cd backend
uvicorn app.main:app --reload
```

### Predict default probability

```bash
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "institution_id": "123e4567-e89b-12d3-a456-426614174000",
    "capital_ratio": 0.10,
    "liquidity_buffer": 0.5,
    "credit_exposure": 200.0,
    "default_probability": 0.02,
    "stress_level": 0.3,
    "risk_appetite": 0.5
  }'
```

### Start training via API

```bash
curl -X POST http://localhost:8000/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_simulations": 100,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hyperparameter_search": false
  }'
```

### Check training status

```bash
curl http://localhost:8000/api/v1/ml/train/status/{task_id}
```

## Running Tests

```bash
cd backend

# Run all ML tests
pytest tests/test_ml/ -v

# Run specific test file
pytest tests/test_ml/test_features.py -v

# Run with coverage
pytest tests/test_ml/ --cov=app.ml --cov-report=html
```

## Configuration

Set environment variables (optional):

```bash
# Enable GPU acceleration
export RUDRA_ML_ENABLE_GPU=true

# Set MLflow tracking URI
export RUDRA_ML_MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Set model path
export RUDRA_ML_MODELS_PATH="./ml_models"

# Confidence threshold
export RUDRA_ML_DEFAULT_PREDICTOR_CONFIDENCE_THRESHOLD=0.7
```

## Model Management

### List models
```bash
curl http://localhost:8000/api/v1/ml/registry/models
```

### Promote model to production
```bash
curl -X POST http://localhost:8000/api/v1/ml/registry/promote \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "default_predictor",
    "version": "1",
    "stage": "Production"
  }'
```

### Rollback model
```bash
curl -X POST http://localhost:8000/api/v1/ml/registry/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "default_predictor",
    "target_version": "0"
  }'
```

## Monitoring with MLflow

Start MLflow UI:

```bash
cd backend
mlflow ui
```

Open http://localhost:5000 to view:
- Experiment runs
- Model metrics
- Parameter comparisons
- Model versions

## Troubleshooting

### "Model not found" error
```bash
# Check model path
ls ml_models/default_predictor/

# Train a model if none exists
python scripts/train_ml_model.py --simulations 50 --epochs 50
```

### Slow training
- Reduce number of simulations: `--simulations 50`
- Use GPU if available: `export RUDRA_ML_ENABLE_GPU=true`
- Reduce dataset size or batch size

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# For torch-geometric issues:
pip install torch-geometric --no-cache-dir
```

### Out of memory
- Reduce batch size: `--batch-size 32`
- Reduce number of simulations
- Close other applications

## Next Steps

1. **Read full documentation**: [ML_ARCHITECTURE.md](../docs/ML_ARCHITECTURE.md)
2. **Explore API docs**: http://localhost:8000/docs (when server is running)
3. **Review examples**: Check `/tests/test_ml/` for usage examples
4. **Train custom models**: Modify hyperparameters in training script
5. **Integrate with frontend**: Use API endpoints from web interface

## Performance Tips

1. **Batch predictions**: Use `/ml/predict/batch` for multiple institutions
2. **Disable confidence**: Set `use_confidence=False` for faster inference
3. **GPU acceleration**: Enable GPU if available
4. **Cache predictions**: Results can be cached for repeated queries
5. **Async training**: Use Celery tasks for background training

## Support

- Documentation: `/docs/ML_ARCHITECTURE.md`
- Tests: `/tests/test_ml/`
- Issues: Open GitHub issue with `[ML]` tag
- Logs: Check `logs/ml_training.log`
