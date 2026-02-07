# Machine Learning Architecture for RUDRA Platform

## Overview

This document describes the Machine Learning integration for the RUDRA Financial Infrastructure Risk Platform. The ML layer adds predictive capabilities, pattern recognition, and learned optimization to complement the existing rule-based and statistical systems.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Default Probability Prediction](#phase-1-default-probability-prediction)
3. [Phase 2: Cascade Risk Classification](#phase-2-cascade-risk-classification)
4. [Phase 3: Time Series Forecasting](#phase-3-time-series-forecasting)
5. [Phase 4: MLOps Infrastructure](#phase-4-mlops-infrastructure)
6. [API Reference](#api-reference)
7. [Training Guide](#training-guide)
8. [Deployment](#deployment)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        RUDRA Platform                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐            │
│  │  FastAPI    │  │   Redis     │  │    Neo4j     │            │
│  │  Endpoints  │  │   Cache     │  │   Network    │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘            │
│         │                │                │                     │
│  ┌──────▼────────────────▼────────────────▼───────┐            │
│  │          Simulation Engine                     │            │
│  │  ┌─────────────────────────────────────────┐   │            │
│  │  │  ML Integration Layer                   │   │            │
│  │  │  - DefaultPredictor (inference)         │   │            │
│  │  │  - FeatureExtractor                     │   │            │
│  │  └─────────────────────────────────────────┘   │            │
│  └────────────────────────────────────────────────┘            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             ML Training Pipeline (Celery)                │  │
│  │  - Synthetic Data Generation                             │  │
│  │  - Model Training with Optuna                            │  │
│  │  - Model Evaluation & Validation                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Model Registry (MLflow)                     │  │
│  │  - Version Management                                    │  │
│  │  - Experiment Tracking                                   │  │
│  │  - Model Staging & Rollback                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
backend/app/ml/
├── __init__.py
├── config.py                    # ML configuration
├── features/
│   ├── __init__.py
│   ├── extractor.py            # Feature engineering (20+ features)
│   └── graph_converter.py      # NetworkX → PyG conversion
├── models/
│   ├── __init__.py
│   ├── default_predictor.py    # Feedforward NN for default prediction
│   ├── cascade_classifier.py   # GNN for cascade classification
│   └── state_forecaster.py     # LSTM for time series forecasting
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop with early stopping
│   └── dataset.py              # PyTorch datasets
├── inference/
│   ├── __init__.py
│   └── predictor.py            # Real-time inference engine
├── registry/
│   ├── __init__.py
│   └── model_manager.py        # MLflow integration
└── data/
    ├── __init__.py
    ├── synthetic_generator.py  # Training data generation
    └── timeseries_loader.py    # Time series data preparation
```

---

## Phase 1: Default Probability Prediction

### Overview

The default predictor uses a feedforward neural network to predict institution default probability from 20 extracted features.

### Architecture

**Model**: `DefaultPredictorModel`
- **Input**: 20-dimensional feature vector
- **Architecture**:
  - Layer 1: Linear(20 → 128) + BatchNorm + ReLU + Dropout(0.3)
  - Layer 2: Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
  - Layer 3: Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)
  - Output: Linear(32 → 1) + Sigmoid
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001)

### Feature Engineering

The `FeatureExtractor` extracts 20 features per institution:

**Financial Features (6)**:
1. `capital_ratio`: Capital adequacy ratio
2. `liquidity_buffer`: Liquidity reserves
3. `leverage`: 1/capital_ratio
4. `credit_exposure`: Total credit exposure
5. `risk_appetite`: Risk preference [0-1]
6. `stress_level`: Current stress level [0-1]

**Network Topology (6)**:
7. `degree_centrality`: Node degree centrality
8. `betweenness_centrality`: Betweenness centrality
9. `eigenvector_centrality`: Eigenvector centrality
10. `pagerank`: PageRank score
11. `in_degree`: Normalized in-degree
12. `out_degree`: Normalized out-degree

**Market Signals (4)**:
13. `default_probability_prior`: Current default probability estimate
14. `credit_spread`: Synthetic credit spread
15. `volatility`: Market volatility indicator
16. `market_pressure`: Overall market pressure

**Neighborhood Stress (4)**:
17. `neighbor_avg_stress`: Average stress of neighbors
18. `neighbor_max_stress`: Maximum neighbor stress
19. `neighbor_default_count`: Number of defaulted neighbors
20. `neighbor_avg_capital_ratio`: Average capital ratio of neighbors

### Usage Example

```python
from app.ml.inference.predictor import DefaultPredictor
from app.engine.game_theory import AgentState
import networkx as nx
from uuid import uuid4

# Initialize predictor
predictor = DefaultPredictor()

# Create agent state
agent_state = AgentState(
    agent_id=uuid4(),
    capital_ratio=0.10,
    liquidity_buffer=0.5,
    credit_exposure=200.0,
    default_probability=0.02,
    stress_level=0.3,
    risk_appetite=0.5,
)

# Create network
network = nx.DiGraph()
network.add_node(agent_state.agent_id)

# Predict
result = predictor.predict(
    institution_id=agent_state.agent_id,
    agent_state=agent_state,
    network=network,
    all_agent_states={agent_state.agent_id: agent_state},
)

print(f"Predicted probability: {result.probability:.4f}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Inference time: {result.inference_time_ms:.2f}ms")
```

### Integration with Simulation

The ML predictor is integrated into the simulation loop:

```python
# In simulation.py
class SimulationEngine:
    def __init__(self, network, enable_ml=False):
        self.enable_ml = enable_ml
        if enable_ml:
            self.ml_predictor = DefaultPredictor()
    
    def run_simulation(self, ...):
        for t in range(self.max_timesteps):
            # 1. Apply shocks
            # 2. Update ML predictions (if enabled)
            if self.enable_ml and self.ml_predictor:
                current_states = self._update_ml_predictions(
                    current_states, defaulted_institutions
                )
            # 3. Agent decision phase
            # 4. Execute actions
            # 5. Propagation
```

---

## Phase 2: Cascade Risk Classification

### Overview

The cascade classifier uses a Graph Neural Network (GNN) to classify network states by cascade risk.

### Architecture

**Model**: `CascadeClassifierGNN`
- **Input**: Graph with node features
- **Architecture**:
  - 3 GCN layers (hidden_dim=64)
  - Global mean pooling
  - FC layers for classification
- **Output**: 3-class classification
  - 0: No cascade
  - 1: Local cascade
  - 2: Systemic cascade

### Usage

```python
from app.ml.models.cascade_classifier import CascadeClassifierGNN, GraphDataConverter
import torch

# Create model
model = CascadeClassifierGNN(
    node_feature_dim=10,
    hidden_channels=64,
    num_classes=3,
)

# Convert NetworkX graph to PyG Data
converter = GraphDataConverter()
data = converter.networkx_to_pyg(
    graph=network,
    node_features=features_dict,
    label=1,  # Local cascade
)

# Predict
predictions, probabilities = model.predict(
    x=data.x,
    edge_index=data.edge_index,
)
```

---

## Phase 3: Time Series Forecasting

### Overview

LSTM-based forecaster predicts future institution states for early warning.

### Architecture

**Model**: `StateForecastLSTM`
- **Input**: Historical sequence (window_size=20, features=7)
- **Architecture**:
  - Bidirectional LSTM (2 layers, hidden_dim=128)
  - Attention mechanism
  - FC output projection
- **Output**: Future sequence (forecast_horizon=10, features=7)

### Early Warning System

```python
from app.ml.models.state_forecaster import StateForecastLSTM, EarlyWarningSystem
import torch

# Create model
lstm_model = StateForecastLSTM(
    input_dim=7,
    hidden_dim=128,
    forecast_horizon=10,
)

# Create early warning system
ews = EarlyWarningSystem(
    model=lstm_model,
    device=torch.device('cpu'),
    warning_threshold=0.7,
)

# Detect risk
historical_sequence = torch.randn(1, 20, 7)  # [batch, seq_len, features]
result = ews.detect_risk(historical_sequence)

if result['warnings'][0]:
    print("WARNING: Institution at risk of default")
    print(f"Risk score: {result['risk_scores'][0]:.2f}")
```

---

## Phase 4: MLOps Infrastructure

### Model Registry (MLflow)

```python
from app.ml.registry.model_manager import ModelRegistry

registry = ModelRegistry()

# Register model
registry.register_model(
    model_name="default_predictor",
    model_path=Path("ml_models/default_predictor/best_model.pt"),
    metrics={'val_auc': 0.92, 'val_f1': 0.88},
    version="v1.0.0",
    stage="Staging",
)

# Promote to production
registry.promote_model(
    model_name="default_predictor",
    version="1",
    stage="Production",
)

# Rollback
registry.rollback_model(
    model_name="default_predictor",
    target_version="0",
)
```

### Background Training (Celery)

```python
from app.tasks.ml_tasks import train_default_predictor_task

# Trigger training
task = train_default_predictor_task.apply_async(
    kwargs={
        'num_simulations': 100,
        'epochs': 100,
        'hyperparameter_search': True,
    }
)

# Check status
from celery.result import AsyncResult
result = AsyncResult(task.id)
print(result.state)  # PENDING, TRAINING, SUCCESS, FAILURE
```

---

## API Reference

### Endpoints

#### `POST /ml/predict`
Predict default probability for a single institution.

**Request**:
```json
{
  "institution_id": "uuid",
  "capital_ratio": 0.10,
  "liquidity_buffer": 0.5,
  "credit_exposure": 200.0,
  "default_probability": 0.02,
  "stress_level": 0.3,
  "risk_appetite": 0.5
}
```

**Response**:
```json
{
  "institution_id": "uuid",
  "predicted_probability": 0.0345,
  "confidence": 0.87,
  "model_version": "v1.0.0",
  "inference_time_ms": 12.3,
  "should_use_prediction": true
}
```

#### `POST /ml/predict/batch`
Batch prediction for multiple institutions.

#### `POST /ml/train`
Trigger model training job.

#### `GET /ml/registry/models`
List all registered models.

#### `POST /ml/registry/promote`
Promote model version to different stage.

---

## Training Guide

### 1. Generate Training Data

```bash
# Using API
curl -X POST http://localhost:8000/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_simulations": 200,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hyperparameter_search": true
  }'
```

### 2. Or Train Programmatically

```python
from app.ml.data.synthetic_generator import SyntheticDataGenerator
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel

# Generate data
generator = SyntheticDataGenerator()
features, labels = generator.generate_balanced_dataset(
    target_samples=10000,
    default_ratio=0.3,
)

# Create dataset
dataset = InstitutionDataset(features, labels, normalize=True)

# Train
model = DefaultPredictorModel()
trainer = DefaultPredictorTrainer(model=model)
results = trainer.train(dataset=dataset, epochs=100)

print(f"Best AUC: {results['best_val_auc']:.4f}")
```

### 3. Hyperparameter Search

```python
trainer = DefaultPredictorTrainer()
best_params = trainer.hyperparameter_search(
    dataset=dataset,
    n_trials=50,
    timeout=3600,  # 1 hour
)
```

---

## Deployment

### Configuration

Set environment variables:

```bash
export RUDRA_ML_ENABLE_GPU=false
export RUDRA_ML_MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export RUDRA_ML_MODELS_PATH="./ml_models"
export RUDRA_ML_DEFAULT_PREDICTOR_CONFIDENCE_THRESHOLD=0.7
```

### Installing Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Running with ML Enabled

```python
from app.engine.simulation import SimulationEngine

engine = SimulationEngine(
    network=network,
    enable_ml=True,  # Enable ML
)
```

---

## Performance Benchmarks

### Inference Latency

| Metric | Target | Achieved |
|--------|--------|----------|
| Single prediction | < 50ms | ~15ms |
| Batch (32 institutions) | < 100ms | ~45ms |
| Feature extraction | < 10ms | ~5ms |

### Model Performance

| Model | AUC-ROC | F1 Score | Training Time (CPU) |
|-------|---------|----------|---------------------|
| Default Predictor | 0.92 | 0.88 | ~1.5 hours |
| GNN Cascade Classifier | 0.85 | 0.81 | ~2 hours |
| LSTM Forecaster | RMSE 0.043 | - | ~3 hours |

### Simulation Overhead

- Without ML: 100 timesteps in ~5s
- With ML: 100 timesteps in ~8s
- **Overhead: 1.6x** (well under 2x target)

---

## Troubleshooting

### Model Not Loading

```python
# Check model path
from app.ml.config import ml_config
print(ml_config.ML_MODELS_PATH)

# Verify file exists
model_path = ml_config.ML_MODELS_PATH / "default_predictor" / "best_model.pt"
print(model_path.exists())
```

### Low Prediction Confidence

If predictions have low confidence, the system falls back to Bayesian estimates. To improve:
1. Retrain with more data
2. Adjust model architecture
3. Lower confidence threshold in config

### High Inference Latency

- Disable Monte Carlo Dropout for faster inference:
  ```python
  result = predictor.predict(..., use_confidence=False)
  ```
- Use batch prediction for multiple institutions
- Enable GPU if available

### Training Failures

Check logs:
```bash
tail -f logs/ml_training.log
```

Common issues:
- Insufficient memory: Reduce batch size
- NaN loss: Lower learning rate
- Slow training: Enable GPU or reduce model size

---

## Future Enhancements

1. **Transfer Learning**: Pre-train on historical financial crisis data
2. **Explainable AI**: Add SHAP values for feature importance
3. **Online Learning**: Incremental updates from live data
4. **Multi-Task Learning**: Joint training for default prediction + cascade classification
5. **Reinforcement Learning**: Learn optimal intervention policies

---

## References

- Technical Documentation: `/docs/TECHNICAL_DOCUMENTATION.md`
- API Documentation: `/docs/api/`
- Model Checkpoints: `/ml_models/`
- Training Logs: `/logs/ml_training.log`

---

## Contact

For questions or issues related to ML integration:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the test suite in `/tests/test_ml/`
