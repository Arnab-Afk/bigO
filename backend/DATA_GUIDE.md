# Data Guide for RUDRA ML Training

## Quick Start: No External Data Needed! ✅

The system generates synthetic training data automatically:

```bash
cd backend
python scripts/train_ml_model.py --simulations 100
```

This creates ~5,000 samples by running financial network simulations.

---

## Option 1: Use Synthetic Data (Recommended to Start)

### Advantages
- ✅ No data collection needed
- ✅ Immediate training
- ✅ Controlled scenarios
- ✅ Perfect for testing and development

### How it Works
1. Creates synthetic financial networks (20 institutions)
2. Runs simulations with shocks (defaults, liquidity freezes, etc.)
3. Extracts 20 features per institution per timestep
4. Labels defaults automatically

### Commands

**Basic training:**
```bash
python scripts/train_ml_model.py --simulations 100
```

**With hyperparameter search:**
```bash
python scripts/train_ml_model.py --simulations 150 --hyperparameter-search
```

**Register to MLflow:**
```bash
python scripts/train_ml_model.py --simulations 100 --register --version v1.0.0
```

---

## Option 2: Load Real Financial Data

### Required Features (20 total)

The model expects these features per institution:

| Category | Features |
|----------|----------|
| **Financial (6)** | capital_ratio, liquidity_buffer, leverage, credit_exposure, risk_appetite, stress_level |
| **Network (6)** | degree_centrality, betweenness_centrality, eigenvector_centrality, pagerank, in_degree, out_degree |
| **Market (4)** | default_probability_prior, credit_spread, volatility, market_pressure |
| **Neighborhood (4)** | neighbor_avg_stress, neighbor_max_stress, neighbor_default_count, neighbor_avg_capital_ratio |

### Data Format Options

#### Format 1: Single CSV File

**Structure:**
```csv
institution_id,timestamp,capital_ratio,liquidity_buffer,leverage,credit_exposure,risk_appetite,stress_level,defaulted
bank_001,2024-01-01,0.12,0.25,8.3,5000,0.6,0.3,0
bank_001,2024-01-02,0.11,0.23,9.1,5200,0.65,0.35,0
bank_002,2024-01-01,0.08,0.15,12.5,8000,0.8,0.7,1
```

**Load and train:**
```bash
python scripts/train_with_real_data.py --csv data/my_data.csv
```

#### Format 2: Network Files (3 separate CSVs)

**institutions.csv:**
```csv
institution_id,name,type,total_assets
bank_001,First National,commercial,10000000
bank_002,State Bank,regional,5000000
```

**exposures.csv:**
```csv
source_id,target_id,exposure_amount
bank_001,bank_002,500000
bank_002,bank_003,300000
```

**states.csv:**
```csv
institution_id,timestamp,capital_ratio,liquidity_buffer,...,defaulted
bank_001,2024-01-01,0.12,0.25,...,0
```

**Load and train:**
```bash
python scripts/train_with_real_data.py --network-data \
    --institutions data/institutions.csv \
    --exposures data/exposures.csv \
    --states data/states.csv
```

---

## Where to Get Real Data

### Public Sources (Free)

#### 1. Federal Reserve Economic Data (FRED)
- **URL:** https://fred.stlouisfed.org/
- **Coverage:** US banking sector
- **Data Types:**
  - Bank equity to total assets (EQTA)
  - Net interest margin (USNIM)
  - Non-performing loans (DDOI08USA156NWDB)
  - Capital adequacy ratios

**Example - Load from FRED:**
```python
from app.ml.data.real_data_loader import RealDataLoader

loader = RealDataLoader()
df = loader.load_from_fred(
    series_ids=['EQTA', 'USNIM', 'DDOI08USA156NWDB'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

#### 2. FFIEC Call Reports
- **URL:** https://cdr.ffiec.gov/public/
- **Coverage:** US banks quarterly reports
- **Data:** Balance sheets, income statements, risk metrics

#### 3. ECB Statistical Data Warehouse
- **URL:** https://sdw.ecb.europa.eu/
- **Coverage:** European banking system
- **Data:** Capital ratios, NPLs, liquidity metrics

#### 4. Bank for International Settlements (BIS)
- **URL:** https://www.bis.org/statistics/
- **Coverage:** Global banking statistics
- **Data:** Cross-border exposures, systemic risk indicators

#### 5. Academic Datasets
- **Federal Reserve Bank of New York:** Systemic risk datasets
- **V-Lab (NYU Stern):** Real-time systemic risk measures
- **Research paper replication data:** Often available on journal websites

### Commercial Sources (Paid)

1. **Bloomberg Terminal** - Comprehensive financial data
2. **S&P Capital IQ** - Bank financials and ratings
3. **Moody's Analytics** - Credit risk data
4. **Refinitiv** - Market and credit data
5. **BankFocus (Bureau van Dijk)** - Global bank financials

---

## How to Load Your Data

### Method 1: Simple CSV

1. **Prepare your CSV** with at least:
   - `institution_id` column
   - `defaulted` column (0 or 1)
   - Numeric feature columns

2. **Train:**
```bash
python scripts/train_with_real_data.py --csv your_data.csv
```

### Method 2: Custom Loading (Python API)

```python
from app.ml.data.real_data_loader import RealDataLoader
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel

# Load data
loader = RealDataLoader()
features, labels = loader.load_from_csv(
    "your_data.csv",
    institution_id_col="bank_id",
    label_col="failed",
    timestamp_col="date"
)

# Create dataset
dataset = InstitutionDataset(features, labels, normalize=True)

# Train
model = DefaultPredictorModel()
trainer = DefaultPredictorTrainer(model)
results = trainer.train(dataset, epochs=100)

print(f"Best AUC: {results['best_val_auc']:.4f}")
```

### Method 3: Test with Sample Data

```bash
# Generate 1000 sample records
python scripts/train_with_real_data.py --create-sample --csv test_data.csv

# Train on sample data
python scripts/train_with_real_data.py --csv test_data.csv --epochs 50
```

---

## Data Quality Requirements

### Minimum Requirements
- ✅ At least **500 samples** (preferably 5,000+)
- ✅ Default ratio between **5-40%** (avoid extreme imbalance)
- ✅ No missing values in critical features
- ✅ Numeric features normalized or on reasonable scales

### Recommended Data Size

| Dataset Size | Use Case |
|--------------|----------|
| 500-1,000 | Quick testing, proof of concept |
| 1,000-5,000 | Development, initial models |
| 5,000-20,000 | Production-ready models |
| 20,000+ | High-performance, robust models |

### Handling Missing Data

If your data has missing values:

```python
import pandas as pd

# Load with pandas
df = pd.read_csv("your_data.csv")

# Fill missing values
df = df.fillna({
    'capital_ratio': 0.1,
    'liquidity_buffer': 0.2,
    'stress_level': 0.0,
    # ... etc
})

# Or drop rows with missing values
df = df.dropna()

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
```

---

## Troubleshooting

### "Not enough data"
- Use synthetic data: `--simulations 200` (creates ~10,000 samples)
- Combine synthetic + real data
- Use data augmentation techniques

### "Imbalanced classes"
- Adjust sampling in training script
- Use class weights in loss function
- Generate balanced synthetic data

### "Features don't match 20 expected"
- The loader pads/truncates automatically
- Map your features to the 20 expected features
- Consider feature engineering to create missing features

### "Poor model performance"
- Check data quality and labels
- Increase training data size
- Try hyperparameter search: `--hyperparameter-search`
- Verify feature scaling/normalization

---

## Next Steps

1. **Start with synthetic data:**
   ```bash
   python scripts/train_ml_model.py --simulations 100
   ```

2. **Try sample data generation:**
   ```bash
   python scripts/train_with_real_data.py --create-sample --csv test.csv
   ```

3. **Prepare your real data** (if available)
   - Format as CSV
   - Ensure required columns
   - Clean missing values

4. **Train with real data:**
   ```bash
   python scripts/train_with_real_data.py --csv your_data.csv --register
   ```

5. **Evaluate results:**
   ```bash
   pytest tests/test_ml/ -v
   ```

For questions, see [ML_ARCHITECTURE.md](../docs/ML_ARCHITECTURE.md) or check the examples in `examples/ml_example.py`.
