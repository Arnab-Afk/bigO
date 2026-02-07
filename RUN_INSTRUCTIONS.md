# Running the CCP Pipeline - Complete Guide

## Prerequisites

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- PyTorch (ML framework)
- NetworkX (graph analysis)
- NumPy, Pandas (data processing)
- scikit-learn (ML utilities)
- scipy (spectral analysis)
- **yfinance (Yahoo Finance API)** ⭐

### 2. Verify Data Files

```bash
ls -la app/ml/data/*.csv
```

You should see all 7 CSV files.

---

## Running Options

### **Option 1: Full Pipeline (Recommended) 🚀**

Run everything end-to-end with real Yahoo Finance market data:

```bash
python scripts/ccp_pipeline.py --full-pipeline --save-report
```

**What it does:**
1. ✅ Loads all 7 datasets (including RBI data)
2. ✅ Trains ML model for default prediction
3. ✅ Builds 3-channel network:
   - Sector channel from RBI exposure data
   - Liquidity channel from maturity profiles
   - **Market channel from Yahoo Finance stock prices** (real-time!)
4. ✅ Performs spectral analysis for systemic risk
5. ✅ Generates CCP risk report

**Output:**
- `ccp_risk_report.txt` - Full risk assessment report
- `ml_models/ccp_default_predictor/` - Trained model files
- Console logs with detailed metrics

**Expected Runtime:** 3-7 minutes (includes Yahoo Finance API calls)

---

### **Option 2: Quick Training Only**

Just train the ML model without network analysis:

```bash
python scripts/ccp_pipeline.py --train
```

Runtime: ~2 minutes

---

### **Option 3: Network Analysis Only**

Skip training, run network and risk analysis:

```bash
python scripts/ccp_pipeline.py --analyze
```

This will:
- Fetch Yahoo Finance data for Indian banks
- Build composite network
- Run spectral analysis
- Generate risk report

Runtime: ~2 minutes

---

### **Option 4: Alternative Training Script**

Train with detailed inspection:

```bash
python scripts/train_rbi_data.py --epochs 150
```

---

## Yahoo Finance Integration

### How It Works

The pipeline automatically:

1. **Maps bank names to ticker symbols**
   - State Bank of India → `SBIN.NS`
   - HDFC Bank → `HDFCBANK.NS`
   - ICICI Bank → `ICICIBANK.NS`
   - Axis Bank → `AXISBANK.NS`
   - And 30+ more Indian banks

2. **Fetches historical price data**
   - Period: Last 1 year
   - Frequency: Daily
   - Exchange: NSE (National Stock Exchange of India)

3. **Computes return correlations**
   - Daily returns calculated
   - Correlation matrix built
   - Used as market channel in network

4. **Automatic fallback**
   - If Yahoo Finance fails → uses synthetic data
   - No disruption to pipeline

### Supported Banks

**Public Sector (PSU):**
- State Bank of India (SBIN.NS)
- Bank of Baroda (BANKBARODA.NS)
- Punjab National Bank (PNB.NS)
- Canara Bank (CANBK.NS)
- Union Bank (UNIONBANK.NS)
- Bank of India (BANKINDIA.NS)
- And 6 more...

**Private Sector:**
- HDFC Bank (HDFCBANK.NS)
- ICICI Bank (ICICIBANK.NS)
- Axis Bank (AXISBANK.NS)
- Kotak Mahindra (KOTAKBANK.NS)
- IndusInd Bank (INDUSINDBK.NS)
- Federal Bank (FEDERALBNK.NS)
- And 12 more...

**Small Finance Banks:**
- AU Small Finance (AUBANK.NS)
- Equitas (EQUITASBNK.NS)
- Ujjivan (UJJIVANSFB.NS)

### Sample Output

```
Building market channel for 30 banks
Fetching real market data from Yahoo Finance...
Fetching SBIN.NS for STATE BANK OF INDIA
✓ Successfully fetched 252 returns for STATE BANK OF INDIA
Fetching HDFCBANK.NS for HDFC BANK LTD.
✓ Successfully fetched 252 returns for HDFC BANK LTD.
Fetching ICICIBANK.NS for ICICI BANK LIMITED
✓ Successfully fetched 252 returns for ICICI BANK LIMITED
...
Successfully fetched market data for 25/30 banks
Market channel built: 30 banks, mean correlation: 0.4523
```

---

## Expected Output

### Console Output

```
======================================================================
CCP-CENTRIC RISK MODELING PIPELINE
======================================================================

LAYER 0: DATA INTEGRATION
----------------------------------------------------------------------
✓ Loaded ML features: (73, 28)
✓ Loaded CRAR data: (1700, 12)
✓ Loaded peer ratios: (618, 10)
✓ Loaded NPA movements: (1991, 10)
✓ Loaded sector exposures: (2008, 6)
✓ Loaded maturity profiles: (1988, 56)

LAYER 1: PARTICIPANT RISK ESTIMATION (ML)
----------------------------------------------------------------------
Training set: 62 samples
Test set: 11 samples
Training...
Epoch [50/100] Loss: 0.3245 | Val Acc: 0.8500
✓ Training complete
Test Accuracy: 0.9000
Test ROC AUC: 0.9234

LAYER 2: INTERDEPENDENCE NETWORK CONSTRUCTION
----------------------------------------------------------------------
Building sector channel...
  Sector channel built: 42 banks

Building liquidity channel...
  Liquidity channel built: 42 banks

Building market channel...
  Fetching Yahoo Finance data...
  ✓ Fetched data for 35/42 banks
  Market channel built: 42 banks

Composite network built: 42 nodes, 487 edges
Mean edge weight: 0.3421

LAYER 3: SYSTEMIC FRAGILITY QUANTIFICATION
----------------------------------------------------------------------
Spectral Analysis:
  Largest Eigenvalue (λ_max): 6.8234
  Spectral Gap: 3.2145
  System Fragility: 58.3%
  Risk Level: HIGH

Top 5 Systemically Important:
  1. STATE BANK OF INDIA: 0.8912
  2. HDFC BANK LTD.: 0.8456
  3. ICICI BANK LIMITED: 0.8123
  4. AXIS BANK LIMITED: 0.7834
  5. PUNJAB NATIONAL BANK: 0.7521

LAYER 4: CCP LOSS ABSORPTION & POLICY RESPONSE
----------------------------------------------------------------------
Assessing 42 clearing members...

System Risk Level: HIGH
Required Default Fund: $2,345,678,901
Coverage Ratio: 1.15x
Status: ✓ ADEQUATE

Risk Tier Distribution:
  Tier 1 (Low):      15 members
  Tier 2 (Moderate): 18 members
  Tier 3 (High):      7 members
  Tier 4 (Critical):  2 members

✓ Report saved to ccp_risk_report.txt

======================================================================
PIPELINE EXECUTION COMPLETE
======================================================================
```

### Report File (ccp_risk_report.txt)

```
================================================================================
                        CCP RISK MANAGEMENT REPORT                              
================================================================================

SYSTEM RISK OVERVIEW
--------------------------------------------------------------------------------
System Risk Level:           HIGH
System Fragility Index:      58.3%
Average Default Probability: 22.4%
Total CCP Exposure:          $10,000,000,000

DEFAULT FUND ADEQUACY
--------------------------------------------------------------------------------
Required Default Fund:       $2,345,678,901
Coverage Ratio:              1.15x
Status:                      ✓ ADEQUATE

MEMBER RISK SUMMARY
--------------------------------------------------------------------------------
Tier 1 (Low Risk)        :  15 members
Tier 2 (Moderate Risk)   :  18 members
Tier 3 (High Risk)       :   7 members
Tier 4 (Critical Risk)   :   2 members

TOP 10 HIGHEST RISK MEMBERS
--------------------------------------------------------------------------------
Rank   Member                          PD    SI    Tier      Action
--------------------------------------------------------------------------------
1      PUNJAB NATIONAL BANK          35.2% 0.75  Tier 3    Enhanced monitoring
2      BANK OF BARODA                28.3% 0.68  Tier 2    Standard monitoring
3      UNION BANK OF INDIA           26.1% 0.64  Tier 2    Standard monitoring
...

PREVENTIVE INTERVENTIONS
--------------------------------------------------------------------------------
1. ⚠️ Increase margin surveillance frequency
2. ⚠️ Run daily stress tests
3. ⚠️ Review member concentration limits
4. FOCUS: 5 systemically important members require enhanced oversight

RECOMMENDED STRESS TEST SCENARIOS
--------------------------------------------------------------------------------
• Scenario 1: Top 2 largest members default simultaneously
• Scenario 2: Sector-wide shock (e.g., real estate crisis)
• Scenario 3: Liquidity squeeze (40% haircut on collateral)
• Scenario 4: Top 3 systemically important members default
• Scenario 5: Cascade event from single critical member

================================================================================
```

---

## Troubleshooting

### Issue: yfinance not found

```bash
pip install yfinance
```

### Issue: No market data fetched

The pipeline will automatically fall back to synthetic data. Check:
- Internet connection
- Yahoo Finance service status
- Ticker symbol mappings in code

### Issue: SSL/Certificate errors

```bash
pip install --upgrade certifi
```

### Issue: Module import errors

```bash
pip install -r requirements.txt --upgrade
```

---

## Advanced Usage

### Custom Market Data Period

Modify in `network_builder.py`:

```python
returns_df = self._fetch_yahoo_finance_data(
    bank_names, 
    period="2y",      # 2 years instead of 1
    interval="1d"     # Daily data
)
```

### Disable Yahoo Finance (use synthetic only)

```python
network = builder.build_composite_network(
    centrality_scores=centrality_dict,
    threshold=0.1,
    use_synthetic=True  # Force synthetic data
)
```

### Test Yahoo Finance Integration

```bash
python -c "
from app.ml.data.network_builder import CompositeNetworkBuilder
builder = CompositeNetworkBuilder('app/ml/data')
banks = ['STATE BANK OF INDIA', 'HDFC BANK LTD.', 'ICICI BANK LIMITED']
bank_names, corr_matrix = builder.build_market_channel(banks, use_synthetic=False)
print(f'Fetched data for {len(bank_names)} banks')
print(f'Correlation matrix shape: {corr_matrix.shape}')
"
```

---

## Quick Commands Reference

```bash
# Full pipeline with real Yahoo Finance data
python scripts/ccp_pipeline.py --full-pipeline --save-report

# View the report
cat ccp_risk_report.txt

# Check if yfinance is installed
python -c "import yfinance; print('✓ yfinance installed')"

# Test individual components
python -c "from app.ml.data.integration_pipeline import DataIntegrationPipeline; print('✓ Data pipeline OK')"
python -c "from app.ml.data.network_builder import CompositeNetworkBuilder; print('✓ Network builder OK')"
python -c "from app.ml.analysis.spectral import SpectralAnalyzer; print('✓ Spectral analysis OK')"
python -c "from app.ml.ccp.risk_manager import CCPRiskManager; print('✓ CCP risk manager OK')"
```

---

## Summary

**One-Command Execution:**

```bash
cd /Users/omlanke/Documents/Programming/GitHub/rudra-datathon/backend && \
pip install yfinance && \
python scripts/ccp_pipeline.py --full-pipeline --save-report && \
echo "✅ Complete! Check ccp_risk_report.txt"
```

This will:
1. Install yfinance
2. Run full CCP pipeline with real market data
3. Generate comprehensive risk report
4. Save all artifacts

**Result:** Production-ready CCP risk assessment with live market data! 🎉
