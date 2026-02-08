# RUDRA Project Setup Guide

## Quick Start (5 Minutes)

This guide will help you run the **CCP ML Dashboard** with the standalone backend.

---

## Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for backend)
- **Terminal** access

---

## Setup Steps

### 1. Backend Setup (CCP ML API)

```bash
# Navigate to backend directory
cd backend/ccp_ml

# Install Python dependencies
pip3 install -r requirements.txt

# Start the API server
python3 api.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend API will be available at: **http://localhost:8000**

**Test it:**
```bash
curl http://localhost:8000/health
```

---

### 2. Frontend Setup (Next.js Dashboard)

Open a **new terminal** window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not done)
npm install

# Start development server
npm run dev
```

**Expected Output:**
```
▲ Next.js 16.1.6
- Local:        http://localhost:3000
- Ready in 2.3s
```

---

### 3. Access the Dashboard

Open your browser to:
**http://localhost:3000/ml-dashboard**

You should see:
- ✅ 72 RBI Banks loaded
- ✅ Network edges count
- ✅ Spectral radius metric
- ✅ Risk distribution

---

## Troubleshooting

### Backend Issues

**Problem: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip3 install -r backend/ccp_ml/requirements.txt
```

**Problem: Port 8000 already in use**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or start on different port
cd backend/ccp_ml
uvicorn api:app --reload --port 8001

# Update frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8001
```

---

### Frontend Issues

**Problem: `NEXT_PUBLIC_API_URL` not found**
- Make sure `.env.local` exists in `/frontend/` directory
- Restart the Next.js dev server after creating `.env.local`

**Problem: API connection errors**
- Verify backend is running: `curl http://localhost:8000/health`
- Check console for CORS errors
- Ensure `.env.local` has correct URL

---

## Project Structure

```
bigO/
├── backend/
│   └── ccp_ml/              # ✅ WORKING - Standalone CCP ML API
│       ├── api.py           # FastAPI application
│       ├── data/            # 7 RBI CSV datasets
│       ├── requirements.txt # Python dependencies
│       └── ...
│
└── frontend/
    ├── src/
    │   ├── app/(main)/ml-dashboard/  # Dashboard pages
    │   ├── hooks/use-ccp-ml.ts       # API integration
    │   └── lib/api/ccp-ml-client.ts  # API client
    ├── .env.local           # Local configuration (YOU JUST CREATED THIS)
    └── package.json
```

---

## Key Features Available

### ✅ Working Now:
- Health check & system status
- 72 RBI banks with real data
- Network analysis (nodes, edges, metrics)
- Spectral analysis (λ_max, Fiedler, contagion index)
- Risk scoring and distribution
- CCP margin calculations
- Real-time simulation engine
- Stress testing (capital, liquidity, market shocks)

### ⚠️ To Be Implemented:
- Individual bank detail pages
- Interactive network graph (D3.js)
- Simulation configuration UI
- Export functionality (CSV, PDF)

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/status` | GET | Current system status |
| `/simulate` | POST | Run complete simulation |
| `/network` | GET | Get network data |
| `/network/nodes` | GET | Get all banks |
| `/network/edges` | GET | Get connections |
| `/risk/scores` | GET | Get all risk scores |
| `/risk/bank` | POST | Get single bank risk |
| `/spectral` | GET | Spectral analysis |
| `/margins` | GET | Margin requirements |
| `/stress-test` | POST | Run stress test |
| `/realtime/init` | POST | Initialize real-time sim |
| `/realtime/step` | POST | Step through simulation |

**View Full API Docs:**
http://localhost:8000/docs (when backend is running)

---

## Development Workflow

### Starting Work:
```bash
# Terminal 1 - Backend
cd backend/ccp_ml && python3 api.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### Making Changes:
- **Backend**: Edit files in `backend/ccp_ml/`, API auto-reloads
- **Frontend**: Edit files in `frontend/src/`, hot-reload automatic

### Testing:
```bash
# Test backend
cd backend/ccp_ml
python3 test_api.py

# Test frontend API calls
# Open browser console at http://localhost:3000/ml-dashboard
```

---

## Next Steps

1. **Explore the Dashboard**: http://localhost:3000/ml-dashboard
2. **Try Features**:
   - Click "Run Simulation" button
   - Apply stress tests (Capital Shock, Liquidity Squeeze)
   - View spectral analysis metrics
3. **Check API Docs**: http://localhost:8000/docs
4. **Build Missing Pages**: See `IMPLEMENTATION_PLAN.md` for task list

---

## Support

- **Documentation**: See `RUN_INSTRUCTIONS.md`, `IMPLEMENTATION_PLAN.md`
- **API Guide**: `backend/ccp_ml/README.md`
- **Frontend Guide**: `frontend/QUICKSTART.md`, `frontend/INTEGRATION_GUIDE.md`

---

## Quick Commands

```bash
# Start backend
cd backend/ccp_ml && python3 api.py

# Start frontend
cd frontend && npm run dev

# Install backend deps
pip3 install -r backend/ccp_ml/requirements.txt

# Install frontend deps
cd frontend && npm install

# Test backend
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

**Ready to go! 🚀**
