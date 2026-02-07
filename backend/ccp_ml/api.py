"""
CCP Risk Simulation API

Real-time backend for CCP risk analysis and network simulation.
Provides REST endpoints for running simulations, stress tests, and retrieving results.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict
import asyncio

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import CCP ML components
from ccp_ml import (
    DataLoader, FeatureEngineer, NetworkBuilder,
    SpectralAnalyzer, CCPRiskModel, CCPEngine
)

# Import new modules
try:
    from .realtime_simulation import RealtimeSimulation
    from .graph_generator import GraphGenerator
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeSimulation = None
    GraphGenerator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="CCP Risk Simulation API",
    description="Real-time Central Counterparty risk analysis and network simulation",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global State (Simulation Cache)
# ============================================

class SimulationState:
    """Holds the current simulation state"""
    def __init__(self):
        self.data = None
        self.features = None
        self.network_builder = None
        self.spectral_analyzer = None
        self.risk_model = None
        self.ccp_engine = None
        self.last_run = None
        self.is_initialized = False
        # New realtime components
        self.realtime_sim = None
        self.graph_generator = None
        self.websocket_clients: List[WebSocket] = []

state = SimulationState()

# ============================================
# Pydantic Models
# ============================================

class SimulationConfig(BaseModel):
    year: Optional[int] = Field(None, description="Target year for analysis")
    sector_weight: float = Field(0.4, ge=0, le=1)
    liquidity_weight: float = Field(0.4, ge=0, le=1)
    market_weight: float = Field(0.2, ge=0, le=1)
    edge_threshold: float = Field(0.05, ge=0, le=1)

class StressTestConfig(BaseModel):
    shock_magnitude: float = Field(0.2, ge=0, le=1, description="Shock magnitude (0-1)")
    target_banks: Optional[List[str]] = Field(None, description="Banks to shock")
    shock_type: str = Field("capital", description="Type: capital, liquidity, or market")

class BankQuery(BaseModel):
    bank_name: str

class RealtimeSimConfig(BaseModel):
    max_timesteps: int = Field(100, ge=1, le=1000)
    shock_config: Optional[Dict] = None
    
class SimStepCommand(BaseModel):
    n_steps: int = Field(1, ge=1, le=100)
    shock_config: Optional[Dict] = None

class GraphRequest(BaseModel):
    graph_type: str = Field(..., description="Type: network, risk_distribution, time_series, spectral")
    format: str = Field("plotly", description="Format: plotly or matplotlib")
    highlight_nodes: Optional[List[str]] = None

# ============================================
# Initialization
# ============================================

def initialize_simulation():
    """Initialize simulation components with data"""
    logger.info("Initializing CCP simulation...")
    
    # Load data
    loader = DataLoader()
    state.data = loader.load_all()
    
    # Feature engineering
    engineer = FeatureEngineer(normalize=True)
    state.features = engineer.create_features(state.data)
    
    # Network builder
    state.network_builder = NetworkBuilder(
        sector_weight=0.4,
        liquidity_weight=0.4,
        market_weight=0.2,
        edge_threshold=0.05
    )
    state.network_builder.build_network(state.data)
    
    # Spectral analyzer
    state.spectral_analyzer = SpectralAnalyzer()
    
    # Risk model
    state.risk_model = CCPRiskModel()
    
    # CCP engine
    state.ccp_engine = CCPEngine(
        risk_model=state.risk_model,
        network_builder=state.network_builder,
        spectral_analyzer=state.spectral_analyzer
    )
    
    # Initialize realtime simulation
    if REALTIME_AVAILABLE and RealtimeSimulation:
        state.realtime_sim = RealtimeSimulation(
            loader, engineer, state.network_builder,
            state.spectral_analyzer, state.risk_model, state.ccp_engine
        )
        state.realtime_sim.initialize(state.features)
        logger.info("Realtime simulation initialized")
    
    # Initialize graph generator
    if REALTIME_AVAILABLE and GraphGenerator:
        state.graph_generator = GraphGenerator()
        logger.info("Graph generator initialized")
    
    state.is_initialized = True
    state.last_run = datetime.now()
    
    logger.info(f"Initialization complete: {len(state.data.banks)} banks loaded")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    initialize_simulation()

# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online" if state.is_initialized else "initializing",
        "service": "CCP Risk Simulation API",
        "initialized": state.is_initialized,
        "last_run": state.last_run.isoformat() if state.last_run else None,
        "version": "1.0.0",
        "features": {
            "realtime_simulation": state.realtime_sim is not None,
            "graph_generation": state.graph_generator is not None,
            "websocket": True
        }
    }

@app.get("/api/status")
async def get_status():
    """Get simulation status"""
    if not state.is_initialized:
        return {"status": "not_initialized", "initialized": False}
    
    num_banks = len(state.data.banks) if state.data else 0
    num_edges = len(state.network_builder.edges) if state.network_builder else 0
    
    return {
        "status": "ready",
        "initialized": True,
        "num_banks": num_banks,
        "n_banks": num_banks,  # Alias for compatibility
        "num_edges": num_edges,
        "n_features": len(state.features.columns) if state.features is not None else 0,
        "n_edges": num_edges,  # Alias for compatibility
        "last_run": state.last_run.isoformat() if state.last_run else None,
        "realtime_available": state.realtime_sim is not None,
        "graph_generator_available": state.graph_generator is not None
    }

@app.post("/api/simulate")
async def run_simulation(config: SimulationConfig = SimulationConfig()):
    """Run full CCP simulation with custom config"""
    try:
        # Rebuild network with new config
        state.network_builder = NetworkBuilder(
            sector_weight=config.sector_weight,
            liquidity_weight=config.liquidity_weight,
            market_weight=config.market_weight,
            edge_threshold=config.edge_threshold
        )
        state.network_builder.build_network(state.data, year=config.year)
        
        # Run spectral analysis
        spectral_results = state.spectral_analyzer.analyze(
            network_builder=state.network_builder
        )
        
        # Run CCP engine
        results = state.ccp_engine.run_full_analysis(
            state.features,
            train=False,
            year=config.year
        )
        
        state.last_run = datetime.now()
        
        return {
            "status": "success",
            "timestamp": state.last_run.isoformat(),
            "config": config.dict(),
            "network": {
                "n_nodes": state.network_builder.graph.number_of_nodes(),
                "n_edges": len(state.network_builder.edges)
            },
            "spectral": {
                "spectral_radius": spectral_results.spectral_radius,
                "fiedler_value": spectral_results.fiedler_value,
                "amplification_risk": spectral_results.amplification_risk,
                "fragmentation_risk": spectral_results.fragmentation_risk,
                "contagion_index": state.spectral_analyzer.compute_contagion_index()
            },
            "ccp": {
                "n_participants": results.get("n_participants"),
                "risk_distribution": results.get("risk_distribution"),
                "margin_summary": results.get("margin_summary"),
                "default_fund": results.get("default_fund")
            },
            "policies": results.get("policies", [])
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/network")
async def get_network():
    """Get network graph data for visualization"""
    if not state.network_builder:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # Get network data
    network_data = state.network_builder.export_to_dict()
    
    # Get metrics
    metrics = state.network_builder.compute_network_metrics()
    if not metrics.empty:
        network_data["metrics"] = metrics.to_dict("records")
    
    return network_data

@app.get("/api/network/nodes")
async def get_nodes():
    """Get all network nodes with metrics"""
    if not state.network_builder:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    metrics = state.network_builder.compute_network_metrics()
    return metrics.to_dict("records") if not metrics.empty else []

@app.get("/api/network/edges")
async def get_edges():
    """Get all network edges"""
    if not state.network_builder:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    return [
        {
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
            "sector_similarity": e.sector_similarity,
            "liquidity_similarity": e.liquidity_similarity,
            "market_correlation": e.market_correlation
        }
        for e in state.network_builder.edges
    ]

@app.get("/api/risk/scores")
async def get_risk_scores():
    """Get risk scores for all banks"""
    if state.features is None:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    # Get risk-related columns
    risk_cols = ["bank_name", "default_probability", "stress_level", "capital_ratio"]
    available_cols = [c for c in risk_cols if c in state.features.columns]
    
    if not available_cols:
        return []
    
    return state.features[available_cols].fillna(0).to_dict("records")

@app.post("/api/risk/bank")
async def get_bank_risk(query: BankQuery):
    """Get detailed risk analysis for a specific bank"""
    if state.features is None:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    bank_data = state.features[
        state.features["bank_name"].str.contains(query.bank_name, case=False, na=False)
    ]
    
    if bank_data.empty:
        raise HTTPException(status_code=404, detail=f"Bank '{query.bank_name}' not found")
    
    # Get network position
    network_metrics = state.network_builder.compute_network_metrics()
    bank_network = network_metrics[
        network_metrics["bank_name"].str.contains(query.bank_name, case=False, na=False)
    ]
    
    return {
        "bank_name": query.bank_name,
        "features": bank_data.to_dict("records")[0],
        "network_position": bank_network.to_dict("records")[0] if not bank_network.empty else None
    }

@app.get("/api/spectral")
async def get_spectral_analysis():
    """Get spectral analysis results"""
    if not state.spectral_analyzer:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    results = state.spectral_analyzer.analyze(network_builder=state.network_builder)
    
    return {
        "spectral_radius": results.spectral_radius,
        "fiedler_value": results.fiedler_value,
        "spectral_gap": results.spectral_gap,
        "effective_rank": results.effective_rank,
        "eigenvalue_entropy": results.eigenvalue_entropy,
        "amplification_risk": results.amplification_risk,
        "fragmentation_risk": results.fragmentation_risk,
        "contagion_index": state.spectral_analyzer.compute_contagion_index()
    }

@app.post("/api/stress-test")
async def run_stress_test(config: StressTestConfig):
    """Run stress test simulation"""
    if not state.is_initialized:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    try:
        # Copy features for stress scenario
        stressed_features = state.features.copy()
        
        # Apply shock based on type
        if config.shock_type == "capital":
            # Reduce capital ratios
            if "capital_ratio" in stressed_features.columns:
                stressed_features["capital_ratio"] *= (1 - config.shock_magnitude)
        
        elif config.shock_type == "liquidity":
            # Increase stress level
            if "stress_level" in stressed_features.columns:
                stressed_features["stress_level"] += config.shock_magnitude
                stressed_features["stress_level"] = stressed_features["stress_level"].clip(0, 1)
        
        elif config.shock_type == "market":
            # Increase market pressure
            if "market_pressure" in stressed_features.columns:
                stressed_features["market_pressure"] += config.shock_magnitude
        
        # Re-run CCP analysis with stressed data
        results = state.ccp_engine.run_full_analysis(stressed_features, train=False)
        
        # Compute impact
        original_fund = state.ccp_engine.results.get("default_fund", {}).get("total_fund", 0) if hasattr(state.ccp_engine, "results") else 0
        stressed_fund = results.get("default_fund", {}).get("total_fund", 0)
        
        return {
            "status": "success",
            "shock_config": config.dict(),
            "baseline": {
                "risk_distribution": state.ccp_engine.results.get("risk_distribution") if hasattr(state.ccp_engine, "results") else None,
                "default_fund": original_fund
            },
            "stressed": {
                "risk_distribution": results.get("risk_distribution"),
                "default_fund": stressed_fund
            },
            "impact": {
                "fund_increase_pct": ((stressed_fund - original_fund) / original_fund * 100) if original_fund > 0 else 0,
                "new_high_risk_count": results.get("risk_distribution", {}).get("high", 0)
            }
        }
    except Exception as e:
        logger.error(f"Stress test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/margins")
async def get_margins():
    """Get margin requirements for all participants using real bank names"""
    if not state.network_builder:
        raise HTTPException(status_code=400, detail="Not initialized")
    
    # Use real bank names from network (sector exposure data)
    network_metrics = state.network_builder.compute_network_metrics()
    
    if network_metrics.empty:
        raise HTTPException(status_code=400, detail="No network data available")
    
    margins = []
    BASE_MARGIN_RATE = 0.02
    MAX_MARGIN_RATE = 0.15
    NETWORK_MARGIN_WEIGHT = 0.3
    
    for _, row in network_metrics.iterrows():
        bank_name = row.get('bank_name', 'Unknown')
        
        # Calculate margin based on network metrics
        pagerank = row.get('pagerank', 0.01)
        degree = row.get('degree_centrality', 0.5)
        eigenvector = row.get('eigenvector_centrality', 0.1)
        
        # Network importance score
        network_importance = min((pagerank * 10 + degree + eigenvector) / 3, 1.0)
        
        # Margin calculation
        base_margin = BASE_MARGIN_RATE * (1 + network_importance)
        network_addon = base_margin * network_importance * NETWORK_MARGIN_WEIGHT
        total_margin = min(base_margin + network_addon, MAX_MARGIN_RATE)
        
        # Explanation
        if network_importance > 0.7:
            explanation = f"High margin due to systemic importance (centrality: {eigenvector:.3f})"
        elif network_importance > 0.4:
            explanation = f"Moderate margin - network score: {network_importance:.2f}"
        else:
            explanation = "Standard margin requirements"
        
        margins.append({
            "bank_name": bank_name,
            "base_margin": round(base_margin, 6),
            "network_addon": round(network_addon, 6),
            "total_margin": round(total_margin, 6),
            "explanation": explanation
        })
    
    # Sort by total margin descending
    margins.sort(key=lambda x: x['total_margin'], reverse=True)
    return margins

@app.get("/api/default-fund")
async def get_default_fund():
    """Get default fund allocation"""
    if not state.ccp_engine:
        raise HTTPException(status_code=400, detail="Run simulation first")
    
    results = state.ccp_engine.run_full_analysis(state.features, train=False)
    return results.get("default_fund", {})

@app.post("/api/reinitialize")
async def reinitialize():
    """Reinitialize simulation with fresh data"""
    try:
        initialize_simulation()
        return {"status": "success", "message": "Simulation reinitialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Real-time Simulation Endpoints
# ============================================

@app.post("/api/realtime/init")
async def init_realtime_simulation(config: RealtimeSimConfig):
    """Initialize real-time simulation"""
    if not state.realtime_sim:
        raise HTTPException(status_code=400, detail="Realtime simulation not available")
    
    try:
        state.realtime_sim.reset(max_timesteps=config.max_timesteps)
        state.realtime_sim.initialize(state.features)
        
        return {
            "status": "success",
            "max_timesteps": config.max_timesteps,
            "current_timestep": 0,
            "message": "Realtime simulation initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/realtime/step")
async def step_realtime_simulation(command: SimStepCommand):
    """Execute simulation timesteps"""
    if not state.realtime_sim:
        raise HTTPException(status_code=400, detail="Realtime simulation not available")
    
    if not state.realtime_sim.initial_features is not None:
        raise HTTPException(status_code=400, detail="Simulation not initialized. Call /api/realtime/init first")
    
    try:
        steps = state.realtime_sim.run_multiple_steps(
            command.n_steps,
            command.shock_config
        )
        
        # Broadcast to WebSocket clients
        for client in state.websocket_clients:
            try:
                await client.send_json({
                    "type": "simulation_update",
                    "data": [asdict(step) for step in steps]
                })
            except:
                pass
        
        return {
            "status": "success",
            "steps_executed": len(steps),
            "current_timestep": state.realtime_sim.current_timestep,
            "latest_state": asdict(steps[-1]) if steps else None
        }
    except Exception as e:
        logger.error(f"Simulation step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/realtime/status")
async def get_realtime_status():
    """Get current realtime simulation status"""
    if not state.realtime_sim:
        return {"available": False}
    
    return {
        "available": True,
        "is_running": state.realtime_sim.is_running,
        "current_timestep": state.realtime_sim.current_timestep,
        "max_timesteps": state.realtime_sim.max_timesteps,
        "history_length": len(state.realtime_sim.history),
        "initialized": state.realtime_sim.initial_features is not None
    }

@app.get("/api/realtime/history")
async def get_simulation_history():
    """Get full simulation history"""
    if not state.realtime_sim:
        raise HTTPException(status_code=400, detail="Realtime simulation not available")
    
    return {
        "history": state.realtime_sim.get_history(),
        "total_timesteps": len(state.realtime_sim.history)
    }

@app.post("/api/realtime/stop")
async def stop_realtime_simulation():
    """Stop running simulation"""
    if not state.realtime_sim:
        raise HTTPException(status_code=400, detail="Realtime simulation not available")
    
    state.realtime_sim.stop()
    return {"status": "success", "message": "Simulation stopped"}

# ============================================
# Graph Generation Endpoints
# ============================================

@app.post("/api/graphs/generate")
async def generate_graph(request: GraphRequest):
    """Generate visualization graph"""
    if not state.graph_generator:
        raise HTTPException(status_code=400, detail="Graph generator not available")
    
    try:
        if request.graph_type == "network":
            if not state.network_builder:
                raise HTTPException(status_code=400, detail="Network not initialized")
            result = state.graph_generator.generate_network_graph(
                state.network_builder,
                highlight_nodes=request.highlight_nodes,
                format=request.format
            )
        
        elif request.graph_type == "risk_distribution":
            if state.features is None:
                raise HTTPException(status_code=400, detail="No risk data available")
            result = state.graph_generator.generate_risk_distribution(
                state.features,
                format=request.format
            )
        
        elif request.graph_type == "time_series":
            if not state.realtime_sim or not state.realtime_sim.history:
                raise HTTPException(status_code=400, detail="No simulation history available")
            result = state.graph_generator.generate_time_series(
                state.realtime_sim.get_history(),
                format=request.format
            )
        
        elif request.graph_type == "spectral":
            if not state.spectral_analyzer or not state.network_builder:
                raise HTTPException(status_code=400, detail="Spectral analysis not available")
            spectral_results = state.spectral_analyzer.analyze(network_builder=state.network_builder)
            result = state.graph_generator.generate_spectral_analysis(
                {
                    'spectral_radius': spectral_results.spectral_radius,
                    'fiedler_value': spectral_results.fiedler_value,
                    'spectral_gap': spectral_results.spectral_gap,
                    'contagion_index': state.spectral_analyzer.compute_contagion_index()
                },
                eigenvalues=spectral_results.eigenvalues if hasattr(spectral_results, 'eigenvalues') else None,
                format=request.format
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown graph type: {request.graph_type}")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graphs/available")
async def get_available_graphs():
    """Get list of available graph types"""
    return {
        "graphs": [
            {
                "type": "network",
                "description": "Banking network topology with centrality metrics",
                "requires": ["network_initialized"]
            },
            {
                "type": "risk_distribution",
                "description": "Risk distribution histograms and pie charts",
                "requires": ["features_available"]
            },
            {
                "type": "time_series",
                "description": "Time series plots from simulation history",
                "requires": ["simulation_history"]
            },
            {
                "type": "spectral",
                "description": "Spectral analysis plots with eigenvalues",
                "requires": ["spectral_analysis"]
            }
        ],
        "formats": ["plotly", "matplotlib"]
    }

# ============================================
# WebSocket for Real-time Updates
# ============================================

@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates"""
    await websocket.accept()
    state.websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(state.websocket_clients)}")
    
    try:
        while True:
            # Wait for messages or commands from client
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif command.get("type") == "subscribe":
                await websocket.send_json({
                    "type": "subscribed",
                    "message": "Subscribed to simulation updates"
                })
    
    except WebSocketDisconnect:
        state.websocket_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(state.websocket_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in state.websocket_clients:
            state.websocket_clients.remove(websocket)

# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
