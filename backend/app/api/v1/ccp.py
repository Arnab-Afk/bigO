"""
CCP (Central Counterparty) Risk Analysis API

Integrates the CCP ML simulation system with the main RUDRA backend.
Provides endpoints for risk analysis, network visualization, and policy recommendations.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

# Add ccp_ml to path
CCP_ML_PATH = Path(__file__).parent.parent.parent.parent / "ccp_ml"
sys.path.insert(0, str(CCP_ML_PATH))

# Import CCP ML components
try:
    from ccp_ml.data_loader import DataLoader
    from ccp_ml.feature_engineering import FeatureEngineer
    from ccp_ml.network_builder import NetworkBuilder
    from ccp_ml.spectral_analyzer import SpectralAnalyzer, SpectralMetrics
    from ccp_ml.risk_model import CCPRiskModel, select_features
    from ccp_ml.ccp_engine import CCPEngine, MarginRequirement, DefaultFundAllocation
    CCP_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import CCP ML components: {e}")
    CCP_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ccp", tags=["ccp-risk-analysis"])

# ============================================================================
# Global State Management
# ============================================================================

class CCPState:
    """Manages CCP simulation state"""
    def __init__(self):
        self.data = None
        self.features = None
        self.network_builder = None
        self.spectral_analyzer = None
        self.risk_model = None
        self.ccp_engine = None
        self.last_run = None
        self.is_initialized = False
        self.network_data = None
        self.risk_scores = None
        self.ccp_results = None

ccp_state = CCPState()

# ============================================================================
# Request/Response Models
# ============================================================================

class SimulationConfig(BaseModel):
    """Configuration for CCP simulation"""
    year: Optional[int] = Field(None, description="Target year for analysis (2008-2025)")
    sector_weight: float = Field(0.4, ge=0, le=1, description="Weight for sector channel")
    liquidity_weight: float = Field(0.4, ge=0, le=1, description="Weight for liquidity channel")
    market_weight: float = Field(0.2, ge=0, le=1, description="Weight for market channel")
    edge_threshold: float = Field(0.05, ge=0, le=1, description="Minimum edge weight threshold")

class StressTestConfig(BaseModel):
    """Configuration for stress testing"""
    shock_magnitude: float = Field(0.2, ge=0, le=1, description="Shock magnitude (0-1)")
    target_banks: Optional[List[str]] = Field(None, description="Banks to apply shock to")
    shock_type: str = Field("capital", description="Shock type: capital, liquidity, or market")

class BankRiskScore(BaseModel):
    """Individual bank risk score"""
    bank_name: str
    default_probability: float
    risk_tier: str
    capital_ratio: float
    stress_level: float
    pagerank: float
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float

class NetworkMetrics(BaseModel):
    """Network-level metrics"""
    total_nodes: int
    total_edges: int
    avg_degree: float
    density: float
    clustering_coefficient: float

class SpectralMetricsResponse(BaseModel):
    """Spectral analysis metrics"""
    spectral_radius: float
    fiedler_value: float
    contagion_index: float
    eigenvalue_entropy: float
    fragility_score: float
    risk_level: str

class CCPMetrics(BaseModel):
    """CCP fund and margin metrics"""
    total_margin_requirement: float
    default_fund_size: float
    cover_n_standard: int
    total_exposure: float
    largest_counterparty_exposure: float

class SimulationSummary(BaseModel):
    """High-level simulation summary"""
    total_banks: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    last_run: Optional[str]
    network_metrics: NetworkMetrics
    spectral_metrics: SpectralMetricsResponse
    ccp_metrics: CCPMetrics

class PolicyRecommendation(BaseModel):
    """Policy recommendation"""
    priority: str
    category: str
    title: str
    description: str
    affected_banks: List[str]
    impact: str
    implementation: str

# ============================================================================
# Helper Functions
# ============================================================================

def initialize_ccp_system(config: SimulationConfig) -> Dict[str, Any]:
    """Initialize or refresh the CCP system with latest data"""
    if not CCP_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CCP ML module not available"
        )
    
    try:
        logger.info("Initializing CCP system...")
        
        # Load data
        data_loader = DataLoader(data_dir=str(CCP_ML_PATH / "data"))
        data = data_loader.load_all()
        
        # Engineer features
        feature_engineer = FeatureEngineer(normalize=True)
        features = feature_engineer.create_features(data, target_year=config.year)
        
        # Build network
        network_builder = NetworkBuilder(
            sector_weight=config.sector_weight,
            liquidity_weight=config.liquidity_weight,
            market_weight=config.market_weight,
            edge_threshold=config.edge_threshold
        )
        network_builder.build_network(data, year=config.year)
        
        # Spectral analysis
        spectral_analyzer = SpectralAnalyzer()
        spectral_metrics = spectral_analyzer.analyze(network_builder=network_builder)
        
        # Train risk model
        risk_model = CCPRiskModel()
        X, y = select_features(features)
        
        # Generate synthetic labels from default_probability_prior for training
        # This allows the model to learn from historical/prior default patterns
        if 'default_probability_prior' in features.columns and len(y.unique()) <= 1:
            logger.info("Generating synthetic training labels from default_probability_prior")
            # Convert probabilities to binary labels using threshold + noise
            priors = features['default_probability_prior'].fillna(0.05)
            # Add some randomness: high prob entities more likely to be labeled as default
            np.random.seed(42)
            synthetic_labels = (priors + np.random.uniform(-0.1, 0.1, len(priors))) > 0.3
            y = pd.Series(synthetic_labels.astype(int), index=features.index)
            logger.info(f"Generated {y.sum()} positive labels from {len(y)} samples")
        
        # Check if we have valid training data
        if len(y.unique()) > 1:
            train_metrics = risk_model.fit(X, y)
            logger.info(f"Risk model trained successfully: AUC={train_metrics.get('train_auc', 0):.4f}")
        else:
            logger.warning("Insufficient target diversity - model will use prior risk scores from features")
        
        # CCP Engine
        ccp_engine = CCPEngine(
            risk_model=risk_model,
            network_builder=network_builder,
            spectral_analyzer=spectral_analyzer
        )
        
        # Store in state
        ccp_state.data = data
        ccp_state.features = features
        ccp_state.network_builder = network_builder
        ccp_state.spectral_analyzer = spectral_analyzer
        ccp_state.risk_model = risk_model
        ccp_state.ccp_engine = ccp_engine
        ccp_state.last_run = datetime.now().isoformat()
        ccp_state.is_initialized = True
        
        logger.info("CCP system initialized successfully")
        
        return {
            "status": "initialized",
            "total_banks": len(features),
            "network_edges": network_builder.graph.number_of_edges(),
            "spectral_radius": spectral_metrics.spectral_radius,
            "timestamp": ccp_state.last_run
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize CCP system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize CCP system: {str(e)}"
        )

# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/simulate", response_model=Dict[str, Any])
async def run_simulation(
    config: SimulationConfig = SimulationConfig(),
    background_tasks: BackgroundTasks = None
):
    """
    Run complete CCP simulation with network analysis and risk assessment
    
    Returns comprehensive results including:
    - Network graph data
    - Risk scores for all banks
    - Spectral analysis metrics
    - CCP fund requirements
    - Policy recommendations
    """
    try:
        # Initialize system
        init_result = initialize_ccp_system(config)
        
        # Run full analysis through CCP Engine
        ccp_results = ccp_state.ccp_engine.run_full_analysis(
            ccp_state.features,
            train=False,  # Already trained during initialization
            year=config.year
        )
        
        # Store results
        ccp_state.ccp_results = ccp_results
        ccp_state.last_run = datetime.now().isoformat()
        
        return {
            "status": "success",
            "initialization": init_result,
            "results": ccp_results,
            "timestamp": ccp_state.last_run
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )

@router.get("/status")
async def get_status():
    """Get current CCP system status"""
    return {
        "initialized": ccp_state.is_initialized,
        "last_run": ccp_state.last_run,
        "available": CCP_AVAILABLE,
        "total_banks": len(ccp_state.features) if ccp_state.features is not None else 0
    }

@router.get("/summary", response_model=SimulationSummary)
async def get_summary():
    """Get high-level simulation summary"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        features = ccp_state.features
        network = ccp_state.network_builder.graph
        spectral_metrics = ccp_state.spectral_analyzer.metrics
        
        # Calculate risk distribution
        risk_scores = ccp_state.risk_model.predict_all(features)
        high_risk = sum(1 for p in risk_scores.values() if p > 0.7)
        medium_risk = sum(1 for p in risk_scores.values() if 0.3 <= p <= 0.7)
        low_risk = sum(1 for p in risk_scores.values() if p < 0.3)
        
        # Network metrics
        import networkx as nx
        avg_degree = sum(dict(network.degree()).values()) / network.number_of_nodes()
        density = nx.density(network)
        clustering = nx.average_clustering(network)
        
        # CCP metrics
        margins = ccp_state.ccp_engine.calculate_margins()
        total_margin = sum(m.total_margin for m in margins.values())
        default_fund = ccp_state.ccp_engine.calculate_default_fund()
        
        return SimulationSummary(
            total_banks=len(features),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            last_run=ccp_state.last_run,
            network_metrics=NetworkMetrics(
                total_nodes=network.number_of_nodes(),
                total_edges=network.number_of_edges(),
                avg_degree=avg_degree,
                density=density,
                clustering_coefficient=clustering
            ),
            spectral_metrics=SpectralMetricsResponse(
                spectral_radius=spectral_metrics.spectral_radius,
                fiedler_value=spectral_metrics.fiedler_value,
                contagion_index=spectral_metrics.contagion_index,
                eigenvalue_entropy=spectral_metrics.eigenvalue_entropy,
                fragility_score=spectral_metrics.spectral_radius / (1 + spectral_metrics.fiedler_value),
                risk_level="HIGH" if spectral_metrics.spectral_radius > 0.9 else "MEDIUM" if spectral_metrics.spectral_radius > 0.7 else "LOW"
            ),
            ccp_metrics=CCPMetrics(
                total_margin_requirement=total_margin,
                default_fund_size=default_fund['total_fund'],
                cover_n_standard=default_fund['cover_n'],
                total_exposure=sum(m.base_margin for m in margins.values()),
                largest_counterparty_exposure=max(m.base_margin for m in margins.values())
            )
        )
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

@router.get("/network")
async def get_network_data():
    """Get network graph data for visualization"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        network = ccp_state.network_builder.graph
        risk_scores = ccp_state.risk_model.predict_all(ccp_state.features)
        
        import networkx as nx
        
        # Calculate centralities
        pagerank = nx.pagerank(network)
        degree_cent = nx.degree_centrality(network)
        
        # Build nodes
        nodes = []
        for node in network.nodes():
            default_prob = risk_scores.get(node, 0.0)
            nodes.append({
                "id": node,
                "name": node,
                "default_probability": float(default_prob),
                "risk_level": "high" if default_prob > 0.7 else "medium" if default_prob > 0.3 else "low",
                "pagerank": float(pagerank.get(node, 0)),
                "degree_centrality": float(degree_cent.get(node, 0)),
                "degree": network.degree(node)
            })
        
        # Build edges
        edges = []
        for u, v, data in network.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": float(data.get('weight', 0)),
                "type": data.get('type', 'composite')
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "density": nx.density(network)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate network data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate network data: {str(e)}"
        )

@router.get("/banks", response_model=List[BankRiskScore])
async def get_all_banks():
    """Get risk scores and metrics for all banks"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        features = ccp_state.features
        risk_scores = ccp_state.risk_model.predict_all(features)
        network = ccp_state.network_builder.graph
        
        import networkx as nx
        pagerank = nx.pagerank(network)
        degree_cent = nx.degree_centrality(network)
        betweenness = nx.betweenness_centrality(network)
        eigenvector = nx.eigenvector_centrality(network, max_iter=1000)
        
        banks = []
        for bank_name in features['bank_name'].unique():
            bank_data = features[features['bank_name'] == bank_name].iloc[-1]
            default_prob = risk_scores.get(bank_name, 0.0)
            
            banks.append(BankRiskScore(
                bank_name=bank_name,
                default_probability=float(default_prob),
                risk_tier="Tier 1" if default_prob > 0.7 else "Tier 2" if default_prob > 0.5 else "Tier 3" if default_prob > 0.3 else "Tier 4",
                capital_ratio=float(bank_data.get('capital_ratio', 0)),
                stress_level=float(bank_data.get('stress_level', 0)),
                pagerank=float(pagerank.get(bank_name, 0)),
                degree_centrality=float(degree_cent.get(bank_name, 0)),
                betweenness_centrality=float(betweenness.get(bank_name, 0)),
                eigenvector_centrality=float(eigenvector.get(bank_name, 0))
            ))
        
        # Sort by default probability descending
        banks.sort(key=lambda x: x.default_probability, reverse=True)
        
        return banks
        
    except Exception as e:
        logger.error(f"Failed to get bank data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bank data: {str(e)}"
        )

@router.get("/banks/{bank_name}")
async def get_bank_detail(bank_name: str):
    """Get detailed information for a specific bank"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        features = ccp_state.features
        bank_data = features[features['bank_name'] == bank_name]
        
        if bank_data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bank '{bank_name}' not found"
            )
        
        risk_scores = ccp_state.risk_model.predict_all(features)
        network = ccp_state.network_builder.graph
        margins = ccp_state.ccp_engine.calculate_margins()
        
        import networkx as nx
        
        # Get latest data
        latest = bank_data.iloc[-1]
        default_prob = risk_scores.get(bank_name, 0.0)
        
        # Network position
        neighbors = list(network.neighbors(bank_name))
        
        # Historical trend
        trend_data = []
        for _, row in bank_data.iterrows():
            trend_data.append({
                "year": int(row.get('year', 0)),
                "capital_ratio": float(row.get('capital_ratio', 0)),
                "stress_level": float(row.get('stress_level', 0)),
                "liquidity_buffer": float(row.get('liquidity_buffer', 0))
            })
        
        return {
            "bank_name": bank_name,
            "current_metrics": {
                "default_probability": float(default_prob),
                "capital_ratio": float(latest.get('capital_ratio', 0)),
                "stress_level": float(latest.get('stress_level', 0)),
                "liquidity_buffer": float(latest.get('liquidity_buffer', 0)),
                "leverage": float(latest.get('leverage', 0))
            },
            "network_position": {
                "neighbors": neighbors,
                "degree": network.degree(bank_name),
                "pagerank": float(nx.pagerank(network).get(bank_name, 0)),
                "betweenness": float(nx.betweenness_centrality(network).get(bank_name, 0))
            },
            "margin_requirement": {
                "base_margin": float(margins[bank_name].base_margin),
                "network_addon": float(margins[bank_name].network_addon),
                "total_margin": float(margins[bank_name].total_margin),
                "explanation": margins[bank_name].explanation
            } if bank_name in margins else None,
            "historical_trend": trend_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bank detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bank detail: {str(e)}"
        )

@router.get("/spectral")
async def get_spectral_metrics():
    """Get spectral analysis metrics"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        metrics = ccp_state.spectral_analyzer.metrics
        
        return {
            "spectral_radius": float(metrics.spectral_radius),
            "fiedler_value": float(metrics.fiedler_value),
            "contagion_index": float(metrics.contagion_index),
            "eigenvalue_entropy": float(metrics.eigenvalue_entropy),
            "fragility_score": float(metrics.spectral_radius / (1 + metrics.fiedler_value)),
            "risk_assessment": {
                "amplification_risk": "HIGH" if metrics.spectral_radius > 0.9 else "MEDIUM" if metrics.spectral_radius > 0.7 else "LOW",
                "fragmentation_risk": "HIGH" if metrics.fiedler_value < 0.1 else "MEDIUM" if metrics.fiedler_value < 0.3 else "LOW",
                "overall_risk": "CRITICAL" if metrics.spectral_radius > 0.9 and metrics.fiedler_value < 0.1 else "HIGH" if metrics.spectral_radius > 0.7 else "MODERATE"
            },
            "eigenvalues": [float(e) for e in metrics.eigenvalues[:10]]  # Top 10
        }
        
    except Exception as e:
        logger.error(f"Failed to get spectral metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get spectral metrics: {str(e)}"
        )

@router.post("/stress-test")
async def run_stress_test(config: StressTestConfig):
    """Run stress test with specified shocks"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        # TODO: Implement stress testing logic
        # For now, return placeholder
        return {
            "status": "completed",
            "config": config.dict(),
            "results": {
                "banks_affected": len(config.target_banks) if config.target_banks else 0,
                "cascade_size": 0,
                "message": "Stress testing implementation pending"
            }
        }
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress test failed: {str(e)}"
        )

@router.get("/policies", response_model=List[PolicyRecommendation])
async def get_policy_recommendations():
    """Get policy recommendations based on CCP analysis"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        policies = ccp_state.ccp_engine.generate_policies()
        
        recommendations = []
        for policy in policies:
            recommendations.append(PolicyRecommendation(
                priority=policy.priority,
                category=policy.category,
                title=policy.title,
                description=policy.description,
                affected_banks=policy.affected_banks,
                impact=policy.impact,
                implementation=policy.implementation
            ))
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate policies: {str(e)}"
        )

@router.get("/margins")
async def get_margin_requirements():
    """Get margin requirements for all banks"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        margins = ccp_state.ccp_engine.calculate_margins()
        
        result = []
        for bank_name, margin in margins.items():
            result.append({
                "bank_name": bank_name,
                "base_margin": float(margin.base_margin),
                "network_addon": float(margin.network_addon),
                "stressed_margin": float(margin.stressed_margin),
                "total_margin": float(margin.total_margin),
                "confidence_level": float(margin.confidence_level),
                "explanation": margin.explanation
            })
        
        # Sort by total margin descending
        result.sort(key=lambda x: x["total_margin"], reverse=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get margins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get margins: {str(e)}"
        )

@router.get("/default-fund")
async def get_default_fund():
    """Get default fund allocation and sizing"""
    if not ccp_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CCP system not initialized. Run /simulate first."
        )
    
    try:
        default_fund = ccp_state.ccp_engine.calculate_default_fund()
        
        return {
            "total_fund_size": float(default_fund.total_fund_size),
            "cover_n": int(default_fund.cover_n),
            "confidence_level": float(default_fund.confidence_level),
            "contributions": [
                {
                    "bank_name": contrib.bank_name,
                    "base_contribution": float(contrib.base_contribution),
                    "systemic_addon": float(contrib.systemic_addon),
                    "total_contribution": float(contrib.total_contribution)
                }
                for contrib in default_fund.contributions
            ],
            "explanation": default_fund.explanation
        }
        
    except Exception as e:
        logger.error(f"Failed to get default fund: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default fund: {str(e)}"
        )
