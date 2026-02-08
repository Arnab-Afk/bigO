"""
Analytics Package

Provides Monte Carlo simulation, VaR/CVaR analysis, sensitivity analysis,
explainability, causal analysis, and policy impact assessment for RUDRA platform.
"""

from app.analytics.distributions import (
    CapitalRatioDistribution,
    DefaultProbabilityDistribution,
    LiquidityDistribution,
    ShockMagnitudeDistribution,
)
from app.analytics.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResults,
    SimulationRun,
)
from app.analytics.var_calculator import (
    VaRCalculator,
    VaRResult,
)
from app.analytics.explainability import ExplainabilityEngine
from app.analytics.causal_analysis import CausalAnalyzer
from app.analytics.policy_impact import PolicyImpactAnalyzer

__all__ = [
    # Distributions
    "CapitalRatioDistribution",
    "DefaultProbabilityDistribution",
    "LiquidityDistribution",
    "ShockMagnitudeDistribution",
    # Monte Carlo
    "MonteCarloEngine",
    "MonteCarloResults",
    "SimulationRun",
    # VaR
    "VaRCalculator",
    "VaRResult",
    # Explainability
    "ExplainabilityEngine",
    "CausalAnalyzer",
    "PolicyImpactAnalyzer",
]
