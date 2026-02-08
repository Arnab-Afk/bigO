"""
Analytics Pydantic Schemas

Request/response schemas for Monte Carlo simulation and VaR analysis endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.common import BaseSchema, IDMixin, TimestampMixin


# ===== Monte Carlo Schemas =====


class MonteCarloSimulationRequest(BaseSchema):
    """Request to create Monte Carlo simulation job"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Simulation job name"
    )

    description: Optional[str] = Field(
        None,
        description="Job description"
    )

    n_simulations: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Number of Monte Carlo runs"
    )

    shocked_institutions: List[UUID] = Field(
        default_factory=list,
        description="Institution IDs to shock initially"
    )

    max_cascade_rounds: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum cascade propagation rounds"
    )

    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )

    # Distribution parameters (optional overrides)
    capital_ratio_mean: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override capital ratio distribution mean"
    )

    capital_ratio_std: Optional[float] = Field(
        None,
        ge=0.0,
        le=0.5,
        description="Override capital ratio distribution std"
    )

    shock_magnitude_max: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override maximum shock magnitude"
    )

    # VaR calculation settings
    compute_var: bool = Field(
        default=True,
        description="Compute VaR/CVaR metrics"
    )

    var_confidence_levels: List[float] = Field(
        default=[0.95, 0.99],
        description="Confidence levels for VaR"
    )

    @field_validator('var_confidence_levels')
    @classmethod
    def validate_confidence_levels(cls, v):
        """Validate confidence levels are between 0 and 1"""
        for level in v:
            if not 0 < level < 1:
                raise ValueError("Confidence levels must be between 0 and 1")
        return v


class MonteCarloJobStatusResponse(BaseSchema, IDMixin, TimestampMixin):
    """Monte Carlo job status"""

    name: str
    status: str = Field(
        description="Job status: pending, running, completed, failed"
    )

    n_simulations: int
    completed_simulations: int = Field(
        default=0,
        description="Number of completed simulation runs"
    )

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress as fraction (0-1)"
    )

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Estimated time
    estimated_time_remaining_seconds: Optional[float] = None

    class Config:
        from_attributes = True


class MonteCarloStatistics(BaseSchema):
    """Statistical summary of Monte Carlo results"""

    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


class VaRMetrics(BaseSchema):
    """VaR and CVaR metrics"""

    var_95: Optional[float] = Field(None, description="VaR at 95% confidence")
    var_99: Optional[float] = Field(None, description="VaR at 99% confidence")
    cvar_95: Optional[float] = Field(None, description="CVaR at 95% confidence")
    cvar_99: Optional[float] = Field(None, description="CVaR at 99% confidence")


class MonteCarloResultsResponse(BaseSchema):
    """Complete Monte Carlo simulation results"""

    job_id: UUID
    name: str
    status: str

    n_simulations: int
    total_execution_time_ms: float

    # Summary statistics
    defaults_statistics: MonteCarloStatistics
    losses_statistics: MonteCarloStatistics
    systemic_stress_mean: float
    systemic_stress_std: float

    # VaR metrics
    var_metrics: Optional[VaRMetrics] = None

    # Sample runs (first 100)
    sample_runs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample of individual simulation runs"
    )

    class Config:
        from_attributes = True


# ===== Sensitivity Analysis Schemas =====


class SensitivityAnalysisRequest(BaseSchema):
    """Request for parameter sensitivity analysis"""

    parameter_name: str = Field(
        ...,
        description="Parameter to vary (e.g., 'capital_ratio_mean', 'fire_sale_impact')"
    )

    parameter_min: float = Field(
        ...,
        description="Minimum parameter value"
    )

    parameter_max: float = Field(
        ...,
        description="Maximum parameter value"
    )

    n_points: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of parameter values to test"
    )

    n_simulations_per_point: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Monte Carlo runs per parameter value"
    )

    shocked_institutions: List[UUID] = Field(
        default_factory=list,
        description="Institution IDs to shock"
    )

    random_seed: Optional[int] = None


class SensitivityDataPoint(BaseSchema):
    """Single data point in sensitivity analysis"""

    parameter_value: float
    mean_defaults: float
    std_defaults: float
    mean_losses: float
    p95_losses: float
    p99_losses: float


class SensitivityAnalysisResponse(BaseSchema):
    """Sensitivity analysis results"""

    parameter_name: str
    parameter_range: List[float]
    results: List[SensitivityDataPoint]


# ===== VaR Calculation Schemas =====


class VaRCalculationRequest(BaseSchema):
    """Request for standalone VaR calculation"""

    loss_data: List[float] = Field(
        ...,
        min_length=10,
        description="Historical loss/return data"
    )

    confidence_levels: List[float] = Field(
        default=[0.95, 0.99],
        description="Confidence levels to calculate"
    )

    methods: List[str] = Field(
        default=["historical", "parametric"],
        description="VaR calculation methods"
    )

    @field_validator('confidence_levels')
    @classmethod
    def validate_confidence_levels(cls, v):
        """Validate confidence levels"""
        for level in v:
            if not 0 < level < 1:
                raise ValueError("Confidence levels must be between 0 and 1")
        return v

    @field_validator('methods')
    @classmethod
    def validate_methods(cls, v):
        """Validate VaR methods"""
        valid_methods = ["historical", "parametric", "monte_carlo", "extreme_value_theory"]
        for method in v:
            if method not in valid_methods:
                raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
        return v


class VaRResultDetail(BaseSchema):
    """Detailed VaR calculation result"""

    method: str
    confidence_level: float
    var: float
    cvar: float

    statistics: Dict[str, float] = Field(
        description="Mean, std, skewness, kurtosis, quartiles"
    )

    extreme_risk: Dict[str, float] = Field(
        description="Max loss, min loss"
    )

    metadata: Dict[str, Any] = Field(
        description="Additional metadata"
    )


class VaRCalculationResponse(BaseSchema):
    """VaR calculation results across methods and confidence levels"""

    results_by_method: Dict[str, Dict[str, VaRResultDetail]] = Field(
        description="Nested dict: {method: {confidence_level: result}}"
    )

    n_samples: int
    data_statistics: Dict[str, float]


# ===== Backtest Schemas =====


class VaRBacktestRequest(BaseSchema):
    """Request for VaR backtesting"""

    loss_data: List[float] = Field(
        ...,
        min_length=100,
        description="Historical loss data"
    )

    var_estimates: List[float] = Field(
        ...,
        min_length=100,
        description="Time series of VaR estimates"
    )

    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level used for VaR"
    )

    @field_validator('var_estimates')
    @classmethod
    def validate_lengths_match(cls, v, info):
        """Ensure loss_data and var_estimates have same length"""
        if 'loss_data' in info.data:
            if len(v) != len(info.data['loss_data']):
                raise ValueError("var_estimates must match loss_data length")
        return v


class VaRBacktestResponse(BaseSchema):
    """VaR backtest results"""

    n_exceedances: int
    expected_exceedances: float
    exceedance_rate: float
    expected_rate: float

    kupiec_lr_stat: float
    kupiec_p_value: float
    passes_kupiec_test: bool

    interpretation: str = Field(
        description="Human-readable interpretation"
    )


# ===== Stress Test Schemas =====


class StressScenario(BaseSchema):
    """Single stress scenario definition"""

    name: str = Field(
        ...,
        description="Scenario name"
    )

    shock_magnitude: float = Field(
        ...,
        ge=-1.0,
        le=10.0,
        description="Multiplicative shock (e.g., 0.2 = 20% increase)"
    )


class StressTestRequest(BaseSchema):
    """Request for stress testing VaR"""

    loss_data: List[float] = Field(
        ...,
        min_length=10,
        description="Baseline loss data"
    )

    stress_scenarios: List[StressScenario] = Field(
        ...,
        min_length=1,
        description="Stress scenarios to apply"
    )

    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0
    )


class StressTestResult(BaseSchema):
    """Result for a single stress scenario"""

    scenario_name: str
    shock_magnitude: float

    baseline_var: float
    stressed_var: float
    var_change: float
    var_change_pct: float

    baseline_cvar: float
    stressed_cvar: float


class StressTestResponse(BaseSchema):
    """Stress test results"""

    scenarios: List[StressTestResult]
    baseline_statistics: Dict[str, float]


# ===== Distribution Info Schemas =====


class DistributionInfo(BaseSchema):
    """Information about a probability distribution"""

    name: str
    distribution_type: str
    parameters: Dict[str, float]

    mean: float
    std: float
    median: Optional[float] = None

    example_samples: List[float] = Field(
        description="5 example samples for illustration"
    )


class DistributionsInfoResponse(BaseSchema):
    """Available distributions and their configurations"""

    distributions: Dict[str, DistributionInfo]
