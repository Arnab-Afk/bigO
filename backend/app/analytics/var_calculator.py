"""
Value at Risk (VaR) Calculator

Provides comprehensive VaR and CVaR calculations with multiple methodologies
for financial risk assessment.

Methods:
- Historical simulation
- Parametric (variance-covariance)
- Monte Carlo simulation
- Extreme Value Theory (EVT)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class VaRMethod(str, Enum):
    """VaR calculation methodology"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    EVT = "extreme_value_theory"


@dataclass
class VaRResult:
    """
    Value at Risk calculation result

    Contains VaR, CVaR, and additional risk metrics
    """

    method: VaRMethod
    confidence_level: float

    # Core metrics
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)

    # Additional statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float

    # Distribution info
    median: float
    q25: float
    q75: float

    # Extreme risk
    max_loss: float
    min_loss: float

    # Metadata
    n_samples: int
    assumed_distribution: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "method": self.method.value,
            "confidence_level": self.confidence_level,
            "var": self.var,
            "cvar": self.cvar,
            "statistics": {
                "mean": self.mean,
                "std": self.std,
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
                "median": self.median,
                "q25": self.q25,
                "q75": self.q75,
            },
            "extreme_risk": {
                "max_loss": self.max_loss,
                "min_loss": self.min_loss,
            },
            "metadata": {
                "n_samples": self.n_samples,
                "assumed_distribution": self.assumed_distribution,
            }
        }

    def exceedance_probability(self) -> float:
        """Probability of exceeding VaR"""
        return 1.0 - self.confidence_level

    def tail_ratio(self) -> float:
        """Ratio of CVaR to VaR (tail heaviness indicator)"""
        return self.cvar / self.var if self.var != 0 else 1.0


class VaRCalculator:
    """
    Comprehensive VaR and CVaR calculator

    Supports multiple calculation methods and confidence levels.
    """

    def __init__(self, loss_data: np.ndarray):
        """
        Initialize VaR calculator

        Args:
            loss_data: Array of loss/return values (positive = loss)
        """
        self.loss_data = np.array(loss_data)
        self.n_samples = len(self.loss_data)

        if self.n_samples == 0:
            raise ValueError("loss_data cannot be empty")

    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> VaRResult:
        """
        Calculate VaR using specified method

        Args:
            confidence_level: Confidence level (0-1)
            method: Calculation method

        Returns:
            VaR result object
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if method == VaRMethod.HISTORICAL:
            return self._historical_var(confidence_level)
        elif method == VaRMethod.PARAMETRIC:
            return self._parametric_var(confidence_level)
        elif method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(confidence_level)
        elif method == VaRMethod.EVT:
            return self._evt_var(confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _historical_var(self, confidence_level: float) -> VaRResult:
        """
        Historical simulation VaR

        Non-parametric method using empirical distribution.
        VaR is simply the percentile of historical losses.
        """
        percentile = confidence_level * 100
        var = float(np.percentile(self.loss_data, percentile))

        # CVaR: mean of losses exceeding VaR
        tail_losses = self.loss_data[self.loss_data > var]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

        return VaRResult(
            method=VaRMethod.HISTORICAL,
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            mean=float(np.mean(self.loss_data)),
            std=float(np.std(self.loss_data)),
            skewness=float(stats.skew(self.loss_data)),
            kurtosis=float(stats.kurtosis(self.loss_data)),
            median=float(np.median(self.loss_data)),
            q25=float(np.percentile(self.loss_data, 25)),
            q75=float(np.percentile(self.loss_data, 75)),
            max_loss=float(np.max(self.loss_data)),
            min_loss=float(np.min(self.loss_data)),
            n_samples=self.n_samples,
            assumed_distribution="empirical",
        )

    def _parametric_var(self, confidence_level: float) -> VaRResult:
        """
        Parametric VaR (variance-covariance method)

        Assumes normal distribution of losses.
        VaR = μ + σ × z_α

        Fast but may underestimate tail risk.
        """
        mean = np.mean(self.loss_data)
        std = np.std(self.loss_data)

        # z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)

        var = float(mean + std * z_score)

        # CVaR for normal distribution
        # CVaR = μ + σ × φ(z_α) / (1 - α)
        # where φ is the standard normal PDF
        phi = stats.norm.pdf(z_score)
        cvar = float(mean + std * phi / (1 - confidence_level))

        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            mean=float(mean),
            std=float(std),
            skewness=float(stats.skew(self.loss_data)),
            kurtosis=float(stats.kurtosis(self.loss_data)),
            median=float(np.median(self.loss_data)),
            q25=float(np.percentile(self.loss_data, 25)),
            q75=float(np.percentile(self.loss_data, 75)),
            max_loss=float(np.max(self.loss_data)),
            min_loss=float(np.min(self.loss_data)),
            n_samples=self.n_samples,
            assumed_distribution="normal",
        )

    def _monte_carlo_var(
        self,
        confidence_level: float,
        n_simulations: int = 10000,
    ) -> VaRResult:
        """
        Monte Carlo VaR

        Fit distribution to data and simulate losses.
        More flexible than parametric but computationally intensive.
        """
        # Fit distribution (using t-distribution for fat tails)
        params = stats.t.fit(self.loss_data)
        df, loc, scale = params

        # Generate simulated losses
        simulated_losses = stats.t.rvs(df, loc=loc, scale=scale, size=n_simulations)

        # Calculate VaR and CVaR from simulated data
        percentile = confidence_level * 100
        var = float(np.percentile(simulated_losses, percentile))

        tail_losses = simulated_losses[simulated_losses > var]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            mean=float(np.mean(self.loss_data)),
            std=float(np.std(self.loss_data)),
            skewness=float(stats.skew(self.loss_data)),
            kurtosis=float(stats.kurtosis(self.loss_data)),
            median=float(np.median(self.loss_data)),
            q25=float(np.percentile(self.loss_data, 25)),
            q75=float(np.percentile(self.loss_data, 75)),
            max_loss=float(np.max(self.loss_data)),
            min_loss=float(np.min(self.loss_data)),
            n_samples=self.n_samples,
            assumed_distribution=f"student-t(df={df:.2f})",
        )

    def _evt_var(self, confidence_level: float, threshold_percentile: float = 0.90) -> VaRResult:
        """
        Extreme Value Theory (EVT) VaR

        Models tail distribution using Generalized Pareto Distribution (GPD).
        Best for extreme quantiles (99%, 99.9%).

        Args:
            confidence_level: Target confidence level
            threshold_percentile: Threshold for defining tail (default 90%)
        """
        # Define threshold for tail
        threshold = np.percentile(self.loss_data, threshold_percentile * 100)

        # Extract exceedances over threshold
        exceedances = self.loss_data[self.loss_data > threshold] - threshold

        if len(exceedances) < 10:
            # Fallback to historical if insufficient tail data
            return self._historical_var(confidence_level)

        # Fit Generalized Pareto Distribution to exceedances
        shape, loc, scale = stats.genpareto.fit(exceedances)

        # Number of exceedances
        n_exceedances = len(exceedances)
        n_total = len(self.loss_data)

        # Empirical probability of exceeding threshold
        p_threshold = n_exceedances / n_total

        # Calculate VaR using EVT formula
        if confidence_level > threshold_percentile:
            # VaR_α = u + (β/ξ) × [((1-α)/p_u)^(-ξ) - 1]
            # where u = threshold, β = scale, ξ = shape, p_u = P(X > u)
            if abs(shape) > 1e-6:  # shape != 0
                var = threshold + (scale / shape) * (
                    ((1 - confidence_level) / p_threshold) ** (-shape) - 1
                )
            else:  # shape ≈ 0 (exponential tail)
                var = threshold - scale * np.log((1 - confidence_level) / p_threshold)

            var = float(var)
        else:
            # Below threshold, use empirical quantile
            var = float(np.percentile(self.loss_data, confidence_level * 100))

        # CVaR calculation for GPD
        if abs(shape) > 1e-6 and shape < 1:
            cvar = float(var / (1 - shape) + (scale - shape * threshold) / (1 - shape))
        else:
            # Fallback to empirical CVaR
            tail_losses = self.loss_data[self.loss_data > var]
            cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

        return VaRResult(
            method=VaRMethod.EVT,
            confidence_level=confidence_level,
            var=var,
            cvar=cvar,
            mean=float(np.mean(self.loss_data)),
            std=float(np.std(self.loss_data)),
            skewness=float(stats.skew(self.loss_data)),
            kurtosis=float(stats.kurtosis(self.loss_data)),
            median=float(np.median(self.loss_data)),
            q25=float(np.percentile(self.loss_data, 25)),
            q75=float(np.percentile(self.loss_data, 75)),
            max_loss=float(np.max(self.loss_data)),
            min_loss=float(np.min(self.loss_data)),
            n_samples=self.n_samples,
            assumed_distribution=f"GPD(ξ={shape:.3f}, β={scale:.3f})",
        )

    def calculate_all_methods(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
    ) -> Dict[str, Dict[str, VaRResult]]:
        """
        Calculate VaR using all methods at multiple confidence levels

        Args:
            confidence_levels: List of confidence levels to calculate

        Returns:
            Nested dict: {method: {confidence_level: VaRResult}}
        """
        results = {}

        for method in VaRMethod:
            results[method.value] = {}
            for conf_level in confidence_levels:
                try:
                    var_result = self.calculate_var(conf_level, method)
                    results[method.value][str(conf_level)] = var_result
                except Exception as e:
                    # Some methods may fail for certain data
                    results[method.value][str(conf_level)] = None

        return results

    def backtest_var(
        self,
        var_values: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Backtest VaR estimates

        Compares actual exceedances to expected exceedances.

        Args:
            var_values: Time series of VaR estimates
            confidence_level: Confidence level used for VaR

        Returns:
            Backtest statistics
        """
        if len(var_values) != self.n_samples:
            raise ValueError("var_values must match loss_data length")

        # Count exceedances
        exceedances = (self.loss_data > var_values).astype(int)
        n_exceedances = np.sum(exceedances)

        # Expected number of exceedances
        expected_exceedances = self.n_samples * (1 - confidence_level)

        # Kupiec test statistic (likelihood ratio test)
        if n_exceedances > 0:
            p_hat = n_exceedances / self.n_samples
            p_expected = 1 - confidence_level

            lr_stat = -2 * (
                n_exceedances * np.log(p_expected / p_hat) +
                (self.n_samples - n_exceedances) * np.log((1 - p_expected) / (1 - p_hat))
            )
        else:
            lr_stat = 0.0

        # p-value from chi-squared distribution (df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        return {
            "n_exceedances": int(n_exceedances),
            "expected_exceedances": float(expected_exceedances),
            "exceedance_rate": float(n_exceedances / self.n_samples),
            "expected_rate": float(1 - confidence_level),
            "kupiec_lr_stat": float(lr_stat),
            "kupiec_p_value": float(p_value),
            "passes_kupiec_test": p_value > 0.05,  # 5% significance level
        }

    def stress_test_var(
        self,
        stress_scenarios: Dict[str, float],
    ) -> Dict[str, VaRResult]:
        """
        Apply stress scenarios to VaR calculation

        Args:
            stress_scenarios: Dict mapping scenario name to shock magnitude

        Returns:
            VaR results under each stress scenario
        """
        results = {}

        for scenario_name, shock_magnitude in stress_scenarios.items():
            # Apply shock to loss data
            stressed_losses = self.loss_data * (1 + shock_magnitude)

            # Create temporary calculator with stressed data
            stressed_calc = VaRCalculator(stressed_losses)

            # Calculate VaR under stress
            var_result = stressed_calc.calculate_var(
                confidence_level=0.95,
                method=VaRMethod.HISTORICAL
            )

            results[scenario_name] = var_result

        return results
