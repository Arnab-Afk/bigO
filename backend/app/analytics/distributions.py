"""
Probability Distributions for Monte Carlo Simulation

Defines stochastic parameter distributions for financial system simulations.
Based on empirical data and Basel III guidelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy import stats


class Distribution(ABC):
    """Base class for probability distributions"""

    @abstractmethod
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the distribution

        Args:
            size: Number of samples
            random_state: Random seed for reproducibility

        Returns:
            Array of samples
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected value of the distribution"""
        pass

    @abstractmethod
    def std(self) -> float:
        """Standard deviation of the distribution"""
        pass

    @abstractmethod
    def percentile(self, q: float) -> float:
        """
        Calculate percentile

        Args:
            q: Percentile (0-100)

        Returns:
            Value at percentile
        """
        pass


@dataclass
class CapitalRatioDistribution(Distribution):
    """
    Capital Adequacy Ratio Distribution

    Models tier 1 capital ratio uncertainty using normal distribution.
    Basel III minimum is 6%, typical bank range: 8-15%.

    Default: N(μ=0.12, σ=0.03)
    """

    mu: float = 0.12  # Mean capital ratio (12%)
    sigma: float = 0.03  # Standard deviation (3%)
    min_ratio: float = 0.00  # Floor (0% - can go negative in stress)
    max_ratio: float = 0.30  # Cap (30%)

    def __post_init__(self):
        """Initialize scipy distribution"""
        self._dist = stats.norm(loc=self.mu, scale=self.sigma)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample capital ratios with bounds"""
        if random_state is not None:
            np.random.seed(random_state)

        samples = self._dist.rvs(size=size)
        # Clip to realistic bounds
        samples = np.clip(samples, self.min_ratio, self.max_ratio)

        return samples

    def mean(self) -> float:
        """Expected capital ratio"""
        return self.mu

    def std(self) -> float:
        """Standard deviation"""
        return self.sigma

    def percentile(self, q: float) -> float:
        """Calculate percentile with bounds"""
        value = self._dist.ppf(q / 100)
        return np.clip(value, self.min_ratio, self.max_ratio)

    def probability_below_threshold(self, threshold: float) -> float:
        """
        Probability that capital ratio falls below threshold

        Args:
            threshold: Capital ratio threshold

        Returns:
            Probability (0-1)
        """
        return self._dist.cdf(threshold)


@dataclass
class DefaultProbabilityDistribution(Distribution):
    """
    Default Probability Distribution

    Models institutional default probability using Beta distribution.
    Beta is bounded on [0,1] and flexible for modeling probabilities.

    Default: Beta(α=2, β=50) → E[X] ≈ 0.038 (3.8% default rate)
    """

    alpha: float = 2.0  # Shape parameter α
    beta: float = 50.0  # Shape parameter β

    def __post_init__(self):
        """Initialize scipy distribution"""
        self._dist = stats.beta(a=self.alpha, b=self.beta)

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample default probabilities"""
        if random_state is not None:
            np.random.seed(random_state)

        return self._dist.rvs(size=size)

    def mean(self) -> float:
        """Expected default probability"""
        return self.alpha / (self.alpha + self.beta)

    def std(self) -> float:
        """Standard deviation"""
        return self._dist.std()

    def percentile(self, q: float) -> float:
        """Calculate percentile"""
        return self._dist.ppf(q / 100)

    def mode(self) -> float:
        """Most likely default probability"""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return 0.0


@dataclass
class LiquidityDistribution(Distribution):
    """
    Liquidity Coverage Ratio Distribution

    Models liquidity buffer using lognormal distribution.
    Lognormal ensures positive values and right-skewed realistic behavior.

    Default: Lognormal(μ=0.0, σ=0.4) → median ≈ 1.0 (100% LCR)
    """

    mu: float = 0.0  # Log-scale location (ln of median)
    sigma: float = 0.4  # Log-scale dispersion
    min_liquidity: float = 0.01  # Floor (1%)
    max_liquidity: float = 5.0  # Cap (500%)

    def __post_init__(self):
        """Initialize scipy distribution"""
        self._dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample liquidity ratios with bounds"""
        if random_state is not None:
            np.random.seed(random_state)

        samples = self._dist.rvs(size=size)
        # Clip to realistic bounds
        samples = np.clip(samples, self.min_liquidity, self.max_liquidity)

        return samples

    def mean(self) -> float:
        """Expected liquidity ratio"""
        return np.exp(self.mu + (self.sigma ** 2) / 2)

    def std(self) -> float:
        """Standard deviation"""
        return self._dist.std()

    def percentile(self, q: float) -> float:
        """Calculate percentile with bounds"""
        value = self._dist.ppf(q / 100)
        return np.clip(value, self.min_liquidity, self.max_liquidity)

    def median(self) -> float:
        """Median liquidity ratio"""
        return np.exp(self.mu)


@dataclass
class ShockMagnitudeDistribution(Distribution):
    """
    Shock Magnitude Distribution

    Models severity of stress scenarios using uniform distribution.
    Range represents minimum to maximum shock impact on asset values.

    Default: Uniform(0.05, 0.30) → 5% to 30% asset value loss
    """

    min_shock: float = 0.05  # Minimum shock (5%)
    max_shock: float = 0.30  # Maximum shock (30%)

    def __post_init__(self):
        """Initialize scipy distribution"""
        self._dist = stats.uniform(
            loc=self.min_shock,
            scale=self.max_shock - self.min_shock
        )

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample shock magnitudes"""
        if random_state is not None:
            np.random.seed(random_state)

        return self._dist.rvs(size=size)

    def mean(self) -> float:
        """Expected shock magnitude"""
        return (self.min_shock + self.max_shock) / 2

    def std(self) -> float:
        """Standard deviation"""
        return self._dist.std()

    def percentile(self, q: float) -> float:
        """Calculate percentile"""
        return self._dist.ppf(q / 100)


@dataclass
class CorrelatedNormalDistribution:
    """
    Correlated Multivariate Normal Distribution

    Models correlated parameter uncertainty (e.g., capital ratios
    of institutions in the same sector tend to move together).
    """

    means: np.ndarray
    covariance_matrix: np.ndarray

    def __post_init__(self):
        """Validate inputs"""
        if len(self.means.shape) != 1:
            raise ValueError("means must be 1D array")

        n = len(self.means)
        if self.covariance_matrix.shape != (n, n):
            raise ValueError(f"covariance_matrix must be {n}x{n}")

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(self.covariance_matrix)
        if not np.all(eigenvalues >= -1e-8):
            raise ValueError("covariance_matrix must be positive semi-definite")

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from multivariate normal

        Returns:
            Array of shape (size, n_variables)
        """
        if random_state is not None:
            np.random.seed(random_state)

        return np.random.multivariate_normal(
            mean=self.means,
            cov=self.covariance_matrix,
            size=size
        )

    def mean_vector(self) -> np.ndarray:
        """Expected values"""
        return self.means

    def covariance(self) -> np.ndarray:
        """Covariance matrix"""
        return self.covariance_matrix

    def correlation(self) -> np.ndarray:
        """Correlation matrix"""
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        return self.covariance_matrix / np.outer(std_devs, std_devs)


def create_default_distributions() -> dict:
    """
    Create default distribution set for Monte Carlo simulation

    Returns:
        Dictionary of distribution objects
    """
    return {
        "capital_ratio": CapitalRatioDistribution(),
        "default_probability": DefaultProbabilityDistribution(),
        "liquidity": LiquidityDistribution(),
        "shock_magnitude": ShockMagnitudeDistribution(),
    }


def create_sector_correlated_capitals(
    n_institutions: int,
    sector_correlation: float = 0.3,
    base_mean: float = 0.12,
    base_std: float = 0.03,
) -> CorrelatedNormalDistribution:
    """
    Create correlated capital ratio distribution for institutions

    Args:
        n_institutions: Number of institutions
        sector_correlation: Correlation between institutions
        base_mean: Mean capital ratio
        base_std: Standard deviation

    Returns:
        Correlated distribution
    """
    means = np.full(n_institutions, base_mean)

    # Create correlation matrix
    correlation_matrix = np.eye(n_institutions)
    correlation_matrix[correlation_matrix == 0] = sector_correlation

    # Convert to covariance
    covariance_matrix = correlation_matrix * (base_std ** 2)

    return CorrelatedNormalDistribution(
        means=means,
        covariance_matrix=covariance_matrix
    )
