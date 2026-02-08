"""
Monte Carlo Simulation Engine

Stochastic simulation framework for financial system stress testing
with parameter uncertainty quantification.

Implements:
- Multi-run stochastic simulation
- Sensitivity analysis via parameter sweeps
- Result aggregation and statistical analysis
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import networkx as nx
import numpy as np
from scipy import stats

from app.analytics.distributions import (
    CapitalRatioDistribution,
    DefaultProbabilityDistribution,
    LiquidityDistribution,
    ShockMagnitudeDistribution,
    create_default_distributions,
)
from app.engine.contagion import ContagionPropagator, PropagationState
from app.core.logging import logger


@dataclass
class SimulationRun:
    """Single Monte Carlo simulation run result"""

    run_id: int
    random_seed: int

    # Input parameters (sampled)
    capital_ratios: Dict[UUID, float]
    liquidity_levels: Dict[UUID, float]
    default_probabilities: Dict[UUID, float]
    shock_magnitude: float

    # Output metrics
    total_defaults: int
    cascade_depth: int
    total_losses: float
    systemic_stress: float
    survival_rate: float
    time_to_first_default: Optional[int]

    # Final state
    final_state: Optional[Dict[str, Any]] = None

    # Execution metadata
    execution_time_ms: float = 0.0


@dataclass
class MonteCarloResults:
    """
    Aggregated Monte Carlo simulation results

    Contains statistical analysis across all simulation runs
    """

    n_simulations: int
    total_execution_time_ms: float

    # Individual runs
    runs: List[SimulationRun] = field(default_factory=list)

    # Aggregated statistics
    mean_defaults: float = 0.0
    std_defaults: float = 0.0
    mean_losses: float = 0.0
    std_losses: float = 0.0
    mean_systemic_stress: float = 0.0
    std_systemic_stress: float = 0.0

    # Percentiles
    p50_defaults: float = 0.0
    p95_defaults: float = 0.0
    p99_defaults: float = 0.0
    p50_losses: float = 0.0
    p95_losses: float = 0.0
    p99_losses: float = 0.0

    # VaR metrics (computed separately)
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None
    cvar_99: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "n_simulations": self.n_simulations,
            "total_execution_time_ms": self.total_execution_time_ms,
            "statistics": {
                "defaults": {
                    "mean": self.mean_defaults,
                    "std": self.std_defaults,
                    "p50": self.p50_defaults,
                    "p95": self.p95_defaults,
                    "p99": self.p99_defaults,
                },
                "losses": {
                    "mean": self.mean_losses,
                    "std": self.std_losses,
                    "p50": self.p50_losses,
                    "p95": self.p95_losses,
                    "p99": self.p99_losses,
                },
                "systemic_stress": {
                    "mean": self.mean_systemic_stress,
                    "std": self.std_systemic_stress,
                },
            },
            "var_metrics": {
                "var_95": self.var_95,
                "var_99": self.var_99,
                "cvar_95": self.cvar_95,
                "cvar_99": self.cvar_99,
            } if self.var_95 is not None else None,
            "run_summary": [
                {
                    "run_id": run.run_id,
                    "total_defaults": run.total_defaults,
                    "total_losses": run.total_losses,
                    "systemic_stress": run.systemic_stress,
                }
                for run in self.runs
            ]
        }


class MonteCarloEngine:
    """
    Monte Carlo Simulation Engine

    Runs stochastic simulations with parameter uncertainty to estimate
    distribution of outcomes and compute risk metrics.
    """

    def __init__(
        self,
        network: nx.DiGraph,
        distributions: Optional[Dict[str, Any]] = None,
        propagator_params: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Monte Carlo engine

        Args:
            network: Financial network graph
            distributions: Custom distributions (uses defaults if None)
            propagator_params: Parameters for ContagionPropagator
        """
        self.network = network
        self.distributions = distributions or create_default_distributions()

        # Contagion propagator parameters
        self.propagator_params = propagator_params or {
            "fire_sale_impact": 0.1,
            "margin_sensitivity": 0.2,
            "information_weight": 0.15,
        }

    def run_stochastic_simulation(
        self,
        n_simulations: int,
        shocked_institutions: List[UUID],
        max_cascade_rounds: int = 10,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo stochastic simulation

        Args:
            n_simulations: Number of simulation runs
            shocked_institutions: Initially shocked institutions
            max_cascade_rounds: Maximum cascade propagation rounds
            random_seed: Base random seed for reproducibility
            progress_callback: Optional callback(current, total)

        Returns:
            Aggregated Monte Carlo results
        """
        logger.info(
            "Starting Monte Carlo simulation",
            n_simulations=n_simulations,
            shocked_count=len(shocked_institutions)
        )

        start_time = time.time()
        runs: List[SimulationRun] = []

        # Set base random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        for i in range(n_simulations):
            run_start = time.time()

            # Generate unique seed for this run
            run_seed = random_seed + i if random_seed is not None else None

            # Run single simulation
            run_result = self._run_single_simulation(
                run_id=i,
                shocked_institutions=shocked_institutions,
                max_cascade_rounds=max_cascade_rounds,
                random_seed=run_seed,
            )

            run_result.execution_time_ms = (time.time() - run_start) * 1000
            runs.append(run_result)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, n_simulations)

            # Log progress
            if (i + 1) % 100 == 0 or (i + 1) == n_simulations:
                logger.info(
                    f"Monte Carlo progress: {i + 1}/{n_simulations} runs completed"
                )

        total_time_ms = (time.time() - start_time) * 1000

        # Aggregate results
        results = self._aggregate_results(runs, total_time_ms)

        logger.info(
            "Monte Carlo simulation completed",
            n_simulations=n_simulations,
            mean_defaults=results.mean_defaults,
            p95_defaults=results.p95_defaults,
            execution_time_ms=total_time_ms
        )

        return results

    def _run_single_simulation(
        self,
        run_id: int,
        shocked_institutions: List[UUID],
        max_cascade_rounds: int,
        random_seed: Optional[int],
    ) -> SimulationRun:
        """
        Execute a single simulation run with sampled parameters

        Args:
            run_id: Run identifier
            shocked_institutions: Initially shocked institutions
            max_cascade_rounds: Max cascade rounds
            random_seed: Random seed for this run

        Returns:
            Single run result
        """
        # Sample parameters from distributions
        nodes = list(self.network.nodes())
        n_nodes = len(nodes)

        capital_dist = self.distributions.get(
            "capital_ratio",
            CapitalRatioDistribution()
        )
        liquidity_dist = self.distributions.get(
            "liquidity",
            LiquidityDistribution()
        )
        default_prob_dist = self.distributions.get(
            "default_probability",
            DefaultProbabilityDistribution()
        )
        shock_dist = self.distributions.get(
            "shock_magnitude",
            ShockMagnitudeDistribution()
        )

        # Sample values
        capital_ratios_array = capital_dist.sample(n_nodes, random_seed)
        liquidity_array = liquidity_dist.sample(n_nodes, random_seed)
        default_probs_array = default_prob_dist.sample(n_nodes, random_seed)
        shock_magnitude = float(shock_dist.sample(1, random_seed)[0])

        # Convert to dictionaries
        capital_ratios = {node: float(cap) for node, cap in zip(nodes, capital_ratios_array)}
        liquidity_levels = {node: float(liq) for node, liq in zip(nodes, liquidity_array)}
        default_probabilities = {node: float(dp) for node, dp in zip(nodes, default_probs_array)}

        # Create initial state
        initial_state = PropagationState(
            capital_levels={
                node: capital_ratios[node] * 10000  # Scale to absolute capital
                for node in nodes
            },
            liquidity_levels=liquidity_levels.copy(),
            stress_levels={node: 0.0 for node in nodes},
            defaulted=set(),
            asset_prices={"default": 1.0 - shock_magnitude}
        )

        # Apply initial shock to shocked institutions
        for inst_id in shocked_institutions:
            if inst_id in initial_state.capital_levels:
                # Reduce capital by shock magnitude
                initial_state.capital_levels[inst_id] *= (1 - shock_magnitude)
                initial_state.stress_levels[inst_id] = shock_magnitude

        # Create propagator and run simulation
        propagator = ContagionPropagator(
            network=self.network,
            **self.propagator_params
        )

        final_state, cascade_history = propagator.propagate_shock(
            initial_state=initial_state,
            shocked_institutions=shocked_institutions,
            max_rounds=max_cascade_rounds,
        )

        # Extract metrics
        total_defaults = len(final_state.defaulted)
        cascade_depth = len(cascade_history)
        total_losses = sum(
            round_data.total_losses
            for round_data in cascade_history
        )
        systemic_stress = float(np.mean(list(final_state.stress_levels.values())))
        survival_rate = 1.0 - (total_defaults / n_nodes) if n_nodes > 0 else 1.0

        # Time to first default
        time_to_first = None
        for round_data in cascade_history:
            if round_data.defaults:
                time_to_first = round_data.round_number
                break

        return SimulationRun(
            run_id=run_id,
            random_seed=random_seed or 0,
            capital_ratios=capital_ratios,
            liquidity_levels=liquidity_levels,
            default_probabilities=default_probabilities,
            shock_magnitude=shock_magnitude,
            total_defaults=total_defaults,
            cascade_depth=cascade_depth,
            total_losses=total_losses,
            systemic_stress=systemic_stress,
            survival_rate=survival_rate,
            time_to_first_default=time_to_first,
        )

    def _aggregate_results(
        self,
        runs: List[SimulationRun],
        total_time_ms: float
    ) -> MonteCarloResults:
        """
        Aggregate statistics across all runs

        Args:
            runs: List of simulation runs
            total_time_ms: Total execution time

        Returns:
            Aggregated results
        """
        n_runs = len(runs)

        # Extract arrays for statistical analysis
        defaults_array = np.array([run.total_defaults for run in runs])
        losses_array = np.array([run.total_losses for run in runs])
        stress_array = np.array([run.systemic_stress for run in runs])

        results = MonteCarloResults(
            n_simulations=n_runs,
            total_execution_time_ms=total_time_ms,
            runs=runs,
            # Defaults statistics
            mean_defaults=float(np.mean(defaults_array)),
            std_defaults=float(np.std(defaults_array)),
            p50_defaults=float(np.percentile(defaults_array, 50)),
            p95_defaults=float(np.percentile(defaults_array, 95)),
            p99_defaults=float(np.percentile(defaults_array, 99)),
            # Losses statistics
            mean_losses=float(np.mean(losses_array)),
            std_losses=float(np.std(losses_array)),
            p50_losses=float(np.percentile(losses_array, 50)),
            p95_losses=float(np.percentile(losses_array, 95)),
            p99_losses=float(np.percentile(losses_array, 99)),
            # Systemic stress
            mean_systemic_stress=float(np.mean(stress_array)),
            std_systemic_stress=float(np.std(stress_array)),
        )

        # Compute VaR/CVaR on losses
        results.var_95 = self.compute_var(losses_array, confidence_level=0.95)
        results.var_99 = self.compute_var(losses_array, confidence_level=0.99)
        results.cvar_95 = self.compute_cvar(losses_array, confidence_level=0.95)
        results.cvar_99 = self.compute_cvar(losses_array, confidence_level=0.99)

        return results

    @staticmethod
    def compute_var(
        loss_distribution: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute Value at Risk (VaR)

        VaR_α = inf{x : P(Loss > x) ≤ 1 - α}

        Args:
            loss_distribution: Array of loss values
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR at specified confidence level
        """
        percentile = confidence_level * 100
        return float(np.percentile(loss_distribution, percentile))

    @staticmethod
    def compute_cvar(
        loss_distribution: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute Conditional Value at Risk (CVaR / Expected Shortfall)

        CVaR_α = E[Loss | Loss > VaR_α]

        Args:
            loss_distribution: Array of loss values
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR at specified confidence level
        """
        var = MonteCarloEngine.compute_var(loss_distribution, confidence_level)

        # Expected value of losses exceeding VaR
        tail_losses = loss_distribution[loss_distribution > var]

        if len(tail_losses) > 0:
            return float(np.mean(tail_losses))
        else:
            # If no losses exceed VaR, return VaR itself
            return var

    def sensitivity_analysis(
        self,
        parameter_name: str,
        parameter_range: np.ndarray,
        shocked_institutions: List[UUID],
        n_simulations_per_point: int = 100,
        max_cascade_rounds: int = 10,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis via parameter sweep

        Args:
            parameter_name: Name of parameter to vary
            parameter_range: Array of parameter values to test
            shocked_institutions: Initially shocked institutions
            n_simulations_per_point: MC runs per parameter value
            max_cascade_rounds: Max cascade rounds
            random_seed: Base random seed

        Returns:
            Sensitivity analysis results
        """
        logger.info(
            "Starting sensitivity analysis",
            parameter=parameter_name,
            n_points=len(parameter_range),
            n_sims_per_point=n_simulations_per_point
        )

        results_per_value = []

        for i, param_value in enumerate(parameter_range):
            # Update distribution parameter
            self._set_distribution_parameter(parameter_name, param_value)

            # Run Monte Carlo at this parameter value
            mc_results = self.run_stochastic_simulation(
                n_simulations=n_simulations_per_point,
                shocked_institutions=shocked_institutions,
                max_cascade_rounds=max_cascade_rounds,
                random_seed=random_seed + i if random_seed else None,
            )

            results_per_value.append({
                "parameter_value": float(param_value),
                "mean_defaults": mc_results.mean_defaults,
                "std_defaults": mc_results.std_defaults,
                "mean_losses": mc_results.mean_losses,
                "p95_losses": mc_results.p95_losses,
                "p99_losses": mc_results.p99_losses,
            })

            logger.info(
                f"Sensitivity point {i + 1}/{len(parameter_range)} completed",
                param_value=param_value,
                mean_defaults=mc_results.mean_defaults
            )

        return {
            "parameter_name": parameter_name,
            "parameter_range": parameter_range.tolist(),
            "results": results_per_value,
        }

    def _set_distribution_parameter(self, parameter_name: str, value: float):
        """
        Update a distribution parameter for sensitivity analysis

        Args:
            parameter_name: Parameter to update
            value: New parameter value
        """
        # Map parameter names to distribution updates
        if parameter_name == "capital_ratio_mean":
            self.distributions["capital_ratio"] = CapitalRatioDistribution(
                mu=value,
                sigma=self.distributions["capital_ratio"].sigma
            )
        elif parameter_name == "capital_ratio_std":
            self.distributions["capital_ratio"] = CapitalRatioDistribution(
                mu=self.distributions["capital_ratio"].mu,
                sigma=value
            )
        elif parameter_name == "liquidity_mu":
            self.distributions["liquidity"] = LiquidityDistribution(
                mu=value,
                sigma=self.distributions["liquidity"].sigma
            )
        elif parameter_name == "shock_magnitude_max":
            self.distributions["shock_magnitude"] = ShockMagnitudeDistribution(
                min_shock=self.distributions["shock_magnitude"].min_shock,
                max_shock=value
            )
        elif parameter_name == "fire_sale_impact":
            self.propagator_params["fire_sale_impact"] = value
        elif parameter_name == "margin_sensitivity":
            self.propagator_params["margin_sensitivity"] = value
        else:
            logger.warning(f"Unknown sensitivity parameter: {parameter_name}")
