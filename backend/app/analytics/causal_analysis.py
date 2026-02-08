"""
Causal Analysis Module for RUDRA

Provides causal inference, mediation analysis, and intervention point identification
to understand the causal mechanisms behind defaults and cascades.

Key capabilities:
1. Direct vs Indirect Effects (Mediation Analysis)
2. Alternative Intervention Discovery (Backtracking)
3. Intervention Point Ranking
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from uuid import UUID
from enum import Enum

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


class EffectType(Enum):
    """Type of causal effect"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    TOTAL = "total"


@dataclass
class CausalEffect:
    """Represents a causal effect between two variables/institutions"""
    source: str  # Source variable or institution
    target: str  # Target variable or institution
    effect_type: EffectType
    magnitude: float  # Strength of effect [-1, 1]
    confidence: float  # Confidence in estimate [0, 1]
    mediators: Optional[List[str]] = None  # Mediating variables (for indirect effects)


@dataclass
class MediationAnalysis:
    """Results of mediation analysis"""
    treatment: str  # Treatment variable (e.g., "Capital Injection")
    outcome: str  # Outcome variable (e.g., "Default")
    mediator: str  # Mediating variable (e.g., "Liquidity")

    # Effect decomposition
    total_effect: float
    direct_effect: float
    indirect_effect: float
    mediation_proportion: float  # % of effect mediated

    explanation: str


@dataclass
class InterventionPoint:
    """A potential intervention point in the system"""
    institution_id: UUID
    institution_name: str
    intervention_type: str  # e.g., "Capital Injection", "Liquidity Support"

    # Effectiveness metrics
    effectiveness_score: float  # How effective this intervention would be [0, 1]
    cascade_prevention_score: float  # Ability to prevent cascades [0, 1]
    cost_effectiveness: float  # Benefit per unit cost

    # Impact analysis
    institutions_saved: int
    expected_defaults_prevented: int
    network_stabilization: float  # Impact on network stability

    # Recommended intervention
    recommended_amount: float
    urgency: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"

    explanation: str


@dataclass
class AlternativeIntervention:
    """An alternative intervention strategy"""
    strategy_name: str
    target_institutions: List[UUID]
    intervention_types: Dict[UUID, str]  # Institution -> Intervention type

    # Effectiveness
    expected_effectiveness: float
    defaults_prevented: int
    total_cost: float
    roi: float

    # Comparison to baseline
    better_than_baseline: bool
    cost_saving: float

    explanation: str


class CausalAnalyzer:
    """
    Causal Analysis Engine

    Provides causal inference capabilities to understand:
    - Why defaults happen (causal mechanisms)
    - How effects propagate (direct vs indirect)
    - Where to intervene (optimal intervention points)
    - Alternative strategies (counterfactual reasoning)
    """

    def __init__(
        self,
        sensitivity_threshold: float = 0.1,
        min_effect_magnitude: float = 0.05,
    ):
        """
        Initialize causal analyzer

        Args:
            sensitivity_threshold: Minimum sensitivity for causal effects
            min_effect_magnitude: Minimum effect magnitude to consider
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.min_effect_magnitude = min_effect_magnitude

        logger.info("CausalAnalyzer initialized")

    def compute_direct_vs_indirect_effects(
        self,
        treatment_variable: str,
        outcome_variable: str,
        mediator_variable: str,
        feature_values: Dict[str, float],
        network_graph: nx.DiGraph,
        institution_id: Optional[UUID] = None,
    ) -> MediationAnalysis:
        """
        Perform mediation analysis to decompose total effect into direct and indirect

        Example: Does capital injection prevent default directly, or through
        improving liquidity (mediator)?

        Args:
            treatment_variable: Treatment (e.g., "Capital Ratio")
            outcome_variable: Outcome (e.g., "Default Probability")
            mediator_variable: Mediator (e.g., "Liquidity Buffer")
            feature_values: Current feature values
            network_graph: Network structure
            institution_id: Optional institution ID for context

        Returns:
            MediationAnalysis with effect decomposition
        """
        logger.info(
            f"Mediation analysis: {treatment_variable} -> {mediator_variable} -> {outcome_variable}"
        )

        # Get baseline values
        treatment_value = feature_values.get(treatment_variable, 0.5)
        mediator_value = feature_values.get(mediator_variable, 0.5)
        outcome_value = feature_values.get(outcome_variable, 0.5)

        # Simulate treatment effect
        # Path 1: Treatment -> Outcome (direct effect)
        # Increase treatment by 10%
        treatment_delta = 0.1
        direct_effect = self._estimate_direct_effect(
            treatment_variable,
            outcome_variable,
            treatment_delta,
            feature_values,
        )

        # Path 2: Treatment -> Mediator -> Outcome (indirect effect)
        # Step 1: Treatment -> Mediator
        mediator_response = self._estimate_treatment_mediator_effect(
            treatment_variable,
            mediator_variable,
            treatment_delta,
            feature_values,
        )

        # Step 2: Mediator -> Outcome
        indirect_effect = self._estimate_mediator_outcome_effect(
            mediator_variable,
            outcome_variable,
            mediator_response,
            feature_values,
        )

        # Total effect
        total_effect = direct_effect + indirect_effect

        # Mediation proportion
        if abs(total_effect) > 1e-6:
            mediation_proportion = abs(indirect_effect) / abs(total_effect)
        else:
            mediation_proportion = 0.0

        # Generate explanation
        explanation = self._generate_mediation_explanation(
            treatment_variable,
            outcome_variable,
            mediator_variable,
            total_effect,
            direct_effect,
            indirect_effect,
            mediation_proportion,
        )

        return MediationAnalysis(
            treatment=treatment_variable,
            outcome=outcome_variable,
            mediator=mediator_variable,
            total_effect=total_effect,
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            mediation_proportion=mediation_proportion,
            explanation=explanation,
        )

    def find_alternative_interventions(
        self,
        baseline_intervention: Dict[UUID, Tuple[str, float]],
        baseline_cost: float,
        baseline_defaults_prevented: int,
        network_graph: nx.DiGraph,
        institution_features: Dict[UUID, Dict[str, float]],
        institution_names: Dict[UUID, str],
        max_alternatives: int = 5,
    ) -> List[AlternativeIntervention]:
        """
        Find alternative intervention strategies through backtracking

        Uses causal reasoning to find cheaper or more effective alternatives
        to a baseline intervention strategy.

        Args:
            baseline_intervention: Baseline strategy {institution_id: (intervention_type, amount)}
            baseline_cost: Total cost of baseline
            baseline_defaults_prevented: Effectiveness of baseline
            network_graph: Network structure
            institution_features: Features for all institutions
            institution_names: Mapping of IDs to names
            max_alternatives: Maximum number of alternatives to return

        Returns:
            List of alternative intervention strategies
        """
        logger.info("Finding alternative interventions via backtracking")

        alternatives = []

        # Strategy 1: Upstream interventions
        # Instead of saving downstream institutions, prevent the cascade at source
        upstream_strategy = self._find_upstream_intervention(
            baseline_intervention,
            network_graph,
            institution_features,
            institution_names,
        )
        if upstream_strategy:
            alternatives.append(upstream_strategy)

        # Strategy 2: Hub-based interventions
        # Target network hubs to maximize impact
        hub_strategy = self._find_hub_intervention(
            baseline_defaults_prevented,
            network_graph,
            institution_features,
            institution_names,
        )
        if hub_strategy:
            alternatives.append(hub_strategy)

        # Strategy 3: Cost-optimized intervention
        # Same effectiveness, lower cost
        cost_optimized = self._find_cost_optimized_intervention(
            baseline_intervention,
            baseline_defaults_prevented,
            baseline_cost,
            network_graph,
            institution_features,
            institution_names,
        )
        if cost_optimized:
            alternatives.append(cost_optimized)

        # Strategy 4: Distributed intervention
        # Spread interventions across multiple institutions
        distributed_strategy = self._find_distributed_intervention(
            baseline_intervention,
            baseline_defaults_prevented,
            network_graph,
            institution_features,
            institution_names,
        )
        if distributed_strategy:
            alternatives.append(distributed_strategy)

        # Sort by ROI and return top alternatives
        alternatives.sort(key=lambda x: x.roi, reverse=True)

        return alternatives[:max_alternatives]

    def rank_intervention_points(
        self,
        network_graph: nx.DiGraph,
        institution_features: Dict[UUID, Dict[str, float]],
        institution_names: Dict[UUID, str],
        default_predictions: Dict[UUID, float],
        at_risk_threshold: float = 0.3,
    ) -> List[InterventionPoint]:
        """
        Rank potential intervention points by effectiveness

        Identifies where interventions would have the greatest impact on
        preventing defaults and stabilizing the network.

        Args:
            network_graph: Network structure
            institution_features: Features for all institutions
            institution_names: Mapping of IDs to names
            default_predictions: Predicted default probabilities
            at_risk_threshold: Threshold for considering institution at risk

        Returns:
            Ranked list of intervention points
        """
        logger.info("Ranking intervention points")

        intervention_points = []

        # Compute network metrics
        centrality_scores = self._compute_intervention_centrality(network_graph)
        vulnerability_scores = self._compute_vulnerability_scores(
            network_graph,
            default_predictions,
        )

        # Analyze each institution
        for inst_id in network_graph.nodes():
            if inst_id not in institution_features:
                continue

            features = institution_features[inst_id]
            name = institution_names.get(inst_id, f"Institution {inst_id}")
            default_prob = default_predictions.get(inst_id, 0.0)

            # Skip if not at risk
            if default_prob < at_risk_threshold:
                continue

            # Compute effectiveness scores
            centrality = centrality_scores.get(inst_id, 0.0)
            vulnerability = vulnerability_scores.get(inst_id, 0.0)

            # Effectiveness score combines multiple factors
            effectiveness_score = (
                0.4 * default_prob +  # How at-risk is this institution
                0.3 * centrality +  # Network importance
                0.3 * vulnerability  # Vulnerability to contagion
            )

            # Cascade prevention score (based on out-degree and centrality)
            out_degree = network_graph.out_degree(inst_id)
            max_out_degree = max(dict(network_graph.out_degree()).values()) or 1
            cascade_prevention = (
                0.6 * (out_degree / max_out_degree) +
                0.4 * centrality
            )

            # Estimate impact
            institutions_saved = self._estimate_institutions_saved(
                inst_id,
                network_graph,
                default_predictions,
            )

            # Determine intervention type and amount
            intervention_type, recommended_amount = self._recommend_intervention(
                features,
                default_prob,
            )

            # Cost-effectiveness (benefit per unit cost)
            estimated_cost = self._estimate_intervention_cost(
                intervention_type,
                recommended_amount,
            )
            estimated_benefit = institutions_saved * 100.0  # Assume $100M per institution saved
            cost_effectiveness = estimated_benefit / estimated_cost if estimated_cost > 0 else 0

            # Network stabilization (reduction in systemic risk)
            network_stabilization = self._estimate_network_stabilization(
                inst_id,
                network_graph,
                centrality,
            )

            # Determine urgency
            if default_prob > 0.7:
                urgency = "CRITICAL"
            elif default_prob > 0.5:
                urgency = "HIGH"
            elif default_prob > 0.3:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"

            # Generate explanation
            explanation = self._generate_intervention_explanation(
                name,
                intervention_type,
                effectiveness_score,
                institutions_saved,
                urgency,
            )

            intervention_point = InterventionPoint(
                institution_id=inst_id,
                institution_name=name,
                intervention_type=intervention_type,
                effectiveness_score=effectiveness_score,
                cascade_prevention_score=cascade_prevention,
                cost_effectiveness=cost_effectiveness,
                institutions_saved=institutions_saved,
                expected_defaults_prevented=institutions_saved,
                network_stabilization=network_stabilization,
                recommended_amount=recommended_amount,
                urgency=urgency,
                explanation=explanation,
            )

            intervention_points.append(intervention_point)

        # Sort by effectiveness score
        intervention_points.sort(key=lambda x: x.effectiveness_score, reverse=True)

        return intervention_points

    # Helper methods

    def _estimate_direct_effect(
        self,
        treatment: str,
        outcome: str,
        delta: float,
        features: Dict[str, float],
    ) -> float:
        """Estimate direct effect of treatment on outcome"""
        # Simplified linear model for demonstration
        # In practice, would use more sophisticated causal inference

        # Effect depends on the variables involved
        effect_matrix = {
            ("Capital Ratio", "Default Probability"): -0.5,
            ("Liquidity Buffer", "Default Probability"): -0.4,
            ("Leverage", "Default Probability"): 0.3,
            ("Stress Level", "Default Probability"): 0.6,
        }

        key = (treatment, outcome)
        base_effect = effect_matrix.get(key, 0.0)

        return base_effect * delta

    def _estimate_treatment_mediator_effect(
        self,
        treatment: str,
        mediator: str,
        delta: float,
        features: Dict[str, float],
    ) -> float:
        """Estimate effect of treatment on mediator"""
        # Effect matrix for treatment -> mediator
        effect_matrix = {
            ("Capital Ratio", "Liquidity Buffer"): 0.6,
            ("Capital Ratio", "Stress Level"): -0.4,
            ("Liquidity Buffer", "Stress Level"): -0.5,
        }

        key = (treatment, mediator)
        base_effect = effect_matrix.get(key, 0.0)

        return base_effect * delta

    def _estimate_mediator_outcome_effect(
        self,
        mediator: str,
        outcome: str,
        mediator_change: float,
        features: Dict[str, float],
    ) -> float:
        """Estimate effect of mediator on outcome"""
        effect_matrix = {
            ("Liquidity Buffer", "Default Probability"): -0.4,
            ("Stress Level", "Default Probability"): 0.6,
        }

        key = (mediator, outcome)
        base_effect = effect_matrix.get(key, 0.0)

        return base_effect * mediator_change

    def _compute_intervention_centrality(
        self,
        network: nx.DiGraph,
    ) -> Dict[UUID, float]:
        """Compute centrality scores for intervention prioritization"""
        try:
            # Use eigenvector centrality as proxy for intervention importance
            centrality = nx.eigenvector_centrality(network, max_iter=1000, weight='weight')
        except:
            # Fallback to degree centrality
            centrality = nx.degree_centrality(network)

        return centrality

    def _compute_vulnerability_scores(
        self,
        network: nx.DiGraph,
        default_predictions: Dict[UUID, float],
    ) -> Dict[UUID, float]:
        """Compute vulnerability to contagion"""
        vulnerability = {}

        for node in network.nodes():
            # Vulnerability based on neighbors' default probabilities
            neighbors = list(network.predecessors(node))

            if neighbors:
                neighbor_risks = [default_predictions.get(n, 0.0) for n in neighbors]
                avg_neighbor_risk = np.mean(neighbor_risks)
                max_neighbor_risk = np.max(neighbor_risks)

                vulnerability[node] = 0.6 * avg_neighbor_risk + 0.4 * max_neighbor_risk
            else:
                vulnerability[node] = 0.0

        return vulnerability

    def _estimate_institutions_saved(
        self,
        institution_id: UUID,
        network: nx.DiGraph,
        default_predictions: Dict[UUID, float],
    ) -> int:
        """Estimate how many institutions would be saved by intervening"""
        # Count downstream institutions at risk
        try:
            descendants = nx.descendants(network, institution_id)
        except:
            descendants = set()

        saved = 0
        for desc in descendants:
            if default_predictions.get(desc, 0.0) > 0.3:
                saved += 1

        # Add the institution itself
        if default_predictions.get(institution_id, 0.0) > 0.3:
            saved += 1

        return saved

    def _recommend_intervention(
        self,
        features: Dict[str, float],
        default_prob: float,
    ) -> Tuple[str, float]:
        """Recommend intervention type and amount"""
        capital_ratio = features.get("Capital Ratio", 0.5)
        liquidity = features.get("Liquidity Buffer", 0.5)

        # Decision logic
        if capital_ratio < 0.1:
            return "Capital Injection", (0.15 - capital_ratio) * 1000.0
        elif liquidity < 0.1:
            return "Liquidity Support", (0.15 - liquidity) * 500.0
        else:
            return "Risk Management Enhancement", 50.0

    def _estimate_intervention_cost(
        self,
        intervention_type: str,
        amount: float,
    ) -> float:
        """Estimate cost of intervention"""
        cost_multipliers = {
            "Capital Injection": 1.0,
            "Liquidity Support": 0.8,
            "Risk Management Enhancement": 0.5,
        }

        multiplier = cost_multipliers.get(intervention_type, 1.0)
        return amount * multiplier

    def _estimate_network_stabilization(
        self,
        institution_id: UUID,
        network: nx.DiGraph,
        centrality: float,
    ) -> float:
        """Estimate impact on network stability"""
        # Higher centrality -> more stabilization
        return centrality

    def _generate_mediation_explanation(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        total: float,
        direct: float,
        indirect: float,
        proportion: float,
    ) -> str:
        """Generate explanation for mediation analysis"""
        lines = []

        lines.append(f"MEDIATION ANALYSIS: {treatment} -> {outcome}")
        lines.append("=" * 80)
        lines.append(f"Mediator: {mediator}")
        lines.append("")
        lines.append(f"Total Effect: {total:.4f}")
        lines.append(f"Direct Effect: {direct:.4f} ({(1-proportion)*100:.1f}%)")
        lines.append(f"Indirect Effect: {indirect:.4f} ({proportion*100:.1f}%)")
        lines.append("")

        if proportion > 0.7:
            lines.append(f"Most of the effect ({proportion*100:.0f}%) is mediated through {mediator}.")
            lines.append(f"To maximize impact, target {mediator} directly.")
        elif proportion > 0.3:
            lines.append(f"The effect is partially mediated through {mediator}.")
            lines.append(f"Both direct and indirect pathways are important.")
        else:
            lines.append(f"The effect is mostly direct ({(1-proportion)*100:.0f}%).")
            lines.append(f"{mediator} plays a minor mediating role.")

        return "\n".join(lines)

    def _generate_intervention_explanation(
        self,
        name: str,
        intervention_type: str,
        effectiveness: float,
        institutions_saved: int,
        urgency: str,
    ) -> str:
        """Generate explanation for intervention point"""
        return (
            f"{name} is a {urgency} priority intervention point. "
            f"{intervention_type} would have effectiveness score of {effectiveness:.2f} "
            f"and could save {institutions_saved} institutions from default."
        )

    # Strategies for alternative interventions

    def _find_upstream_intervention(
        self,
        baseline: Dict[UUID, Tuple[str, float]],
        network: nx.DiGraph,
        features: Dict[UUID, Dict[str, float]],
        names: Dict[UUID, str],
    ) -> Optional[AlternativeIntervention]:
        """Find upstream intervention alternative"""
        # Find institutions that have paths to baseline targets
        target_institutions = set()

        for inst_id in baseline.keys():
            try:
                predecessors = nx.ancestors(network, inst_id)
                target_institutions.update(predecessors)
            except:
                pass

        if not target_institutions:
            return None

        # Pick top 2 upstream institutions by centrality
        centrality = self._compute_intervention_centrality(network)
        upstream_ranked = sorted(
            target_institutions,
            key=lambda x: centrality.get(x, 0),
            reverse=True
        )[:2]

        # Estimate cost and effectiveness
        total_cost = 100.0 * len(upstream_ranked)  # Simplified
        defaults_prevented = len(baseline)
        roi = defaults_prevented * 100.0 / total_cost

        intervention_types = {
            inst_id: "Capital Injection" for inst_id in upstream_ranked
        }

        explanation = (
            f"Upstream strategy: Intervene at {len(upstream_ranked)} upstream "
            f"institutions to prevent cascade before it reaches {len(baseline)} targets."
        )

        return AlternativeIntervention(
            strategy_name="Upstream Prevention",
            target_institutions=list(upstream_ranked),
            intervention_types=intervention_types,
            expected_effectiveness=0.85,
            defaults_prevented=defaults_prevented,
            total_cost=total_cost,
            roi=roi,
            better_than_baseline=True,
            cost_saving=len(baseline) * 100.0 - total_cost,
            explanation=explanation,
        )

    def _find_hub_intervention(
        self,
        baseline_effectiveness: int,
        network: nx.DiGraph,
        features: Dict[UUID, Dict[str, float]],
        names: Dict[UUID, str],
    ) -> Optional[AlternativeIntervention]:
        """Find hub-based intervention"""
        centrality = self._compute_intervention_centrality(network)

        # Select top 3 hubs
        hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        hub_ids = [h[0] for h in hubs]

        total_cost = 150.0 * len(hub_ids)
        defaults_prevented = int(baseline_effectiveness * 1.2)  # 20% more effective
        roi = defaults_prevented * 100.0 / total_cost

        intervention_types = {
            inst_id: "Liquidity Support" for inst_id in hub_ids
        }

        explanation = (
            f"Hub strategy: Target {len(hub_ids)} network hubs to maximize "
            f"systemic stability and prevent {defaults_prevented} defaults."
        )

        return AlternativeIntervention(
            strategy_name="Hub-Based Intervention",
            target_institutions=hub_ids,
            intervention_types=intervention_types,
            expected_effectiveness=0.9,
            defaults_prevented=defaults_prevented,
            total_cost=total_cost,
            roi=roi,
            better_than_baseline=roi > 1.0,
            cost_saving=0.0,
            explanation=explanation,
        )

    def _find_cost_optimized_intervention(
        self,
        baseline: Dict[UUID, Tuple[str, float]],
        baseline_effectiveness: int,
        baseline_cost: float,
        network: nx.DiGraph,
        features: Dict[UUID, Dict[str, float]],
        names: Dict[UUID, str],
    ) -> Optional[AlternativeIntervention]:
        """Find cost-optimized intervention"""
        # Reduce intervention amounts by 30%
        optimized = {}
        total_cost = 0.0

        for inst_id, (intervention_type, amount) in baseline.items():
            reduced_amount = amount * 0.7
            optimized[inst_id] = intervention_type
            total_cost += self._estimate_intervention_cost(intervention_type, reduced_amount)

        # Assume slightly lower effectiveness
        defaults_prevented = int(baseline_effectiveness * 0.85)
        roi = defaults_prevented * 100.0 / total_cost

        explanation = (
            f"Cost-optimized strategy: Reduce intervention amounts by 30% "
            f"while maintaining 85% effectiveness, saving ${baseline_cost - total_cost:.0f}M."
        )

        return AlternativeIntervention(
            strategy_name="Cost-Optimized",
            target_institutions=list(baseline.keys()),
            intervention_types=optimized,
            expected_effectiveness=0.85,
            defaults_prevented=defaults_prevented,
            total_cost=total_cost,
            roi=roi,
            better_than_baseline=total_cost < baseline_cost,
            cost_saving=baseline_cost - total_cost,
            explanation=explanation,
        )

    def _find_distributed_intervention(
        self,
        baseline: Dict[UUID, Tuple[str, float]],
        baseline_effectiveness: int,
        network: nx.DiGraph,
        features: Dict[UUID, Dict[str, float]],
        names: Dict[UUID, str],
    ) -> Optional[AlternativeIntervention]:
        """Find distributed intervention strategy"""
        # Spread interventions across more institutions with smaller amounts
        num_targets = len(baseline) * 2
        all_institutions = list(network.nodes())[:num_targets]

        intervention_types = {
            inst_id: "Risk Management Enhancement" for inst_id in all_institutions
        }

        total_cost = 50.0 * num_targets
        defaults_prevented = int(baseline_effectiveness * 0.9)
        roi = defaults_prevented * 100.0 / total_cost

        explanation = (
            f"Distributed strategy: Spread interventions across {num_targets} "
            f"institutions with smaller individual investments for network resilience."
        )

        return AlternativeIntervention(
            strategy_name="Distributed Intervention",
            target_institutions=all_institutions,
            intervention_types=intervention_types,
            expected_effectiveness=0.75,
            defaults_prevented=defaults_prevented,
            total_cost=total_cost,
            roi=roi,
            better_than_baseline=roi > 1.5,
            cost_saving=0.0,
            explanation=explanation,
        )
