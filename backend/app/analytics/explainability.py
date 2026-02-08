"""
Explainability Engine for RUDRA

Provides SHAP-based feature importance, causal attribution, and natural language
explanations for ML predictions and simulation outcomes.

Integrates with:
- DefaultPredictorModel for SHAP analysis
- NetworkAnalyzer for contagion sources
- SimulationEngine for counterfactual analysis
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import numpy as np
import torch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.features.extractor import FeatureExtractor, InstitutionFeatures

logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """SHAP values and feature importance for a single prediction"""
    institution_id: UUID
    institution_name: str
    prediction: float  # Default probability
    base_value: float  # Expected value (baseline)
    shap_values: np.ndarray  # SHAP values for each feature
    feature_names: List[str]
    feature_values: np.ndarray  # Actual feature values

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float, float]]:
        """
        Get top N features by absolute SHAP value

        Returns:
            List of (feature_name, shap_value, feature_value) tuples
        """
        abs_shap = np.abs(self.shap_values)
        top_indices = np.argsort(abs_shap)[-n:][::-1]

        return [
            (
                self.feature_names[i],
                float(self.shap_values[i]),
                float(self.feature_values[i])
            )
            for i in top_indices
        ]


@dataclass
class DefaultCauseAnalysis:
    """Analysis of why an institution defaulted or is at risk"""
    institution_id: UUID
    institution_name: str
    default_probability: float
    risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"

    # Primary causes
    primary_causes: List[Dict[str, Any]]  # Top factors contributing to default

    # Contributing factors
    network_factors: Dict[str, float]  # Network-related risks
    financial_factors: Dict[str, float]  # Balance sheet weaknesses
    contagion_factors: Dict[str, float]  # Contagion effects

    # Natural language explanation
    explanation_text: str


@dataclass
class ContagionSource:
    """Identifies which institutions caused a cascade"""
    cascade_id: str
    trigger_institution_id: UUID
    trigger_institution_name: str

    # Cascade statistics
    total_defaults: int
    cascade_depth: int  # How many hops from trigger
    affected_institutions: List[UUID]

    # Propagation path
    propagation_paths: List[List[UUID]]  # Paths from trigger to each defaulted bank

    # Contribution scores
    institution_contributions: Dict[UUID, float]  # How much each bank contributed to cascade

    explanation: str


@dataclass
class PolicyImpactExplanation:
    """Explains the impact of a policy intervention"""
    policy_name: str
    baseline_defaults: int
    counterfactual_defaults: int
    defaults_prevented: int
    effectiveness: float  # Percentage reduction

    # Institution-level impact
    saved_institutions: List[UUID]
    still_at_risk: List[UUID]

    # Cost-benefit metrics
    estimated_cost: float
    estimated_benefit: float
    roi: float

    explanation: str


class ExplainabilityEngine:
    """
    Main engine for explainability and interpretability

    Provides:
    1. SHAP-based feature importance
    2. Causal attribution for defaults
    3. Contagion source identification
    4. Policy impact analysis
    5. Natural language explanations
    """

    # Feature names matching FeatureExtractor order
    FEATURE_NAMES = [
        # Financial (6)
        "Capital Ratio",
        "Liquidity Buffer",
        "Leverage",
        "Credit Exposure",
        "Risk Appetite",
        "Stress Level",
        # Network (6)
        "Degree Centrality",
        "Betweenness Centrality",
        "Eigenvector Centrality",
        "PageRank",
        "In-Degree",
        "Out-Degree",
        # Market (4)
        "Default Probability Prior",
        "Credit Spread",
        "Volatility",
        "Market Pressure",
        # Neighborhood (4)
        "Neighbor Avg Stress",
        "Neighbor Max Stress",
        "Neighbor Default Count",
        "Neighbor Avg Capital",
    ]

    def __init__(
        self,
        model: DefaultPredictorModel,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize explainability engine

        Args:
            model: Trained DefaultPredictorModel
            feature_extractor: FeatureExtractor instance
            device: Torch device
        """
        self.model = model
        self.model.eval()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE:
            self._init_shap_explainer()

        logger.info("ExplainabilityEngine initialized")

    def _init_shap_explainer(self):
        """Initialize SHAP explainer with model"""
        try:
            # Create a wrapper for the PyTorch model
            def model_predict(x):
                """Wrapper for SHAP that accepts numpy arrays"""
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    predictions = self.model(x_tensor)
                return predictions.cpu().numpy()

            # Use DeepExplainer for neural networks
            # For now, we'll use KernelExplainer which is model-agnostic
            self.model_predict = model_predict
            logger.info("SHAP model wrapper initialized")

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def compute_shap_values(
        self,
        institution_features: InstitutionFeatures,
        institution_name: str,
        background_samples: Optional[np.ndarray] = None,
        nsamples: int = 100,
    ) -> SHAPExplanation:
        """
        Compute SHAP values for feature importance

        Args:
            institution_features: Features for the institution
            institution_name: Name of the institution
            background_samples: Background dataset for SHAP (optional)
            nsamples: Number of samples for SHAP kernel

        Returns:
            SHAPExplanation with SHAP values and feature importance
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, returning zero importance")
            feature_array = institution_features.to_array()
            return SHAPExplanation(
                institution_id=institution_features.institution_id,
                institution_name=institution_name,
                prediction=0.5,
                base_value=0.5,
                shap_values=np.zeros_like(feature_array),
                feature_names=self.FEATURE_NAMES,
                feature_values=feature_array,
            )

        # Get feature array
        feature_array = institution_features.to_array().reshape(1, -1)

        # Get prediction
        with torch.no_grad():
            x_tensor = torch.FloatTensor(feature_array).to(self.device)
            prediction = self.model(x_tensor).cpu().numpy()[0, 0]

        # Compute SHAP values
        try:
            # If no background samples provided, use the current sample
            if background_samples is None:
                background_samples = feature_array

            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                self.model_predict,
                background_samples,
                link="identity"
            )

            # Compute SHAP values
            shap_values = explainer.shap_values(
                feature_array,
                nsamples=nsamples,
                silent=True
            )

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = shap_values.flatten()
            base_value = explainer.expected_value

            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}, using gradient-based fallback")
            # Fallback to gradient-based importance
            x_tensor.requires_grad = True
            output = self.model(x_tensor)
            output.backward()
            shap_values = (x_tensor.grad * x_tensor).detach().cpu().numpy().flatten()
            base_value = 0.5

        return SHAPExplanation(
            institution_id=institution_features.institution_id,
            institution_name=institution_name,
            prediction=float(prediction),
            base_value=base_value,
            shap_values=shap_values,
            feature_names=self.FEATURE_NAMES,
            feature_values=feature_array.flatten(),
        )

    def attribute_default_causes(
        self,
        institution_features: InstitutionFeatures,
        institution_name: str,
        shap_explanation: Optional[SHAPExplanation] = None,
    ) -> DefaultCauseAnalysis:
        """
        Analyze why an institution defaulted or is at high risk

        Args:
            institution_features: Features of the institution
            institution_name: Name of the institution
            shap_explanation: Pre-computed SHAP explanation (optional)

        Returns:
            DefaultCauseAnalysis with root causes and explanations
        """
        # Compute SHAP if not provided
        if shap_explanation is None:
            shap_explanation = self.compute_shap_values(
                institution_features,
                institution_name
            )

        # Classify risk level
        prob = shap_explanation.prediction
        if prob < 0.15:
            risk_level = "LOW"
        elif prob < 0.30:
            risk_level = "MODERATE"
        elif prob < 0.50:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Get top contributing features
        top_features = shap_explanation.get_top_features(n=10)

        # Categorize causes
        primary_causes = []
        network_factors = {}
        financial_factors = {}
        contagion_factors = {}

        for feature_name, shap_value, feature_value in top_features:
            cause = {
                "feature": feature_name,
                "shap_contribution": float(shap_value),
                "feature_value": float(feature_value),
                "impact": "Increases risk" if shap_value > 0 else "Decreases risk"
            }
            primary_causes.append(cause)

            # Categorize
            if any(net in feature_name for net in ["Centrality", "Degree", "PageRank"]):
                network_factors[feature_name] = float(shap_value)
            elif any(fin in feature_name for fin in ["Capital", "Liquidity", "Leverage", "Exposure"]):
                financial_factors[feature_name] = float(shap_value)
            elif "Neighbor" in feature_name:
                contagion_factors[feature_name] = float(shap_value)

        # Generate explanation text
        explanation = self.generate_explanation_text(
            institution_name,
            risk_level,
            prob,
            primary_causes,
            network_factors,
            financial_factors,
            contagion_factors,
        )

        return DefaultCauseAnalysis(
            institution_id=institution_features.institution_id,
            institution_name=institution_name,
            default_probability=prob,
            risk_level=risk_level,
            primary_causes=primary_causes,
            network_factors=network_factors,
            financial_factors=financial_factors,
            contagion_factors=contagion_factors,
            explanation_text=explanation,
        )

    def identify_contagion_sources(
        self,
        cascade_id: str,
        trigger_institution_id: UUID,
        trigger_institution_name: str,
        defaulted_institutions: List[UUID],
        propagation_graph: Dict[UUID, List[UUID]],
        institution_names: Dict[UUID, str],
    ) -> ContagionSource:
        """
        Identify which institutions caused a cascade event

        Args:
            cascade_id: Unique identifier for the cascade
            trigger_institution_id: Institution that triggered the cascade
            trigger_institution_name: Name of trigger institution
            defaulted_institutions: List of all defaulted institution IDs
            propagation_graph: Graph showing how default propagated
            institution_names: Mapping of institution IDs to names

        Returns:
            ContagionSource analysis
        """
        # Compute propagation paths using BFS
        propagation_paths = []
        institution_contributions = {trigger_institution_id: 1.0}

        # BFS to find paths
        from collections import deque, defaultdict

        visited = set()
        queue = deque([(trigger_institution_id, [trigger_institution_id], 0)])
        depth_by_institution = defaultdict(lambda: float('inf'))
        depth_by_institution[trigger_institution_id] = 0

        max_depth = 0

        while queue:
            current_id, path, depth = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)
            max_depth = max(max_depth, depth)
            depth_by_institution[current_id] = min(depth_by_institution[current_id], depth)

            # Record path if this institution defaulted
            if current_id in defaulted_institutions and current_id != trigger_institution_id:
                propagation_paths.append(path)

            # Explore neighbors
            if current_id in propagation_graph:
                for neighbor_id in propagation_graph[current_id]:
                    if neighbor_id not in visited:
                        new_path = path + [neighbor_id]
                        queue.append((neighbor_id, new_path, depth + 1))

                        # Contribution score (decays with distance)
                        contribution = 1.0 / (depth + 2)
                        if neighbor_id in institution_contributions:
                            institution_contributions[neighbor_id] += contribution
                        else:
                            institution_contributions[neighbor_id] = contribution

        # Generate explanation
        explanation = self._generate_contagion_explanation(
            trigger_institution_name,
            len(defaulted_institutions),
            max_depth,
            institution_contributions,
            institution_names,
        )

        return ContagionSource(
            cascade_id=cascade_id,
            trigger_institution_id=trigger_institution_id,
            trigger_institution_name=trigger_institution_name,
            total_defaults=len(defaulted_institutions),
            cascade_depth=max_depth,
            affected_institutions=defaulted_institutions,
            propagation_paths=propagation_paths,
            institution_contributions=institution_contributions,
            explanation=explanation,
        )

    def analyze_policy_impact(
        self,
        policy_name: str,
        baseline_defaults: int,
        counterfactual_defaults: int,
        saved_institutions: List[UUID],
        still_at_risk: List[UUID],
        estimated_cost: float,
        estimated_benefit: float,
    ) -> PolicyImpactExplanation:
        """
        Analyze the impact of a policy intervention

        Args:
            policy_name: Name of the policy
            baseline_defaults: Number of defaults without intervention
            counterfactual_defaults: Number of defaults with intervention
            saved_institutions: Institutions saved by the policy
            still_at_risk: Institutions still at risk
            estimated_cost: Cost of implementing policy
            estimated_benefit: Estimated benefit from preventing defaults

        Returns:
            PolicyImpactExplanation
        """
        defaults_prevented = baseline_defaults - counterfactual_defaults
        effectiveness = (defaults_prevented / baseline_defaults * 100) if baseline_defaults > 0 else 0
        roi = (estimated_benefit - estimated_cost) / estimated_cost if estimated_cost > 0 else 0

        # Generate explanation
        explanation = self._generate_policy_explanation(
            policy_name,
            baseline_defaults,
            counterfactual_defaults,
            defaults_prevented,
            effectiveness,
            roi,
        )

        return PolicyImpactExplanation(
            policy_name=policy_name,
            baseline_defaults=baseline_defaults,
            counterfactual_defaults=counterfactual_defaults,
            defaults_prevented=defaults_prevented,
            effectiveness=effectiveness,
            saved_institutions=saved_institutions,
            still_at_risk=still_at_risk,
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            roi=roi,
            explanation=explanation,
        )

    def generate_explanation_text(
        self,
        institution_name: str,
        risk_level: str,
        default_probability: float,
        primary_causes: List[Dict[str, Any]],
        network_factors: Dict[str, float],
        financial_factors: Dict[str, float],
        contagion_factors: Dict[str, float],
    ) -> str:
        """
        Generate natural language explanation for default risk

        Args:
            institution_name: Name of the institution
            risk_level: Risk classification
            default_probability: Predicted default probability
            primary_causes: Top contributing factors
            network_factors: Network-related factors
            financial_factors: Financial factors
            contagion_factors: Contagion factors

        Returns:
            Human-readable explanation text
        """
        lines = []

        # Header
        lines.append(f"DEFAULT RISK ANALYSIS: {institution_name}")
        lines.append("=" * 80)
        lines.append(f"Risk Level: {risk_level}")
        lines.append(f"Default Probability: {default_probability:.2%}")
        lines.append("")

        # Primary causes
        lines.append("PRIMARY RISK FACTORS:")
        lines.append("-" * 80)
        for i, cause in enumerate(primary_causes[:5], 1):
            feature = cause["feature"]
            shap = cause["shap_contribution"]
            value = cause["feature_value"]
            impact = cause["impact"]

            lines.append(f"{i}. {feature}: {value:.3f}")
            lines.append(f"   Contribution: {shap:+.4f} ({impact})")
        lines.append("")

        # Category analysis
        if financial_factors:
            lines.append("FINANCIAL FACTORS:")
            total_financial = sum(abs(v) for v in financial_factors.values())
            lines.append(f"  Total contribution: {total_financial:.4f}")
            for factor, value in sorted(financial_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                lines.append(f"  - {factor}: {value:+.4f}")
            lines.append("")

        if network_factors:
            lines.append("NETWORK FACTORS:")
            total_network = sum(abs(v) for v in network_factors.values())
            lines.append(f"  Total contribution: {total_network:.4f}")
            for factor, value in sorted(network_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                lines.append(f"  - {factor}: {value:+.4f}")
            lines.append("")

        if contagion_factors:
            lines.append("CONTAGION FACTORS:")
            total_contagion = sum(abs(v) for v in contagion_factors.values())
            lines.append(f"  Total contribution: {total_contagion:.4f}")
            for factor, value in sorted(contagion_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                lines.append(f"  - {factor}: {value:+.4f}")
            lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append("-" * 80)
        if risk_level == "CRITICAL":
            lines.append(f"{institution_name} is at CRITICAL risk of default.")
            lines.append("IMMEDIATE intervention required.")
        elif risk_level == "HIGH":
            lines.append(f"{institution_name} is at HIGH risk of default.")
            lines.append("Enhanced monitoring and preventive measures recommended.")
        elif risk_level == "MODERATE":
            lines.append(f"{institution_name} shows MODERATE default risk.")
            lines.append("Standard monitoring procedures should be maintained.")
        else:
            lines.append(f"{institution_name} shows LOW default risk.")
            lines.append("Routine oversight is sufficient.")

        return "\n".join(lines)

    def _generate_contagion_explanation(
        self,
        trigger_name: str,
        total_defaults: int,
        cascade_depth: int,
        contributions: Dict[UUID, float],
        institution_names: Dict[UUID, str],
    ) -> str:
        """Generate explanation for contagion cascade"""
        lines = []

        lines.append(f"CONTAGION CASCADE ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Trigger Institution: {trigger_name}")
        lines.append(f"Total Defaults: {total_defaults}")
        lines.append(f"Cascade Depth: {cascade_depth} hops")
        lines.append("")

        lines.append("TOP CONTRIBUTORS TO CASCADE:")
        lines.append("-" * 80)

        # Sort by contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (inst_id, score) in enumerate(sorted_contributions, 1):
            name = institution_names.get(inst_id, f"Institution {inst_id}")
            lines.append(f"{i}. {name}: {score:.3f}")

        lines.append("")
        lines.append(f"The cascade was triggered by {trigger_name} and propagated")
        lines.append(f"through {cascade_depth} levels of the network, affecting {total_defaults} institutions.")

        return "\n".join(lines)

    def _generate_policy_explanation(
        self,
        policy_name: str,
        baseline: int,
        counterfactual: int,
        prevented: int,
        effectiveness: float,
        roi: float,
    ) -> str:
        """Generate explanation for policy impact"""
        lines = []

        lines.append(f"POLICY IMPACT ANALYSIS: {policy_name}")
        lines.append("=" * 80)
        lines.append(f"Baseline Defaults: {baseline}")
        lines.append(f"With Policy: {counterfactual}")
        lines.append(f"Defaults Prevented: {prevented}")
        lines.append(f"Effectiveness: {effectiveness:.1f}%")
        lines.append(f"ROI: {roi:.2f}x")
        lines.append("")

        if effectiveness > 70:
            lines.append(f"ASSESSMENT: {policy_name} is HIGHLY EFFECTIVE")
            lines.append(f"The policy prevents {effectiveness:.0f}% of defaults with strong ROI.")
        elif effectiveness > 40:
            lines.append(f"ASSESSMENT: {policy_name} is MODERATELY EFFECTIVE")
            lines.append(f"The policy shows meaningful impact but may need refinement.")
        elif effectiveness > 0:
            lines.append(f"ASSESSMENT: {policy_name} is MARGINALLY EFFECTIVE")
            lines.append(f"Consider alternative interventions for better results.")
        else:
            lines.append(f"ASSESSMENT: {policy_name} is INEFFECTIVE")
            lines.append(f"This policy does not prevent defaults and should be reconsidered.")

        return "\n".join(lines)
