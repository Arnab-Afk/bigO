"""
ML-Based Risk Mitigation Advisor

This module provides intelligent risk reduction strategies for agents using ML predictions.
Each agent uses ML to:
1. Predict systemic risk exposure
2. Identify optimal risk-reducing actions
3. Dynamically adjust policies to minimize network-wide risk
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import numpy as np
import torch
from scipy.optimize import minimize

from app.ml.inference.predictor import DefaultPredictor, PredictionResult
from app.engine.game_theory import AgentState, ActionType, AgentAction

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification"""
    CRITICAL = "critical"  # >0.7 default probability
    HIGH = "high"          # 0.4-0.7
    MEDIUM = "medium"      # 0.2-0.4
    LOW = "low"            # <0.2


@dataclass
class RiskMitigationAction:
    """Risk mitigation recommendation from ML model"""
    action_type: ActionType
    target_id: Optional[UUID]
    magnitude: float
    expected_risk_reduction: float
    confidence: float
    reasoning: str


@dataclass
class RiskAssessment:
    """Complete risk assessment for an agent"""
    agent_id: UUID
    current_risk_level: RiskLevel
    default_probability: float
    systemic_importance: float
    ml_confidence: float
    exposures_at_risk: List[Tuple[UUID, float]]  # (counterparty_id, risk_score)
    recommended_actions: List[RiskMitigationAction]


class MLRiskMitigationAdvisor:
    """
    ML-powered risk mitigation advisor for agents.
    
    Uses trained ML models to:
    - Predict default probabilities
    - Assess network contagion risk
    - Recommend optimal policy adjustments
    - Prioritize actions by risk-reduction impact
    """
    
    def __init__(
        self,
        default_predictor: Optional[DefaultPredictor] = None,
        risk_aversion: float = 0.3,  # Lower default for stability
        min_confidence_threshold: float = 0.6,
    ):
        """
        Args:
            default_predictor: ML model for default prediction
            risk_aversion: How aggressively to reduce risk (0-1)
            min_confidence_threshold: Minimum ML confidence to use predictions
        """
        self.default_predictor = default_predictor
        self.risk_aversion = risk_aversion
        self.min_confidence_threshold = min_confidence_threshold
        
        logger.info(
            f"MLRiskMitigationAdvisor initialized with risk_aversion={risk_aversion}"
        )
    
    def assess_risk(
        self,
        agent_id: UUID,
        agent_state: AgentState,
        network,
        all_agent_states: Dict[UUID, AgentState],
        centralities: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment using ML predictions.
        
        Args:
            agent_id: Agent to assess
            agent_state: Current agent state
            network: Network graph
            all_agent_states: All agent states
            centralities: Network centrality measures
        
        Returns:
            RiskAssessment with recommendations
        """
        # Get ML prediction
        if self.default_predictor and self.default_predictor.model:
            prediction = self.default_predictor.predict(
                institution_id=agent_id,
                agent_state=agent_state,
                network=network,
                all_agent_states=all_agent_states,
                centralities=centralities,
                use_confidence=True,
            )
            default_prob = prediction.probability
            ml_confidence = prediction.confidence
        else:
            # Fallback to heuristic
            default_prob = self._heuristic_default_probability(agent_state)
            ml_confidence = 0.5
        
        # Classify risk level
        risk_level = self._classify_risk_level(default_prob)
        
        # Assess systemic importance
        systemic_importance = self._compute_systemic_importance(
            agent_id, network, centralities
        )
        
        # Identify risky exposures
        exposures_at_risk = self._identify_risky_exposures(
            agent_id, network, all_agent_states
        )
        
        # Generate risk mitigation recommendations
        recommended_actions = self._generate_mitigation_actions(
            agent_id=agent_id,
            agent_state=agent_state,
            risk_level=risk_level,
            default_prob=default_prob,
            exposures_at_risk=exposures_at_risk,
            systemic_importance=systemic_importance,
            ml_confidence=ml_confidence,
        )
        
        return RiskAssessment(
            agent_id=agent_id,
            current_risk_level=risk_level,
            default_probability=default_prob,
            systemic_importance=systemic_importance,
            ml_confidence=ml_confidence,
            exposures_at_risk=exposures_at_risk,
            recommended_actions=recommended_actions,
        )
    
    def _classify_risk_level(self, default_prob: float) -> RiskLevel:
        """Classify default probability into risk levels"""
        if default_prob >= 0.7:
            return RiskLevel.CRITICAL
        elif default_prob >= 0.4:
            return RiskLevel.HIGH
        elif default_prob >= 0.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _compute_systemic_importance(
        self,
        agent_id: UUID,
        network,
        centralities: Optional[Dict] = None,
    ) -> float:
        """
        Compute how systemically important this agent is.
        Higher = more critical to network stability.
        """
        if centralities and agent_id in centralities:
            # Weighted combination of centrality measures
            betweenness = centralities.get('betweenness', {}).get(agent_id, 0)
            pagerank = centralities.get('pagerank', {}).get(agent_id, 0)
            degree = centralities.get('degree', {}).get(agent_id, 0)
            
            systemic_importance = (
                0.4 * betweenness +
                0.3 * pagerank +
                0.3 * degree
            )
        else:
            # Fallback: use degree centrality
            try:
                degree = network.degree(agent_id)
                total_nodes = len(network.nodes())
                systemic_importance = degree / max(total_nodes - 1, 1)
            except:
                systemic_importance = 0.5
        
        return min(1.0, systemic_importance)
    
    def _identify_risky_exposures(
        self,
        agent_id: UUID,
        network,
        all_agent_states: Dict[UUID, AgentState],
    ) -> List[Tuple[UUID, float]]:
        """
        Identify counterparties that pose high risk based on their state.
        Returns list of (counterparty_id, risk_score) sorted by risk.
        """
        risky_exposures = []
        
        try:
            # Check outgoing edges (exposures to others)
            for successor in network.successors(agent_id):
                if successor in all_agent_states:
                    counterparty_state = all_agent_states[successor]
                    
                    # Risk score based on counterparty health
                    risk_score = (
                        (1 - counterparty_state.capital_ratio / 10) * 0.3 +
                        (1 - counterparty_state.liquidity_buffer) * 0.3 +
                        counterparty_state.default_probability * 0.4
                    )
                    risk_score = min(1.0, max(0.0, risk_score))
                    
                    if risk_score > 0.3:  # Only flag significant risks
                        risky_exposures.append((successor, risk_score))
        except Exception as e:
            logger.warning(f"Error identifying risky exposures: {e}")
        
        # Sort by risk score (highest first)
        risky_exposures.sort(key=lambda x: x[1], reverse=True)
        return risky_exposures
    
    def _generate_mitigation_actions(
        self,
        agent_id: UUID,
        agent_state: AgentState,
        risk_level: RiskLevel,
        default_prob: float,
        exposures_at_risk: List[Tuple[UUID, float]],
        systemic_importance: float,
        ml_confidence: float,
    ) -> List[RiskMitigationAction]:
        """
        Generate prioritized list of risk mitigation actions.
        Uses ML-guided decision making to recommend optimal actions.
        """
        actions = []
        
        # Action 1: Reduce credit exposure to risky counterparties (GRADUAL)
        if exposures_at_risk:
            for counterparty_id, risk_score in exposures_at_risk[:2]:  # Only top 2
                # Much smaller reductions (max 10%)
                reduction_magnitude = min(0.1, risk_score * self.risk_aversion * 0.2)
                expected_risk_reduction = risk_score * reduction_magnitude * 0.15
                
                actions.append(RiskMitigationAction(
                    action_type=ActionType.ADJUST_CREDIT_LIMIT,
                    target_id=counterparty_id,
                    magnitude=-reduction_magnitude,  # Negative = reduce
                    expected_risk_reduction=expected_risk_reduction,
                    confidence=ml_confidence * 0.9,
                    reasoning=f"Gradually reduce exposure to high-risk counterparty "
                              f"(risk score: {risk_score:.2f})"
                ))
        
        # Action 2: Increase liquidity buffer if below threshold (GRADUAL)
        if agent_state.liquidity_buffer < 0.3:  # Lower threshold
            # Much smaller increases (max 10%)
            liquidity_increase = min(0.1, (0.3 - agent_state.liquidity_buffer) * 0.5)
            expected_risk_reduction = liquidity_increase * 0.1
            
            actions.append(RiskMitigationAction(
                action_type=ActionType.LIQUIDITY_DECISION,
                target_id=None,
                magnitude=liquidity_increase,
                expected_risk_reduction=expected_risk_reduction,
                confidence=0.85,
                reasoning=f"Build liquidity buffer from {agent_state.liquidity_buffer:.2f} "
                          f"to safer levels"
            ))
        
        # Action 3: Adjust margin requirements if risk is high (GRADUAL)
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Smaller margin increases (5-10%)
            margin_increase = 0.05 if risk_level == RiskLevel.HIGH else 0.1
            expected_risk_reduction = margin_increase * 0.08
            
            actions.append(RiskMitigationAction(
                action_type=ActionType.MODIFY_MARGIN,
                target_id=None,
                magnitude=margin_increase,
                expected_risk_reduction=expected_risk_reduction,
                confidence=ml_confidence,
                reasoning=f"Increase margins due to {risk_level.value} risk level "
                          f"(default prob: {default_prob:.2f})"
            ))
        
        # Action 4: Make collateral calls if capital ratio is low (only if critical)
        if agent_state.capital_ratio < 0.9 and risk_level == RiskLevel.CRITICAL:  # Only when really needed
            collateral_call_amount = (1.0 - agent_state.capital_ratio) * 1.0  # Less aggressive
            expected_risk_reduction = collateral_call_amount * 0.2
            
            actions.append(RiskMitigationAction(
                action_type=ActionType.COLLATERAL_CALL,
                target_id=None,
                magnitude=collateral_call_amount,
                expected_risk_reduction=expected_risk_reduction,
                confidence=0.9,
                reasoning=f"Shore up capital ratio from {agent_state.capital_ratio:.2f} "
                          f"to meet regulatory requirements"
            ))
        
        # Action 5: Reroute trades away from stressed nodes (only if very high risk)
        if exposures_at_risk and systemic_importance > 0.7 and risk_level == RiskLevel.CRITICAL:
            for counterparty_id, risk_score in exposures_at_risk[:1]:  # Only worst one
                if risk_score > 0.6:  # Only very high risk
                    expected_risk_reduction = risk_score * 0.1
                    
                    actions.append(RiskMitigationAction(
                        action_type=ActionType.REROUTE_TRADE,
                        target_id=counterparty_id,
                        magnitude=0.3,  # Reroute only 30% of trades
                        expected_risk_reduction=expected_risk_reduction,
                        confidence=ml_confidence * 0.6,
                        reasoning=f"Reroute some trades from critically stressed counterparty "
                                  f"(risk: {risk_score:.2f})"
                    ))
        
        # Sort actions by expected risk reduction (highest first)
        actions.sort(key=lambda x: x.expected_risk_reduction, reverse=True)
        
        return actions
    
    def _heuristic_default_probability(self, agent_state: AgentState) -> float:
        """
        Fallback heuristic for default probability when ML model is unavailable.
        """
        # Simple heuristic based on capital ratio and liquidity
        capital_risk = max(0, (1.0 - agent_state.capital_ratio / 1.08)) * 0.5
        liquidity_risk = max(0, (0.3 - agent_state.liquidity_buffer)) * 0.3
        stress_risk = agent_state.stress_level * 0.2
        
        default_prob = min(1.0, capital_risk + liquidity_risk + stress_risk)
        return default_prob
    
    def optimize_policy_parameters(
        self,
        agent_state: AgentState,
        risk_assessment: RiskAssessment,
        current_policies: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Use optimization to find best policy parameters that minimize risk
        while maintaining operational viability.
        
        Args:
            agent_state: Current agent state
            risk_assessment: Risk assessment from ML
            current_policies: Current policy parameters
        
        Returns:
            Optimized policy parameters
        """
        # Define objective: CAPITAL PRESERVATION first, then risk minimization
        def objective(params):
            credit_limit, risk_appetite, interbank_limit = params
            
            # Risk component (want to minimize)
            risk_cost = risk_assessment.default_probability * 5.0
            
            # Operational viability - CRITICAL to maintain lending (revenue)
            # Penalize heavily if credit limit drops below 70%
            if credit_limit < 0.7:
                activity_penalty = (0.7 - credit_limit) * 10.0
            else:
                activity_penalty = 0.0
            
            # Stability bonus - prefer gradual changes
            current_credit = current_policies.get('credit_supply_limit', 1.0)
            change_penalty = abs(credit_limit - current_credit) * 2.0
            
            # Conservative adjustment based on risk level (less extreme)
            if risk_assessment.current_risk_level == RiskLevel.CRITICAL:
                risk_weight = 1.5
            elif risk_assessment.current_risk_level == RiskLevel.HIGH:
                risk_weight = 1.2
            else:
                risk_weight = 1.0
            
            return risk_weight * risk_cost + activity_penalty + change_penalty
        
        # Initial values
        x0 = [
            current_policies.get('credit_supply_limit', 1.0),
            current_policies.get('risk_appetite', 0.5),
            current_policies.get('interbank_limit', 0.5),
        ]
        
        # Bounds: keep parameters reasonable - don't go too extreme
        # Credit limit: 70% to 110% (don't kill lending)
        # Risk appetite: 20% to 90%
        # Interbank: 50% to 120%
        bounds = [(0.7, 1.1), (0.2, 0.9), (0.5, 1.2)]
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
            )
            
            if result.success:
                optimized = {
                    'credit_supply_limit': result.x[0],
                    'risk_appetite': result.x[1],
                    'interbank_limit': result.x[2],
                }
                
                logger.debug(f"Optimized policies: {optimized}")
                return optimized
            else:
                logger.warning("Policy optimization failed, using conservative defaults")
                return self._conservative_defaults(risk_assessment)
        
        except Exception as e:
            logger.error(f"Error in policy optimization: {e}")
            return self._conservative_defaults(risk_assessment)
    
    def _conservative_defaults(self, risk_assessment: RiskAssessment) -> Dict[str, float]:
        """Return conservative default policies based on risk level"""
        # MUCH less aggressive - banks need lending to survive
        if risk_assessment.current_risk_level == RiskLevel.CRITICAL:
            return {
                'credit_supply_limit': 0.85,  # Still allow 85% of normal lending
                'risk_appetite': 0.3,
                'interbank_limit': 0.7,
            }
        elif risk_assessment.current_risk_level == RiskLevel.HIGH:
            return {
                'credit_supply_limit': 0.90,
                'risk_appetite': 0.4,
                'interbank_limit': 0.8,
            }
        elif risk_assessment.current_risk_level == RiskLevel.MEDIUM:
            return {
                'credit_supply_limit': 0.95,
                'risk_appetite': 0.5,
                'interbank_limit': 0.9,
            }
        else:  # LOW
            return {
                'credit_supply_limit': 1.0,
                'risk_appetite': 0.7,
                'interbank_limit': 1.0,
            }


# Global singleton instance
_global_advisor: Optional[MLRiskMitigationAdvisor] = None


def get_risk_advisor() -> MLRiskMitigationAdvisor:
    """Get or create global risk advisor instance"""
    global _global_advisor
    if _global_advisor is None:
        _global_advisor = MLRiskMitigationAdvisor()
    return _global_advisor


def initialize_risk_advisor(
    default_predictor: Optional[DefaultPredictor] = None,
    risk_aversion: float = 0.3,  # Lower default
) -> MLRiskMitigationAdvisor:
    """
    Initialize global risk advisor with custom parameters.
    
    Args:
        default_predictor: ML model for default prediction
        risk_aversion: How aggressively to reduce risk (0-1)
    
    Returns:
        Configured MLRiskMitigationAdvisor instance
    """
    global _global_advisor
    _global_advisor = MLRiskMitigationAdvisor(
        default_predictor=default_predictor,
        risk_aversion=risk_aversion,
    )
    return _global_advisor
