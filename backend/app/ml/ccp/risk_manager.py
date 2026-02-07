"""
CCP Loss Absorption and Policy Response Module

Implements CCP-centric decision making based on:
- ML-predicted default probabilities
- Network amplification factors
- Capital & liquidity buffers
- Systemic fragility metrics

Reference: ML_Flow.md Section 6 & 7
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tier classification for clearing members"""
    TIER_1_LOW = "Tier 1 (Low Risk)"
    TIER_2_MODERATE = "Tier 2 (Moderate Risk)"
    TIER_3_HIGH = "Tier 3 (High Risk)"
    TIER_4_CRITICAL = "Tier 4 (Critical Risk)"


@dataclass
class MemberRiskProfile:
    """Risk profile for a clearing member"""
    member_id: str
    member_name: str
    
    # Core risk metrics
    default_probability: float          # From ML model
    systemic_importance: float          # From spectral analysis
    capital_buffer: float               # CRAR / leverage
    liquidity_buffer: float             # Liquidity ratio
    
    # Network effects
    cascade_trigger_prob: float         # Probability of triggering cascade
    contagion_vulnerability: float      # Exposure to others' defaults
    
    # CCP assessment
    risk_tier: RiskTier
    expected_loss: float                # Expected loss to CCP
    margin_add_on: float                # Additional margin requirement (%)
    recommended_action: str             # CCP policy recommendation


@dataclass
class SystemRiskAssessment:
    """System-wide risk assessment"""
    
    # Aggregate metrics
    system_fragility_index: float       # From spectral analysis
    avg_default_probability: float
    total_exposure: float
    
    # Member profiles
    member_profiles: List[MemberRiskProfile]
    
    # Policy recommendations
    system_risk_level: str              # LOW, MEDIUM, HIGH, CRITICAL
    preventive_interventions: List[str]
    stress_test_scenarios: List[str]
    
    # CCP capital adequacy
    required_default_fund: float
    current_coverage_ratio: float


class CCPRiskManager:
    """
    CCP Risk Manager for clearing member oversight
    
    CCP Objective: Minimize Expected Systemic Loss
    
    Formula from ML_Flow.md:
    min Σ_i Expected_Systemic_Loss_i
    
    Where Expected_Systemic_Loss_i accounts for:
    - Direct exposure to member i
    - Network amplification effects
    - Recovery expectations
    """
    
    def __init__(
        self,
        base_margin_rate: float = 0.02,  # 2% base margin
        risk_tier_multipliers: Optional[Dict[RiskTier, float]] = None,
        cascade_weight: float = 0.3,      # Weight for cascade risk
        systemic_weight: float = 0.4,     # Weight for systemic importance
        default_weight: float = 0.3       # Weight for default probability
    ):
        """
        Initialize CCP risk manager
        
        Args:
            base_margin_rate: Base margin requirement
            risk_tier_multipliers: Margin multipliers by tier
            cascade_weight: Weight for cascade risk in scoring
            systemic_weight: Weight for systemic importance
            default_weight: Weight for default probability
        """
        self.base_margin_rate = base_margin_rate
        
        if risk_tier_multipliers is None:
            self.risk_tier_multipliers = {
                RiskTier.TIER_1_LOW: 1.0,
                RiskTier.TIER_2_MODERATE: 1.5,
                RiskTier.TIER_3_HIGH: 2.5,
                RiskTier.TIER_4_CRITICAL: 4.0
            }
        else:
            self.risk_tier_multipliers = risk_tier_multipliers
        
        # Scoring weights
        self.cascade_weight = cascade_weight
        self.systemic_weight = systemic_weight
        self.default_weight = default_weight
        
        # Validate weights
        total_weight = cascade_weight + systemic_weight + default_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.cascade_weight /= total_weight
            self.systemic_weight /= total_weight
            self.default_weight /= total_weight
        
        logger.info(f"CCP Risk Manager initialized")
        logger.info(f"  Base margin: {base_margin_rate:.1%}")
        logger.info(f"  Weights: cascade={self.cascade_weight:.2f}, systemic={self.systemic_weight:.2f}, default={self.default_weight:.2f}")
    
    def assess_member_risk(
        self,
        member_id: str,
        member_name: str,
        default_probability: float,
        systemic_importance: float,
        capital_buffer: float,
        liquidity_buffer: float,
        cascade_trigger_prob: float = 0.0,
        contagion_vulnerability: float = 0.0,
        exposure_amount: float = 1e6  # Default 1M units
    ) -> MemberRiskProfile:
        """
        Assess individual member risk and determine CCP response
        
        Args:
            member_id: Unique identifier
            member_name: Member name
            default_probability: PD from ML model [0, 1]
            systemic_importance: From eigenvector centrality [0, 1]
            capital_buffer: Capital adequacy ratio [0, 1+]
            liquidity_buffer: Liquidity ratio [0, 1]
            cascade_trigger_prob: Prob of triggering cascade [0, 1]
            contagion_vulnerability: Vulnerability score [0, 1]
            exposure_amount: CCP exposure to this member
        
        Returns:
            MemberRiskProfile with CCP assessment
        """
        # 1. Compute composite risk score
        risk_score = (
            self.default_weight * default_probability +
            self.systemic_weight * systemic_importance +
            self.cascade_weight * cascade_trigger_prob
        )
        
        # 2. Adjust for buffers (lower score if well-capitalized)
        buffer_adjustment = (capital_buffer + liquidity_buffer) / 2
        buffer_adjustment = np.clip(buffer_adjustment, 0.5, 1.5)  # Don't over-adjust
        risk_score = risk_score / buffer_adjustment
        
        # 3. Classify risk tier
        risk_tier = self._classify_risk_tier(risk_score, default_probability)
        
        # 4. Calculate margin add-on
        tier_multiplier = self.risk_tier_multipliers[risk_tier]
        margin_add_on = self.base_margin_rate * tier_multiplier
        
        # 5. Calculate expected loss
        # Expected Loss = PD × Exposure × (1 - Recovery Rate)
        recovery_rate = 0.4  # Assume 40% recovery
        expected_loss = default_probability * exposure_amount * (1 - recovery_rate)
        
        # Amplify by systemic importance
        systemic_multiplier = 1 + systemic_importance
        expected_loss *= systemic_multiplier
        
        # 6. Generate policy recommendation
        recommended_action = self._generate_recommendation(
            risk_tier, 
            default_probability,
            systemic_importance,
            cascade_trigger_prob,
            capital_buffer,
            liquidity_buffer
        )
        
        # Create risk profile
        profile = MemberRiskProfile(
            member_id=member_id,
            member_name=member_name,
            default_probability=default_probability,
            systemic_importance=systemic_importance,
            capital_buffer=capital_buffer,
            liquidity_buffer=liquidity_buffer,
            cascade_trigger_prob=cascade_trigger_prob,
            contagion_vulnerability=contagion_vulnerability,
            risk_tier=risk_tier,
            expected_loss=expected_loss,
            margin_add_on=margin_add_on,
            recommended_action=recommended_action
        )
        
        return profile
    
    def _classify_risk_tier(
        self,
        risk_score: float,
        default_probability: float
    ) -> RiskTier:
        """Classify member into risk tier"""
        
        # Tier 4: Critical (immediate intervention needed)
        if default_probability > 0.5 or risk_score > 0.7:
            return RiskTier.TIER_4_CRITICAL
        
        # Tier 3: High (enhanced monitoring)
        elif default_probability > 0.3 or risk_score > 0.5:
            return RiskTier.TIER_3_HIGH
        
        # Tier 2: Moderate (standard monitoring)
        elif default_probability > 0.15 or risk_score > 0.3:
            return RiskTier.TIER_2_MODERATE
        
        # Tier 1: Low (routine oversight)
        else:
            return RiskTier.TIER_1_LOW
    
    def _generate_recommendation(
        self,
        risk_tier: RiskTier,
        default_prob: float,
        systemic_importance: float,
        cascade_prob: float,
        capital_buffer: float,
        liquidity_buffer: float
    ) -> str:
        """Generate CCP policy recommendation"""
        
        recommendations = []
        
        # Tier-based base recommendations
        if risk_tier == RiskTier.TIER_4_CRITICAL:
            recommendations.append("IMMEDIATE: Require additional collateral")
            recommendations.append("IMMEDIATE: Increase margin requirements")
            recommendations.append("IMMEDIATE: Daily porting/risk review")
            
            if systemic_importance > 0.7:
                recommendations.append("ALERT: Systemically critical institution")
                recommendations.append("ACTION: Coordinate with regulatory authorities")
        
        elif risk_tier == RiskTier.TIER_3_HIGH:
            recommendations.append("Enhanced monitoring frequency")
            recommendations.append("Request capital/liquidity improvement plan")
            
            if cascade_prob > 0.3:
                recommendations.append("CAUTION: High cascade trigger risk")
        
        elif risk_tier == RiskTier.TIER_2_MODERATE:
            recommendations.append("Standard monitoring")
            recommendations.append("Review margin adequacy quarterly")
        
        else:  # TIER_1_LOW
            recommendations.append("Routine oversight")
        
        # Specific metric-based recommendations
        if capital_buffer < 0.1:
            recommendations.append("⚠️ Low capital buffer - Request capital injection")
        
        if liquidity_buffer < 0.05:
            recommendations.append("⚠️ Low liquidity - Monitor intraday exposures")
        
        if default_prob > 0.4 and systemic_importance > 0.6:
            recommendations.append("🚨 HIGH PRIORITY: Systemically important member at high default risk")
        
        return " | ".join(recommendations)
    
    def assess_system_risk(
        self,
        member_profiles: List[MemberRiskProfile],
        system_fragility: float,
        total_exposure: float = 1e9
    ) -> SystemRiskAssessment:
        """
        Assess system-wide risk from CCP perspective
        
        Args:
            member_profiles: List of all member risk profiles
            system_fragility: Fragility index from spectral analysis
            total_exposure: Total CCP exposure
        
        Returns:
            SystemRiskAssessment with system-level recommendations
        """
        logger.info(f"Assessing system risk for {len(member_profiles)} members")
        
        # Aggregate metrics
        avg_default_prob = np.mean([p.default_probability for p in member_profiles])
        
        # Determine system risk level
        system_risk_level = self._classify_system_risk(
            system_fragility,
            avg_default_prob,
            member_profiles
        )
        
        # Generate preventive interventions
        preventive_interventions = self._generate_preventive_interventions(
            system_risk_level,
            system_fragility,
            member_profiles
        )
        
        # Suggest stress test scenarios
        stress_scenarios = self._suggest_stress_scenarios(
            system_risk_level,
            member_profiles
        )
        
        # Calculate required default fund
        required_fund = self._calculate_default_fund(
            member_profiles,
            total_exposure
        )
        
        # Mock current fund (in practice, would be actual CCP fund)
        current_fund = required_fund * 1.2  # Assume 120% coverage
        coverage_ratio = current_fund / required_fund if required_fund > 0 else 1.0
        
        assessment = SystemRiskAssessment(
            system_fragility_index=system_fragility,
            avg_default_probability=avg_default_prob,
            total_exposure=total_exposure,
            member_profiles=member_profiles,
            system_risk_level=system_risk_level,
            preventive_interventions=preventive_interventions,
            stress_test_scenarios=stress_scenarios,
            required_default_fund=required_fund,
            current_coverage_ratio=coverage_ratio
        )
        
        return assessment
    
    def _classify_system_risk(
        self,
        fragility: float,
        avg_default_prob: float,
        member_profiles: List[MemberRiskProfile]
    ) -> str:
        """Classify overall system risk level"""
        
        # Count critical members
        critical_count = sum(1 for p in member_profiles if p.risk_tier == RiskTier.TIER_4_CRITICAL)
        critical_ratio = critical_count / len(member_profiles) if member_profiles else 0
        
        # System is CRITICAL if:
        if fragility > 0.8 or avg_default_prob > 0.4 or critical_ratio > 0.2:
            return "CRITICAL"
        
        # System is HIGH if:
        elif fragility > 0.6 or avg_default_prob > 0.25 or critical_ratio > 0.1:
            return "HIGH"
        
        # System is MEDIUM if:
        elif fragility > 0.4 or avg_default_prob > 0.15:
            return "MEDIUM"
        
        # Otherwise LOW
        else:
            return "LOW"
    
    def _generate_preventive_interventions(
        self,
        system_risk_level: str,
        fragility: float,
        member_profiles: List[MemberRiskProfile]
    ) -> List[str]:
        """Generate preventive intervention recommendations"""
        
        interventions = []
        
        if system_risk_level == "CRITICAL":
            interventions.append("🚨 IMMEDIATE: Convene risk committee")
            interventions.append("🚨 IMMEDIATE: Increase default fund contributions")
            interventions.append("🚨 IMMEDIATE: Implement enhanced collateral requirements")
            interventions.append("🚨 IMMEDIATE: Coordinate with systemic risk regulators")
        
        elif system_risk_level == "HIGH":
            interventions.append("⚠️ Increase margin surveillance frequency")
            interventions.append("⚠️ Run daily stress tests")
            interventions.append("⚠️ Review member concentration limits")
        
        elif system_risk_level == "MEDIUM":
            interventions.append("Monitor key risk indicators")
            interventions.append("Prepare contingency funding plans")
        
        else:  # LOW
            interventions.append("Maintain standard oversight procedures")
        
        # Specific interventions based on fragility
        if fragility > 0.7:
            interventions.append("   High network fragility detected")
            interventions.append("ACTION: Implement circuit breakers for cascade prevention")
        
        # Interventions for specific member issues
        high_systemic = [p for p in member_profiles if p.systemic_importance > 0.7]
        if high_systemic:
            interventions.append(f"FOCUS: {len(high_systemic)} systemically important members require enhanced oversight")
        
        return interventions
    
    def _suggest_stress_scenarios(
        self,
        system_risk_level: str,
        member_profiles: List[MemberRiskProfile]
    ) -> List[str]:
        """Suggest appropriate stress test scenarios"""
        
        scenarios = []
        
        # Base scenarios
        scenarios.append("Scenario 1: Top 2 largest members default simultaneously")
        scenarios.append("Scenario 2: Sector-wide shock (e.g., real estate crisis)")
        scenarios.append("Scenario 3: Liquidity squeeze (40% haircut on collateral)")
        
        # Risk-level specific scenarios
        if system_risk_level in ["HIGH", "CRITICAL"]:
            scenarios.append("Scenario 4: Top 3 systemically important members default")
            scenarios.append("Scenario 5: Cascade event from single critical member")
        
        # Member-specific scenarios
        critical_members = [p for p in member_profiles if p.risk_tier == RiskTier.TIER_4_CRITICAL]
        if critical_members:
            top_critical = critical_members[0].member_name
            scenarios.append(f"Scenario 6: {top_critical} default with contagion effects")
        
        return scenarios
    
    def _calculate_default_fund(
        self,
        member_profiles: List[MemberRiskProfile],
        total_exposure: float
    ) -> float:
        """
        Calculate required default fund size
        
        Uses Cover-2 approach: fund must cover default of 2 largest members
        Plus additional buffer for systemic risk
        """
        if not member_profiles:
            return 0.0
        
        # Sort by expected loss
        sorted_profiles = sorted(
            member_profiles,
            key=lambda p: p.expected_loss,
            reverse=True
        )
        
        # Cover-2: Top 2 members
        cover_2_loss = sum(p.expected_loss for p in sorted_profiles[:2])
        
        # Additional systemic buffer (20% of cover-2)
        systemic_buffer = 0.2 * cover_2_loss
        
        # Total required fund
        required_fund = cover_2_loss + systemic_buffer
        
        logger.info(f"Required default fund: {required_fund:,.0f}")
        logger.info(f"  Cover-2 loss: {cover_2_loss:,.0f}")
        logger.info(f"  Systemic buffer: {systemic_buffer:,.0f}")
        
        return required_fund
    
    def generate_risk_report(
        self,
        assessment: SystemRiskAssessment
    ) -> str:
        """
        Generate comprehensive CCP risk report
        
        Args:
            assessment: SystemRiskAssessment object
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CCP RISK MANAGEMENT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System overview
        report.append("SYSTEM RISK OVERVIEW")
        report.append("-" * 80)
        report.append(f"System Risk Level:           {assessment.system_risk_level}")
        report.append(f"System Fragility Index:      {assessment.system_fragility_index:.2%}")
        report.append(f"Average Default Probability: {assessment.avg_default_probability:.2%}")
        report.append(f"Total CCP Exposure:          ${assessment.total_exposure:,.0f}")
        report.append("")
        
        # Default fund adequacy
        report.append("DEFAULT FUND ADEQUACY")
        report.append("-" * 80)
        report.append(f"Required Default Fund:       ${assessment.required_default_fund:,.0f}")
        report.append(f"Coverage Ratio:              {assessment.current_coverage_ratio:.2f}x")
        status = "✓ ADEQUATE" if assessment.current_coverage_ratio >= 1.0 else "✗ INSUFFICIENT"
        report.append(f"Status:                      {status}")
        report.append("")
        
        # Member risk summary
        report.append("MEMBER RISK SUMMARY")
        report.append("-" * 80)
        
        # Count by tier
        tier_counts = {}
        for profile in assessment.member_profiles:
            tier = profile.risk_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for tier in RiskTier:
            count = tier_counts.get(tier, 0)
            report.append(f"{tier.value:25s}: {count:3d} members")
        report.append("")
        
        # Top 10 highest risk members
        report.append("TOP 10 HIGHEST RISK MEMBERS")
        report.append("-" * 80)
        sorted_members = sorted(
            assessment.member_profiles,
            key=lambda p: p.default_probability * p.systemic_importance,
            reverse=True
        )[:10]
        
        report.append(f"{'Rank':<6} {'Member':<25} {'PD':>8} {'SI':>8} {'Tier':<10} {'Action'}")
        report.append("-" * 80)
        for i, profile in enumerate(sorted_members, 1):
            report.append(
                f"{i:<6} {profile.member_name[:24]:<25} "
                f"{profile.default_probability:>7.1%} "
                f"{profile.systemic_importance:>7.2f} "
                f"{profile.risk_tier.name:<10} "
                f"{profile.recommended_action[:40]}"
            )
        report.append("")
        
        # Preventive interventions
        report.append("PREVENTIVE INTERVENTIONS")
        report.append("-" * 80)
        for i, intervention in enumerate(assessment.preventive_interventions, 1):
            report.append(f"{i}. {intervention}")
        report.append("")
        
        # Stress test scenarios
        report.append("RECOMMENDED STRESS TEST SCENARIOS")
        report.append("-" * 80)
        for scenario in assessment.stress_test_scenarios:
            report.append(f"• {scenario}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
