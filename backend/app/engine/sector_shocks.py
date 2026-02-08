"""
Multi-Sector Shock Framework
=============================
Models shocks across different economic sectors with quantifiable conditions.

Each sector has specific health indicators and shock transmission mechanisms
that affect banks differently based on their exposure composition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SectorType(str, Enum):
    """Economic sectors in the financial system"""
    REAL_ESTATE = "real_estate"
    INFRASTRUCTURE = "infrastructure"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"
    TECHNOLOGY = "technology"
    EXPORT_ORIENTED = "export_oriented"
    RETAIL_TRADE = "retail_trade"
    MSME = "msme"  # Micro, Small, Medium Enterprises


class ShockSeverity(str, Enum):
    """Severity levels for shocks"""
    MILD = "mild"           # -10% to -20% health impact
    MODERATE = "moderate"   # -20% to -40% health impact
    SEVERE = "severe"       # -40% to -60% health impact
    CRISIS = "crisis"       # -60% to -80% health impact


@dataclass
class SectorHealthIndicators:
    """
    Quantifiable health metrics for a sector
    All metrics normalized to [0, 1] scale where 1.0 = excellent health
    """
    # Core health
    economic_health: float = 0.8              # Overall sector vitality
    
    # Financial metrics
    debt_service_coverage: float = 1.5        # Cash flow / debt payments (>1.5 is healthy)
    leverage_ratio: float = 0.40              # Debt / Assets (lower is better, <40% is healthy)
    profitability_margin: float = 0.15        # Operating margin (15% baseline)
    
    # Market conditions
    demand_index: float = 0.75                # Current demand vs baseline
    capacity_utilization: float = 0.70        # % of capacity in use
    price_index: float = 1.0                  # Price level (1.0 = baseline)
    
    # Credit quality
    default_rate: float = 0.03                # % of sector loans in default (3% baseline)
    restructuring_rate: float = 0.05          # % of loans restructured (5% baseline)
    
    # Sentiment
    business_confidence: float = 0.65         # Business sentiment index
    investment_growth: float = 0.05           # YoY investment growth rate
    
    def compute_overall_health(self) -> float:
        """
        Aggregate health score [0-1]
        Weighted combination of all indicators
        """
        # Financial strength (40%)
        financial_health = (
            0.5 * min(1.0, self.debt_service_coverage / 2.0) +
            0.3 * (1.0 - min(1.0, self.leverage_ratio / 0.6)) +
            0.2 * min(1.0, self.profitability_margin / 0.20)
        )
        
        # Market conditions (30%)
        market_health = (
            0.4 * self.demand_index +
            0.4 * self.capacity_utilization +
            0.2 * min(1.0, max(0.0, 2.0 - abs(self.price_index - 1.0)))
        )
        
        # Credit quality (20%)
        credit_health = (
            0.6 * (1.0 - min(1.0, self.default_rate / 0.10)) +
            0.4 * (1.0 - min(1.0, self.restructuring_rate / 0.15))
        )
        
        # Sentiment (10%)
        sentiment_health = (
            0.7 * self.business_confidence +
            0.3 * min(1.0, max(0.0, (self.investment_growth + 0.05) / 0.20))
        )
        
        overall = (
            0.40 * financial_health +
            0.30 * market_health +
            0.20 * credit_health +
            0.10 * sentiment_health
        )
        
        return overall
    
    def to_dict(self) -> Dict[str, float]:
        """Export to dictionary"""
        return {
            'economic_health': self.economic_health,
            'debt_service_coverage': self.debt_service_coverage,
            'leverage_ratio': self.leverage_ratio,
            'profitability_margin': self.profitability_margin,
            'demand_index': self.demand_index,
            'capacity_utilization': self.capacity_utilization,
            'price_index': self.price_index,
            'default_rate': self.default_rate,
            'restructuring_rate': self.restructuring_rate,
            'business_confidence': self.business_confidence,
            'investment_growth': self.investment_growth,
            'overall_health': self.compute_overall_health()
        }


@dataclass
class SectorShockScenario:
    """Defines a shock scenario for a specific sector"""
    sector: SectorType
    severity: ShockSeverity
    trigger_event: str                        # Human-readable description
    
    # Direct impacts on health indicators
    demand_shock: float = 0.0                 # Change in demand index
    price_shock: float = 0.0                  # Change in price index
    leverage_increase: float = 0.0            # Increase in leverage ratio
    default_rate_increase: float = 0.0        # Increase in default rate
    confidence_drop: float = 0.0              # Drop in business confidence
    
    # Spillover effects
    correlated_sectors: List[SectorType] = field(default_factory=list)
    spillover_intensity: float = 0.3          # How much shock spills to correlated sectors
    
    # Duration
    recovery_periods: int = 8                 # Timesteps to recover (if no further shocks)


@dataclass
class SectorState:
    """Complete state of a sector including all metrics"""
    sector_type: SectorType
    indicators: SectorHealthIndicators
    total_exposure: float = 0.0               # Total bank lending to this sector ($)
    num_banks_exposed: int = 0                # Number of banks with exposure
    concentration_hhi: float = 0.0            # HHI of exposure concentration
    
    # Shock history
    active_shocks: List[SectorShockScenario] = field(default_factory=list)
    recovery_counter: int = 0                 # Periods since last shock
    
    def apply_shock(self, shock: SectorShockScenario) -> Dict[str, float]:
        """
        Apply shock to sector, updating health indicators
        Returns: Dict of changes
        """
        changes = {}
        
        # Apply demand shock
        if shock.demand_shock != 0:
            old_demand = self.indicators.demand_index
            self.indicators.demand_index = max(0.0, min(1.0, old_demand + shock.demand_shock))
            changes['demand_index'] = self.indicators.demand_index - old_demand
        
        # Apply price shock
        if shock.price_shock != 0:
            old_price = self.indicators.price_index
            self.indicators.price_index = max(0.5, min(2.0, old_price + shock.price_shock))
            changes['price_index'] = self.indicators.price_index - old_price
        
        # Increase leverage
        if shock.leverage_increase != 0:
            old_leverage = self.indicators.leverage_ratio
            self.indicators.leverage_ratio = min(0.90, old_leverage + shock.leverage_increase)
            changes['leverage_ratio'] = self.indicators.leverage_ratio - old_leverage
        
        # Increase default rate
        if shock.default_rate_increase != 0:
            old_default = self.indicators.default_rate
            self.indicators.default_rate = min(0.30, old_default + shock.default_rate_increase)
            changes['default_rate'] = self.indicators.default_rate - old_default
        
        # Drop confidence
        if shock.confidence_drop != 0:
            old_conf = self.indicators.business_confidence
            self.indicators.business_confidence = max(0.1, old_conf - shock.confidence_drop)
            changes['business_confidence'] = self.indicators.business_confidence - old_conf
        
        # Secondary effects: debt service coverage drops with higher defaults
        old_dsc = self.indicators.debt_service_coverage
        self.indicators.debt_service_coverage = max(
            0.5,
            old_dsc * (1.0 - shock.default_rate_increase * 2.0)
        )
        changes['debt_service_coverage'] = self.indicators.debt_service_coverage - old_dsc
        
        # Profitability drops with demand/price shocks
        old_profit = self.indicators.profitability_margin
        profit_impact = shock.demand_shock * 0.5 + shock.price_shock * 0.3
        self.indicators.profitability_margin = max(
            -0.05,  # Can go negative (losses)
            old_profit + profit_impact
        )
        changes['profitability_margin'] = self.indicators.profitability_margin - old_profit
        
        # Update overall health
        self.indicators.economic_health = self.indicators.compute_overall_health()
        changes['economic_health'] = self.indicators.economic_health
        
        # Record shock
        self.active_shocks.append(shock)
        self.recovery_counter = 0
        
        # Only log severe shocks
        # logger.info(
        #     f"Applied {shock.severity.value.upper()} shock to {self.sector_type.value}: "
        #     f"Health {self.indicators.economic_health:.2%}"
        # )
        
        return changes
    
    def recover_step(self, recovery_rate: float = 0.05):
        """
        Natural recovery over time if no new shocks
        """
        if self.recovery_counter > 0:
            # Gradual recovery toward healthy baseline
            self.indicators.demand_index = min(0.85, self.indicators.demand_index + recovery_rate * 0.5)
            self.indicators.default_rate = max(0.03, self.indicators.default_rate - recovery_rate * 0.3)
            self.indicators.business_confidence = min(0.75, self.indicators.business_confidence + recovery_rate * 0.8)
            self.indicators.leverage_ratio = max(0.35, self.indicators.leverage_ratio - recovery_rate * 0.2)
            
            self.indicators.economic_health = self.indicators.compute_overall_health()
        
        self.recovery_counter += 1


# ========================================================================================
# PRE-DEFINED SHOCK SCENARIOS FOR EACH SECTOR
# ========================================================================================

SECTOR_SHOCK_LIBRARY: Dict[SectorType, Dict[ShockSeverity, SectorShockScenario]] = {
    
    SectorType.REAL_ESTATE: {
        ShockSeverity.MILD: SectorShockScenario(
            sector=SectorType.REAL_ESTATE,
            severity=ShockSeverity.MILD,
            trigger_event="Interest rate hike + regulatory tightening",
            demand_shock=-0.15,
            price_shock=-0.10,
            leverage_increase=0.05,
            default_rate_increase=0.02,
            confidence_drop=0.10,
            correlated_sectors=[SectorType.INFRASTRUCTURE, SectorType.MANUFACTURING],
            spillover_intensity=0.25,
            recovery_periods=10
        ),
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.REAL_ESTATE,
            severity=ShockSeverity.MODERATE,
            trigger_event="Property bubble burst + credit crunch",
            demand_shock=-0.30,
            price_shock=-0.25,
            leverage_increase=0.10,
            default_rate_increase=0.05,
            confidence_drop=0.25,
            correlated_sectors=[SectorType.INFRASTRUCTURE, SectorType.MANUFACTURING, SectorType.RETAIL_TRADE],
            spillover_intensity=0.4,
            recovery_periods=16
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.REAL_ESTATE,
            severity=ShockSeverity.SEVERE,
            trigger_event="Real estate crisis + developer defaults cascade",
            demand_shock=-0.50,
            price_shock=-0.40,
            leverage_increase=0.15,
            default_rate_increase=0.10,
            confidence_drop=0.45,
            correlated_sectors=[SectorType.INFRASTRUCTURE, SectorType.MANUFACTURING, SectorType.RETAIL_TRADE, SectorType.MSME],
            spillover_intensity=0.6,
            recovery_periods=24
        ),
        ShockSeverity.CRISIS: SectorShockScenario(
            sector=SectorType.REAL_ESTATE,
            severity=ShockSeverity.CRISIS,
            trigger_event="Systemic real estate collapse (2008-style)",
            demand_shock=-0.70,
            price_shock=-0.60,
            leverage_increase=0.20,
            default_rate_increase=0.18,
            confidence_drop=0.65,
            correlated_sectors=[SectorType.INFRASTRUCTURE, SectorType.MANUFACTURING, SectorType.RETAIL_TRADE, SectorType.MSME, SectorType.SERVICES],
            spillover_intensity=0.8,
            recovery_periods=36
        ),
    },
    
    SectorType.INFRASTRUCTURE: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.INFRASTRUCTURE,
            severity=ShockSeverity.MODERATE,
            trigger_event="Government project delays + funding cuts",
            demand_shock=-0.25,
            price_shock=0.05,  # Input cost pressures
            leverage_increase=0.12,
            default_rate_increase=0.06,
            confidence_drop=0.20,
            correlated_sectors=[SectorType.MANUFACTURING, SectorType.REAL_ESTATE],
            spillover_intensity=0.35,
            recovery_periods=20
        ),
    },
    
    SectorType.MANUFACTURING: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.MANUFACTURING,
            severity=ShockSeverity.MODERATE,
            trigger_event="Supply chain disruption + input cost spike",
            demand_shock=-0.20,
            price_shock=0.15,  # Cost-push inflation
            leverage_increase=0.08,
            default_rate_increase=0.04,
            confidence_drop=0.30,
            correlated_sectors=[SectorType.EXPORT_ORIENTED, SectorType.MSME, SectorType.RETAIL_TRADE],
            spillover_intensity=0.45,
            recovery_periods=12
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.MANUFACTURING,
            severity=ShockSeverity.SEVERE,
            trigger_event="Industrial recession + demand collapse",
            demand_shock=-0.45,
            price_shock=-0.15,  # Deflation from excess capacity
            leverage_increase=0.14,
            default_rate_increase=0.09,
            confidence_drop=0.50,
            correlated_sectors=[SectorType.EXPORT_ORIENTED, SectorType.MSME, SectorType.RETAIL_TRADE, SectorType.SERVICES],
            spillover_intensity=0.65,
            recovery_periods=20
        ),
    },
    
    SectorType.AGRICULTURE: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.AGRICULTURE,
            severity=ShockSeverity.MODERATE,
            trigger_event="Poor monsoon / drought conditions",
            demand_shock=0.0,  # Food demand is inelastic
            price_shock=-0.20,  # Crop prices fall
            leverage_increase=0.10,
            default_rate_increase=0.08,  # High default rate in agriculture
            confidence_drop=0.25,
            correlated_sectors=[SectorType.MSME, SectorType.RETAIL_TRADE],
            spillover_intensity=0.30,
            recovery_periods=4  # Seasonal recovery
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.AGRICULTURE,
            severity=ShockSeverity.SEVERE,
            trigger_event="Multi-year drought + rural income collapse",
            demand_shock=0.0,
            price_shock=-0.40,
            leverage_increase=0.18,
            default_rate_increase=0.15,
            confidence_drop=0.45,
            correlated_sectors=[SectorType.MSME, SectorType.RETAIL_TRADE, SectorType.SERVICES],
            spillover_intensity=0.50,
            recovery_periods=8
        ),
    },
    
    SectorType.ENERGY: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.ENERGY,
            severity=ShockSeverity.MODERATE,
            trigger_event="Oil price shock (supply disruption)",
            demand_shock=-0.10,
            price_shock=0.40,  # Energy prices spike
            leverage_increase=0.05,
            default_rate_increase=0.03,
            confidence_drop=0.15,
            correlated_sectors=[SectorType.MANUFACTURING, SectorType.EXPORT_ORIENTED, SectorType.RETAIL_TRADE],
            spillover_intensity=0.55,  # High spillover - energy affects everything
            recovery_periods=8
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.ENERGY,
            severity=ShockSeverity.SEVERE,
            trigger_event="Energy crisis + power shortages",
            demand_shock=-0.30,
            price_shock=0.70,
            leverage_increase=0.12,
            default_rate_increase=0.07,
            confidence_drop=0.40,
            correlated_sectors=[SectorType.MANUFACTURING, SectorType.EXPORT_ORIENTED, SectorType.RETAIL_TRADE, SectorType.INFRASTRUCTURE],
            spillover_intensity=0.75,
            recovery_periods=16
        ),
    },
    
    SectorType.EXPORT_ORIENTED: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.EXPORT_ORIENTED,
            severity=ShockSeverity.MODERATE,
            trigger_event="Global recession + export demand collapse",
            demand_shock=-0.35,
            price_shock=-0.10,
            leverage_increase=0.10,
            default_rate_increase=0.06,
            confidence_drop=0.35,
            correlated_sectors=[SectorType.MANUFACTURING],
            spillover_intensity=0.40,
            recovery_periods=12
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.EXPORT_ORIENTED,
            severity=ShockSeverity.SEVERE,
            trigger_event="Trade war + tariff barriers",
            demand_shock=-0.55,
            price_shock=-0.25,
            leverage_increase=0.16,
            default_rate_increase=0.12,
            confidence_drop=0.55,
            correlated_sectors=[SectorType.MANUFACTURING, SectorType.MSME],
            spillover_intensity=0.60,
            recovery_periods=20
        ),
    },
    
    SectorType.SERVICES: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.SERVICES,
            severity=ShockSeverity.MODERATE,
            trigger_event="Economic slowdown + discretionary spending cuts",
            demand_shock=-0.25,
            price_shock=-0.08,
            leverage_increase=0.07,
            default_rate_increase=0.04,
            confidence_drop=0.20,
            correlated_sectors=[SectorType.RETAIL_TRADE, SectorType.TECHNOLOGY],
            spillover_intensity=0.30,
            recovery_periods=10
        ),
    },
    
    SectorType.MSME: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.MSME,
            severity=ShockSeverity.MODERATE,
            trigger_event="Credit squeeze + working capital crisis",
            demand_shock=-0.20,
            price_shock=0.10,  # Input costs rise but can't pass through
            leverage_increase=0.15,  # MSMEs are highly leveraged
            default_rate_increase=0.10,  # MSME default rates are high
            confidence_drop=0.35,
            correlated_sectors=[SectorType.RETAIL_TRADE, SectorType.SERVICES],
            spillover_intensity=0.25,
            recovery_periods=12
        ),
        ShockSeverity.SEVERE: SectorShockScenario(
            sector=SectorType.MSME,
            severity=ShockSeverity.SEVERE,
            trigger_event="MSME crisis + mass bankruptcies",
            demand_shock=-0.45,
            price_shock=0.05,
            leverage_increase=0.22,
            default_rate_increase=0.18,
            confidence_drop=0.60,
            correlated_sectors=[SectorType.RETAIL_TRADE, SectorType.SERVICES, SectorType.MANUFACTURING],
            spillover_intensity=0.40,
            recovery_periods=18
        ),
    },
    
    SectorType.TECHNOLOGY: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.TECHNOLOGY,
            severity=ShockSeverity.MODERATE,
            trigger_event="Tech bubble burst + funding drought",
            demand_shock=-0.30,
            price_shock=-0.20,
            leverage_increase=0.08,
            default_rate_increase=0.05,
            confidence_drop=0.40,
            correlated_sectors=[SectorType.SERVICES],
            spillover_intensity=0.20,
            recovery_periods=14
        ),
    },
    
    SectorType.RETAIL_TRADE: {
        ShockSeverity.MODERATE: SectorShockScenario(
            sector=SectorType.RETAIL_TRADE,
            severity=ShockSeverity.MODERATE,
            trigger_event="Consumer spending slowdown + inventory buildup",
            demand_shock=-0.28,
            price_shock=-0.12,
            leverage_increase=0.09,
            default_rate_increase=0.06,
            confidence_drop=0.25,
            correlated_sectors=[SectorType.MSME, SectorType.SERVICES],
            spillover_intensity=0.35,
            recovery_periods=10
        ),
    },
}


def create_default_sector_states() -> Dict[SectorType, SectorState]:
    """Create baseline healthy states for all sectors"""
    return {
        sector: SectorState(
            sector_type=sector,
            indicators=SectorHealthIndicators()  # Default healthy values
        )
        for sector in SectorType
    }


def get_shock_scenario(
    sector: SectorType,
    severity: ShockSeverity
) -> Optional[SectorShockScenario]:
    """Retrieve pre-defined shock scenario"""
    return SECTOR_SHOCK_LIBRARY.get(sector, {}).get(severity)


def compute_bank_loss_from_sector_shock(
    sector_state: SectorState,
    bank_exposure: float,
    bank_total_assets: float
) -> float:
    """
    Compute bank loss from sector deterioration
    
    Args:
        sector_state: Current sector state
        bank_exposure: Bank's lending to this sector ($)
        bank_total_assets: Bank's total assets for normalization
    
    Returns:
        Loss amount ($)
    """
    if bank_exposure <= 0 or sector_state.indicators.economic_health >= 0.7:
        return 0.0
    
    # Loss rate based on sector health and default rate
    health_factor = max(0.0, 0.7 - sector_state.indicators.economic_health) / 0.7
    base_loss_rate = sector_state.indicators.default_rate * health_factor
    
    # Amplify for severe stress
    if sector_state.indicators.economic_health < 0.3:
        base_loss_rate *= 2.0
    
    # Loss = Exposure × Loss Rate × Stress Severity
    loss = bank_exposure * base_loss_rate
    
    return loss
