"""
Geopolitical and Macro-Prudential State Module
==============================================
Models international factors affecting regulatory decisions:
- Geopolitical tensions and stability
- Forex reserves and currency pressure
- Treasury bond markets (domestic and US)
- Gold reserves as stability buffer
- International trade conditions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GeopoliticalTension(str, Enum):
    """Level of international geopolitical tension"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"


class CurrencyPressure(str, Enum):
    """Direction and magnitude of currency pressure"""
    STRONG_APPRECIATION = "strong_appreciation"
    MILD_APPRECIATION = "mild_appreciation"
    STABLE = "stable"
    MILD_DEPRECIATION = "mild_depreciation"
    STRONG_DEPRECIATION = "strong_depreciation"


@dataclass
class ForexReserves:
    """Foreign Exchange Reserve Holdings"""
    total_usd: float  # Total reserves in USD billions
    us_treasury_bills: float  # Holdings of US T-bills
    us_treasury_bonds: float  # Holdings of US T-bonds
    gold_reserves_tonnes: float  # Physical gold in tonnes
    gold_value_usd: float  # Market value of gold
    other_currencies: Dict[str, float] = field(default_factory=dict)  # EUR, GBP, JPY, etc.
    
    # Import cover (months of imports that reserves can finance)
    import_cover_months: float = 6.0
    
    # Adequacy metrics
    reserves_to_gdp_ratio: float = 0.15  # Reserves / GDP
    reserves_to_short_term_debt: float = 1.5  # Coverage ratio
    
    def compute_adequacy_score(self) -> float:
        """
        Compute reserve adequacy score [0-1]
        Based on IMF metrics: Import cover, GDP ratio, debt coverage
        """
        # Import cover (>3 months is good, >6 months is excellent)
        import_score = min(1.0, self.import_cover_months / 6.0)
        
        # Reserves to GDP (>15% is good)
        gdp_score = min(1.0, self.reserves_to_gdp_ratio / 0.15)
        
        # Debt coverage (>1.5 is good)
        debt_score = min(1.0, self.reserves_to_short_term_debt / 1.5)
        
        # Weighted average
        adequacy = 0.4 * import_score + 0.3 * gdp_score + 0.3 * debt_score
        
        return adequacy
    
    def apply_capital_outflow(self, outflow_usd: float):
        """Simulate capital outflow reducing reserves"""
        self.total_usd -= outflow_usd
        # Liquidate assets proportionally
        if self.total_usd > 0:
            reduction_factor = outflow_usd / (self.total_usd + outflow_usd)
            self.us_treasury_bills *= (1 - reduction_factor)
            self.us_treasury_bonds *= (1 - reduction_factor)


@dataclass
class TreasuryBondMarket:
    """Domestic Treasury Bond Market State"""
    outstanding_bonds_usd: float  # Total domestic govt debt
    avg_yield_10yr: float  # 10-year benchmark yield (%)
    yield_spread_vs_us: float  # Spread over US 10Y (bps)
    debt_to_gdp_ratio: float  # Government debt / GDP
    
    # Market liquidity
    bid_ask_spread_bps: float = 5.0
    daily_turnover_usd: float = 1000.0  # Daily trading volume
    
    # Investor composition
    domestic_banks_pct: float = 40.0
    foreign_investors_pct: float = 20.0
    central_bank_pct: float = 25.0
    insurance_pension_pct: float = 15.0
    
    def compute_sovereign_stress(self) -> float:
        """
        Compute sovereign debt stress indicator [0-1]
        Higher = more stress
        """
        # Debt to GDP stress (>80% is concerning)
        debt_stress = min(1.0, self.debt_to_gdp_ratio / 0.80)
        
        # Yield level stress (>8% is high)
        yield_stress = min(1.0, self.avg_yield_10yr / 8.0)
        
        # Spread stress (>400bps is high)
        spread_stress = min(1.0, abs(self.yield_spread_vs_us) / 400.0)
        
        return 0.4 * debt_stress + 0.3 * yield_stress + 0.3 * spread_stress
    
    def simulate_yield_shock(self, shock_bps: float):
        """Apply yield shock (positive = yields rise, prices fall)"""
        self.avg_yield_10yr += shock_bps / 100.0
        self.yield_spread_vs_us += shock_bps


@dataclass
class GeopoliticalState:
    """
    Complete geopolitical and macro-prudential state
    
    Captures external factors affecting domestic financial stability
    """
    # Geopolitical factors
    tension_level: GeopoliticalTension = GeopoliticalTension.MODERATE
    regional_conflicts: List[str] = field(default_factory=list)
    trade_war_active: bool = False
    sanctions_exposure: float = 0.0  # 0-1 scale
    
    # Currency factors
    currency_pressure: CurrencyPressure = CurrencyPressure.STABLE
    exchange_rate_volatility: float = 0.02  # 30-day volatility
    capital_flows_net_usd: float = 0.0  # Net inflows (positive) or outflows (negative)
    
    # Reserves and buffers
    forex_reserves: ForexReserves = None
    
    # Domestic bond market
    treasury_market: TreasuryBondMarket = None
    
    # US Treasury exposure (international benchmark)
    us_10yr_yield: float = 4.5  # US 10-year treasury yield (%)
    us_dollar_index: float = 100.0  # DXY index
    
    # Gold market
    gold_price_usd_per_oz: float = 2000.0
    
    # Trade conditions
    trade_balance_usd: float = 0.0  # Monthly trade balance
    export_growth_yoy: float = 0.05  # Year-on-year export growth
    import_growth_yoy: float = 0.07
    
    # Global risk sentiment
    vix_index: float = 15.0  # S&P 500 volatility index
    global_risk_appetite: float = 0.7  # 0 (risk-off) to 1 (risk-on)
    
    def __post_init__(self):
        """Initialize sub-components if not provided"""
        if self.forex_reserves is None:
            self.forex_reserves = ForexReserves(
                total_usd=600.0,  # $600B reserves (India-like scale)
                us_treasury_bills=150.0,
                us_treasury_bonds=100.0,
                gold_reserves_tonnes=800.0,
                gold_value_usd=50.0,
                other_currencies={'EUR': 50.0, 'GBP': 20.0, 'JPY': 30.0}
            )
        
        if self.treasury_market is None:
            self.treasury_market = TreasuryBondMarket(
                outstanding_bonds_usd=2000.0,
                avg_yield_10yr=7.0,
                yield_spread_vs_us=250.0,  # 250 bps over US
                debt_to_gdp_ratio=0.60
            )
    
    def compute_systemic_risk_multiplier(self) -> float:
        """
        Compute how geopolitical factors amplify domestic systemic risk
        
        Returns multiplier: 1.0 = no effect, >1.0 = amplifies risk, <1.0 = dampens
        """
        multiplier = 1.0
        
        # Geopolitical tension effect
        tension_factors = {
            GeopoliticalTension.LOW: 0.95,
            GeopoliticalTension.MODERATE: 1.0,
            GeopoliticalTension.HIGH: 1.15,
            GeopoliticalTension.CRISIS: 1.40
        }
        multiplier *= tension_factors[self.tension_level]
        
        # Currency pressure effect (depreciation increases risk)
        pressure_factors = {
            CurrencyPressure.STRONG_APPRECIATION: 0.95,
            CurrencyPressure.MILD_APPRECIATION: 0.98,
            CurrencyPressure.STABLE: 1.0,
            CurrencyPressure.MILD_DEPRECIATION: 1.10,
            CurrencyPressure.STRONG_DEPRECIATION: 1.25
        }
        multiplier *= pressure_factors[self.currency_pressure]
        
        # Forex reserve adequacy (low reserves = higher risk)
        reserve_adequacy = self.forex_reserves.compute_adequacy_score()
        if reserve_adequacy < 0.7:
            multiplier *= (1.0 + (0.7 - reserve_adequacy) * 0.5)
        
        # Sovereign stress
        sovereign_stress = self.treasury_market.compute_sovereign_stress()
        multiplier *= (1.0 + sovereign_stress * 0.3)
        
        # Global risk sentiment (low appetite = higher risk)
        multiplier *= (1.0 + (1.0 - self.global_risk_appetite) * 0.2)
        
        return multiplier
    
    def update_from_shocks(self, shocks: Dict[str, float]):
        """
        Update geopolitical state from external shocks
        
        Expected shock keys:
        - 'geopolitical_tension': -1 to 1 (increase in tension)
        - 'capital_outflow': USD billions
        - 'yield_shock_bps': basis points increase
        - 'gold_price_shock_pct': % change in gold price
        - 'us_yield_shock_bps': change in US yields
        """
        # Geopolitical tension
        if 'geopolitical_tension' in shocks:
            tension_change = shocks['geopolitical_tension']
            if tension_change > 0.5:
                self.tension_level = GeopoliticalTension.CRISIS
            elif tension_change > 0.2:
                self.tension_level = GeopoliticalTension.HIGH
            elif tension_change > -0.2:
                self.tension_level = GeopoliticalTension.MODERATE
            else:
                self.tension_level = GeopoliticalTension.LOW
        
        # Capital flows
        if 'capital_outflow' in shocks:
            outflow = shocks['capital_outflow']
            self.capital_flows_net_usd -= outflow
            self.forex_reserves.apply_capital_outflow(outflow)
            
            # Trigger currency pressure
            if outflow > 10.0:  # >$10B outflow
                self.currency_pressure = CurrencyPressure.STRONG_DEPRECIATION
            elif outflow > 5.0:
                self.currency_pressure = CurrencyPressure.MILD_DEPRECIATION
        
        # Bond market stress
        if 'yield_shock_bps' in shocks:
            self.treasury_market.simulate_yield_shock(shocks['yield_shock_bps'])
        
        # Gold price changes
        if 'gold_price_shock_pct' in shocks:
            self.gold_price_usd_per_oz *= (1.0 + shocks['gold_price_shock_pct'])
            # Update gold reserve value
            self.forex_reserves.gold_value_usd = (
                self.forex_reserves.gold_reserves_tonnes * 32150.75 / 1000000 
                * self.gold_price_usd_per_oz
            )
        
        # US yield changes (affects spreads)
        if 'us_yield_shock_bps' in shocks:
            self.us_10yr_yield += shocks['us_yield_shock_bps'] / 100.0
            self.treasury_market.yield_spread_vs_us = (
                (self.treasury_market.avg_yield_10yr - self.us_10yr_yield) * 100
            )
    
    def get_regulatory_constraints(self) -> Dict[str, float]:
        """
        Generate regulatory constraints based on geopolitical state
        
        Returns dictionary of constraint tightening factors (1.0 = normal, >1.0 = tighter)
        """
        constraints = {
            'capital_requirement_multiplier': 1.0,
            'liquidity_requirement_multiplier': 1.0,
            'foreign_exposure_limit_multiplier': 1.0,
            'leverage_limit_multiplier': 1.0,
            'concentration_limit_multiplier': 1.0
        }
        
        # Tighten based on tension
        if self.tension_level == GeopoliticalTension.CRISIS:
            constraints['capital_requirement_multiplier'] = 1.3
            constraints['liquidity_requirement_multiplier'] = 1.4
            constraints['foreign_exposure_limit_multiplier'] = 0.7  # Reduce limits
        elif self.tension_level == GeopoliticalTension.HIGH:
            constraints['capital_requirement_multiplier'] = 1.15
            constraints['liquidity_requirement_multiplier'] = 1.2
            constraints['foreign_exposure_limit_multiplier'] = 0.85
        
        # Tighten if reserves are low
        reserve_adequacy = self.forex_reserves.compute_adequacy_score()
        if reserve_adequacy < 0.7:
            constraints['foreign_exposure_limit_multiplier'] *= 0.8
            constraints['liquidity_requirement_multiplier'] *= 1.2
        
        # Tighten if sovereign stress is high
        sovereign_stress = self.treasury_market.compute_sovereign_stress()
        if sovereign_stress > 0.6:
            constraints['capital_requirement_multiplier'] *= 1.2
            constraints['leverage_limit_multiplier'] = 0.85
        
        return constraints
    
    def to_dict(self) -> Dict:
        """Export state as dictionary"""
        # Map enum values to numeric scores for API serialization
        tension_scores = {
            GeopoliticalTension.LOW: 0.0,
            GeopoliticalTension.MODERATE: 0.33,
            GeopoliticalTension.HIGH: 0.66,
            GeopoliticalTension.CRISIS: 1.0
        }
        
        currency_scores = {
            CurrencyPressure.STRONG_APPRECIATION: -1.0,
            CurrencyPressure.MILD_APPRECIATION: -0.5,
            CurrencyPressure.STABLE: 0.0,
            CurrencyPressure.MILD_DEPRECIATION: 0.5,
            CurrencyPressure.STRONG_DEPRECIATION: 1.0
        }
        
        return {
            'tension_level': tension_scores.get(self.tension_level, 0.0),
            'currency_pressure': currency_scores.get(self.currency_pressure, 0.0),
            'forex_reserves_usd': self.forex_reserves.total_usd,
            'forex_adequacy': self.forex_reserves.compute_adequacy_score(),
            'gold_reserves_tonnes': self.forex_reserves.gold_reserves_tonnes,
            'gold_value_usd': self.forex_reserves.gold_value_usd,
            'us_treasury_holdings_usd': (
                self.forex_reserves.us_treasury_bills + 
                self.forex_reserves.us_treasury_bonds
            ),
            'domestic_10yr_yield': self.treasury_market.avg_yield_10yr,
            'us_10yr_yield': self.us_10yr_yield,
            'yield_spread_bps': self.treasury_market.yield_spread_vs_us,
            'sovereign_stress': self.treasury_market.compute_sovereign_stress(),
            'systemic_risk_multiplier': self.compute_systemic_risk_multiplier(),
            'trade_balance_usd': self.trade_balance_usd,
            'vix_index': self.vix_index,
            'global_risk_appetite': self.global_risk_appetite
        }


# Default initialization for India-like economy
def create_default_indian_geopolitical_state() -> GeopoliticalState:
    """
    Create realistic geopolitical state for India
    Based on 2025-2026 projections
    """
    return GeopoliticalState(
        tension_level=GeopoliticalTension.MODERATE,
        regional_conflicts=['border_tensions', 'trade_negotiations'],
        trade_war_active=False,
        sanctions_exposure=0.1,
        currency_pressure=CurrencyPressure.MILD_DEPRECIATION,
        exchange_rate_volatility=0.03,
        capital_flows_net_usd=5.0,  # $5B monthly net inflows
        forex_reserves=ForexReserves(
            total_usd=640.0,  # $640B (realistic for India)
            us_treasury_bills=180.0,
            us_treasury_bonds=120.0,
            gold_reserves_tonnes=800.0,
            gold_value_usd=55.0,
            other_currencies={'EUR': 60.0, 'GBP': 25.0, 'JPY': 35.0, 'CNY': 20.0},
            import_cover_months=8.5,
            reserves_to_gdp_ratio=0.17,
            reserves_to_short_term_debt=2.0
        ),
        treasury_market=TreasuryBondMarket(
            outstanding_bonds_usd=2500.0,
            avg_yield_10yr=7.1,
            yield_spread_vs_us=260.0,
            debt_to_gdp_ratio=0.58,
            domestic_banks_pct=42.0,
            foreign_investors_pct=18.0,
            central_bank_pct=23.0,
            insurance_pension_pct=17.0
        ),
        us_10yr_yield=4.5,
        us_dollar_index=103.0,
        gold_price_usd_per_oz=2050.0,
        trade_balance_usd=-8.5,  # Monthly deficit
        export_growth_yoy=0.08,
        import_growth_yoy=0.10,
        vix_index=16.5,
        global_risk_appetite=0.65
    )
