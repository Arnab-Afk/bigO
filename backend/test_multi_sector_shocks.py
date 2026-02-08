"""
Multi-Sector Shock Demonstration
=================================
Demonstrates all sector-specific shocks with quantifiable conditions.

Shows how different sectors respond to shocks based on:
- Financial metrics (debt service, leverage, profitability)
- Market conditions (demand, capacity, prices)
- Credit quality (default rates, restructuring)
- Sentiment (confidence, investment)
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.engine.sector_shocks import (
    SectorType, ShockSeverity, SectorState, SectorHealthIndicators,
    get_shock_scenario, create_default_sector_states,
    compute_bank_loss_from_sector_shock, SECTOR_SHOCK_LIBRARY
)
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section"""
    logger.info(\"\n\" + \"=\" * 80)
    logger.info(title.center(80))
    logger.info(\"=\" * 80)


def print_sector_state(sector_state: SectorState, title: str = None):
    \"\"\"Pretty print sector state\"\"\"
    if title:
        print_section(title)
    
    ind = sector_state.indicators
    
    logger.info(f\"\\n📊 SECTOR: {sector_state.sector_type.value.upper()}\")
    logger.info(f\"{'─' * 80}\")
    
    logger.info(f\"\\n💰 FINANCIAL METRICS:\")
    logger.info(f\"  Overall Health:           {ind.economic_health:.1%}\")
    logger.info(f\"  Debt Service Coverage:    {ind.debt_service_coverage:.2f}x\")
    logger.info(f\"  Leverage Ratio:           {ind.leverage_ratio:.1%} (Debt/Assets)\")
    logger.info(f\"  Profitability Margin:     {ind.profitability_margin:.1%}\")
    
    logger.info(f\"\\n📈 MARKET CONDITIONS:\")
    logger.info(f\"  Demand Index:             {ind.demand_index:.1%}\")
    logger.info(f\"  Capacity Utilization:     {ind.capacity_utilization:.1%}\")
    logger.info(f\"  Price Index:              {ind.price_index:.2f} (1.0 = baseline)\")
    
    logger.info(f\"\\n⚠️  CREDIT QUALITY:\")
    logger.info(f\"  Default Rate:             {ind.default_rate:.1%}\")
    logger.info(f\"  Restructuring Rate:       {ind.restructuring_rate:.1%}\")
    
    logger.info(f\"\\n🎯 SENTIMENT:\")
    logger.info(f\"  Business Confidence:      {ind.business_confidence:.1%}\")
    logger.info(f\"  Investment Growth:        {ind.investment_growth:+.1%}\")
    
    logger.info(f\"\\n🏦 BANKING EXPOSURE:\")
    logger.info(f\"  Total Exposure:           ${sector_state.total_exposure:,.0f}M\")
    logger.info(f\"  Banks Exposed:            {sector_state.num_banks_exposed}\")
    
    if sector_state.active_shocks:
        logger.info(f\"\\n🔴 ACTIVE SHOCKS: {len(sector_state.active_shocks)}\")
        for shock in sector_state.active_shocks:
            logger.info(f\"  • {shock.severity.value.upper()}: {shock.trigger_event}\")


def demo_sector_shock(sector_type: SectorType, severity: ShockSeverity):
    \"\"\"Demonstrate a specific sector shock\"\"\"
    
    # Get shock scenario
    scenario = get_shock_scenario(sector_type, severity)
    if not scenario:
        logger.warning(f\"No {severity.value} shock defined for {sector_type.value}\")
        return
    
    print_section(f\"{sector_type.value.upper()} - {severity.value.upper()} SHOCK\")
    
    logger.info(f\"\\n🔥 TRIGGER EVENT: {scenario.trigger_event}\")
    logger.info(f\"\\n📋 SHOCK PARAMETERS:\")
    logger.info(f\"  Demand Shock:             {scenario.demand_shock:+.1%}\")
    logger.info(f\"  Price Shock:              {scenario.price_shock:+.1%}\")
    logger.info(f\"  Leverage Increase:        {scenario.leverage_increase:+.1%}\")
    logger.info(f\"  Default Rate Increase:    {scenario.default_rate_increase:+.1%}\")
    logger.info(f\"  Confidence Drop:          {scenario.confidence_drop:.1%}\")
    logger.info(f\"  Recovery Periods:         {scenario.recovery_periods} timesteps\")
    
    if scenario.correlated_sectors:
        logger.info(f\"\\n🔗 SPILLOVER TO:\")
        for corr_sector in scenario.correlated_sectors:
            logger.info(f\"  • {corr_sector.value} (intensity: {scenario.spillover_intensity:.0%})\")
    
    # Create sector state and apply shock
    sector_state = SectorState(
        sector_type=sector_type,
        indicators=SectorHealthIndicators(),
        total_exposure=50000.0,  # $50B exposure
        num_banks_exposed=12
    )
    
    logger.info(f\"\\n\\n📸 BEFORE SHOCK:\")
    logger.info(f\"{'─' * 80}\")
    print_sector_state(sector_state, title=None)
    
    # Apply shock
    changes = sector_state.apply_shock(scenario)
    
    logger.info(f\"\\n\\n📸 AFTER SHOCK:\")
    logger.info(f\"{'─' * 80}\")
    print_sector_state(sector_state, title=None)
    
    logger.info(f\"\\n\\n📉 CHANGES:\")
    logger.info(f\"{'─' * 80}\")
    for metric, change in changes.items():
        if metric != 'economic_health':
            logger.info(f\"  {metric:.<30} {change:+.3f}\")
    
    # Compute bank losses
    bank_exposure = 5000.0  # $5B individual bank exposure
    bank_assets = 100000.0  # $100B bank assets
    loss = compute_bank_loss_from_sector_shock(sector_state, bank_exposure, bank_assets)
    
    logger.info(f\"\\n\\n💸 BANK IMPACT:\")
    logger.info(f\"{'─' * 80}\")
    logger.info(f\"  Bank Exposure:            ${bank_exposure:,.0f}M\")
    logger.info(f\"  Computed Loss:            ${loss:,.0f}M\")
    logger.info(f\"  Loss Rate:                {loss/bank_exposure if bank_exposure > 0 else 0:.2%}\")
    logger.info(f\"  Impact on Bank Capital:   {loss/bank_assets if bank_assets > 0 else 0:.2%}\")


def demo_all_sectors_overview():
    \"\"\"Show overview of all sector shock definitions\"\"\"
    print_section(\"MULTI-SECTOR SHOCK LIBRARY OVERVIEW\")
    
    logger.info(f\"\\nTotal sector types: {len(SectorType)}\")
    logger.info(f\"Total shock scenarios: {sum(len(shocks) for shocks in SECTOR_SHOCK_LIBRARY.values())}\")
    
    logger.info(f\"\\n{'SECTOR':<20} | {'MILD':<8} | {'MODERATE':<8} | {'SEVERE':<8} | {'CRISIS':<8}\")
    logger.info(f\"{'-' * 80}\")
    
    for sector in SectorType:
        row = f\"{sector.value:<20} |\"
        for severity in [ShockSeverity.MILD, ShockSeverity.MODERATE, ShockSeverity.SEVERE, ShockSeverity.CRISIS]:
            has_shock = get_shock_scenario(sector, severity) is not None
            row += f\" {'✓' if has_shock else '-':^8} |\"
        logger.info(row)
    
    logger.info(f\"\\n\\n🔍 SECTOR DETAILS:\")
    logger.info(f\"{'-' * 80}\")
    
    for sector_type, severity_dict in SECTOR_SHOCK_LIBRARY.items():
        logger.info(f\"\\n{sector_type.value.upper()}:\")
        for severity, scenario in severity_dict.items():
            logger.info(f\"  [{severity.value:8}] {scenario.trigger_event}\")


def demo_recovery():
    \"\"\"Demonstrate natural recovery process\"\"\"
    print_section(\"NATURAL RECOVERY DEMONSTRATION\")
    
    logger.info(f\"\\nShowing recovery of REAL ESTATE after SEVERE shock over 24 timesteps...\")
    
    # Create shocked sector
    sector_state = SectorState(
        sector_type=SectorType.REAL_ESTATE,
        indicators=SectorHealthIndicators()
    )
    
    scenario = get_shock_scenario(SectorType.REAL_ESTATE, ShockSeverity.SEVERE)
    sector_state.apply_shock(scenario)
    
    logger.info(f\"\\n{'Timestep':<10} | {'Health':<10} | {'Demand':<10} | {'Default Rate':<12} | {'Confidence':<12}\")
    logger.info(f\"{'-' * 70}\")
    
    for t in range(25):
        if t == 0:
            label = \"(shocked)\"
        else:
            label = \"\"
        
        logger.info(
            f\"{t:<10} | {sector_state.indicators.economic_health:<10.1%} | \"
            f\"{sector_state.indicators.demand_index:<10.1%} | \"
            f\"{sector_state.indicators.default_rate:<12.1%} | \"
            f\"{sector_state.indicators.business_confidence:<12.1%} {label}\"
        )
        
        if t > 0:
            sector_state.recover_step(recovery_rate=0.05)


def demo_spillover():
    \"\"\"Demonstrate spillover effects between correlated sectors\"\"\"
    print_section(\"SPILLOVER EFFECTS DEMONSTRATION\")
    
    logger.info(f\"\\nShowing how REAL ESTATE crisis spills over to correlated sectors...\")
    
    # Create all sector states
    all_sectors = create_default_sector_states()
    
    # Apply severe real estate shock
    re_scenario = get_shock_scenario(SectorType.REAL_ESTATE, ShockSeverity.SEVERE)
    re_state = all_sectors[SectorType.REAL_ESTATE]
    re_state.apply_shock(re_scenario)
    
    logger.info(f\"\\n✅ PRIMARY SHOCK: Real Estate health dropped to {re_state.indicators.economic_health:.1%}\")
    
    logger.info(f\"\\n🔗 SPILLOVER IMPACTS (intensity: {re_scenario.spillover_intensity:.0%}):\\n\")
    logger.info(f\"{'Sector':<20} | {'Initial Health':<15} | {'After Spillover':<15} | {'Change':<10}\")
    logger.info(f\"{'-' * 70}\")
    
    for correlated_sector in re_scenario.correlated_sectors:
        spillover_state = all_sectors[correlated_sector]
        initial_health = spillover_state.indicators.economic_health
        
        # Apply spillover (simplified)
        spillover_impact = re_scenario.spillover_intensity * 0.5
        spillover_state.indicators.demand_index *= (1.0 - spillover_impact * 0.3)
        spillover_state.indicators.business_confidence *= (1.0 - spillover_impact * 0.4)
        spillover_state.indicators.economic_health = spillover_state.indicators.compute_overall_health()
        
        change = spillover_state.indicators.economic_health - initial_health
        
        logger.info(
            f\"{correlated_sector.value:<20} | {initial_health:<15.1%} | \"
            f\"{spillover_state.indicators.economic_health:<15.1%} | {change:+.1%}\"
        )


def main():
    \"\"\"Run all demonstrations\"\"\"
    logger.info(\"\\n\" + \"█\" * 80)
    logger.info(\"MULTI-SECTOR SHOCK FRAMEWORK DEMONSTRATION\".center(80))
    logger.info(\"█\" * 80)
    
    # Overview
    demo_all_sectors_overview()
    input(\"\\nPress Enter to continue...\")
    
    # Demonstrate specific shocks
    logger.info(\"\\n\\n\")
    demo_sector_shock(SectorType.REAL_ESTATE, ShockSeverity.SEVERE)
    input(\"\\nPress Enter to continue...\")
    
    logger.info(\"\\n\\n\")
    demo_sector_shock(SectorType.ENERGY, ShockSeverity.MODERATE)
    input(\"\\nPress Enter to continue...\")
    
    logger.info(\"\\n\\n\")
    demo_sector_shock(SectorType.AGRICULTURE, ShockSeverity.MODERATE)
    input(\"\\nPress Enter to continue...\")
    
    logger.info(\"\\n\\n\")
    demo_sector_shock(SectorType.MSME, ShockSeverity.SEVERE)
    input(\"\\nPress Enter to continue...\")
    
    logger.info(\"\\n\\n\")
    demo_sector_shock(SectorType.MANUFACTURING, ShockSeverity.MODERATE)
    input(\"\\nPress Enter to continue...\")
    
    # Recovery demonstration
    logger.info(\"\\n\\n\")
    demo_recovery()
    input(\"\\nPress Enter to continue...\")
    
    # Spillover demonstration
    logger.info(\"\\n\\n\")
    demo_spillover()
    
    print_section(\"SUMMARY\")
    
    logger.info(f\"\\n✅ Framework Features:\")
    logger.info(f\"  • {len(SectorType)} distinct economic sectors\")
    logger.info(f\"  • {len(ShockSeverity)} severity levels (mild → crisis)\")
    logger.info(f\"  • {sum(len(s) for s in SECTOR_SHOCK_LIBRARY.values())} pre-defined scenarios\")
    logger.info(f\"  • 11 quantifiable health indicators per sector\")
    logger.info(f\"  • Automatic spillover to correlated sectors\")
    logger.info(f\"  • Natural recovery over time\")
    logger.info(f\"  • Bank loss computation based on exposures\")
    
    logger.info(f\"\\n📊 Quantifiable Metrics:\")
    logger.info(f\"  Financial: Debt service coverage, leverage, profitability\")
    logger.info(f\"  Market: Demand, capacity utilization, prices\")
    logger.info(f\"  Credit: Default rates, restructuring rates\")
    logger.info(f\"  Sentiment: Business confidence, investment growth\")
    
    logger.info(f\"\\n🎯 Use Cases:\")
    logger.info(f\"  • Test bank resilience to sector-specific shocks\")
    logger.info(f\"  • Analyze contagion through supply chains\")
    logger.info(f\"  • Optimize portfolio diversification\")
    logger.info(f\"  • Regulatory stress testing\")
    logger.info(f\"  • Policy impact analysis\")
    
    logger.info(f\"\\n\" + \"=\"*80)
    logger.info(f\"Demonstration complete!\")


if __name__ == \"__main__\":
    main()
