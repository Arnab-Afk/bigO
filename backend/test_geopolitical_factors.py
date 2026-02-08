"""
Test Geopolitical Factors and Regulator Response
=================================================
Demonstrates how regulators respond to:
- Geopolitical tensions
- Forex reserve depletion
- Treasury bond market stress
- Capital outflows
- Currency depreciation
- Gold price shocks
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.engine.geopolitical_state import (
    GeopoliticalState,
    create_default_indian_geopolitical_state,
    GeopoliticalTension,
    CurrencyPressure
)
from app.engine.agents import RegulatorAgent, BankAgent
import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header"""
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def print_geopolitical_state(geo_state: GeopoliticalState, title="Geopolitical State"):
    """Pretty print geopolitical state"""
    print_section(title)
    
    logger.info(f"\n🌍 GEOPOLITICAL FACTORS:")
    logger.info(f"  Tension Level:        {geo_state.tension_level.value.upper()}")
    logger.info(f"  Currency Pressure:    {geo_state.currency_pressure.value.replace('_', ' ').title()}")
    logger.info(f"  Trade War Active:     {'YES' if geo_state.trade_war_active else 'NO'}")
    logger.info(f"  Sanctions Exposure:   {geo_state.sanctions_exposure:.1%}")
    
    logger.info(f"\n💵 FOREX RESERVES:")
    logger.info(f"  Total Reserves:       ${geo_state.forex_reserves.total_usd:.1f}B")
    logger.info(f"  US Treasury Bills:    ${geo_state.forex_reserves.us_treasury_bills:.1f}B")
    logger.info(f"  US Treasury Bonds:    ${geo_state.forex_reserves.us_treasury_bonds:.1f}B")
    logger.info(f"  Gold (tonnes):        {geo_state.forex_reserves.gold_reserves_tonnes:.0f}t")
    logger.info(f"  Gold Value:           ${geo_state.forex_reserves.gold_value_usd:.1f}B")
    logger.info(f"  Import Cover:         {geo_state.forex_reserves.import_cover_months:.1f} months")
    logger.info(f"  Reserve Adequacy:     {geo_state.forex_reserves.compute_adequacy_score():.1%}")
    
    logger.info(f"\n🏛️  TREASURY BOND MARKET:")
    logger.info(f"  Outstanding Debt:     ${geo_state.treasury_market.outstanding_bonds_usd:.0f}B")
    logger.info(f"  10Y Yield:            {geo_state.treasury_market.avg_yield_10yr:.2f}%")
    logger.info(f"  Debt/GDP Ratio:       {geo_state.treasury_market.debt_to_gdp_ratio:.1%}")
    logger.info(f"  Sovereign Stress:     {geo_state.treasury_market.compute_sovereign_stress():.1%}")
    
    logger.info(f"\n📊 INTERNATIONAL BENCHMARKS:")
    logger.info(f"  US 10Y Yield:         {geo_state.us_10yr_yield:.2f}%")
    logger.info(f"  Yield Spread:         {geo_state.treasury_market.yield_spread_vs_us:.0f} bps")
    logger.info(f"  Gold Price:           ${geo_state.gold_price_usd_per_oz:.0f}/oz")
    logger.info(f"  VIX Index:            {geo_state.vix_index:.1f}")
    logger.info(f"  Risk Appetite:        {geo_state.global_risk_appetite:.1%}")
    
    logger.info(f"\n⚠️  RISK ASSESSMENT:")
    multiplier = geo_state.compute_systemic_risk_multiplier()
    logger.info(f"  Systemic Risk Multiplier: {multiplier:.2f}x")
    if multiplier > 1.3:
        logger.info(f"  └─ HIGH RISK: Geopolitical factors significantly amplify domestic risk")
    elif multiplier > 1.1:
        logger.info(f"  └─ MODERATE RISK: Some amplification from external factors")
    else:
        logger.info(f"  └─ LOW RISK: Stable external environment")


def print_regulator_response(regulator: RegulatorAgent, decisions: dict):
    """Print regulator's policy response"""
    print_section("REGULATOR POLICY RESPONSE")
    
    logger.info(f"\n🏦 MONETARY POLICY:")
    logger.info(f"  Repo Rate:            {regulator.base_repo_rate:.2f}%")
    logger.info(f"  Policy Stance:        {regulator.policy_stance.upper()}")
    logger.info(f"  Rate Change:          {decisions.get('rate_change', 0):.2f}%")
    
    logger.info(f"\n📈 MACRO-PRUDENTIAL POLICY:")
    logger.info(f"  Base CRAR Requirement:      {regulator.base_min_crar:.2f}%")
    logger.info(f"  Countercyclical Buffer:     {regulator.countercyclical_buffer:.2f}%")
    logger.info(f"  Adjusted CRAR Requirement:  {decisions['adjusted_crar_requirement']:.2f}%")
    logger.info(f"  Foreign Currency Limit:     {regulator.foreign_currency_limit:.1%}")
    
    logger.info(f"\n🚨 SYSTEM STATUS:")
    logger.info(f"  System Risk Score:    {regulator.system_wide_risk_score:.2%}")
    logger.info(f"  System Liquidity:     {regulator.system_liquidity:.2%}")
    logger.info(f"  Violations:           {len(regulator.violations)}")
    logger.info(f"  Interventions:        {regulator.intervention_count}")
    logger.info(f"  Forex Interventions:  {regulator.forex_interventions_count}")
    
    if regulator.violations:
        logger.info(f"\n  Violations:")
        for v in regulator.violations[:5]:
            logger.info(f"    • {v}")
        if len(regulator.violations) > 5:
            logger.info(f"    ... and {len(regulator.violations) - 5} more")


def scenario_normal_conditions():
    """Baseline scenario: Normal conditions"""
    print_section("SCENARIO 1: NORMAL CONDITIONS")
    
    # Create baseline geopolitical state
    geo_state = create_default_indian_geopolitical_state()
    print_geopolitical_state(geo_state)
    
    # Create regulator
    regulator = RegulatorAgent(
        agent_id='RBI',
        base_repo_rate=6.5,
        min_crar=9.0,
        geopolitical_state=geo_state
    )
    
    # Create simple network
    G = nx.DiGraph()
    for i in range(3):
        bank = BankAgent(
            agent_id=f'BANK_{i}',
            initial_capital=10000,
            initial_assets=80000,
            initial_liquidity=2000,
            initial_crar=11.5
        )
        G.add_node(bank.agent_id, agent=bank)
    
    # Regulator perception and decision
    global_state = {}
    regulator.perceive(G, global_state)
    decisions = regulator.decide()
    
    print_regulator_response(regulator, decisions)
    
    logger.info(f"\n✅ ASSESSMENT: Stable conditions, normal policy stance")


def scenario_geopolitical_crisis():
    """Scenario: Major geopolitical crisis"""
    print_section("SCENARIO 2: GEOPOLITICAL CRISIS")
    
    logger.info("\n🔴 CRISIS EVENT: Major regional conflict + trade war + capital flight")
    
    # Create crisis state
    geo_state = create_default_indian_geopolitical_state()
    geo_state.tension_level = GeopoliticalTension.CRISIS
    geo_state.trade_war_active = True
    geo_state.regional_conflicts = ['border_war', 'trade_sanctions', 'cyber_attacks']
    geo_state.currency_pressure = CurrencyPressure.STRONG_DEPRECIATION
    
    # Apply shocks
    shocks = {
        'geopolitical_tension': 0.8,
        'capital_outflow': 50.0,  # $50B outflow
        'yield_shock_bps': 150,  # 150 bps spike
        'us_yield_shock_bps': 50
    }
    geo_state.update_from_shocks(shocks)
    
    print_geopolitical_state(geo_state, "POST-CRISIS Geopolitical State")
    
    # Create regulator
    regulator = RegulatorAgent(
        agent_id='RBI',
        base_repo_rate=6.5,
        min_crar=9.0,
        geopolitical_state=geo_state
    )
    
    # Create stressed banks
    G = nx.DiGraph()
    for i in range(3):
        bank = BankAgent(
            agent_id=f'BANK_{i}',
            initial_capital=10000,
            initial_assets=80000,
            initial_liquidity=1000,  # Lower liquidity
            initial_crar=10.0,  # Lower CRAR
            initial_npa_ratio=8.0  # High NPAs
        )
        G.add_node(bank.agent_id, agent=bank)
    
    # Regulator perception and decision
    global_state = {}
    regulator.perceive(G, global_state)
    decisions = regulator.decide()
    
    print_regulator_response(regulator, decisions)
    
    logger.info(f"\n🚨 ASSESSMENT: Crisis response - emergency measures activated")


def scenario_forex_stress():
    """Scenario: Forex reserve depletion"""
    print_section("SCENARIO 3: FOREX RESERVE CRISIS")
    
    logger.info("\n⚠️  EVENT: Massive capital outflows depleting forex reserves")
    
    geo_state = create_default_indian_geopolitical_state()
    
    # Deplete reserves through multiple outflows
    logger.info("\nSimulating sustained capital outflows...")
    for month in range(1, 7):
        outflow = 40.0  # $40B per month
        geo_state.forex_reserves.apply_capital_outflow(outflow)
        logger.info(f"  Month {month}: ${outflow:.0f}B outflow → Reserves: ${geo_state.forex_reserves.total_usd:.1f}B")
    
    geo_state.currency_pressure = CurrencyPressure.STRONG_DEPRECIATION
    
    print_geopolitical_state(geo_state, "POST-OUTFLOW Geopolitical State")
    
    regulator = RegulatorAgent(
        agent_id='RBI',
        base_repo_rate=6.5,
        min_crar=9.0,
        geopolitical_state=geo_state
    )
    
    G = nx.DiGraph()
    bank = BankAgent('BANK_1', 10000, 80000, 1500, 11.0)
    G.add_node(bank.agent_id, agent=bank)
    
    global_state = {}
    regulator.perceive(G, global_state)
    decisions = regulator.decide()
    
    print_regulator_response(regulator, decisions)
    
    logger.info(f"\n🔒 ASSESSMENT: Capital controls and forex intervention required")


def scenario_sovereign_debt_crisis():
    """Scenario: Sovereign debt market stress"""
    print_section("SCENARIO 4: SOVEREIGN DEBT CRISIS")
    
    logger.info("\n📉 EVENT: Bond market sell-off + fiscal crisis")
    
    geo_state = create_default_indian_geopolitical_state()
    
    # Spike in yields, debt concerns
    geo_state.treasury_market.avg_yield_10yr = 9.5  # Up from 7.1%
    geo_state.treasury_market.yield_spread_vs_us = 500.0  # 500 bps spread
    geo_state.treasury_market.debt_to_gdp_ratio = 0.85  # High debt
    geo_state.treasury_market.foreign_investors_pct = 8.0  # FII exit
    
    # Global risk-off
    geo_state.vix_index = 35.0
    geo_state.global_risk_appetite = 0.2
    
    print_geopolitical_state(geo_state, "SOVEREIGN CRISIS State")
    
    regulator = RegulatorAgent(
        agent_id='RBI',
        base_repo_rate=7.0,
        min_crar=9.0,
        geopolitical_state=geo_state
    )
    
    G = nx.DiGraph()
    # Banks holding government bonds face mark-to-market losses
    for i in range(2):
        bank = BankAgent(f'BANK_{i}', 10000, 80000, 1200, 10.5)
        bank.npa_ratio = 6.0
        G.add_node(bank.agent_id, agent=bank)
    
    global_state = {}
    regulator.perceive(G, global_state)
    decisions = regulator.decide()
    
    print_regulator_response(regulator, decisions)
    
    logger.info(f"\n💥 ASSESSMENT: Sovereign stress requires coordinated fiscal-monetary response")


def scenario_gold_rally():
    """Scenario: Gold price surge strengthening reserves"""
    print_section("SCENARIO 5: GOLD PRICE RALLY")
    
    logger.info("\n🥇 EVENT: Gold surges to $2500/oz amid global uncertainty")
    
    geo_state = create_default_indian_geopolitical_state()
    
    # Gold price shock +25%
    geo_state.update_from_shocks({'gold_price_shock_pct': 0.25})
    
    logger.info(f"\nGold price: ${geo_state.gold_price_usd_per_oz:.0f}/oz")
    logger.info(f"Gold reserves value increased to: ${geo_state.forex_reserves.gold_value_usd:.1f}B")
    
    print_geopolitical_state(geo_state, "POST-GOLD RALLY State")
    
    regulator = RegulatorAgent(
        agent_id='RBI',
        base_repo_rate=6.5,
        min_crar=9.0,
        geopolitical_state=geo_state
    )
    
    G = nx.DiGraph()
    bank = BankAgent('BANK_1', 10000, 80000, 1800, 11.5)
    G.add_node(bank.agent_id, agent=bank)
    
    global_state = {}
    regulator.perceive(G, global_state)
    decisions = regulator.decide()
    
    print_regulator_response(regulator, decisions)
    
    logger.info(f"\n✅ ASSESSMENT: Stronger reserves provide policy flexibility")


def main():
    """Run all scenarios"""
    logger.info("\n" + "█" * 80)
    logger.info("GEOPOLITICAL FACTORS & REGULATORY RESPONSE DEMONSTRATION")
    logger.info("█" * 80)
    
    logger.info("\nThis demonstrates how regulators respond to:")
    logger.info("  • Geopolitical tensions and conflicts")
    logger.info("  • Forex reserve adequacy")
    logger.info("  • Treasury bond market stress")
    logger.info("  • Capital flows and currency pressure")
    logger.info("  • Gold reserves as stability buffer")
    logger.info("  • US Treasury holdings and spreads")
    
    # Run scenarios
    scenario_normal_conditions()
    input("\nPress Enter to continue to next scenario...")    
    scenario_geopolitical_crisis()
    input("\nPress Enter to continue to next scenario...")    
    scenario_forex_stress()
    input("\nPress Enter to continue to next scenario...")    
    scenario_sovereign_debt_crisis()
    input("\nPress Enter to continue to next scenario...")    
    scenario_gold_rally()
    
    print_section("SUMMARY")
    logger.info("\n✅ All scenarios completed successfully!")
    logger.info("\nKey Findings:")
    logger.info("  1. Regulators adjust policy dynamically based on geopolitical factors")
    logger.info("  2. Forex reserves act as first line of defense against external shocks")
    logger.info("  3. Treasury bond market stress forces tighter regulations")
    logger.info("  4. Geopolitical tensions amplify systemic risk (multiplier effect)")
    logger.info("  5. Gold reserves provide stability buffer during crises")
    logger.info("  6. CRAR requirements increase automatically under stress")
    logger.info("  7. Capital controls triggered by severe forex depletion")


if __name__ == "__main__":
    main()
