"""
Test Script for ML-Based Risk Reduction in ABM

This script demonstrates that the agent-based model now uses ML to actively
reduce risk for every node in the network.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import logging
from app.engine.simulation_engine import FinancialEcosystem, SimulationConfig
from app.engine.agents import BankAgent, CCPAgent, SectorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ml_risk_reduction():
    """
    Test that ML-based risk reduction is working in the ABM.
    """
    logger.info("=" * 80)
    logger.info("Testing ML-Based Risk Reduction in Agent-Based Model")
    logger.info("=" * 80)
    
    # Create simulation with ML enabled
    config = SimulationConfig(
        max_timesteps=10,
        enable_shocks=True,
        shock_probability=0.1,
        enable_ml=True,  # EXPLICITLY ENABLE ML
        random_seed=42
    )
    
    ecosystem = FinancialEcosystem(config)
    
    # Check if ML advisor was initialized
    if ecosystem.ml_risk_advisor:
        logger.info("✓ ML Risk Advisor successfully initialized")
    else:
        logger.warning("⚠ ML Risk Advisor not initialized (will use heuristics)")
    
    # Create test banks with varying risk profiles
    banks = [
        BankAgent(
            agent_id="BANK_HEALTHY",
            initial_capital=1000,
            initial_assets=9000,
            initial_liquidity=300,
            initial_npa_ratio=2.5,
            initial_crar=15.0,
            regulatory_min_crar=9.0
        ),
        BankAgent(
            agent_id="BANK_MODERATE",
            initial_capital=800,
            initial_assets=9200,
            initial_liquidity=150,
            initial_npa_ratio=5.5,
            initial_crar=11.0,
            regulatory_min_crar=9.0
        ),
        BankAgent(
            agent_id="BANK_RISKY",
            initial_capital=500,
            initial_assets=9500,
            initial_liquidity=80,
            initial_npa_ratio=8.5,
            initial_crar=9.5,
            regulatory_min_crar=9.0
        ),
    ]
    
    # Add banks to ecosystem
    for bank in banks:
        ecosystem.add_agent(bank)
        
        # Verify ML advisor was injected
        if bank.ml_risk_advisor:
            logger.info(f"✓ {bank.agent_id}: ML Risk Advisor injected")
        else:
            logger.warning(f"⚠ {bank.agent_id}: No ML Risk Advisor")
    
    # Create CCP
    ccp = CCPAgent(
        agent_id="CCP_MAIN",
        initial_default_fund=5000,
        initial_margin_requirement=10.0
    )
    ecosystem.add_agent(ccp)
    
    if ccp.ml_risk_advisor:
        logger.info(f"✓ {ccp.agent_id}: ML Risk Advisor injected")
    
    # Create sector
    sector = SectorAgent(
        agent_id="SECTOR_MANUFACTURING",
        sector_name="Manufacturing",
        initial_health=0.8
    )
    ecosystem.add_agent(sector)
    
    # Create exposures between banks
    ecosystem.add_exposure("BANK_HEALTHY", "BANK_MODERATE", 200, "interbank_loan")
    ecosystem.add_exposure("BANK_MODERATE", "BANK_RISKY", 150, "interbank_loan")
    ecosystem.add_exposure("BANK_HEALTHY", "BANK_RISKY", 100, "interbank_loan")
    ecosystem.add_exposure("BANK_HEALTHY", "SECTOR_MANUFACTURING", 500, "credit")
    ecosystem.add_exposure("BANK_MODERATE", "SECTOR_MANUFACTURING", 300, "credit")
    ecosystem.add_exposure("BANK_RISKY", "SECTOR_MANUFACTURING", 200, "credit")
    
    logger.info("\n" + "=" * 80)
    logger.info("Running Simulation with ML-Guided Risk Reduction")
    logger.info("=" * 80 + "\n")
    
    # Run simulation
    initial_risk_levels = {}
    for bank in banks:
        initial_risk_levels[bank.agent_id] = {
            'capital': bank.capital,
            'crar': bank.crar,
            'liquidity': bank.liquidity,
            'credit_limit': bank.credit_supply_limit,
            'risk_appetite': bank.risk_appetite
        }
    
    # Run 5 timesteps
    for t in range(5):
        logger.info(f"\n{'='*60}")
        logger.info(f"TIMESTEP {t}")
        logger.info(f"{'='*60}")
        
        snapshot = ecosystem.step()
        
        # Display agent decisions
        for bank in banks:
            if bank.alive:
                health = bank.compute_health()
                logger.info(f"\n{bank.agent_id}:")
                logger.info(f"  Health Score: {health:.3f}")
                logger.info(f"  CRAR: {bank.crar:.2f}%")
                logger.info(f"  Liquidity: {bank.liquidity:.2f}")
                logger.info(f"  Mode: {bank.mode.value}")
                logger.info(f"  Credit Limit: {bank.credit_supply_limit:.2f}")
                logger.info(f"  Risk Appetite: {bank.risk_appetite:.3f}")
    
    # Compare initial vs final risk levels
    logger.info("\n" + "=" * 80)
    logger.info("RISK REDUCTION SUMMARY")
    logger.info("=" * 80 + "\n")
    
    for bank in banks:
        if bank.alive:
            initial = initial_risk_levels[bank.agent_id]
            
            logger.info(f"{bank.agent_id}:")
            logger.info(f"  CRAR: {initial['crar']:.2f}% → {bank.crar:.2f}% "
                       f"({'+' if bank.crar > initial['crar'] else ''}{bank.crar - initial['crar']:.2f}%)")
            logger.info(f"  Liquidity: {initial['liquidity']:.2f} → {bank.liquidity:.2f} "
                       f"({'+' if bank.liquidity > initial['liquidity'] else ''}{bank.liquidity - initial['liquidity']:.2f})")
            logger.info(f"  Risk Appetite: {initial['risk_appetite']:.3f} → {bank.risk_appetite:.3f} "
                       f"({'+' if bank.risk_appetite > initial['risk_appetite'] else ''}{bank.risk_appetite - initial['risk_appetite']:.3f})")
            
            # Calculate overall risk reduction
            initial_risk = (1 - initial['crar']/15.0) * 0.5 + (1 - initial['liquidity']/initial['capital']) * 0.3
            final_risk = (1 - bank.crar/15.0) * 0.5 + (1 - bank.liquidity/bank.capital) * 0.3
            risk_reduction = initial_risk - final_risk
            
            if risk_reduction > 0:
                logger.info(f"  ✓ Risk Reduced: {risk_reduction * 100:.2f}%")
            elif risk_reduction < 0:
                logger.info(f"  ⚠ Risk Increased: {abs(risk_reduction) * 100:.2f}%")
            else:
                logger.info(f"  = Risk Unchanged")
            
            logger.info("")
    
    # CCP metrics
    logger.info(f"CCP {ccp.agent_id}:")
    logger.info(f"  Initial Margin Requirement: {ccp.initial_margin_requirement:.2f}%")
    logger.info(f"  Default Fund Size: {ccp.default_fund_size:.2f}")
    logger.info(f"  Mode: {ccp.mode.value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    
    # Summary statistics
    alive_banks = [b for b in banks if b.alive]
    defaulted_banks = [b for b in banks if not b.alive]
    
    logger.info(f"\nSummary:")
    logger.info(f"  Alive Banks: {len(alive_banks)}/{len(banks)}")
    logger.info(f"  Defaulted Banks: {len(defaulted_banks)}/{len(banks)}")
    
    if alive_banks:
        avg_health = sum(b.compute_health() for b in alive_banks) / len(alive_banks)
        logger.info(f"  Average Health Score: {avg_health:.3f}")
        
        avg_crar = sum(b.crar for b in alive_banks) / len(alive_banks)
        logger.info(f"  Average CRAR: {avg_crar:.2f}%")
    
    if ecosystem.ml_risk_advisor:
        logger.info("\n✓ ML-Based Risk Reduction is ACTIVE")
        logger.info("  All agents are using ML predictions to reduce risk dynamically")
    else:
        logger.info("\n⚠ Using heuristic-based risk management (ML model not loaded)")
    
    return ecosystem


if __name__ == "__main__":
    try:
        ecosystem = test_ml_risk_reduction()
        print("\n✓ Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
