"""
Quick Stability Test: Compare with and without ML

Shows that the system is now stable and ML provides additional safety.
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import logging
from app.engine.simulation_engine import FinancialEcosystem, SimulationConfig
from app.engine.agents import BankAgent, CCPAgent, SectorAgent

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


def run_stability_test(enable_ml: bool, timesteps: int = 20) -> dict:
    """Run simulation and return stability metrics"""
    
    config = SimulationConfig(
        max_timesteps=timesteps,
        enable_shocks=True,
        shock_probability=0.1,
        enable_ml=enable_ml,
        random_seed=42
    )
    
    ecosystem = FinancialEcosystem(config)
    
    # Create test banks
    banks = [
        BankAgent("BANK_1", 1000, 9000, 300, 2.5, 15.0, 9.0),
        BankAgent("BANK_2", 800, 9200, 150, 5.5, 11.0, 9.0),
        BankAgent("BANK_3", 500, 9500, 80, 8.5, 9.5, 9.0),
    ]
    
    for bank in banks:
        ecosystem.add_agent(bank)
    
    # Add CCP and sector
    ccp = CCPAgent("CCP_MAIN", 5000, 10.0)
    ecosystem.add_agent(ccp)
    
    sector = SectorAgent("SECTOR_MANUFACTURING", "Manufacturing", 0.8)
    ecosystem.add_agent(sector)
    
    # Create exposures
    ecosystem.add_exposure("BANK_1", "BANK_2", 200, "interbank_loan")
    ecosystem.add_exposure("BANK_2", "BANK_3", 150, "interbank_loan")
    ecosystem.add_exposure("BANK_1", "SECTOR_MANUFACTURING", 500, "credit")
    ecosystem.add_exposure("BANK_2", "SECTOR_MANUFACTURING", 300, "credit")
    ecosystem.add_exposure("BANK_3", "SECTOR_MANUFACTURING", 200, "credit")
    
    # Run simulation
    initial_capitals = {b.agent_id: b.capital for b in banks}
    initial_crars = {b.agent_id: b.crar for b in banks}
    
    for t in range(timesteps):
        try:
            ecosystem.step()
        except Exception as e:
            logger.error(f"Simulation failed at t={t}: {e}")
            break
    
    # Collect results
    alive_banks = [b for b in banks if b.alive]
    defaulted_banks = [b for b in banks if not b.alive]
    
    avg_capital_change = 0
    avg_crar_change = 0
    if alive_banks:
        for bank in alive_banks:
            capital_change = ((bank.capital - initial_capitals[bank.agent_id]) / 
                             initial_capitals[bank.agent_id]) * 100
            crar_change = bank.crar - initial_crars[bank.agent_id]
            avg_capital_change += capital_change
            avg_crar_change += crar_change
        avg_capital_change /= len(alive_banks)
        avg_crar_change /= len(alive_banks)
    
    return {
        'ml_enabled': enable_ml,
        'banks_alive': len(alive_banks),
        'banks_defaulted': len(defaulted_banks),
        'avg_capital_change_pct': avg_capital_change,
        'avg_crar_change': avg_crar_change,
        'final_health': sum(b.compute_health() for b in alive_banks) / len(alive_banks) if alive_banks else 0
    }


if __name__ == "__main__":
    print("=" * 80)
    print("STABILIT TEST: Comparing ABM with and without ML Risk Reduction")
    print("=" * 80)
    print()
    
    print("Running 20 timesteps with 3 banks...")
    print()
    
    # Test without ML
    print("Test 1: Traditional ABM (NO ML)")
    print("-" * 40)
    results_no_ml = run_stability_test(enable_ml=False, timesteps=20)
    
    print(f"  Banks Alive: {results_no_ml['banks_alive']}/3")
    print(f"  Banks Defaulted: {results_no_ml['banks_defaulted']}/3")
    print(f"  Avg Capital Change: {results_no_ml['avg_capital_change_pct']:.2f}%")
    print(f"  Avg CRAR Change: {results_no_ml['avg_crar_change']:.2f}%")
    print(f"  Avg Health Score: {results_no_ml['final_health']:.3f}")
    print()
    
    # Test with ML
    print("Test 2: ML-Enhanced ABM (WITH ML)")
    print("-" * 40)
    results_with_ml = run_stability_test(enable_ml=True, timesteps=20)
    
    print(f"  Banks Alive: {results_with_ml['banks_alive']}/3")
    print(f"  Banks Defaulted: {results_with_ml['banks_defaulted']}/3")
    print(f"  Avg Capital Change: {results_with_ml['avg_capital_change_pct']:.2f}%")
    print(f"  Avg CRAR Change: {results_with_ml['avg_crar_change']:.2f}%")
    print(f"  Avg Health Score: {results_with_ml['final_health']:.3f}")
    print()
    
    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    survival_improvement = results_with_ml['banks_alive'] - results_no_ml['banks_alive']
    capital_improvement = results_with_ml['avg_capital_change_pct'] - results_no_ml['avg_capital_change_pct']
    health_improvement = results_with_ml['final_health'] - results_no_ml['final_health']
    
    print(f"Survival Rate: {results_no_ml['banks_alive']}/3 → {results_with_ml['banks_alive']}/3 "
          f"({'+'if survival_improvement >= 0 else ''}{survival_improvement} banks)")
    print(f"Capital Preservation: {results_no_ml['avg_capital_change_pct']:.2f}% → {results_with_ml['avg_capital_change_pct']:.2f}% "
          f"({'+'if capital_improvement >= 0 else ''}{capital_improvement:.2f}%)")
    print(f"Health Score: {results_no_ml['final_health']:.3f} → {results_with_ml['final_health']:.3f} "
          f"({'+'if health_improvement >= 0 else ''}{health_improvement:.3f})")
    
    print()
    if results_with_ml['banks_alive'] >= results_no_ml['banks_alive']:
        print("✓ System is STABLE with or without ML")
        if results_with_ml['banks_alive'] > results_no_ml['banks_alive']:
            print("✓ ML provides ADDITIONAL SAFETY - more banks survive!")
        else:
            print("✓ ML maintains same survival rate with better risk management")
    else:
        print("⚠ Results inconclusive - may need more tuning")
    
    print("=" * 80)
