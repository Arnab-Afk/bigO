"""
Agent-Based Model Example Script
=================================
Demonstrates the complete ABM pipeline from initialization to visualization.

Usage:
    python examples/abm_example.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.engine.simulation_engine import FinancialEcosystem, SimulationConfig, ShockType
from app.engine.initial_state_loader import load_ecosystem_from_data
from app.engine.visualization import NetworkVisualizer, prepare_dashboard_data
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_simulation():
    """
    Example 1: Basic simulation with synthetic data
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Simulation with Synthetic Data")
    print("="*70 + "\n")
    
    # Create configuration
    config = SimulationConfig(
        max_timesteps=20,
        enable_shocks=True,
        shock_probability=0.1,
        random_seed=42
    )
    
    # Create ecosystem with synthetic data
    from app.engine.simulation_engine import SimulationFactory
    ecosystem = SimulationFactory.create_default_scenario(config)
    
    print(f"✓ Ecosystem initialized")
    print(f"  Banks: {len([a for a in ecosystem.agents.values() if a.agent_type.value == 'bank'])}")
    print(f"  Sectors: {len([a for a in ecosystem.agents.values() if a.agent_type.value == 'sector'])}")
    print(f"  Network edges: {ecosystem.network.number_of_edges()}")
    
    # Run simulation
    print(f"\n▶ Running simulation for 10 steps...")
    snapshots = ecosystem.run(steps=10)
    
    print(f"✓ Simulation completed")
    print(f"  Final timestep: {ecosystem.timestep}")
    print(f"  Bank survival rate: {ecosystem.global_state['survival_rate']:.1%}")
    print(f"  Average CRAR: {ecosystem.global_state['avg_crar']:.2f}%")
    print(f"  Total defaults: {ecosystem.global_state['total_defaults']}")
    
    # Visualization
    if snapshots:
        latest = snapshots[-1].to_dict()
        d3_data = NetworkVisualizer.convert_to_d3(latest)
        
        print(f"\n✓ Visualization data prepared")
        print(f"  Nodes: {len(d3_data['nodes'])}")
        print(f"  Links: {len(d3_data['links'])}")
        
        # Save to JSON
        output_file = Path("simulation_output_basic.json")
        with open(output_file, 'w') as f:
            json.dump(d3_data, f, indent=2)
        print(f"  Saved to: {output_file}")


def example_with_real_data():
    """
    Example 2: Load ecosystem from CSV data
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Simulation with Real RBI Data")
    print("="*70 + "\n")
    
    # Check if data directory exists
    data_dir = Path("backend/ccp_ml/data")
    if not data_dir.exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Skipping this example. Run with synthetic data instead.")
        return
    
    # Load ecosystem
    print(f"Loading data from: {data_dir}")
    ecosystem = load_ecosystem_from_data(
        str(data_dir),
        max_timesteps=50,
        enable_shocks=True,
        random_seed=42
    )
    
    print(f"\n✓ Ecosystem loaded from real data")
    network_stats = ecosystem.get_network_stats()
    print(f"  Banks: {network_stats['num_banks']}")
    print(f"  Sectors: {network_stats['num_sectors']}")
    print(f"  Total exposures: {network_stats['num_edges']}")
    
    # Run simulation
    print(f"\n▶ Running simulation for 15 steps...")
    snapshots = ecosystem.run(steps=15)
    
    print(f"\n✓ Simulation completed")
    print(f"  Final survival rate: {ecosystem.global_state['survival_rate']:.1%}")
    print(f"  System NPA: {ecosystem.global_state.get('system_npa', 0):.2f}%")


def example_with_shocks():
    """
    Example 3: Apply manual shocks and observe cascade
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Manual Shock Application and Contagion")
    print("="*70 + "\n")
    
    # Create ecosystem
    config = SimulationConfig(
        max_timesteps=30,
        enable_shocks=False,  # Disable random shocks
        random_seed=42
    )
    
    from app.engine.simulation_engine import SimulationFactory
    ecosystem = SimulationFactory.create_default_scenario(config)
    
    print(f"✓ Ecosystem initialized (shocks disabled)")
    
    # Run normally for 5 steps
    print(f"\n▶ Running 5 normal steps...")
    ecosystem.run(steps=5)
    
    initial_survival = ecosystem.global_state['survival_rate']
    print(f"  Initial survival rate: {initial_survival:.1%}")
    
    # Apply shock to Real Estate sector
    print(f"\n💥 Applying SECTOR_CRISIS shock to Real Estate...")
    shock_event = ecosystem.apply_shock(
        shock_type=ShockType.SECTOR_CRISIS,
        target="SECTOR_REAL_ESTATE",
        magnitude=-0.4  # 40% crash
    )
    
    print(f"  Shock applied: {shock_event.get('new_health', 'N/A')}")
    
    # Run for 10 more steps to observe contagion
    print(f"\n▶ Running 10 more steps to observe contagion...")
    ecosystem.run(steps=10)
    
    final_survival = ecosystem.global_state['survival_rate']
    print(f"\n✓ Post-shock analysis:")
    print(f"  Initial survival: {initial_survival:.1%}")
    print(f"  Final survival: {final_survival:.1%}")
    print(f"  Impact: {(initial_survival - final_survival):.1%} failure rate increase")
    print(f"  Total defaults: {ecosystem.global_state['total_defaults']}")


def example_policy_intervention():
    """
    Example 4: CCP policy intervention
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CCP Policy Intervention")
    print("="*70 + "\n")
    
    # Create ecosystem
    config = SimulationConfig(max_timesteps=50, enable_shocks=True, random_seed=42)
    from app.engine.simulation_engine import SimulationFactory
    ecosystem = SimulationFactory.create_default_scenario(config)
    
    # Get CCP agent
    ccp = ecosystem.get_agent("CCP_MAIN")
    
    print(f"✓ Ecosystem initialized")
    print(f"  CCP: {ccp.agent_id}")
    print(f"  Initial margin requirement: {ccp.initial_margin_requirement}%")
    
    # Add a policy rule: If system NPA > 5%, increase haircuts
    print(f"\n➕ Adding policy rule: IF system_npa > 5% THEN increase haircuts")
    
    def stress_condition(agent):
        return ecosystem.global_state.get('system_npa', 0) > 5.0
    
    def stress_action():
        ccp.haircut_rate = min(0.5, ccp.haircut_rate + 0.05)
        logger.info(f"CCP increased haircut rate to {ccp.haircut_rate:.2%}")
        return {'haircut_rate': ccp.haircut_rate}
    
    ccp.add_policy_rule({
        'name': 'Stress Haircut Rule',
        'condition': stress_condition,
        'action': stress_action
    })
    
    print(f"  Policy rule added")
    
    # Run simulation
    print(f"\n▶ Running simulation for 20 steps...")
    snapshots = ecosystem.run(steps=20)
    
    print(f"\n✓ Simulation completed with policy intervention")
    print(f"  Final CCP haircut rate: {ccp.haircut_rate:.2%}")
    print(f"  System NPA: {ecosystem.global_state.get('system_npa', 0):.2f}%")


def example_visualization_dashboard():
    """
    Example 5: Generate complete dashboard data
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Dashboard Data Generation")
    print("="*70 + "\n")
    
    # Create and run simulation
    config = SimulationConfig(max_timesteps=30, enable_shocks=True, random_seed=42)
    from app.engine.simulation_engine import SimulationFactory
    ecosystem = SimulationFactory.create_default_scenario(config)
    
    print(f"✓ Running simulation...")
    ecosystem.run(steps=20)
    
    # Generate dashboard data
    print(f"\n▶ Generating dashboard data...")
    history_dicts = [s.to_dict() for s in ecosystem.history]
    dashboard_data = prepare_dashboard_data(history_dicts)
    
    print(f"\n✓ Dashboard data prepared:")
    print(f"  D3 network nodes: {len(dashboard_data['d3_network']['nodes'])}")
    print(f"  Time series metrics: {len([k for k in dashboard_data.keys() if '_ts' in k])}")
    print(f"  Critical nodes: {len(dashboard_data['critical_nodes'])}")
    print(f"  Network density: {dashboard_data['network_metrics'].get('density', 0):.3f}")
    
    # Save dashboard data
    output_file = Path("dashboard_data.json")
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"  Saved to: {output_file}")
    
    # Print critical nodes
    if dashboard_data['critical_nodes']:
        print(f"\n⚠ Critical nodes detected:")
        for node_id in dashboard_data['critical_nodes']:
            print(f"    - {node_id}")


def example_time_series_analysis():
    """
    Example 6: Time series extraction for specific metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Time Series Analysis")
    print("="*70 + "\n")
    
    # Create and run simulation
    config = SimulationConfig(max_timesteps=50, enable_shocks=True, random_seed=42)
    from app.engine.simulation_engine import SimulationFactory
    ecosystem = SimulationFactory.create_default_scenario(config)
    
    print(f"Running simulation...")
    ecosystem.run(steps=25)
    
    # Extract time series
    history_dicts = [s.to_dict() for s in ecosystem.history]
    
    survival_ts = NetworkVisualizer.create_time_series(history_dicts, 'survival_rate')
    crar_ts = NetworkVisualizer.create_time_series(history_dicts, 'avg_crar')
    
    print(f"\n✓ Time series extracted:")
    print(f"  Survival Rate:")
    print(f"    Initial: {survival_ts['values'][0]:.1%}")
    print(f"    Final: {survival_ts['values'][-1]:.1%}")
    print(f"  Average CRAR:")
    print(f"    Initial: {crar_ts['values'][0]:.2f}%")
    print(f"    Final: {crar_ts['values'][-1]:.2f}%")
    
    # Track specific bank
    bank_id = "BANK_1"
    bank_crar_ts = NetworkVisualizer.create_agent_time_series(history_dicts, bank_id, 'crar')
    
    print(f"\n  {bank_id} CRAR over time:")
    print(f"    Initial: {bank_crar_ts['values'][0]:.2f}%")
    print(f"    Final: {bank_crar_ts['values'][-1]:.2f}%")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUDRA Agent-Based Model - Example Suite")
    print("="*70)
    
    try:
        example_basic_simulation()
        example_with_real_data()
        example_with_shocks()
        example_policy_intervention()
        example_visualization_dashboard()
        example_time_series_analysis()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70 + "\n")
        
        print("Next steps:")
        print("  1. Check the generated JSON files for visualization data")
        print("  2. Start the FastAPI server: uvicorn app.main:app --reload")
        print("  3. Access API docs: http://localhost:8000/docs")
        print("  4. Try the /abm endpoints for interactive simulation")
        print()
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        sys.exit(1)
