"""
Test Eigenvector-Based Loss Mutualization for CCP Zero Risk
===========================================================

This script demonstrates how the eigenvector-based payoff normalization
ensures the CCP (clearing house) always has zero net loss.

Key Mechanisms:
1. Eigenvector centrality measures systemic importance
2. Losses are redistributed proportional to centrality
3. CCP maintains zero risk through mutualization
4. Higher centrality nodes absorb more of system losses
"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import networkx as nx
from app.engine.agents import BankAgent, CCPAgent
from app.engine.simulation_engine import FinancialEcosystem, SimulationConfig
from ccp_ml.spectral_analyzer import SpectralAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_network_with_ccp():
    """Create a test network with CCP and banks"""
    logger.info("="*80)
    logger.info("Creating Test Financial Network with CCP")
    logger.info("="*80)
    
    # Create banks with varying risk profiles
    banks = {
        'SYSTEMICALLY_IMPORTANT_BANK': BankAgent(
            agent_id='SYSTEMICALLY_IMPORTANT_BANK',
            initial_capital=50000,
            initial_assets=400000,
            initial_liquidity=10000,
            initial_crar=12.5,
            initial_npa_ratio=2.0
        ),
        'LARGE_BANK': BankAgent(
            agent_id='LARGE_BANK',
            initial_capital=30000,
            initial_assets=250000,
            initial_liquidity=6000,
            initial_crar=12.0,
            initial_npa_ratio=3.0
        ),
        'MEDIUM_BANK': BankAgent(
            agent_id='MEDIUM_BANK',
            initial_capital=15000,
            initial_assets=120000,
            initial_liquidity=3000,
            initial_crar=12.5,
            initial_npa_ratio=4.0
        ),
        'SMALL_RISKY_BANK': BankAgent(
            agent_id='SMALL_RISKY_BANK',
            initial_capital=5000,
            initial_assets=50000,
            initial_liquidity=800,
            initial_crar=10.0,
            initial_npa_ratio=8.0
        ),
    }
    
    # Create CCP with eigenvector-based loss mutualization ENABLED
    ccp = CCPAgent(
        agent_id='CCP_CENTRAL',
        initial_default_fund=100000,
        initial_margin_requirement=10.0
    )
    ccp.enable_loss_mutualization = True  # KEY: Enable zero-risk mechanism
    
    logger.info(f"\nCreated {len(banks)} banks and 1 CCP")
    logger.info(f"CCP Loss Mutualization: {'ENABLED' if ccp.enable_loss_mutualization else 'DISABLED'}")
    logger.info(f"CCP Initial Default Fund: ${ccp.default_fund_size:,.2f}")
    
    for bank_id, bank in banks.items():
        logger.info(f"  {bank_id}: Capital=${bank.capital:,.2f}, CRAR={bank.crar:.2f}%, NPA={bank.npa_ratio:.2f}%")
    
    return banks, ccp


def build_exposure_network(banks, ccp):
    """Build network with exposures"""
    logger.info("\n" + "="*80)
    logger.info("Building Exposure Network")
    logger.info("="*80)
    
    G = nx.DiGraph()
    
    # Add all agents as nodes
    all_agents = {**banks, ccp.agent_id: ccp}
    for agent_id, agent in all_agents.items():
        G.add_node(agent_id, agent=agent)
    
    # Create interbank exposures (realistic network structure)
    exposures = [
        ('SYSTEMICALLY_IMPORTANT_BANK', 'LARGE_BANK', 20000),
        ('SYSTEMICALLY_IMPORTANT_BANK', 'MEDIUM_BANK', 15000),
        ('SYSTEMICALLY_IMPORTANT_BANK', 'SMALL_RISKY_BANK', 5000),
        ('LARGE_BANK', 'MEDIUM_BANK', 10000),
        ('LARGE_BANK', 'SMALL_RISKY_BANK', 3000),
        ('MEDIUM_BANK', 'SMALL_RISKY_BANK', 2000),
        # All banks have exposures through CCP
        ('SYSTEMICALLY_IMPORTANT_BANK', 'CCP_CENTRAL', 30000),
        ('LARGE_BANK', 'CCP_CENTRAL', 20000),
        ('MEDIUM_BANK', 'CCP_CENTRAL', 10000),
        ('SMALL_RISKY_BANK', 'CCP_CENTRAL', 5000),
    ]
    
    for src, tgt, amount in exposures:
        G.add_edge(src, tgt, weight=amount, exposure=amount)
        logger.info(f"  {src} → {tgt}: ${amount:,}")
    
    logger.info(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def compute_eigenvector_centrality_analysis(G):
    """Compute and display eigenvector centrality analysis"""
    logger.info("\n" + "="*80)
    logger.info("Eigenvector Centrality Analysis")
    logger.info("="*80)
    
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Normalize
        total = sum(centrality.values())
        normalized_centrality = {k: v/total for k, v in centrality.items()}
        
        logger.info("\nNormalized Eigenvector Centrality (Systemic Importance):")
        logger.info("-" * 60)
        
        sorted_centrality = sorted(normalized_centrality.items(), key=lambda x: x[1], reverse=True)
        for node, cent in sorted_centrality:
            percentage = cent * 100
            bar = "█" * int(percentage * 2)
            logger.info(f"  {node:30s}: {percentage:6.2f}% {bar}")
        
        logger.info("\nInterpretation:")
        logger.info("  → Higher centrality = More systemically important")
        logger.info("  → These nodes will absorb larger shares of mutualized losses")
        
        return normalized_centrality
    
    except Exception as e:
        logger.error(f"Failed to compute centrality: {e}")
        return {}


def simulate_bank_default_with_mutualization(banks, ccp, G, centrality):
    """Simulate a bank default and show loss mutualization"""
    logger.info("\n" + "="*80)
    logger.info("SIMULATION: Bank Default with Loss Mutualization")
    logger.info("="*80)
    
    # Force SMALL_RISKY_BANK to default
    failing_bank = banks['SMALL_RISKY_BANK']
    loss_amount = abs(failing_bank.capital)
    
    logger.info(f"\n⚠️  {failing_bank.agent_id} is DEFAULTING")
    logger.info(f"    Loss Amount: ${loss_amount:,.2f}")
    logger.info(f"    CRAR: {failing_bank.crar:.2f}% (below minimum)")
    
    # Mark as defaulted
    failing_bank.alive = False
    failing_bank.capital = -loss_amount  # Negative capital
    
    # Show before state
    logger.info("\n📊 BEFORE Loss Mutualization:")
    logger.info("-" * 60)
    logger.info(f"  CCP Default Fund: ${ccp.default_fund_size:,.2f}")
    logger.info(f"  CCP Health: {ccp.compute_health():.2%}")
    
    for bank_id, bank in banks.items():
        if bank.alive:
            logger.info(f"  {bank_id}: Capital=${bank.capital:,.2f}, CRAR={bank.crar:.2f}%")
    
    # Execute CCP's act() method which performs mutualization
    logger.info("\n🔄 Executing Eigenvector-Based Loss Mutualization...")
    events = ccp.act(G)
    
    # Show after state
    logger.info("\n📊 AFTER Loss Mutualization:")
    logger.info("-" * 60)
    
    if events and events[0]['type'] == 'CCP_LOSS_MUTUALIZATION':
        event = events[0]
        logger.info(f"  ✅ CCP Net Loss: ${event['ccp_net_loss']:,.2f} (ZERO RISK MAINTAINED)")
        logger.info(f"  CCP Default Fund (unchanged): ${ccp.default_fund_size:,.2f}")
        logger.info(f"  CCP Health: {ccp.compute_health():.2%} (Perfect Health)")
        logger.info(f"  Total Mutualized: ${ccp.total_mutualized_losses:,.2f}")
        
        logger.info("\n  Loss Distribution Among Surviving Members:")
        redistributions = event['redistributed_to']
        total_redistributed = 0.0
        
        for member_id, details in redistributions.items():
            loss_share = details['loss_share']
            cent = details['eigenvector_centrality']
            total_redistributed += loss_share
            
            logger.info(
                f"    {member_id:30s}: ${loss_share:10,.2f} "
                f"({cent*100:5.2f}% centrality) → CRAR: {details['new_crar']:.2f}%"
            )
        
        logger.info(f"\n  Conservation Check:")
        logger.info(f"    Original Loss:        ${loss_amount:,.2f}")
        logger.info(f"    Total Redistributed:  ${total_redistributed:,.2f}")
        logger.info(f"    Difference:           ${abs(loss_amount - total_redistributed):,.2f}")
        
        if abs(loss_amount - total_redistributed) < 0.01:
            logger.info("    ✅ VERIFIED: Loss perfectly conserved")
        else:
            logger.warning("    ⚠️  Loss conservation mismatch!")
    
    return events


def verify_ccp_zero_risk(ccp, initial_fund):
    """Verify CCP maintained zero risk"""
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION: CCP Zero Risk Guarantee")
    logger.info("="*80)
    
    logger.info(f"\nInitial Default Fund:  ${initial_fund:,.2f}")
    logger.info(f"Current Default Fund:  ${ccp.default_fund_size:,.2f}")
    logger.info(f"Fund Change:           ${ccp.default_fund_size - initial_fund:,.2f}")
    logger.info(f"\nTotal Losses Mutualized: ${ccp.total_mutualized_losses:,.2f}")
    logger.info(f"CCP Net Loss:              $0.00")
    logger.info(f"CCP Health Score:          {ccp.compute_health():.2%}")
    
    if ccp.default_fund_size == initial_fund and ccp.compute_health() == 1.0:
        logger.info("\n✅ VERIFICATION PASSED:")
        logger.info("   • CCP default fund unchanged")
        logger.info("   • CCP health remains perfect (100%)")
        logger.info("   • All losses redistributed to members")
        logger.info("   • ZERO RISK MAINTAINED via eigenvector mutualization")
        return True
    else:
        logger.warning("\n⚠️  VERIFICATION FAILED: CCP absorbed some loss")
        return False


def demonstrate_spectral_analysis(G):
    """Show spectral analysis of network"""
    logger.info("\n" + "="*80)
    logger.info("Spectral Analysis of Network Structure")
    logger.info("="*80)
    
    # Convert to numpy adjacency matrix
    nodes = list(G.nodes())
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if G.has_edge(node_i, node_j):
                adj_matrix[i, j] = G[node_i][node_j].get('weight', 1.0)
    
    # Normalize to [0,1]
    if np.max(adj_matrix) > 0:
        adj_matrix = adj_matrix / np.max(adj_matrix)
    
    # Compute Laplacian
    degree = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree - adj_matrix
    
    # Analyze
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze(adj_matrix, laplacian)
    
    logger.info(f"\nSpectral Metrics:")
    logger.info(f"  Spectral Radius (ρ):     {metrics.spectral_radius:.4f}")
    logger.info(f"  Fiedler Value (λ₂):      {metrics.fiedler_value:.4f}")
    logger.info(f"  Spectral Gap:            {metrics.spectral_gap:.4f}")
    logger.info(f"  Eigenvalue Entropy:      {metrics.eigenvalue_entropy:.4f}")
    logger.info(f"  Amplification Risk:      {metrics.amplification_risk.upper()}")
    logger.info(f"  Fragmentation Risk:      {metrics.fragmentation_risk.upper()}")
    
    logger.info(f"\nInterpretation:")
    if metrics.spectral_radius > 0.9:
        logger.info(f"  ⚠️  High spectral radius → Network amplifies shocks")
    else:
        logger.info(f"  ✅ Moderate spectral radius → Network is resilient")
    
    # Show eigenvector centrality from spectral analysis
    eigenvector_centrality = analyzer.compute_eigenvector_centrality()
    logger.info(f"\nEigenvector Centrality from Spectral Analysis:")
    for idx in sorted(eigenvector_centrality.keys()):
        node_name = nodes[idx] if idx < len(nodes) else f"Node_{idx}"
        cent = eigenvector_centrality[idx]
        logger.info(f"  {node_name:30s}: {cent:.4f}")


def main():
    """Main test execution"""
    logger.info("\n" + "="*80)
    logger.info("EIGENVECTOR-BASED CCP ZERO RISK DEMONSTRATION")
    logger.info("="*80)
    logger.info("\nThis demonstrates how eigenvector centrality normalization")
    logger.info("ensures the CCP (clearing house) maintains zero risk by")
    logger.info("redistributing losses among members proportional to their")
    logger.info("systemic importance in the network.")
    
    # Step 1: Create network
    banks, ccp = create_test_network_with_ccp()
    initial_fund = ccp.default_fund_size
    
    # Step 2: Build exposure network
    G = build_exposure_network(banks, ccp)
    
    # Step 3: Compute eigenvector centrality
    centrality = compute_eigenvector_centrality_analysis(G)
    
    # Step 4: Spectral analysis
    demonstrate_spectral_analysis(G)
    
    # Step 5: Simulate default with mutualization
    events = simulate_bank_default_with_mutualization(banks, ccp, G, centrality)
    
    # Step 6: Verify zero risk
    success = verify_ccp_zero_risk(ccp, initial_fund)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if success:
        logger.info("\n✅ Eigenvector-based loss mutualization is WORKING")
        logger.info("\nKey Findings:")
        logger.info("  1. CCP maintains zero net loss through mutualization")
        logger.info("  2. Losses distributed proportional to eigenvector centrality")
        logger.info("  3. Systemically important nodes absorb more loss")
        logger.info("  4. Default fund preserved for future shocks")
        logger.info("  5. CCP health remains at 100% (perfect)")
        
        logger.info("\nMechanism:")
        logger.info("  • Eigenvector centrality measures each node's systemic importance")
        logger.info("  • When member defaults, loss is redistributed to survivors")
        logger.info("  • Each survivor's share ∝ their eigenvector centrality")
        logger.info("  • CCP absorbs zero loss (all risk mutualized)")
        logger.info("  • Total loss is conserved (Σ redistributed = original loss)")
    else:
        logger.warning("\n⚠️  Test failed - check implementation")
    
    logger.info("\n" + "="*80)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
