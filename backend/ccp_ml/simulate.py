#!/usr/bin/env python3
"""
CCP Simulation Runner

Interactive script to run CCP risk simulation with network visualization.
Exports results in formats usable for further analysis and visualization.
"""

import json
import logging
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_simulation():
    """Run complete CCP simulation with network graph export."""
    
    from ccp_ml import (
        DataLoader, FeatureEngineer, NetworkBuilder, 
        SpectralAnalyzer, CCPRiskModel, CCPEngine,
        select_features
    )
    
    print("\n" + "="*70)
    print("🏦 CCP RISK SIMULATION - RUDRA Financial Infrastructure")
    print("="*70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'simulation_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================
    # STEP 1: Load Data
    # ========================
    print("\n📊 STEP 1: Loading all datasets...")
    loader = DataLoader()
    data = loader.load_all()
    
    print(f"   ✓ {len(data.banks)} banks loaded")
    print(f"   ✓ Years: {min(data.years)}-{max(data.years)}")
    print(f"   ✓ ML ready: {data.ml_ready.shape if data.ml_ready is not None else 'N/A'}")
    
    # ========================
    # STEP 2: Engineer Features
    # ========================
    print("\n🔧 STEP 2: Engineering features...")
    engineer = FeatureEngineer(normalize=True)
    features = engineer.create_features(data)
    
    print(f"   ✓ {len(features.columns)} features created")
    print(f"   ✓ {len(features)} samples")
    
    # ========================
    # STEP 3: Build Network
    # ========================
    print("\n🔗 STEP 3: Building interdependence network...")
    # Use lower threshold to get more edges
    builder = NetworkBuilder(
        sector_weight=0.4,
        liquidity_weight=0.4, 
        market_weight=0.2,
        edge_threshold=0.05  # Very low to ensure edges
    )
    graph = builder.build_network(data)
    
    print(f"   ✓ {graph.number_of_nodes()} nodes")
    print(f"   ✓ {len(builder.edges)} edges")
    
    # Compute network metrics
    network_metrics = builder.compute_network_metrics()
    if not network_metrics.empty:
        top_central = network_metrics.nlargest(5, 'pagerank')
        print(f"\n   Top 5 systemically important banks (by PageRank):")
        for _, row in top_central.iterrows():
            print(f"      - {row['bank_name']}: {row['pagerank']:.4f}")
    
    # ========================
    # STEP 4: Spectral Analysis
    # ========================
    print("\n📐 STEP 4: Spectral fragility analysis...")
    spectral = SpectralAnalyzer()
    spectral_metrics = spectral.analyze(network_builder=builder)
    
    print(f"   ✓ Spectral radius (ρ): {spectral_metrics.spectral_radius:.4f}")
    print(f"   ✓ Fiedler value (λ2): {spectral_metrics.fiedler_value:.4f}")
    print(f"   ✓ Amplification risk: {spectral_metrics.amplification_risk}")
    print(f"   ✓ Fragmentation risk: {spectral_metrics.fragmentation_risk}")
    print(f"   ✓ Contagion index: {spectral.compute_contagion_index():.4f}")
    
    # ========================
    # STEP 5: Train Risk Model
    # ========================
    print("\n🤖 STEP 5: Training risk model...")
    X, y = select_features(features)
    
    # Check if we have both classes
    unique_labels = y.unique()
    print(f"   Target distribution: {dict(y.value_counts())}")
    
    model = CCPRiskModel()
    
    if len(unique_labels) > 1:
        train_metrics = model.fit(X, y)
        print(f"   ✓ Training AUC: {train_metrics['train_auc']:.4f}")
        print(f"   ✓ Positive rate: {train_metrics['positive_rate']:.2%}")
        
        # Get feature importance
        importance = model.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   Top 5 risk factors:")
        for feat, imp in top_features:
            print(f"      - {feat}: {imp:.4f}")
        
        # Add predictions to features
        features['default_probability'] = model.predict_proba(X)[:, 1]
    else:
        print(f"   ⚠️ Only one class in target - using synthetic risk scores")
        # Create synthetic risk based on features
        features['default_probability'] = (
            features.get('stress_level', 0.1) * 0.3 +
            features.get('leverage', 0.5) * 0.3 +
            (1 - features.get('capital_ratio', 0.1)) * 0.4
        ).clip(0, 1)
    
    # ========================
    # STEP 6: CCP Engine
    # ========================
    print("\n⚙️ STEP 6: Running CCP engine...")
    engine = CCPEngine(
        risk_model=model,
        network_builder=builder,
        spectral_analyzer=spectral
    )
    
    results = engine.run_full_analysis(features, train=False)
    
    print(f"   ✓ Margins calculated for {len(engine.margin_requirements)} participants")
    print(f"   ✓ Default fund: {results['default_fund']['total_fund']:,.0f}")
    print(f"   ✓ Policies generated: {len(results['policies'])}")
    
    # ========================
    # STEP 7: Export Results
    # ========================
    print("\n💾 STEP 7: Exporting results...")
    
    # Export network graph
    network_file = os.path.join(output_dir, 'network_graph.json')
    network_data = builder.export_to_dict()
    network_data['metrics'] = network_metrics.to_dict('records') if not network_metrics.empty else []
    with open(network_file, 'w') as f:
        json.dump(network_data, f, indent=2)
    print(f"   ✓ Network graph: {network_file}")
    
    # Export risk scores
    risk_file = os.path.join(output_dir, 'risk_scores.json')
    risk_data = features[['bank_name', 'default_probability']].to_dict('records') if 'bank_name' in features.columns else []
    with open(risk_file, 'w') as f:
        json.dump(risk_data, f, indent=2)
    print(f"   ✓ Risk scores: {risk_file}")
    
    # Export CCP results
    ccp_file = os.path.join(output_dir, 'ccp_results.json')
    engine.export_results(ccp_file)
    print(f"   ✓ CCP results: {ccp_file}")
    
    # Export summary
    summary_file = os.path.join(output_dir, 'simulation_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_banks': len(data.banks),
        'years_covered': data.years,
        'n_features': len(features.columns),
        'n_edges': len(builder.edges),
        'spectral_metrics': {
            'spectral_radius': spectral_metrics.spectral_radius,
            'fiedler_value': spectral_metrics.fiedler_value,
            'contagion_index': spectral.compute_contagion_index(),
            'amplification_risk': spectral_metrics.amplification_risk,
            'fragmentation_risk': spectral_metrics.fragmentation_risk
        },
        'ccp_metrics': {
            'total_margin': sum(m.total_margin for m in engine.margin_requirements),
            'default_fund_size': results['default_fund']['total_fund'],
            'cover_n': results['default_fund']['cover_n']
        },
        'policies': results['policies']
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✓ Summary: {summary_file}")
    
    # ========================
    # FINAL SUMMARY
    # ========================
    print("\n" + "="*70)
    print("🎉 SIMULATION COMPLETE")
    print("="*70)
    
    print(f"""
📈 KEY FINDINGS:

   Network Structure:
   - {graph.number_of_nodes()} banks connected by {len(builder.edges)} relationships
   - Spectral radius: {spectral_metrics.spectral_radius:.4f} → {spectral_metrics.amplification_risk} amplification risk
   - Fiedler value: {spectral_metrics.fiedler_value:.4f} → {spectral_metrics.fragmentation_risk} fragmentation risk

   Risk Distribution:
   - High risk banks: {len([1 for p in features.get('default_probability', []) if p > 0.5])}
   - Medium risk: {len([1 for p in features.get('default_probability', []) if 0.3 <= p <= 0.5])}
   - Low risk: {len([1 for p in features.get('default_probability', []) if p < 0.3])}

   CCP Requirements:
   - Total margin requirement: {sum(m.total_margin for m in engine.margin_requirements):.4f}
   - Default fund size: ₹{results['default_fund']['total_fund']:,.0f}
   
   Output files saved to: {output_dir}
""")
    
    return {
        'data': data,
        'features': features,
        'network': builder,
        'spectral': spectral_metrics,
        'engine': engine,
        'results': results
    }


if __name__ == '__main__':
    run_simulation()
