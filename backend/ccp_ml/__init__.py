"""
CCP ML Package

Central Counterparty (CCP) centric risk modeling for financial infrastructure.

Implements ML_Flow.md specification with 4 layers:
1. Participant Risk Estimation (ML)
2. Interdependence Network Construction
3. Systemic Fragility Quantification (Spectral Analysis)
4. CCP Loss Absorption & Policy Response

Usage:
    from ccp_ml import CCPPipeline
    
    pipeline = CCPPipeline()
    results = pipeline.run()
"""

import logging

# Data Loading
from .data_loader import DataLoader, DatasetContainer, load_data

# Feature Engineering
from .feature_engineering import FeatureEngineer

# Network Analysis
from .network_builder import NetworkBuilder, NetworkEdge

# Spectral Analysis
from .spectral_analyzer import SpectralAnalyzer, SpectralMetrics

# Risk Modeling
from .risk_model import CCPRiskModel, ModelType, select_features

# CCP Engine
from .ccp_engine import (
    CCPEngine, 
    MarginRequirement, 
    DefaultFundAllocation, 
    PolicyRecommendation
)

__version__ = '1.0.0'

logger = logging.getLogger(__name__)


class CCPPipeline:
    """
    Complete CCP risk analysis pipeline.
    
    Orchestrates the full ML_Flow.md specification:
    1. Load and validate data
    2. Engineer features
    3. Build network
    4. Analyze spectral properties
    5. Train risk model
    6. Generate CCP outputs
    
    Example:
        >>> pipeline = CCPPipeline()
        >>> results = pipeline.run()
        >>> print(results['policies'])
    """
    
    def __init__(self, data_dir: str = None, config: dict = None):
        """
        Initialize the CCP pipeline.
        
        Args:
            data_dir: Path to data directory (uses default if None)
            config: Optional configuration overrides
        """
        self.config = config or {}
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = FeatureEngineer(
            normalize=self.config.get('normalize', True)
        )
        self.network_builder = NetworkBuilder(
            sector_weight=self.config.get('sector_weight', 0.3),
            liquidity_weight=self.config.get('liquidity_weight', 0.4),
            market_weight=self.config.get('market_weight', 0.3)
        )
        self.spectral_analyzer = SpectralAnalyzer()
        self.risk_model = CCPRiskModel(
            model_type=self.config.get('model_type', ModelType.GRADIENT_BOOSTING)
        )
        self.ccp_engine = CCPEngine(
            risk_model=self.risk_model,
            network_builder=self.network_builder,
            spectral_analyzer=self.spectral_analyzer
        )
        
        # State
        self.data = None
        self.features = None
        self.results = None
    
    def run(
        self, 
        target_year: int = None,
        train: bool = True,
        include_market_data: bool = False
    ) -> dict:
        """
        Run the complete CCP risk analysis pipeline.
        
        Args:
            target_year: Specific year to analyze (None for all)
            train: Whether to train the risk model
            include_market_data: Whether to fetch market data
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("="*60)
        logger.info("STARTING CCP RISK ANALYSIS PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load data
        logger.info("\n📊 Step 1/6: Loading data...")
        self.data = self.data_loader.load_all()
        logger.info(f"  Loaded {len(self.data.banks)} banks, years: {self.data.years}")
        
        # Step 2: Load market data if requested
        market_data = None
        if include_market_data:
            logger.info("\n📈 Step 2/6: Loading market data...")
            market_data = self.data_loader.load_market_data()
            if market_data is not None:
                logger.info(f"  Loaded market data for {len(market_data.get('returns', []))} tickers")
        else:
            logger.info("\n📈 Step 2/6: Skipping market data (disabled)")
        
        # Step 3: Engineer features
        logger.info("\n🔧 Step 3/6: Engineering features...")
        self.features = self.feature_engineer.create_features(self.data, target_year)
        logger.info(f"  Created {len(self.features.columns)} features for {len(self.features)} samples")
        
        # Step 4: Build network
        logger.info("\n🔗 Step 4/6: Building interdependence network...")
        self.network_builder.build_network(self.data, target_year, market_data)
        network_metrics = self.network_builder.compute_network_metrics()
        
        if not network_metrics.empty:
            # Merge network metrics into features
            if 'bank_name' in self.features.columns:
                self.features = self.features.merge(
                    network_metrics, 
                    on='bank_name', 
                    how='left',
                    suffixes=('', '_network')
                )
            logger.info(f"  Built network with {len(self.network_builder.edges)} edges")
        
        # Step 5: Spectral analysis
        logger.info("\n📐 Step 5/6: Performing spectral analysis...")
        spectral_results = self.spectral_analyzer.analyze(
            network_builder=self.network_builder
        )
        logger.info(f"  Spectral radius: {spectral_results.spectral_radius:.4f}")
        logger.info(f"  Fiedler value: {spectral_results.fiedler_value:.4f}")
        
        # Step 6: Run CCP engine
        logger.info("\n⚙️ Step 6/6: Running CCP engine...")
        self.results = self.ccp_engine.run_full_analysis(
            self.features, 
            train=train,
            year=target_year
        )
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return self.results
    
    def get_participant_report(self, bank_name: str) -> dict:
        """Get detailed report for a specific participant."""
        return self.ccp_engine.get_participant_summary(bank_name)
    
    def export_results(self, output_path: str) -> None:
        """Export results to JSON file."""
        self.ccp_engine.export_results(output_path)


# Convenience function for quick analysis
def run_analysis(data_dir: str = None, **kwargs) -> dict:
    """
    Quick function to run CCP analysis.
    
    Args:
        data_dir: Path to data directory
        **kwargs: Additional arguments passed to CCPPipeline.run()
        
    Returns:
        Analysis results dictionary
    """
    pipeline = CCPPipeline(data_dir=data_dir)
    return pipeline.run(**kwargs)


__all__ = [
    # Data
    'DataLoader', 'DatasetContainer', 'load_data',
    # Features
    'FeatureEngineer',
    # Network
    'NetworkBuilder', 'NetworkEdge',
    # Spectral
    'SpectralAnalyzer', 'SpectralMetrics',
    # Model
    'CCPRiskModel', 'ModelType', 'select_features',
    # Engine
    'CCPEngine', 'MarginRequirement', 'DefaultFundAllocation', 'PolicyRecommendation',
    # Pipeline
    'CCPPipeline', 'run_analysis'
]
