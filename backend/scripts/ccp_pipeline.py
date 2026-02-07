#!/usr/bin/env python3
"""
End-to-End CCP-Centric Risk Modeling Pipeline

Implements the complete workflow from ML_Flow.md:
1. Load and integrate all 7 datasets
2. Train ML model for default prediction
3. Build composite multi-channel network
4. Perform spectral analysis for systemic fragility
5. CCP loss absorption and policy response

Usage:
    python scripts/ccp_pipeline.py --train
    python scripts/ccp_pipeline.py --analyze
    python scripts/ccp_pipeline.py --full-pipeline --save-report
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from app.ml.data.integration_pipeline import DataIntegrationPipeline, IntegratedDataset
from app.ml.data.network_builder import CompositeNetworkBuilder
from app.ml.data.real_data_loader import RealDataLoader
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.analysis.spectral import SpectralAnalyzer, analyze_systemic_fragility
from app.ml.ccp.risk_manager import CCPRiskManager, RiskTier
from app.ml.registry.model_manager import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CCPPipeline:
    """
    End-to-end CCP risk modeling pipeline
    
    Follows ML_Flow.md architecture:
    - Layer 1: Participant Risk Estimation (ML)
    - Layer 2: Interdependence Network Construction
    - Layer 3: Systemic Fragility Quantification
    - Layer 4: CCP Loss Absorption & Policy Response
    """
    
    def __init__(
        self,
        data_dir: str = "app/ml/data",
        model_name: str = "ccp_default_predictor",
        save_artifacts: bool = True
    ):
        """
        Initialize CCP pipeline
        
        Args:
            data_dir: Directory containing data files
            model_name: Name for ML model
            save_artifacts: Whether to save intermediate artifacts
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.save_artifacts = save_artifacts
        
        # Initialize components
        self.data_pipeline = DataIntegrationPipeline(data_dir)
        self.network_builder = CompositeNetworkBuilder(data_dir)
        self.spectral_analyzer = SpectralAnalyzer()
        self.ccp_manager = CCPRiskManager()
        
        # State
        self.integrated_data = None
        self.trained_model = None
        self.network = None
        self.spectral_metrics = None
        self.risk_assessment = None
        
        logger.info("="*70)
        logger.info("CCP-CENTRIC RISK MODELING PIPELINE")
        logger.info("="*70)
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Model name: {model_name}")
        logger.info("")
    
    def run_full_pipeline(self):
        """Execute complete end-to-end pipeline"""
        logger.info("Starting full CCP pipeline execution...")
        logger.info("")
        
        # Layer 0: Data Integration
        logger.info("LAYER 0: DATA INTEGRATION")
        logger.info("-"*70)
        self.integrated_data = self.data_pipeline.load_all_datasets()
        print(self.data_pipeline.generate_summary(self.integrated_data))
        logger.info("")
        
        # Layer 1: ML-based Risk Estimation
        logger.info("LAYER 1: PARTICIPANT RISK ESTIMATION (ML)")
        logger.info("-"*70)
        self.trained_model = self.train_default_predictor()
        logger.info("")
        
        # Layer 2: Network Construction
        logger.info("LAYER 2: INTERDEPENDENCE NETWORK CONSTRUCTION")
        logger.info("-"*70)
        self.network = self.build_composite_network()
        logger.info("")
        
        # Layer 3: Spectral Analysis
        logger.info("LAYER 3: SYSTEMIC FRAGILITY QUANTIFICATION")
        logger.info("-"*70)
        self.spectral_metrics = self.analyze_systemic_fragility()
        logger.info("")
        
        # Layer 4: CCP Decision Making
        logger.info("LAYER 4: CCP LOSS ABSORPTION & POLICY RESPONSE")
        logger.info("-"*70)
        self.risk_assessment = self.assess_ccp_risk()
        logger.info("")
        
        logger.info("="*70)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*70)
        
        return self.risk_assessment
    
    def train_default_predictor(self):
        """Layer 1: Train ML model for default prediction"""
        logger.info("Training default predictor model...")
        
        # Create train/val/test splits
        train_df, val_df, test_df = self.data_pipeline.create_training_dataset(
            self.integrated_data,
            test_year='2025',
            validation_split=0.15
        )
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Load data using RealDataLoader
        loader = RealDataLoader()
        
        # Convert to features
        train_features, train_labels = self._df_to_features(train_df, loader)
        val_features, val_labels = self._df_to_features(val_df, loader)
        test_features, test_labels = self._df_to_features(test_df, loader)
        
        # Create datasets
        train_dataset = InstitutionDataset(train_features, train_labels)
        val_dataset = InstitutionDataset(val_features, val_labels)
        test_dataset = InstitutionDataset(test_features, test_labels)
        
        # Initialize model
        model = DefaultPredictorModel(
            input_dim=train_dataset.num_features,
            hidden_dims=[128, 64, 32]
        )
        
        # Train
        trainer = DefaultPredictorTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=16,
            learning_rate=0.001,
            epochs=100,
            early_stopping_patience=15
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        
        logger.info(f"Test Metrics:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Save model
        if self.save_artifacts:
            registry = ModelRegistry()
            registry.register_model(
                model=model,
                model_name=self.model_name,
                model_type="default_predictor",
                metrics=test_metrics,
                metadata={
                    'architecture': 'MLP',
                    'input_dim': train_dataset.num_features,
                    'hidden_dims': [128, 64, 32],
                    'training_samples': len(train_df),
                    'data_source': 'RBI Banks'
                }
            )
            logger.info(f"Model saved as '{self.model_name}'")
        
        return model
    
    def _df_to_features(self, df, loader):
        """Convert dataframe to feature objects"""
        # Use real data loader to convert
        # Simplified: extract feature columns directly
        feature_cols = [
            'capital_ratio', 'liquidity_buffer', 'leverage', 'credit_exposure',
            'risk_appetite', 'stress_level', 'degree_centrality', 'betweenness_centrality',
            'eigenvector_centrality', 'pagerank', 'in_degree', 'out_degree',
            'default_probability_prior', 'credit_spread', 'volatility', 'market_pressure',
            'neighbor_avg_stress', 'neighbor_max_stress', 'neighbor_default_count',
            'neighbor_avg_capital_ratio'
        ]
        
        # Fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        features_array = df[feature_cols].fillna(0).values
        labels = df['defaulted'].values if 'defaulted' in df.columns else np.zeros(len(df))
        
        # Convert to feature objects
        from app.ml.features.extractor import InstitutionFeatures
        
        features_list = []
        for i in range(len(features_array)):
            feat = InstitutionFeatures(
                institution_id=str(df.iloc[i]['institution_id']),
                timestamp=df.iloc[i].get('timestamp', '2025-03-31'),
                capital_ratio=float(features_array[i, 0]),
                liquidity_buffer=float(features_array[i, 1]),
                leverage=float(features_array[i, 2]),
                credit_exposure=float(features_array[i, 3]),
                risk_appetite=float(features_array[i, 4]),
                stress_level=float(features_array[i, 5]),
                degree_centrality=float(features_array[i, 6]),
                betweenness_centrality=float(features_array[i, 7]),
                eigenvector_centrality=float(features_array[i, 8]),
                pagerank=float(features_array[i, 9]),
                in_degree=float(features_array[i, 10]),
                out_degree=float(features_array[i, 11]),
                default_probability_prior=float(features_array[i, 12]),
                credit_spread=float(features_array[i, 13]),
                volatility=float(features_array[i, 14]),
                market_pressure=float(features_array[i, 15]),
                neighbor_avg_stress=float(features_array[i, 16]),
                neighbor_max_stress=float(features_array[i, 17]),
                neighbor_default_count=float(features_array[i, 18]),
                neighbor_avg_capital_ratio=float(features_array[i, 19])
            )
            features_list.append(feat)
        
        labels_list = labels.tolist()
        
        return features_list, labels_list
    
    def build_composite_network(self):
        """Layer 2: Build multi-channel network"""
        logger.info("Building composite multi-channel network...")
        
        # Get centrality scores from ML features
        ml_df = self.integrated_data.ml_features
        centrality_dict = {}
        
        if 'eigenvector_centrality' in ml_df.columns:
            for _, row in ml_df.iterrows():
                bank_name = row['bank_name']
                centrality = row['eigenvector_centrality']
                centrality_dict[bank_name] = float(centrality)
        
        # Build network
        network = self.network_builder.build_composite_network(
            centrality_scores=centrality_dict,
            threshold=0.1
        )
        
        logger.info(f"Network built:")
        logger.info(f"  Nodes: {network.number_of_nodes()}")
        logger.info(f"  Edges: {network.number_of_edges()}")
        logger.info(f"  Density: {network.number_of_edges() / (network.number_of_nodes() * (network.number_of_nodes() - 1) / 2):.4f}")
        
        return network
    
    def analyze_systemic_fragility(self):
        """Layer 3: Spectral analysis of network"""
        logger.info("Performing spectral analysis...")
        
        # Analyze network
        metrics = analyze_systemic_fragility(self.network, verbose=True)
        
        return metrics
    
    def assess_ccp_risk(self):
        """Layer 4: CCP risk assessment and policy response"""
        logger.info("Assessing CCP risk and generating policy recommendations...")
        
        # Get predictions for all members
        ml_df = self.integrated_data.ml_features
        
        # Prepare member profiles
        member_profiles = []
        
        for _, row in ml_df.iterrows():
            member_id = str(row['institution_id'])
            member_name = row['bank_name']
            
            # Default probability (use model if available, else use prior)
            if self.trained_model:
                # Would normally predict here, but for now use stored value
                default_prob = float(row.get('default_probability_prior', 0.3))
            else:
                default_prob = float(row.get('default_probability_prior', 0.3))
            
            # Systemic importance from spectral analysis
            if self.spectral_metrics and member_name in self.spectral_metrics.eigenvector_centrality:
                systemic_importance = self.spectral_metrics.eigenvector_centrality[member_name]
            else:
                systemic_importance = float(row.get('eigenvector_centrality', 0.5))
            
            # Buffers
            capital_buffer = float(row.get('capital_ratio', 0.1))
            liquidity_buffer = float(row.get('liquidity_buffer', 0.1))
            
            # Assess member
            profile = self.ccp_manager.assess_member_risk(
                member_id=member_id,
                member_name=member_name,
                default_probability=default_prob,
                systemic_importance=systemic_importance,
                capital_buffer=capital_buffer,
                liquidity_buffer=liquidity_buffer,
                cascade_trigger_prob=default_prob * systemic_importance,  # Approximation
                contagion_vulnerability=systemic_importance,
                exposure_amount=1e8  # 100M per member
            )
            
            member_profiles.append(profile)
        
        # System-wide assessment
        system_fragility = self.spectral_metrics.fragility_index if self.spectral_metrics else 0.5
        
        assessment = self.ccp_manager.assess_system_risk(
            member_profiles=member_profiles,
            system_fragility=system_fragility,
            total_exposure=1e10  # 10B total
        )
        
        # Generate report
        report = self.ccp_manager.generate_risk_report(assessment)
        print(report)
        
        # Save report if requested
        if self.save_artifacts:
            report_path = Path("ccp_risk_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
        
        return assessment


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CCP-Centric Risk Modeling Pipeline'
    )
    
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete end-to-end pipeline'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train ML model only'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze network and generate report (assumes model exists)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='app/ml/data',
        help='Data directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='ccp_default_predictor',
        help='Model name'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save risk report to file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CCPPipeline(
        data_dir=args.data_dir,
        model_name=args.model_name,
        save_artifacts=args.save_report
    )
    
    try:
        if args.full_pipeline or (not args.train and not args.analyze):
            # Run full pipeline
            pipeline.run_full_pipeline()
        
        elif args.train:
            # Train only
            pipeline.integrated_data = pipeline.data_pipeline.load_all_datasets()
            pipeline.train_default_predictor()
        
        elif args.analyze:
            # Analyze only
            pipeline.integrated_data = pipeline.data_pipeline.load_all_datasets()
            pipeline.network = pipeline.build_composite_network()
            pipeline.spectral_metrics = pipeline.analyze_systemic_fragility()
            pipeline.assess_ccp_risk()
        
        logger.info("\n✓ Pipeline execution successful!")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline execution failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
