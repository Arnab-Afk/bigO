"""
CCP Engine Module

Central Counterparty risk engine implementing Layer 4 of ML_Flow.md:
- Loss absorption logic
- Default fund sizing
- Margin requirements
- Policy response generation

CCP Perspective:
- We are the risk ABSORBER, not profit optimizer
- Focus on tail risk and systemic stability
- Conservative estimates for regulatory compliance
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np
import pandas as pd

from .risk_model import CCPRiskModel, select_features
from .network_builder import NetworkBuilder
from .spectral_analyzer import SpectralAnalyzer, SpectralMetrics

logger = logging.getLogger(__name__)


@dataclass
class MarginRequirement:
    """Initial margin requirement for a participant"""
    bank_name: str
    base_margin: float          # Based on default probability
    network_addon: float        # Based on network position
    stressed_margin: float      # Under stress scenarios
    total_margin: float         # Sum of components
    confidence_level: float     # VaR confidence level (e.g., 99%)
    explanation: str


@dataclass
class DefaultFundAllocation:
    """Default fund contribution for a participant"""
    bank_name: str
    base_contribution: float    # Based on risk profile
    systemic_addon: float       # Based on network importance
    total_contribution: float
    proportional_share: float   # Percentage of total fund


@dataclass
class PolicyRecommendation:
    """Policy recommendation based on risk assessment"""
    priority: str               # 'critical', 'high', 'medium', 'low'
    category: str               # 'margin', 'monitoring', 'exposure', 'capital'
    recommendation: str         # Human-readable recommendation
    affected_banks: List[str]   # Banks this applies to
    rationale: str              # Why this recommendation
    metrics: Dict[str, float]   # Supporting metrics


class CCPEngine:
    """
    CCP Engine: Orchestrates risk assessment and policy generation.
    
    This is the main entry point that combines:
    - Participant risk estimates
    - Network structure analysis
    - Spectral fragility metrics
    - Loss absorption calculations
    """
    
    # Default fund sizing parameters
    COVER_N = 2  # Cover-N standard (cover N largest defaults)
    CONFIDENCE_LEVEL = 0.99  # 99% VaR
    STRESS_MULTIPLIER = 1.5  # Multiplier for stressed scenarios
    
    # Margin calculation parameters
    BASE_MARGIN_RATE = 0.02  # 2% of exposure as base
    MAX_MARGIN_RATE = 0.15   # 15% maximum
    NETWORK_MARGIN_WEIGHT = 0.3  # Network contribution to margin
    
    def __init__(
        self,
        risk_model: CCPRiskModel = None,
        network_builder: NetworkBuilder = None,
        spectral_analyzer: SpectralAnalyzer = None
    ):
        """
        Initialize CCP Engine.
        
        Args:
            risk_model: Pre-trained risk model (will create if None)
            network_builder: Network builder (will create if None)
            spectral_analyzer: Spectral analyzer (will create if None)
        """
        self.risk_model = risk_model or CCPRiskModel()
        self.network_builder = network_builder or NetworkBuilder()
        self.spectral_analyzer = spectral_analyzer or SpectralAnalyzer()
        
        # Cached results
        self.risk_scores = None
        self.network_metrics = None
        self.spectral_metrics = None
        self.margin_requirements = None
        self.default_fund = None
    
    def run_full_analysis(
        self,
        features: pd.DataFrame,
        train: bool = True,
        year: int = None
    ) -> Dict[str, Any]:
        """
        Run complete CCP risk analysis pipeline.
        
        Args:
            features: Feature DataFrame with all participant data
            train: Whether to train the risk model
            year: Specific year for analysis
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting CCP risk analysis pipeline...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_participants': len(features),
            'year': year
        }
        
        # Step 1: Participant risk estimation
        logger.info("Step 1/4: Participant risk estimation")
        X, y = select_features(features)
        
        if train and len(y.unique()) > 1:
            train_metrics = self.risk_model.fit(X, y)
            results['training_metrics'] = train_metrics
        
        if self.risk_model.is_fitted:
            probs = self.risk_model.predict_proba(X)[:, 1]
            features['default_probability'] = probs
            self.risk_scores = pd.DataFrame({
                'bank_name': features['bank_name'].values if 'bank_name' in features.columns else range(len(features)),
                'default_probability': probs,
                'risk_level': pd.cut(probs, bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                     labels=['low', 'medium', 'high', 'critical'])
            })
            results['risk_distribution'] = self.risk_scores['risk_level'].value_counts().to_dict()
        
        # Step 2: Network construction (using DatasetContainer would be passed externally)
        logger.info("Step 2/4: Network analysis")
        # Network metrics would typically be passed in or computed from full data
        if 'degree_centrality' in features.columns:
            self.network_metrics = features[[
                c for c in features.columns 
                if 'centrality' in c.lower() or 'pagerank' in c.lower() or 'degree' in c.lower()
            ]].copy()
            if 'bank_name' in features.columns:
                self.network_metrics['bank_name'] = features['bank_name']
        
        # Step 3: Spectral analysis
        logger.info("Step 3/4: Spectral fragility analysis")
        if self.network_builder.graph is not None:
            self.spectral_metrics = self.spectral_analyzer.analyze(
                network_builder=self.network_builder
            )
            results['spectral_metrics'] = {
                'spectral_radius': self.spectral_metrics.spectral_radius,
                'fiedler_value': self.spectral_metrics.fiedler_value,
                'amplification_risk': self.spectral_metrics.amplification_risk,
                'fragmentation_risk': self.spectral_metrics.fragmentation_risk
            }
        
        # Step 4: Loss absorption & policy
        logger.info("Step 4/4: Loss absorption and policy generation")
        self.margin_requirements = self._calculate_margins(features)
        self.default_fund = self._size_default_fund(features)
        
        results['margin_summary'] = {
            'total_margin': sum(m.total_margin for m in self.margin_requirements),
            'avg_margin_rate': np.mean([m.total_margin for m in self.margin_requirements]),
            'high_margin_count': sum(1 for m in self.margin_requirements 
                                    if m.total_margin > 0.05 * len(features))
        }
        
        results['default_fund'] = {
            'total_fund': sum(d.total_contribution for d in self.default_fund),
            'cover_n': self.COVER_N,
            'largest_contributions': sorted(
                [(d.bank_name, d.total_contribution) for d in self.default_fund],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        # Generate policy recommendations
        results['policies'] = self._generate_policies(features)
        
        logger.info("CCP analysis complete")
        return results
    
    def _calculate_margins(self, features: pd.DataFrame) -> List[MarginRequirement]:
        """
        Calculate initial margin requirements for each participant.
        
        Margin = base_rate × (1 + risk_factor + network_factor)
        """
        margins = []
        
        for idx, row in features.iterrows():
            bank_name = row.get('bank_name', f'Bank_{idx}')
            
            # Get default probability
            default_prob = row.get('default_probability', 0.1)
            
            # Get network importance
            network_importance = row.get('pagerank', 0) + row.get('degree_centrality', 0)
            network_importance = min(network_importance / 2, 1.0)  # Normalize
            
            # Calculate margin components
            risk_factor = default_prob * 2  # Higher risk = higher margin
            base_margin = self.BASE_MARGIN_RATE * (1 + risk_factor)
            network_addon = base_margin * network_importance * self.NETWORK_MARGIN_WEIGHT
            stressed_margin = (base_margin + network_addon) * self.STRESS_MULTIPLIER
            
            total_margin = min(base_margin + network_addon, self.MAX_MARGIN_RATE)
            
            # Generate explanation
            if default_prob > 0.5:
                explanation = f"High margin due to elevated default probability ({default_prob:.1%})"
            elif network_importance > 0.5:
                explanation = f"Higher margin due to systemic importance (network score: {network_importance:.2f})"
            else:
                explanation = "Standard margin requirements"
            
            margins.append(MarginRequirement(
                bank_name=bank_name,
                base_margin=base_margin,
                network_addon=network_addon,
                stressed_margin=stressed_margin,
                total_margin=total_margin,
                confidence_level=self.CONFIDENCE_LEVEL,
                explanation=explanation
            ))
        
        return margins
    
    def _size_default_fund(self, features: pd.DataFrame) -> List[DefaultFundAllocation]:
        """
        Size default fund using Cover-N methodology.
        
        Total fund should cover N largest participant defaults.
        Allocation is proportional to risk contribution.
        """
        allocations = []
        
        # Get risk-weighted exposure for each participant
        risk_weights = []
        for idx, row in features.iterrows():
            default_prob = row.get('default_probability', 0.1)
            network_importance = row.get('pagerank', 0.1)
            
            # Risk weight = probability × systemic importance
            weight = default_prob * (1 + network_importance)
            risk_weights.append({
                'bank_name': row.get('bank_name', f'Bank_{idx}'),
                'weight': weight,
                'default_prob': default_prob,
                'network_importance': network_importance
            })
        
        # Sort by weight
        risk_weights = sorted(risk_weights, key=lambda x: x['weight'], reverse=True)
        
        # Total fund = Cover top N exposures
        top_n_weights = sum(w['weight'] for w in risk_weights[:self.COVER_N])
        base_fund_multiplier = 1e6  # Scaling factor (would be based on actual exposures)
        total_fund = top_n_weights * base_fund_multiplier
        
        # Allocate proportionally
        total_weight = sum(w['weight'] for w in risk_weights)
        
        for rw in risk_weights:
            proportion = rw['weight'] / total_weight if total_weight > 0 else 1 / len(risk_weights)
            base = proportion * total_fund * 0.7  # 70% based on individual risk
            systemic = proportion * total_fund * 0.3 * (1 + rw['network_importance'])
            
            allocations.append(DefaultFundAllocation(
                bank_name=rw['bank_name'],
                base_contribution=base,
                systemic_addon=systemic,
                total_contribution=base + systemic,
                proportional_share=proportion
            ))
        
        return allocations
    
    def _generate_policies(self, features: pd.DataFrame) -> List[Dict]:
        """
        Generate policy recommendations based on analysis.
        """
        policies = []
        
        # Policy 1: High-risk participant monitoring
        if self.risk_scores is not None:
            high_risk = self.risk_scores[
                self.risk_scores['risk_level'].isin(['high', 'critical'])
            ]['bank_name'].tolist()
            
            if high_risk:
                policies.append({
                    'priority': 'high',
                    'category': 'monitoring',
                    'recommendation': 'Enhanced monitoring for high-risk participants',
                    'affected_banks': high_risk,
                    'rationale': f'{len(high_risk)} participants show elevated default probability',
                    'actions': [
                        'Daily margin call review',
                        'Weekly exposure limit review',
                        'Monthly stress testing'
                    ]
                })
        
        # Policy 2: Systemic risk response
        if self.spectral_metrics is not None:
            if self.spectral_metrics.amplification_risk == 'high':
                policies.append({
                    'priority': 'critical',
                    'category': 'systemic',
                    'recommendation': 'Network structure indicates high contagion potential',
                    'affected_banks': [],
                    'rationale': f'Spectral radius {self.spectral_metrics.spectral_radius:.2f} suggests shock amplification',
                    'actions': [
                        'Consider position limits on highly connected participants',
                        'Increase default fund size by 20%',
                        'Activate stress monitoring dashboard'
                    ]
                })
        
        # Policy 3: Concentration risk
        if self.margin_requirements:
            top_margins = sorted(self.margin_requirements, 
                               key=lambda m: m.total_margin, reverse=True)[:3]
            top_share = sum(m.total_margin for m in top_margins) / sum(
                m.total_margin for m in self.margin_requirements
            ) if self.margin_requirements else 0
            
            if top_share > 0.5:
                policies.append({
                    'priority': 'medium',
                    'category': 'concentration',
                    'recommendation': 'High concentration in top participants',
                    'affected_banks': [m.bank_name for m in top_margins],
                    'rationale': f'Top 3 participants represent {top_share:.1%} of total margin requirements',
                    'actions': [
                        'Review position limits for top participants',
                        'Consider diversification incentives',
                        'Increase surveillance frequency'
                    ]
                })
        
        # Policy 4: Capital adequacy concerns
        if 'capital_ratio' in features.columns:
            low_capital = features[features['capital_ratio'] < 0.10]['bank_name'].tolist()
            if low_capital:
                policies.append({
                    'priority': 'high',
                    'category': 'capital',
                    'recommendation': 'Participants below minimum capital threshold',
                    'affected_banks': low_capital,
                    'rationale': f'{len(low_capital)} participants have capital ratio below 10%',
                    'actions': [
                        'Request capital adequacy documentation',
                        'Consider additional margin requirements',
                        'Notify risk committee'
                    ]
                })
        
        return policies
    
    def get_participant_summary(self, bank_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary for a specific participant.
        """
        summary = {'bank_name': bank_name}
        
        # Risk score
        if self.risk_scores is not None:
            bank_risk = self.risk_scores[self.risk_scores['bank_name'] == bank_name]
            if len(bank_risk) > 0:
                summary['default_probability'] = float(bank_risk['default_probability'].values[0])
                summary['risk_level'] = str(bank_risk['risk_level'].values[0])
        
        # Network metrics
        if self.network_metrics is not None:
            bank_network = self.network_metrics[self.network_metrics['bank_name'] == bank_name]
            if len(bank_network) > 0:
                for col in bank_network.columns:
                    if col != 'bank_name':
                        summary[f'network_{col}'] = float(bank_network[col].values[0])
        
        # Margin requirement
        if self.margin_requirements:
            for margin in self.margin_requirements:
                if margin.bank_name == bank_name:
                    summary['margin'] = {
                        'base': margin.base_margin,
                        'network_addon': margin.network_addon,
                        'total': margin.total_margin,
                        'explanation': margin.explanation
                    }
                    break
        
        # Default fund allocation
        if self.default_fund:
            for allocation in self.default_fund:
                if allocation.bank_name == bank_name:
                    summary['default_fund'] = {
                        'contribution': allocation.total_contribution,
                        'share': allocation.proportional_share
                    }
                    break
        
        return summary
    
    def export_results(self, output_path: str = None) -> Dict[str, Any]:
        """
        Export all results to a dictionary (optionally save to file).
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'risk_scores': self.risk_scores.to_dict('records') if self.risk_scores is not None else [],
            'margins': [
                {
                    'bank_name': m.bank_name,
                    'base_margin': m.base_margin,
                    'network_addon': m.network_addon,
                    'total_margin': m.total_margin,
                    'explanation': m.explanation
                }
                for m in (self.margin_requirements or [])
            ],
            'default_fund': [
                {
                    'bank_name': d.bank_name,
                    'contribution': d.total_contribution,
                    'share': d.proportional_share
                }
                for d in (self.default_fund or [])
            ],
            'spectral': self.spectral_analyzer.to_dict() if self.spectral_metrics else {}
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results exported to {output_path}")
        
        return results


if __name__ == "__main__":
    # Test CCP Engine
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic test data
    np.random.seed(42)
    n_banks = 20
    
    features = pd.DataFrame({
        'bank_name': [f'Bank_{i}' for i in range(n_banks)],
        'capital_ratio': np.random.uniform(0.08, 0.20, n_banks),
        'liquidity_buffer': np.random.uniform(0.15, 0.40, n_banks),
        'stress_level': np.random.uniform(0.1, 0.8, n_banks),
        'gross_npa': np.random.uniform(0.02, 0.12, n_banks),
        'degree_centrality': np.random.uniform(0.1, 0.9, n_banks),
        'pagerank': np.random.uniform(0.02, 0.15, n_banks),
        'defaulted': np.random.binomial(1, 0.15, n_banks)
    })
    
    # Run CCP analysis
    engine = CCPEngine()
    results = engine.run_full_analysis(features, train=True)
    
    print("\n" + "="*60)
    print("CCP RISK ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nParticipants analyzed: {results['n_participants']}")
    print(f"Risk distribution: {results.get('risk_distribution', {})}")
    
    if 'margin_summary' in results:
        print(f"\nMargin Summary:")
        print(f"  Total margin: {results['margin_summary']['total_margin']:.4f}")
        print(f"  High-margin participants: {results['margin_summary']['high_margin_count']}")
    
    if 'default_fund' in results:
        print(f"\nDefault Fund:")
        print(f"  Total size: {results['default_fund']['total_fund']:,.0f}")
        print(f"  Cover-N: {results['default_fund']['cover_n']}")
    
    if 'policies' in results:
        print(f"\nPolicy Recommendations ({len(results['policies'])}):")
        for policy in results['policies']:
            print(f"  [{policy['priority'].upper()}] {policy['recommendation']}")
    
    # Get summary for one participant
    print(f"\nParticipant Summary (Bank_0):")
    summary = engine.get_participant_summary('Bank_0')
    for key, value in summary.items():
        print(f"  {key}: {value}")
