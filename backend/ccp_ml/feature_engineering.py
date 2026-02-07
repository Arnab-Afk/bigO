"""
Feature Engineering Module

Creates ML features according to ML_Flow.md specification:
- Capital & solvency: CRAR, Tier1, capital_ratio, leverage
- Liquidity & stress: liquidity_buffer, stress_level  
- Credit risk: Gross NPA, Net NPA, default_probability_prior
- Network metrics: degree_centrality, betweenness_centrality, pagerank
- Derived features: NPA growth, writeoff intensity, sector concentration
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .data_loader import DatasetContainer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for CCP risk modeling.
    
    Enriches ML features with additional signals from all datasets.
    """
    
    # Feature categories
    CAPITAL_FEATURES = ['capital_ratio', 'crar_tier1', 'crar_tier2', 'crar_total', 'leverage']
    LIQUIDITY_FEATURES = ['liquidity_buffer', 'stress_level']
    CREDIT_FEATURES = ['gross_npa', 'net_npa', 'npa_growth_rate', 'writeoff_intensity', 
                       'default_probability_prior']
    NETWORK_FEATURES = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality',
                        'pagerank', 'in_degree', 'out_degree']
    SECTOR_FEATURES = ['capital_market_exposure', 'real_estate_exposure', 'sector_concentration']
    
    def __init__(self, normalize: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.fitted = False
    
    def create_features(
        self, 
        data: DatasetContainer,
        target_year: int = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set from all data sources.
        
        Args:
            data: DatasetContainer with all loaded data
            target_year: Specific year to create features for (None = all years)
            
        Returns:
            DataFrame with all features
        """
        # Start with ML ready data
        if data.ml_ready is None:
            raise ValueError("ML ready data required but not loaded")
        
        features = data.ml_ready.copy()
        
        # Filter by year if specified
        if target_year is not None:
            features = features[features['timestamp'].str.contains(str(target_year))]
        
        # Enrich with CRAR data
        if data.crar is not None:
            features = self._enrich_crar(features, data.crar)
        
        # Enrich with NPA data
        if data.npa_movements is not None:
            features = self._enrich_npa(features, data.npa_movements)
        
        # Enrich with sector exposure data
        if data.sector_exposures is not None:
            features = self._enrich_sector(features, data.sector_exposures)
        
        # Enrich with peer ratio context
        if data.peer_ratios is not None:
            features = self._enrich_peer_context(features, data.peer_ratios)
        
        # Create derived features
        features = self._create_derived_features(features)
        
        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)
        
        logger.info(f"Created {len(features.columns)} features for {len(features)} samples")
        return features
    
    def _enrich_crar(self, features: pd.DataFrame, crar: pd.DataFrame) -> pd.DataFrame:
        """Enrich features with CRAR data"""
        # Create mapping from bank name to CRAR data
        crar_latest = crar.sort_values('year', ascending=False).drop_duplicates('bank_name')
        
        # Map using fuzzy matching on bank names
        crar_map = crar_latest.set_index('bank_name')[['crar_tier1', 'crar_tier2', 'crar_total']].to_dict('index')
        
        # Add CRAR features
        features['crar_tier1_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, crar_map, 'crar_tier1')
        )
        features['crar_tier2_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, crar_map, 'crar_tier2')
        )
        features['crar_total_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, crar_map, 'crar_total')
        )
        
        return features
    
    def _enrich_npa(self, features: pd.DataFrame, npa: pd.DataFrame) -> pd.DataFrame:
        """Enrich features with NPA data"""
        # Get latest NPA data per bank
        npa_latest = npa.sort_values('year', ascending=False).drop_duplicates('bank_name')
        
        npa_map = npa_latest.set_index('bank_name')[
            ['gross_npa_closing', 'net_npa_closing', 'npa_growth_rate', 'writeoff_intensity']
        ].to_dict('index')
        
        features['gross_npa_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, npa_map, 'gross_npa_closing')
        )
        features['net_npa_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, npa_map, 'net_npa_closing')
        )
        features['npa_growth_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, npa_map, 'npa_growth_rate')
        )
        features['writeoff_intensity_enriched'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, npa_map, 'writeoff_intensity')
        )
        
        return features
    
    def _enrich_sector(self, features: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
        """Enrich features with sector exposure data"""
        # Get latest sector data per bank
        sector_latest = sector.sort_values('year', ascending=False).drop_duplicates('bank_name')
        
        sector_map = sector_latest.set_index('bank_name')[
            ['capital_market', 'real_estate', 'total', 'capital_market_pct', 'real_estate_pct']
        ].to_dict('index')
        
        features['capital_market_exposure'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, sector_map, 'capital_market_pct')
        )
        features['real_estate_exposure'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, sector_map, 'real_estate_pct')
        )
        features['total_sensitive_exposure'] = features['bank_name'].apply(
            lambda x: self._fuzzy_lookup(x, sector_map, 'total')
        )
        
        return features
    
    def _enrich_peer_context(self, features: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
        """Add peer group context as reference (not direct predictors)"""
        # Pivot peer ratios for easy access
        # This provides group-level baselines for sanity checks
        
        # Get key ratios for the most recent year
        peer_latest = peer[peer['year'] == peer['year'].max()]
        
        # Extract key ratios
        key_ratios = ['Credit - Deposit Ratio', 'Return on assets', 'Return on equity']
        
        for ratio in key_ratios:
            ratio_data = peer_latest[peer_latest['ratio_name'].str.contains(ratio, case=False, na=False)]
            if not ratio_data.empty:
                # Store group averages as reference
                clean_name = ratio.lower().replace(' ', '_').replace('-', '_')
                features[f'peer_avg_{clean_name}'] = ratio_data['all_scb'].values[0] if len(ratio_data) > 0 else np.nan
        
        return features
    
    def _fuzzy_lookup(self, bank_name, mapping: Dict, key: str) -> Optional[float]:
        """Fuzzy lookup for bank name in mapping"""
        # Handle non-string input
        if pd.isna(bank_name) or not isinstance(bank_name, str):
            return np.nan
        
        # Direct match
        if bank_name in mapping:
            return mapping[bank_name].get(key, np.nan)
        
        # Fuzzy matching
        bank_upper = str(bank_name).upper()
        for mapped_name in mapping.keys():
            # Skip non-string keys (e.g., NaN)
            if not isinstance(mapped_name, str):
                continue
            if mapped_name.upper() in bank_upper or bank_upper in mapped_name.upper():
                return mapping[mapped_name].get(key, np.nan)
        
        return np.nan
    
    def _create_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create derived/interaction features"""
        
        # Capital adequacy stress indicator
        if 'capital_ratio' in features.columns and 'stress_level' in features.columns:
            features['capital_stress_ratio'] = features['capital_ratio'] / (features['stress_level'] + 0.01)
        
        # Leverage risk indicator
        if 'leverage' in features.columns and 'capital_ratio' in features.columns:
            features['leverage_risk'] = features['leverage'] * (1 - features['capital_ratio'])
        
        # Network influence score
        network_cols = [c for c in features.columns if 'centrality' in c.lower() or 'pagerank' in c.lower()]
        if network_cols:
            features['network_influence'] = features[network_cols].mean(axis=1)
        
        # Sector concentration (Herfindahl-like)
        sector_cols = ['capital_market_exposure', 'real_estate_exposure']
        sector_cols = [c for c in sector_cols if c in features.columns]
        if sector_cols:
            features['sector_concentration'] = (features[sector_cols] ** 2).sum(axis=1)
        
        # Composite risk score (weighted combination)
        risk_cols = ['stress_level', 'default_probability_prior', 'npa_growth_enriched']
        risk_cols = [c for c in risk_cols if c in features.columns]
        if risk_cols:
            features['composite_risk'] = features[risk_cols].fillna(0).mean(axis=1)
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        # Identify numeric columns (excluding IDs and targets)
        exclude_cols = ['institution_id', 'bank_name', 'timestamp', 'defaulted']
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if not self.fitted:
            self.scaler.fit(features[numeric_cols].fillna(0))
            self.fitted = True
        
        features[numeric_cols] = self.scaler.transform(features[numeric_cols].fillna(0))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for modeling"""
        return (
            self.CAPITAL_FEATURES + 
            self.LIQUIDITY_FEATURES + 
            self.CREDIT_FEATURES + 
            self.NETWORK_FEATURES + 
            self.SECTOR_FEATURES
        )
    
    def create_train_test_split(
        self,
        features: pd.DataFrame,
        test_year: int = 2025,
        validation_split: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-aware train/validation/test splits.
        
        Following ML_Flow.md guidelines:
        - Train: 2022-2024
        - Validation: Random 15% from train
        - Test: 2025
        
        Args:
            features: Feature DataFrame
            test_year: Year to use for test set
            validation_split: Fraction for validation
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Extract year from timestamp
        if 'timestamp' in features.columns:
            features['_year'] = pd.to_datetime(features['timestamp'], errors='coerce').dt.year
        else:
            # Try to extract from institution_id if it contains year
            features['_year'] = features['institution_id'].astype(str).str.extract(r'(\d{4})').astype(float)
        
        # Split by year
        test = features[features['_year'] == test_year].drop('_year', axis=1)
        train_val = features[features['_year'] < test_year].drop('_year', axis=1)
        
        # Random validation split
        val_mask = np.random.random(len(train_val)) < validation_split
        validation = train_val[val_mask]
        train = train_val[~val_mask]
        
        logger.info(f"Split: Train={len(train)}, Val={len(validation)}, Test={len(test)}")
        return train, validation, test


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    from .data_loader import load_data
    
    data = load_data()
    engineer = FeatureEngineer()
    features = engineer.create_features(data)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"\nColumns: {features.columns.tolist()}")
