"""
Real Data Loader for External Financial Data

Supports loading real-world financial institution data from various formats.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import pandas as pd
import numpy as np
import networkx as nx

from app.engine.game_theory import AgentState
from app.ml.features.extractor import FeatureExtractor, InstitutionFeatures

logger = logging.getLogger(__name__)


class RealDataLoader:
    """
    Load real financial data from external sources
    
    Supports:
    - CSV files with institution states
    - Network edge lists
    - Time series data
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None):
        self.feature_extractor = feature_extractor or FeatureExtractor()
    
    def load_from_csv(
        self,
        filepath: str,
        institution_id_col: str = "institution_id",
        timestamp_col: Optional[str] = "timestamp",
        label_col: str = "defaulted",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[List[InstitutionFeatures], List[int]]:
        """
        Load data from a single CSV file
        
        Args:
            filepath: Path to CSV file
            institution_id_col: Column name for institution ID
            timestamp_col: Column name for timestamp (optional)
            label_col: Column name for default label
            feature_cols: List of feature column names (uses all if None)
        
        Returns:
            Tuple of (features_list, labels_list)
        
        Example CSV format:
            institution_id,timestamp,capital_ratio,liquidity_buffer,...,defaulted
            bank_001,2024-01-01,0.12,0.25,...,0
            bank_001,2024-01-02,0.11,0.23,...,0
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [institution_id_col, label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Determine feature columns
        if feature_cols is None:
            # Use all numeric columns except ID and label
            exclude_cols = {institution_id_col, label_col}
            if timestamp_col:
                exclude_cols.add(timestamp_col)
            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
        
        # Extract features and labels
        features_list = []
        labels_list = []
        
        for idx, row in df.iterrows():
            # Extract feature values
            feature_values = row[feature_cols].values.astype(np.float32)
            
            # Pad or truncate to 20 features
            if len(feature_values) < 20:
                feature_values = np.pad(
                    feature_values,
                    (0, 20 - len(feature_values)),
                    mode='constant',
                    constant_values=0.0
                )
            elif len(feature_values) > 20:
                feature_values = feature_values[:20]
            
            # Create InstitutionFeatures object
            institution_id = uuid4()  # Generate UUID for institution
            
            features = InstitutionFeatures(
                institution_id=institution_id,
                # Financial features (6)
                capital_ratio=feature_values[0] if len(feature_values) > 0 else 0.1,
                liquidity_buffer=feature_values[1] if len(feature_values) > 1 else 0.2,
                leverage=feature_values[2] if len(feature_values) > 2 else 10.0,
                credit_exposure=feature_values[3] if len(feature_values) > 3 else 1000.0,
                risk_appetite=feature_values[4] if len(feature_values) > 4 else 0.5,
                stress_level=feature_values[5] if len(feature_values) > 5 else 0.0,
                # Network features (6)
                degree_centrality=feature_values[6] if len(feature_values) > 6 else 0.0,
                betweenness_centrality=feature_values[7] if len(feature_values) > 7 else 0.0,
                eigenvector_centrality=feature_values[8] if len(feature_values) > 8 else 0.0,
                pagerank=feature_values[9] if len(feature_values) > 9 else 0.0,
                in_degree=feature_values[10] if len(feature_values) > 10 else 0.0,
                out_degree=feature_values[11] if len(feature_values) > 11 else 0.0,
                # Market signals (4)
                default_probability_prior=feature_values[12] if len(feature_values) > 12 else 0.01,
                credit_spread=feature_values[13] if len(feature_values) > 13 else 0.0,
                volatility=feature_values[14] if len(feature_values) > 14 else 0.0,
                market_pressure=feature_values[15] if len(feature_values) > 15 else 0.0,
                # Neighborhood features (4)
                neighbor_avg_stress=feature_values[16] if len(feature_values) > 16 else 0.0,
                neighbor_max_stress=feature_values[17] if len(feature_values) > 17 else 0.0,
                neighbor_default_count=int(feature_values[18]) if len(feature_values) > 18 else 0,
                neighbor_avg_capital_ratio=feature_values[19] if len(feature_values) > 19 else 0.1,
            )
            
            features_list.append(features)
            labels_list.append(int(row[label_col]))
        
        logger.info(
            f"Loaded {len(features_list)} samples "
            f"(Defaults: {sum(labels_list)}, Non-defaults: {len(labels_list) - sum(labels_list)})"
        )
        
        return features_list, labels_list
    
    def load_from_network_data(
        self,
        institutions_file: str,
        exposures_file: str,
        states_file: str,
    ) -> Tuple[List[InstitutionFeatures], List[int]]:
        """
        Load data from separate network files
        
        Args:
            institutions_file: CSV with institution attributes
                Format: institution_id,name,type,total_assets
            exposures_file: CSV with bilateral exposures (edges)
                Format: source_id,target_id,exposure_amount
            states_file: CSV with time series states
                Format: institution_id,timestamp,capital_ratio,...,defaulted
        
        Returns:
            Tuple of (features_list, labels_list)
        """
        logger.info("Loading network data from multiple files")
        
        # Load institutions
        institutions_df = pd.read_csv(institutions_file)
        inst_id_map = {
            row['institution_id']: uuid4()
            for _, row in institutions_df.iterrows()
        }
        
        # Load network
        exposures_df = pd.read_csv(exposures_file)
        network = nx.DiGraph()
        
        for _, row in exposures_df.iterrows():
            source = inst_id_map.get(row['source_id'])
            target = inst_id_map.get(row['target_id'])
            if source and target:
                network.add_edge(
                    source,
                    target,
                    exposure=row['exposure_amount']
                )
        
        logger.info(f"Loaded network: {len(network.nodes())} nodes, {len(network.edges())} edges")
        
        # Load states
        states_df = pd.read_csv(states_file)
        
        features_list = []
        labels_list = []
        
        # Group by timestamp
        for timestamp, group in states_df.groupby('timestamp'):
            # Create agent states for this timestep
            agent_states = {}
            defaulted_institutions = set()
            
            for _, row in group.iterrows():
                inst_id = inst_id_map.get(row['institution_id'])
                if not inst_id:
                    continue
                
                agent_state = AgentState(
                    institution_id=inst_id,
                    capital_ratio=row.get('capital_ratio', 0.1),
                    liquidity_buffer=row.get('liquidity_buffer', 0.2),
                    credit_exposure=row.get('credit_exposure', 1000.0),
                    risk_appetite=row.get('risk_appetite', 0.5),
                    stress_level=row.get('stress_level', 0.0),
                    default_probability=row.get('default_probability', 0.01),
                )
                agent_states[inst_id] = agent_state
                
                if row.get('defaulted', 0) == 1:
                    defaulted_institutions.add(inst_id)
            
            # Extract features for all institutions
            for inst_id, agent_state in agent_states.items():
                features = self.feature_extractor.extract_features(
                    institution_id=inst_id,
                    agent_state=agent_state,
                    network=network,
                    all_agent_states=agent_states,
                    defaulted_institutions=defaulted_institutions,
                )
                
                label = 1 if inst_id in defaulted_institutions else 0
                
                features_list.append(features)
                labels_list.append(label)
        
        logger.info(
            f"Extracted {len(features_list)} samples from network data "
            f"(Defaults: {sum(labels_list)})"
        )
        
        return features_list, labels_list
    
    def load_from_fred(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load data from FRED (Federal Reserve Economic Data)
        
        Args:
            series_ids: List of FRED series IDs to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with time series data
        
        Example:
            series_ids = [
                'EQTA', # Bank equity to total assets
                'USNIM', # Net interest margin
                'DDOI08USA156NWDB', # Non-performing loans
            ]
        
        Note: Requires pandas_datareader:
            pip install pandas-datareader
        """
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            raise ImportError(
                "pandas_datareader required for FRED data. "
                "Install with: pip install pandas-datareader"
            )
        
        logger.info(f"Fetching {len(series_ids)} series from FRED")
        
        dfs = []
        for series_id in series_ids:
            try:
                df = pdr.DataReader(series_id, 'fred', start_date, end_date)
                df.columns = [series_id]
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        
        if not dfs:
            raise ValueError("No data fetched from FRED")
        
        # Merge all series
        result = pd.concat(dfs, axis=1)
        logger.info(f"Fetched data shape: {result.shape}")
        
        return result


def create_sample_csv(output_path: str, num_samples: int = 1000):
    """
    Create a sample CSV file for testing
    
    Args:
        output_path: Path to save CSV
        num_samples: Number of samples to generate
    """
    np.random.seed(42)
    
    data = {
        'institution_id': [f'bank_{i:03d}' for i in range(num_samples)],
        'timestamp': pd.date_range('2024-01-01', periods=num_samples, freq='D'),
        'capital_ratio': np.random.uniform(0.05, 0.20, num_samples),
        'liquidity_buffer': np.random.uniform(0.10, 0.40, num_samples),
        'leverage': np.random.uniform(5, 20, num_samples),
        'credit_exposure': np.random.uniform(1000, 10000, num_samples),
        'risk_appetite': np.random.uniform(0.3, 0.8, num_samples),
        'stress_level': np.random.uniform(0, 1, num_samples),
        'defaulted': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created sample CSV with {num_samples} samples at {output_path}")
    return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_path = "sample_data.csv"
    create_sample_csv(sample_path, num_samples=500)
    
    # Load the data
    loader = RealDataLoader()
    features, labels = loader.load_from_csv(
        sample_path,
        institution_id_col="institution_id",
        timestamp_col="timestamp",
        label_col="defaulted",
    )
    
    print(f"Loaded {len(features)} samples")
    print(f"Defaults: {sum(labels)}, Non-defaults: {len(labels) - sum(labels)}")
