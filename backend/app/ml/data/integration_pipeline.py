"""
Comprehensive Data Integration Pipeline

Integrates all 7 datasets according to ML_Flow.md specification:
1. rbi_banks_ml_ready.csv - Core ML dataset
2. 3.Bank-wise Capital Adequacy Ratios (CRAR) - Capital strength
3. 10.Bank Group-wise Select Ratios - Macro-prudential context
4. 6.Movement of NPAs - Asset quality signals
5. 8.Exposure to Sensitive Sectors - Network edges
6. 9.Maturity Profile - Liquidity channel
7. Yahoo Finance / Market Data - Market correlation

Reference: ML_Flow.md Section 3
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDataset:
    """Container for integrated multi-source data"""
    
    # Core ML features
    ml_features: pd.DataFrame          # From rbi_banks_ml_ready.csv
    
    # Enrichment features
    crar_data: pd.DataFrame            # Capital adequacy ratios
    peer_ratios: pd.DataFrame          # Group-wise benchmarks
    npa_movements: pd.DataFrame        # Asset quality trends
    sector_exposures: pd.DataFrame     # Sensitive sector data
    maturity_profiles: pd.DataFrame    # ALM data
    market_data: Optional[pd.DataFrame] = None  # Market correlations
    
    # Metadata
    bank_list: List[str] = None
    time_period: Tuple[str, str] = None
    
    def __post_init__(self):
        if self.bank_list is None:
            self.bank_list = self.ml_features['bank_name'].unique().tolist()
        if self.time_period is None and 'timestamp' in self.ml_features.columns:
            dates = pd.to_datetime(self.ml_features['timestamp'])
            self.time_period = (str(dates.min()), str(dates.max()))


class DataIntegrationPipeline:
    """
    Integrate all RBI datasets for CCP risk modeling
    
    CCP Perspective:
    - Each dataset provides different risk signals
    - Cross-validation across datasets
    - Time-aligned for temporal consistency
    """
    
    def __init__(self, data_dir: str = "app/ml/data"):
        """
        Initialize pipeline
        
        Args:
            data_dir: Directory containing all CSV files
        """
        self.data_dir = Path(data_dir)
        
        # Define file paths
        self.file_paths = {
            'ml_ready': self.data_dir / 'rbi_banks_ml_ready.csv',
            'crar': self.data_dir / '3.Bank-wise Capital Adequacy Ratios (CRAR) of Scheduled Commercial Banks.csv',
            'ratios': self.data_dir / '10.Bank Group-wise Select Ratios of Scheduled Commercial Banks.csv',
            'npa': self.data_dir / '6.Movement of Non Performing Assets (NPAs) of Scheduled Commercial Banks.csv',
            'sectors': self.data_dir / '8.Exposure to Sensitive Sectors of Scheduled Commercial Banks.csv',
            'maturity': self.data_dir / '9.Maturity Profile of Select Items of Liabilities and Assets of Scheduled Commercial Banks.csv',
        }
        
        logger.info(f"Data integration pipeline initialized with data directory: {self.data_dir}")
    
    def load_all_datasets(self) -> IntegratedDataset:
        """
        Load and integrate all 7 datasets
        
        Returns:
            IntegratedDataset with all data aligned
        """
        logger.info("="*60)
        logger.info("LOADING ALL DATASETS")
        logger.info("="*60)
        
        # 1. Load core ML dataset
        ml_features = self._load_ml_ready()
        logger.info(f"✓ Loaded ML features: {ml_features.shape}")
        
        # 2. Load CRAR data
        crar_data = self._load_crar()
        logger.info(f"✓ Loaded CRAR data: {crar_data.shape}")
        
        # 3. Load group ratios
        peer_ratios = self._load_peer_ratios()
        logger.info(f"✓ Loaded peer ratios: {peer_ratios.shape}")
        
        # 4. Load NPA movements
        npa_movements = self._load_npa_movements()
        logger.info(f"✓ Loaded NPA movements: {npa_movements.shape}")
        
        # 5. Load sector exposures
        sector_exposures = self._load_sector_exposures()
        logger.info(f"✓ Loaded sector exposures: {sector_exposures.shape}")
        
        # 6. Load maturity profiles
        maturity_profiles = self._load_maturity_profiles()
        logger.info(f"✓ Loaded maturity profiles: {maturity_profiles.shape}")
        
        # 7. Market data (optional, can be added later)
        market_data = None
        
        logger.info("="*60)
        logger.info("ALL DATASETS LOADED SUCCESSFULLY")
        logger.info("="*60)
        
        # Create integrated dataset
        integrated = IntegratedDataset(
            ml_features=ml_features,
            crar_data=crar_data,
            peer_ratios=peer_ratios,
            npa_movements=npa_movements,
            sector_exposures=sector_exposures,
            maturity_profiles=maturity_profiles,
            market_data=market_data
        )
        
        return integrated
    
    def _load_ml_ready(self) -> pd.DataFrame:
        """Load Dataset 1: rbi_banks_ml_ready.csv"""
        df = pd.read_csv(self.file_paths['ml_ready'])
        
        # Ensure required columns
        required = ['institution_id', 'bank_name', 'defaulted']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"ML dataset missing required columns: {missing}")
        
        return df
    
    def _load_crar(self) -> pd.DataFrame:
        """Load Dataset 2: Capital Adequacy Ratios"""
        df = pd.read_csv(self.file_paths['crar'])
        
        # Find data start (skip header rows)
        data_start = 0
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        # Extract clean data
        df_clean = df.iloc[data_start:].copy()
        
        # Set proper column names
        # Expected: Year, Bank Name, Basel-I (3 cols), Basel-II (3 cols), Basel-III (3 cols)
        df_clean.columns = [
            'Year', 'Bank', 
            'Basel1_Tier1', 'Basel1_Tier2', 'Basel1_Total',
            'Basel2_Tier1', 'Basel2_Tier2', 'Basel2_Total',
            'Basel3_Tier1', 'Basel3_Tier2', 'Basel3_Total',
            'Extra1'  # Extra column if exists
        ] if len(df_clean.columns) >= 11 else df_clean.columns
        
        # Remove aggregate rows
        df_clean = df_clean[df_clean['Bank'].notna()]
        df_clean = df_clean[~df_clean['Bank'].str.contains('SECTOR BANKS|PUBLIC SECTOR|PRIVATE SECTOR', 
                                                             case=False, na=False)]
        
        # Convert numeric columns
        for col in df_clean.columns[2:]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace('..', 'NaN').str.replace(',', ''),
                    errors='coerce'
                )
        
        return df_clean
    
    def _load_peer_ratios(self) -> pd.DataFrame:
        """Load Dataset 3: Bank Group-wise Select Ratios"""
        df = pd.read_csv(self.file_paths['ratios'])
        
        # Find data start
        data_start = 0
        for idx, row in df.iterrows():
            # Look for year column with actual data (in column 2 after 2 empty columns)
            if '2024-25' in str(row.iloc[2]) or '2023-24' in str(row.iloc[2]):
                data_start = idx
                break
        
        df_clean = df.iloc[data_start:].copy()
        
        # The CSV has 2 empty leading columns, so we drop them first
        # Then rename the remaining columns
        # Structure: Year, Ratios, SBI, Nationalized, PSB, Private, Foreign, Small Finance, Payments, All SCBs
        if len(df_clean.columns) > 10:
            # Drop first 2 empty columns
            df_clean = df_clean.iloc[:, 2:]
        
        expected_cols = [
            'Year', 'Ratio', 'SBI', 'Nationalized', 'PSB', 'Private', 
            'Foreign', 'SmallFinance', 'Payments', 'All_SCB'
        ]
        
        if len(df_clean.columns) == len(expected_cols):
            df_clean.columns = expected_cols
        
        return df_clean
    
    def _load_npa_movements(self) -> pd.DataFrame:
        """Load Dataset 4: Movement of NPAs"""
        df = pd.read_csv(self.file_paths['npa'])
        
        # Find data start
        data_start = 0
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        df_clean = df.iloc[data_start:].copy()
        
        # Set column names
        df_clean.columns = [
            'Year', 'Bank', 
            'Gross_Opening', 'Gross_Addition', 'Gross_Reduction', 
            'Gross_Writeoff', 'Gross_Closing',
            'Net_Opening', 'Net_Closing'
        ] if len(df_clean.columns) >= 9 else df_clean.columns
        
        # Clean data
        df_clean = df_clean[df_clean['Bank'].notna()]
        df_clean = df_clean[~df_clean['Bank'].str.contains('SECTOR BANKS', case=False, na=False)]
        
        # Convert numeric columns
        for col in df_clean.columns[2:]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', '').str.replace('-', '0'),
                    errors='coerce'
                )
        
        # Compute NPA growth rate
        df_clean['NPA_Growth_Rate'] = (
            (df_clean['Gross_Closing'] - df_clean['Gross_Opening']) / 
            df_clean['Gross_Opening'].replace(0, np.nan)
        )
        
        return df_clean
    
    def _load_sector_exposures(self) -> pd.DataFrame:
        """Load Dataset 5: Exposure to Sensitive Sectors"""
        df = pd.read_csv(self.file_paths['sectors'])
        
        # Find data start
        data_start = 0
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        df_clean = df.iloc[data_start:].copy()
        
        # Set column names
        df_clean.columns = [
            'Year', 'Bank', 'Capital_Market', 'Real_Estate', 'Commodities', 'Total'
        ] if len(df_clean.columns) >= 6 else df_clean.columns
        
        # Clean data
        df_clean = df_clean[df_clean['Bank'].notna()]
        df_clean = df_clean[~df_clean['Bank'].str.contains('SECTOR BANKS', case=False, na=False)]
        
        # Convert numeric columns
        for col in ['Capital_Market', 'Real_Estate', 'Commodities', 'Total']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(',', '').str.replace('-', '0'),
                    errors='coerce'
                )
        
        return df_clean
    
    def _load_maturity_profiles(self) -> pd.DataFrame:
        """Load Dataset 6: Maturity Profile"""
        df = pd.read_csv(self.file_paths['maturity'])
        
        # Find data start
        data_start = 0
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        df_clean = df.iloc[data_start:].copy()
        
        # This file has many columns (54+) - keep all for now
        # Will be processed by network builder
        
        return df_clean
    
    def enrich_ml_features(
        self,
        integrated_data: IntegratedDataset
    ) -> pd.DataFrame:
        """
        Enrich ML features with additional signals from other datasets
        
        Args:
            integrated_data: IntegratedDataset with all loaded data
        
        Returns:
            Enhanced feature dataframe
        """
        logger.info("Enriching ML features with cross-dataset signals")
        
        ml_df = integrated_data.ml_features.copy()
        
        # Add NPA growth rate
        npa_df = integrated_data.npa_movements.copy()
        if 'NPA_Growth_Rate' in npa_df.columns:
            # Merge NPA growth
            npa_lookup = npa_df.set_index('Bank')['NPA_Growth_Rate'].to_dict()
            ml_df['npa_growth_rate'] = ml_df['bank_name'].map(npa_lookup).fillna(0)
        
        # Add sector concentration score
        sector_df = integrated_data.sector_exposures.copy()
        if 'Total' in sector_df.columns:
            # Compute Herfindahl index for sector concentration
            def compute_concentration(row):
                if row['Total'] > 0:
                    shares = [
                        row['Capital_Market'] / row['Total'],
                        row['Real_Estate'] / row['Total'],
                        row.get('Commodities', 0) / row['Total']
                    ]
                    return sum(s**2 for s in shares)
                return 0
            
            sector_df['sector_concentration'] = sector_df.apply(compute_concentration, axis=1)
            sector_lookup = sector_df.set_index('Bank')['sector_concentration'].to_dict()
            ml_df['sector_concentration'] = ml_df['bank_name'].map(sector_lookup).fillna(0)
        
        logger.info(f"Enriched features: {ml_df.shape[1]} columns")
        
        return ml_df
    
    def create_training_dataset(
        self,
        integrated_data: IntegratedDataset,
        test_year: str = '2025',
        validation_split: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits following ML_Flow.md guidelines
        
        Time-aware split:
        - Train: 2022-2024
        - Validation: Random 15% from train
        - Test: 2025
        
        Args:
            integrated_data: Integrated dataset
            test_year: Year to use for test set
            validation_split: Fraction of training data for validation
        
        Returns:
            train_df, val_df, test_df
        """
        logger.info("Creating train/validation/test splits")
        
        # Enrich features first
        df = self.enrich_ml_features(integrated_data)
        
        # Extract year from timestamp if available
        if 'timestamp' in df.columns:
            df['year'] = pd.to_datetime(df['timestamp']).dt.year.astype(str)
        elif 'institution_id' in df.columns:
            # Extract from institution_id (e.g., "2025", "2024_aug0")
            df['year'] = df['institution_id'].astype(str).str.extract(r'(\d{4})')[0]
        else:
            raise ValueError("Cannot determine year for time-based split")
        
        # Test set: specified year
        test_df = df[df['year'] == test_year].copy()
        
        # Train set: all other years
        train_val_df = df[df['year'] != test_year].copy()
        
        # Random split for validation
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=validation_split,
            random_state=42,
            stratify=train_val_df['defaulted'] if 'defaulted' in train_val_df.columns else None
        )
        
        logger.info(f"Split complete:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val:   {len(val_df)} samples")
        logger.info(f"  Test:  {len(test_df)} samples")
        
        if 'defaulted' in df.columns:
            logger.info(f"  Train default rate: {train_df['defaulted'].mean():.2%}")
            logger.info(f"  Val default rate:   {val_df['defaulted'].mean():.2%}")
            logger.info(f"  Test default rate:  {test_df['defaulted'].mean():.2%}")
        
        return train_df, val_df, test_df
    
    def generate_summary(self, integrated_data: IntegratedDataset) -> str:
        """
        Generate summary report of integrated data
        
        Args:
            integrated_data: Integrated dataset
        
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("="*70)
        summary.append("DATA INTEGRATION SUMMARY")
        summary.append("="*70)
        summary.append("")
        
        # Dataset sizes
        summary.append("Dataset Sizes:")
        summary.append(f"  1. ML Features:        {integrated_data.ml_features.shape[0]:5d} rows × {integrated_data.ml_features.shape[1]:2d} cols")
        summary.append(f"  2. CRAR Data:          {integrated_data.crar_data.shape[0]:5d} rows × {integrated_data.crar_data.shape[1]:2d} cols")
        summary.append(f"  3. Peer Ratios:        {integrated_data.peer_ratios.shape[0]:5d} rows × {integrated_data.peer_ratios.shape[1]:2d} cols")
        summary.append(f"  4. NPA Movements:      {integrated_data.npa_movements.shape[0]:5d} rows × {integrated_data.npa_movements.shape[1]:2d} cols")
        summary.append(f"  5. Sector Exposures:   {integrated_data.sector_exposures.shape[0]:5d} rows × {integrated_data.sector_exposures.shape[1]:2d} cols")
        summary.append(f"  6. Maturity Profiles:  {integrated_data.maturity_profiles.shape[0]:5d} rows × {integrated_data.maturity_profiles.shape[1]:2d} cols")
        if integrated_data.market_data is not None:
            summary.append(f"  7. Market Data:        {integrated_data.market_data.shape[0]:5d} rows × {integrated_data.market_data.shape[1]:2d} cols")
        else:
            summary.append(f"  7. Market Data:        Not loaded (optional)")
        summary.append("")
        
        # Coverage
        summary.append(f"Coverage:")
        summary.append(f"  Banks:       {len(integrated_data.bank_list)}")
        if integrated_data.time_period:
            summary.append(f"  Time Period: {integrated_data.time_period[0]} to {integrated_data.time_period[1]}")
        summary.append("")
        
        # Sample banks
        summary.append(f"Sample Banks:")
        for i, bank in enumerate(integrated_data.bank_list[:10], 1):
            summary.append(f"  {i:2d}. {bank}")
        if len(integrated_data.bank_list) > 10:
            summary.append(f"  ... and {len(integrated_data.bank_list) - 10} more")
        summary.append("")
        
        summary.append("="*70)
        
        return "\n".join(summary)
