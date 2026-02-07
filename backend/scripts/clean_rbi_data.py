#!/usr/bin/env python3
"""
RBI Data Cleaner

Cleans and consolidates Reserve Bank of India (RBI) banking data into ML-ready format.
Processes multiple CSV files with banking statistics into a unified training dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RBIDataCleaner:
    """Clean RBI banking data files"""
    
    def __init__(self, data_dir: str = "app/ml/data"):
        self.data_dir = Path(data_dir)
        self.banks_data = {}
        
    def clean_crar_data(self):
        """Clean Capital Adequacy Ratio data"""
        logger.info("Cleaning CRAR data...")
        
        file_path = self.data_dir / "3.Bank-wise Capital Adequacy Ratios (CRAR) of Scheduled Commercial Banks.csv"
        
        # Read raw data, skip header rows
        df = pd.read_csv(file_path, skiprows=8)
        
        # Use first row as column names
        df.columns = ['Year', 'Bank', 'Basel1_T1', 'Basel1_T2', 'Basel1_Total', 
                      'Basel2_T1', 'Basel2_T2', 'Basel2_Total',
                      'Basel3_T1', 'Basel3_T2', 'Basel3_Total', 'Extra']
        
        # Drop extra columns and rows with missing bank names
        df = df[['Year', 'Bank', 'Basel3_T1', 'Basel3_T2', 'Basel3_Total']].copy()
        df = df[df['Bank'].notna() & (df['Bank'] != '')]
        
        # Skip category headers
        df = df[~df['Bank'].str.contains('SECTOR BANKS|BANKS', case=False, na=False)]
        
        # Convert to numeric
        for col in ['Basel3_T1', 'Basel3_T2', 'Basel3_Total']:
            df[col] = pd.to_numeric(df[col].replace('..', np.nan), errors='coerce')
        
        # Store by bank
        for _, row in df.iterrows():
            bank = row['Bank'].strip()
            if bank not in self.banks_data:
                self.banks_data[bank] = {}
            self.banks_data[bank]['tier1_capital'] = row['Basel3_T1']
            self.banks_data[bank]['tier2_capital'] = row['Basel3_T2']
            self.banks_data[bank]['total_crar'] = row['Basel3_Total']
        
        logger.info(f"Processed CRAR for {len(self.banks_data)} banks")
        return df
    
    def clean_npa_data(self):
        """Clean Non-Performing Assets data"""
        logger.info("Cleaning NPA data...")
        
        file_path = self.data_dir / "6.Movement of Non Performing Assets (NPAs) of Scheduled Commercial Banks.csv"
        
        # Read raw data
        df = pd.read_csv(file_path, skiprows=8)
        
        # Set column names
        df.columns = ['Year', 'Bank', 'Gross_Opening', 'Gross_Addition', 'Gross_Reduction', 
                      'Gross_Writeoff', 'Gross_Closing', 'Net_Opening', 'Net_Closing', 'Extra']
        
        df = df[['Year', 'Bank', 'Gross_Closing', 'Net_Closing']].copy()
        df = df[df['Bank'].notna() & (df['Bank'] != '')]
        df = df[~df['Bank'].str.contains('SECTOR BANKS|BANKS', case=False, na=False)]
        
        # Clean numbers (remove commas)
        for col in ['Gross_Closing', 'Net_Closing']:
            df[col] = df[col].astype(str).str.replace(',', '').replace('-', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Store by bank
        for _, row in df.iterrows():
            bank = row['Bank'].strip()
            if bank in self.banks_data:
                self.banks_data[bank]['gross_npa'] = row['Gross_Closing']
                self.banks_data[bank]['net_npa'] = row['Net_Closing']
        
        logger.info(f"Processed NPA for {len(df)} banks")
        return df
    
    def clean_exposure_data(self):
        """Clean Sensitive Sector Exposures"""
        logger.info("Cleaning exposure data...")
        
        file_path = self.data_dir / "8.Exposure to Sensitive Sectors of Scheduled Commercial Banks.csv"
        
        df = pd.read_csv(file_path, skiprows=6)
        
        df.columns = ['Year', 'Bank', 'Capital_Market', 'Real_Estate', 'Commodities', 'Total_Exposure', 'Extra']
        df = df[['Year', 'Bank', 'Capital_Market', 'Real_Estate', 'Total_Exposure']].copy()
        df = df[df['Bank'].notna() & (df['Bank'] != '')]
        df = df[~df['Bank'].str.contains('SECTOR BANKS|BANKS', case=False, na=False)]
        
        # Clean numbers
        for col in ['Capital_Market', 'Real_Estate', 'Total_Exposure']:
            df[col] = df[col].astype(str).str.replace(',', '').replace('-', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Store by bank
        for _, row in df.iterrows():
            bank = row['Bank'].strip()
            if bank in self.banks_data:
                self.banks_data[bank]['capital_market_exp'] = row['Capital_Market']
                self.banks_data[bank]['real_estate_exp'] = row['Real_Estate']
                self.banks_data[bank]['total_sensitive_exp'] = row['Total_Exposure']
        
        logger.info(f"Processed exposure for {len(df)} banks")
        return df
    
    def create_ml_features(self):
        """Convert bank data to ML features"""
        logger.info("Creating ML features...")
        
        ml_records = []
        
        for bank_name, data in self.banks_data.items():
            # Skip if essential data missing
            if 'total_crar' not in data or pd.isna(data['total_crar']):
                continue
            
            # Extract features with defaults for missing values
            tier1 = data.get('tier1_capital', 15.0)
            tier2 = data.get('tier2_capital', 2.0)
            total_crar = data.get('total_crar', 17.0)
            gross_npa = data.get('gross_npa', 5000.0)
            net_npa = data.get('net_npa', 1000.0)
            capital_mkt = data.get('capital_market_exp', 0.0)
            real_estate = data.get('real_estate_exp', 50000.0)
            total_exp = data.get('total_sensitive_exp', 50000.0)
            
            # Normalize/scale features
            capital_ratio = total_crar / 100.0  # Convert percentage to ratio
            tier1_ratio = tier1 / 100.0
            
            # Estimate liquidity (inverse of NPA ratio)
            npa_ratio = min(net_npa / max(gross_npa, 1), 1.0)
            liquidity_buffer = max(0.1, 0.5 - npa_ratio)
            
            # Leverage (inverse of capital)
            leverage = 1.0 / max(capital_ratio, 0.01)
            
            # Credit exposure (normalized by sensitive sector exposure)
            credit_exposure = np.log1p(total_exp / 1000.0)  # Log scale, in thousands
            
            # Risk appetite (based on sensitive sector exposure)
            risk_appetite = min(0.9, (capital_mkt + real_estate) / max(total_exp, 1))
            
            # Stress level (based on NPA and capital adequacy)
            # Higher NPA and lower capital = higher stress
            stress_level = min(1.0, (npa_ratio * 2) + (1 - min(capital_ratio / 0.15, 1)))
            
            # Network features (synthetic - would need actual network data)
            # For now, use bank size as proxy
            bank_size = np.log1p(gross_npa + real_estate) / 20.0  # Log scaled
            degree_centrality = np.clip(bank_size, 0, 1)
            betweenness = degree_centrality * 0.5
            eigenvector = degree_centrality * 0.8
            pagerank = degree_centrality * 0.7
            in_degree = degree_centrality * 0.6
            out_degree = degree_centrality * 0.6
            
            # Market signals
            default_prob_prior = min(0.5, stress_level * 0.3)  # Prior estimate
            credit_spread = stress_level * 0.05  # Synthetic spread
            volatility = stress_level * 0.3
            market_pressure = (stress_level + risk_appetite) / 2
            
            # Neighborhood features (synthetic - would need network data)
            neighbor_avg_stress = stress_level * np.random.uniform(0.8, 1.2)
            neighbor_max_stress = min(1.0, stress_level * 1.3)
            neighbor_default_count = int(stress_level * 3)  # 0-3 neighbors defaulted
            neighbor_avg_capital = capital_ratio * np.random.uniform(0.9, 1.1)
            
            # Create label: default indicator
            # Banks with very low capital, high NPA, high stress -> label as risky (1)
            # This is synthetic but realistic
            default_indicator = int(
                (capital_ratio < 0.10) or 
                (npa_ratio > 0.8) or 
                (stress_level > 0.75)
            )
            
            record = {
                'institution_id': bank_name.lower().replace(' ', '_').replace('.', ''),
                'bank_name': bank_name,
                'timestamp': '2025-03-31',  # Latest data point
                # Financial features (6)
                'capital_ratio': capital_ratio,
                'liquidity_buffer': liquidity_buffer,
                'leverage': leverage,
                'credit_exposure': credit_exposure,
                'risk_appetite': risk_appetite,
                'stress_level': stress_level,
                # Network features (6)
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness,
                'eigenvector_centrality': eigenvector,
                'pagerank': pagerank,
                'in_degree': in_degree,
                'out_degree': out_degree,
                # Market signals (4)
                'default_probability_prior': default_prob_prior,
                'credit_spread': credit_spread,
                'volatility': volatility,
                'market_pressure': market_pressure,
                # Neighborhood features (4)
                'neighbor_avg_stress': neighbor_avg_stress,
                'neighbor_max_stress': neighbor_max_stress,
                'neighbor_default_count': neighbor_default_count,
                'neighbor_avg_capital_ratio': neighbor_avg_capital,
                # Label
                'defaulted': default_indicator,
                # Raw data for reference
                'raw_crar': total_crar,
                'raw_tier1': tier1,
                'raw_gross_npa': gross_npa,
                'raw_net_npa': net_npa,
            }
            
            ml_records.append(record)
        
        df = pd.DataFrame(ml_records)
        logger.info(f"Created {len(df)} ML-ready records")
        logger.info(f"Default rate: {df['defaulted'].mean():.2%}")
        
        return df
    
    def augment_with_noise(self, df: pd.DataFrame, n_copies: int = 5) -> pd.DataFrame:
        """Create augmented samples with noise for more training data"""
        logger.info(f"Augmenting data with {n_copies} noisy copies per bank...")
        
        augmented = []
        feature_cols = [col for col in df.columns if col not in 
                       ['institution_id', 'bank_name', 'timestamp', 'defaulted', 
                        'raw_crar', 'raw_tier1', 'raw_gross_npa', 'raw_net_npa']]
        
        for _, row in df.iterrows():
            # Add original
            augmented.append(row)
            
            # Add noisy copies
            for i in range(n_copies):
                noisy_row = row.copy()
                
                # Add Gaussian noise to features (5-10% std)
                for col in feature_cols:
                    if pd.notna(noisy_row[col]):
                        noise = np.random.normal(0, 0.05 * abs(noisy_row[col]))
                        noisy_row[col] = max(0, noisy_row[col] + noise)
                
                # Update institution ID
                noisy_row['institution_id'] = f"{row['institution_id']}_aug{i}"
                noisy_row['bank_name'] = f"{row['bank_name']} (Aug {i})"
                
                # Recalculate label based on noisy features
                noisy_row['defaulted'] = int(
                    (noisy_row['capital_ratio'] < 0.10) or 
                    (noisy_row['stress_level'] > 0.75)
                )
                
                augmented.append(noisy_row)
        
        result = pd.DataFrame(augmented)
        logger.info(f"Augmented to {len(result)} total samples")
        return result
    
    def process_all(self, augment: bool = True, n_copies: int = 5):
        """Process all data files and create final dataset"""
        logger.info("Starting RBI data processing...")
        
        # Clean each file
        self.clean_crar_data()
        self.clean_npa_data()
        self.clean_exposure_data()
        
        # Create ML features
        df = self.create_ml_features()
        
        # Augment data if requested
        if augment:
            df = self.augment_with_noise(df, n_copies=n_copies)
        
        # Reorder columns for ML training
        feature_cols = [
            'institution_id', 'bank_name', 'timestamp',
            # 20 ML features
            'capital_ratio', 'liquidity_buffer', 'leverage', 'credit_exposure', 
            'risk_appetite', 'stress_level',
            'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality',
            'pagerank', 'in_degree', 'out_degree',
            'default_probability_prior', 'credit_spread', 'volatility', 'market_pressure',
            'neighbor_avg_stress', 'neighbor_max_stress', 'neighbor_default_count',
            'neighbor_avg_capital_ratio',
            # Label
            'defaulted',
            # Reference columns
            'raw_crar', 'raw_tier1', 'raw_gross_npa', 'raw_net_npa',
        ]
        
        df = df[feature_cols]
        
        return df


def main():
    logger.info("=" * 60)
    logger.info("RBI Banking Data Cleaner")
    logger.info("=" * 60)
    
    # Process data
    cleaner = RBIDataCleaner()
    df = cleaner.process_all(augment=True, n_copies=5)
    
    # Save output
    output_file = cleaner.data_dir / "rbi_banks_ml_ready.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Saved cleaned data to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique banks: {df['bank_name'].str.extract(r'^([^(]+)')[0].nunique()}")
    logger.info(f"Default rate: {df['defaulted'].mean():.2%}")
    logger.info(f"Defaulted: {df['defaulted'].sum()}")
    logger.info(f"Non-defaulted: {(df['defaulted'] == 0).sum()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE STATISTICS")
    logger.info("=" * 60)
    
    feature_cols = [
        'capital_ratio', 'liquidity_buffer', 'leverage', 'credit_exposure',
        'stress_level', 'risk_appetite'
    ]
    
    for col in feature_cols:
        logger.info(f"{col:25s}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Train model:")
    logger.info(f"   python scripts/train_with_real_data.py --csv {output_file}")
    logger.info("\n2. Or train with hyperparameter search:")
    logger.info(f"   python scripts/train_with_real_data.py --csv {output_file} \\")
    logger.info("       --hyperparameter-search --epochs 100 --register")
    logger.info("\n3. Evaluate:")
    logger.info("   pytest tests/test_ml/ -v")


if __name__ == "__main__":
    main()
