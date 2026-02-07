"""
Data Loader Module

Loads and preprocesses all 7 datasets according to ML_Flow.md:
1. rbi_banks_ml_ready.csv - Core ML dataset
2. CRAR data - Capital strength
3. Bank Group-wise Ratios - Peer benchmarking
4. NPA Movements - Asset quality
5. Sensitive Sector Exposures - Network edges (sector)
6. Maturity Profile - Liquidity contagion
7. Yahoo Finance - Market data (optional)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetContainer:
    """Container for all loaded datasets"""
    ml_ready: pd.DataFrame = None
    crar: pd.DataFrame = None
    peer_ratios: pd.DataFrame = None
    npa_movements: pd.DataFrame = None
    sector_exposures: pd.DataFrame = None
    maturity_profile: pd.DataFrame = None
    market_data: Optional[pd.DataFrame] = None
    
    # Metadata
    banks: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    load_timestamp: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> str:
        """Generate summary of loaded data"""
        lines = [
            "=" * 60,
            "DATASET SUMMARY",
            "=" * 60,
            f"Load timestamp: {self.load_timestamp}",
            f"Banks: {len(self.banks)}",
            f"Years: {self.years}",
            "",
            "Dataset shapes:",
        ]
        
        if self.ml_ready is not None:
            lines.append(f"  - ML Ready: {self.ml_ready.shape}")
        if self.crar is not None:
            lines.append(f"  - CRAR: {self.crar.shape}")
        if self.peer_ratios is not None:
            lines.append(f"  - Peer Ratios: {self.peer_ratios.shape}")
        if self.npa_movements is not None:
            lines.append(f"  - NPA Movements: {self.npa_movements.shape}")
        if self.sector_exposures is not None:
            lines.append(f"  - Sector Exposures: {self.sector_exposures.shape}")
        if self.maturity_profile is not None:
            lines.append(f"  - Maturity Profile: {self.maturity_profile.shape}")
        if self.market_data is not None:
            lines.append(f"  - Market Data: {self.market_data.shape}")
            
        lines.append("=" * 60)
        return "\n".join(lines)


class DataLoader:
    """
    Load all RBI datasets for CCP risk modeling.
    
    CCP Perspective:
    - Each dataset provides different risk signals
    - Cross-validation across datasets
    - Time-aligned for temporal consistency
    """
    
    # File name mappings
    FILE_NAMES = {
        'ml_ready': 'rbi_banks_ml_ready.csv',
        'crar': '3.Bank-wise Capital Adequacy Ratios (CRAR) of Scheduled Commercial Banks.csv',
        'peer_ratios': '10.Bank Group-wise Select Ratios of Scheduled Commercial Banks.csv',
        'npa': '6.Movement of Non Performing Assets (NPAs) of Scheduled Commercial Banks.csv',
        'sector': '8.Exposure to Sensitive Sectors of Scheduled Commercial Banks.csv',
        'maturity': '9.Maturity Profile of Select Items of Liabilities and Assets of Scheduled Commercial Banks.csv',
    }
    
    def __init__(self, data_dir: str = None):
        """
        Initialize loader with data directory.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        if data_dir is None:
            # Default to data directory within this package
            data_dir = os.path.join(
                os.path.dirname(__file__), 
                'data'
            )
        self.data_dir = Path(data_dir)
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
    
    def load_all(self) -> DatasetContainer:
        """
        Load all datasets and return container.
        
        Returns:
            DatasetContainer with all loaded data
        """
        container = DatasetContainer()
        
        # Load each dataset
        container.ml_ready = self._load_ml_ready()
        container.crar = self._load_crar()
        container.peer_ratios = self._load_peer_ratios()
        container.npa_movements = self._load_npa()
        container.sector_exposures = self._load_sector_exposures()
        container.maturity_profile = self._load_maturity_profile()
        
        # Extract metadata
        if container.ml_ready is not None:
            container.banks = container.ml_ready['bank_name'].unique().tolist()
            
        if container.crar is not None:
            container.years = sorted(container.crar['year'].unique().tolist())
        
        logger.info(f"Loaded {len(container.banks)} banks across years {container.years}")
        return container
    
    def _load_ml_ready(self) -> pd.DataFrame:
        """Load Dataset 1: Core ML-ready dataset"""
        filepath = self.data_dir / self.FILE_NAMES['ml_ready']
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded ML ready data: {df.shape}")
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"ML ready file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading ML ready data: {e}")
            return None
    
    def _load_crar(self) -> pd.DataFrame:
        """
        Load Dataset 2: Capital Adequacy Ratios (CRAR)
        
        Purpose: Capital strength and loss-absorption calibration
        """
        filepath = self.data_dir / self.FILE_NAMES['crar']
        
        try:
            # Read CSV, skipping header rows
            df = pd.read_csv(filepath, skiprows=7)
            
            # Rename columns based on structure
            df.columns = [
                'empty', 'year', 'bank_name',
                'basel1_tier1', 'basel1_tier2', 'basel1_total',
                'basel2_tier1', 'basel2_tier2', 'basel2_total',
                'basel3_tier1', 'basel3_tier2', 'basel3_total'
            ]
            
            # Drop empty column
            df = df.drop('empty', axis=1)
            
            # Clean bank names
            df['bank_name'] = df['bank_name'].str.strip()
            
            # Forward fill year
            df['year'] = df['year'].ffill()
            
            # Convert year to int where possible
            df = df[df['year'].notna()]
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'].notna()]
            df['year'] = df['year'].astype(int)
            
            # Replace '..' with NaN and convert numerics
            for col in ['basel1_tier1', 'basel1_tier2', 'basel1_total',
                       'basel2_tier1', 'basel2_tier2', 'basel2_total',
                       'basel3_tier1', 'basel3_tier2', 'basel3_total']:
                df[col] = pd.to_numeric(df[col].replace('..', np.nan), errors='coerce')
            
            # Filter out category headers (rows where bank_name contains "BANKS" or "SECTOR")
            category_keywords = ['BANKS', 'SECTOR', 'ASSOCIATES']
            df = df[~df['bank_name'].str.contains('|'.join(category_keywords), case=False, na=False)]
            
            # Create unified CRAR columns (prioritize Basel III, then II, then I)
            df['crar_tier1'] = df['basel3_tier1'].fillna(df['basel2_tier1']).fillna(df['basel1_tier1'])
            df['crar_tier2'] = df['basel3_tier2'].fillna(df['basel2_tier2']).fillna(df['basel1_tier2'])
            df['crar_total'] = df['basel3_total'].fillna(df['basel2_total']).fillna(df['basel1_total'])
            
            logger.info(f"Loaded CRAR data: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"CRAR file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading CRAR data: {e}")
            return None
    
    def _load_peer_ratios(self) -> pd.DataFrame:
        """
        Load Dataset 3: Bank Group-wise Select Ratios
        
        Purpose: Macro-prudential context and peer benchmarking
        """
        filepath = self.data_dir / self.FILE_NAMES['peer_ratios']
        
        try:
            # Read CSV, skipping header rows
            df = pd.read_csv(filepath, skiprows=6)
            
            # Clean columns
            df.columns = ['empty1', 'empty2', 'year', 'ratio_name', 
                         'sbi_associates', 'nationalised', 'public_sector',
                         'private_sector', 'foreign', 'small_finance', 
                         'payments', 'all_scb']
            
            # Drop empty columns
            df = df.drop(['empty1', 'empty2'], axis=1)
            
            # Forward fill year
            df['year'] = df['year'].ffill()
            
            # Clean year (extract year from string like "2024-25")
            df['year'] = df['year'].astype(str).str.extract(r'(\d{4})').astype(float)
            df = df[df['year'].notna()]
            df['year'] = df['year'].astype(int) + 1  # "2024-25" means FY ending 2025
            
            # Clean ratio names
            df['ratio_name'] = df['ratio_name'].str.strip()
            df = df[df['ratio_name'].notna()]
            
            # Convert numeric columns
            numeric_cols = ['sbi_associates', 'nationalised', 'public_sector',
                           'private_sector', 'foreign', 'small_finance', 
                           'payments', 'all_scb']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded peer ratios data: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"Peer ratios file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading peer ratios data: {e}")
            return None
    
    def _load_npa(self) -> pd.DataFrame:
        """
        Load Dataset 4: Movement of Non-Performing Assets
        
        Purpose: Asset quality deterioration signal
        """
        filepath = self.data_dir / self.FILE_NAMES['npa']
        
        try:
            # Read CSV, skipping header rows
            df = pd.read_csv(filepath, skiprows=8)
            
            # Rename columns
            df.columns = ['empty', 'year', 'bank_name', 
                         'gross_npa_opening', 'gross_npa_additions',
                         'gross_npa_reductions', 'gross_npa_writeoffs',
                         'gross_npa_closing', 'net_npa_opening', 'net_npa_closing']
            
            # Drop empty column
            df = df.drop('empty', axis=1)
            
            # Forward fill year
            df['year'] = df['year'].ffill()
            
            # Convert year
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'].notna()]
            df['year'] = df['year'].astype(int)
            
            # Clean bank names
            df['bank_name'] = df['bank_name'].str.strip()
            
            # Filter out category headers
            category_keywords = ['BANKS', 'SECTOR', 'FINANCE', 'PAYMENT']
            df = df[~df['bank_name'].str.contains('|'.join(category_keywords), case=False, na=False)]
            df = df[df['bank_name'].notna()]
            
            # Clean numeric columns (remove commas)
            numeric_cols = ['gross_npa_opening', 'gross_npa_additions',
                           'gross_npa_reductions', 'gross_npa_writeoffs',
                           'gross_npa_closing', 'net_npa_opening', 'net_npa_closing']
            
            for col in numeric_cols:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('-', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Compute NPA growth rate
            df['npa_growth_rate'] = (
                (df['gross_npa_closing'] - df['gross_npa_opening']) / 
                df['gross_npa_opening'].replace(0, np.nan)
            )
            
            # Compute write-off intensity
            df['writeoff_intensity'] = (
                df['gross_npa_writeoffs'] / 
                df['gross_npa_opening'].replace(0, np.nan)
            )
            
            logger.info(f"Loaded NPA data: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"NPA file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading NPA data: {e}")
            return None
    
    def _load_sector_exposures(self) -> pd.DataFrame:
        """
        Load Dataset 5: Exposure to Sensitive Sectors
        
        Purpose: Primary network edge construction dataset (sector similarity)
        """
        filepath = self.data_dir / self.FILE_NAMES['sector']
        
        try:
            # Read CSV, skipping header rows
            df = pd.read_csv(filepath, skiprows=7)
            
            # Rename columns
            df.columns = ['empty', 'year', 'bank_name',
                         'capital_market', 'real_estate', 'commodities', 'total']
            
            # Drop empty column
            df = df.drop('empty', axis=1)
            
            # Forward fill year
            df['year'] = df['year'].ffill()
            
            # Convert year
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[df['year'].notna()]
            df['year'] = df['year'].astype(int)
            
            # Clean bank names
            df['bank_name'] = df['bank_name'].str.strip()
            
            # Filter out category headers
            category_keywords = ['BANKS', 'SECTOR']
            df = df[~df['bank_name'].str.contains('|'.join(category_keywords), case=False, na=False)]
            df = df[df['bank_name'].notna()]
            
            # Clean numeric columns
            numeric_cols = ['capital_market', 'real_estate', 'commodities', 'total']
            for col in numeric_cols:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('-', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Compute sector exposure proportions
            df['capital_market_pct'] = df['capital_market'] / df['total'].replace(0, np.nan)
            df['real_estate_pct'] = df['real_estate'] / df['total'].replace(0, np.nan)
            df['commodities_pct'] = df['commodities'] / df['total'].replace(0, np.nan)
            
            logger.info(f"Loaded sector exposure data: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"Sector exposure file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading sector exposure data: {e}")
            return None
    
    def _load_maturity_profile(self) -> pd.DataFrame:
        """
        Load Dataset 6: Maturity Profile of Select Items
        
        Purpose: Liquidity contagion channel
        """
        filepath = self.data_dir / self.FILE_NAMES['maturity']
        
        try:
            # This is a complex file - read and process
            df = pd.read_csv(filepath, low_memory=False)
            
            # For now, return raw data - will process in feature engineering
            logger.info(f"Loaded maturity profile data: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.warning(f"Maturity profile file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading maturity profile data: {e}")
            return None
    
    def load_market_data(
        self, 
        tickers: List[str] = None,
        start_date: str = '2018-01-01',
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load Dataset 7: Market data from Yahoo Finance
        
        Purpose: Market-implied systemic risk channel
        
        Args:
            tickers: List of stock tickers (bank stocks)
            start_date: Start date for data
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with price and return data
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed. Run: pip install yfinance")
            return None
        
        if tickers is None:
            # Default Indian bank tickers
            tickers = [
                'SBIN.NS',  # State Bank of India
                'HDFCBANK.NS',  # HDFC Bank
                'ICICIBANK.NS',  # ICICI Bank
                'KOTAKBANK.NS',  # Kotak Mahindra Bank
                'AXISBANK.NS',  # Axis Bank
                'INDUSINDBK.NS',  # IndusInd Bank
                'BANKBARODA.NS',  # Bank of Baroda
                'PNB.NS',  # Punjab National Bank
                'CANBK.NS',  # Canara Bank
                'IDFCFIRSTB.NS',  # IDFC First Bank
            ]
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Download data
            raw_df = yf.download(tickers, start=start_date, end=end_date)
            
            # Handle different yfinance column formats
            if isinstance(raw_df.columns, pd.MultiIndex):
                # New format: multi-level columns
                if 'Adj Close' in raw_df.columns.get_level_values(0):
                    df = raw_df['Adj Close']
                elif 'Close' in raw_df.columns.get_level_values(0):
                    df = raw_df['Close']
                else:
                    # Just get the first price column
                    df = raw_df.iloc[:, :len(tickers)]
            else:
                # Old format: single-level columns
                if 'Adj Close' in raw_df.columns:
                    df = raw_df['Adj Close']
                elif 'Close' in raw_df.columns:
                    df = raw_df['Close']
                else:
                    df = raw_df
            
            # Compute log returns
            returns = np.log(df / df.shift(1)).dropna()
            
            # Compute rolling volatility (21-day)
            volatility = returns.rolling(window=21).std() * np.sqrt(252)
            
            # Compute rolling correlations (latest snapshot)
            correlations = returns.corr()
            
            result = {
                'prices': df,
                'returns': returns,
                'volatility': volatility,
                'correlation': correlations
            }
            
            logger.info(f"Loaded market data for {len(tickers)} tickers")
            return result
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return None


# Convenience function
def load_data(data_dir: str = None) -> DatasetContainer:
    """
    Convenience function to load all data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DatasetContainer with all loaded data
    """
    loader = DataLoader(data_dir)
    return loader.load_all()


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    
    data = load_data()
    print(data.summary())
