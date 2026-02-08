"""
Initial State Loader for Financial Ecosystem
============================================
Calibrates the simulation using real-world data from CSV files.

Data Mapping:
- bank_crar.csv -> Initial capital and risk-weighted assets
- bank_npa.csv -> Initial NPA ratios
- bank_sensitive_sector.csv -> Bank-Sector exposure edges
- bank_maturity_profile.csv -> Liquidity mismatch calculation
- reverse_repo.csv -> Global liquidity state
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .agents import BankAgent, SectorAgent, CCPAgent, RegulatorAgent
from .simulation_engine import FinancialEcosystem, SimulationConfig

logger = logging.getLogger(__name__)


class InitialStateLoader:
    """
    Loads CSV data and constructs a calibrated FinancialEcosystem.
    """
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        
        # Data containers
        self.bank_crar_df: Optional[pd.DataFrame] = None
        self.bank_npa_df: Optional[pd.DataFrame] = None
        self.bank_sector_df: Optional[pd.DataFrame] = None
        self.bank_maturity_df: Optional[pd.DataFrame] = None
        self.reverse_repo_df: Optional[pd.DataFrame] = None
        
        logger.info(f"InitialStateLoader initialized with data_dir: {data_dir}")
    
    def load_all_data(self) -> None:
        """
        Load all CSV files into memory.
        """
        try:
            # Bank CRAR (Capital Adequacy)
            crar_path = self.data_dir / "bank_crar.csv"
            if crar_path.exists():
                self.bank_crar_df = pd.read_csv(crar_path)
                logger.info(f"Loaded bank_crar.csv: {len(self.bank_crar_df)} rows")
            else:
                logger.warning(f"bank_crar.csv not found at {crar_path}")
            
            # Bank NPA (Non-Performing Assets)
            npa_path = self.data_dir / "bank_npa.csv"
            if npa_path.exists():
                self.bank_npa_df = pd.read_csv(npa_path)
                logger.info(f"Loaded bank_npa.csv: {len(self.bank_npa_df)} rows")
            else:
                logger.warning(f"bank_npa.csv not found at {npa_path}")
            
            # Bank-Sector Exposures
            sector_path = self.data_dir / "bank_sensitive_sector.csv"
            if sector_path.exists():
                self.bank_sector_df = pd.read_csv(sector_path)
                logger.info(f"Loaded bank_sensitive_sector.csv: {len(self.bank_sector_df)} rows")
            else:
                logger.warning(f"bank_sensitive_sector.csv not found at {sector_path}")
            
            # Bank Maturity Profile
            maturity_path = self.data_dir / "bank_maturity_profile.csv"
            if maturity_path.exists():
                self.bank_maturity_df = pd.read_csv(maturity_path)
                logger.info(f"Loaded bank_maturity_profile.csv: {len(self.bank_maturity_df)} rows")
            else:
                logger.warning(f"bank_maturity_profile.csv not found at {maturity_path}")
            
            # Reverse Repo (Liquidity)
            rrepo_path = self.data_dir / "reverse_repo.csv"
            if rrepo_path.exists():
                self.reverse_repo_df = pd.read_csv(rrepo_path)
                logger.info(f"Loaded reverse_repo.csv: {len(self.reverse_repo_df)} rows")
            else:
                logger.warning(f"reverse_repo.csv not found at {rrepo_path}")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_latest_year_data(self, df: pd.DataFrame, year_column: str = 'Year') -> pd.DataFrame:
        """
        Extract data from the most recent year in the dataset.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        if year_column in df.columns:
            latest_year = df[year_column].max()
            return df[df[year_column] == latest_year]
        else:
            # If no year column, return all data
            return df
    
    def create_bank_agents(self) -> Dict[str, BankAgent]:
        """
        Create BankAgent instances from CRAR and NPA data.
        
        Returns:
            Dict mapping bank_id -> BankAgent
        """
        banks = {}
        
        if self.bank_crar_df is None:
            logger.warning("No CRAR data available. Using synthetic banks.")
            return self._create_synthetic_banks()
        
        # Get latest year data
        crar_latest = self.get_latest_year_data(self.bank_crar_df)
        
        # Try to identify bank name column (varies by dataset)
        possible_bank_cols = ['Bank', 'Bank Name', 'BankName', 'Institution', 'bank_name']
        bank_col = None
        for col in possible_bank_cols:
            if col in crar_latest.columns:
                bank_col = col
                break
        
        if bank_col is None:
            logger.error("Could not identify bank name column in CRAR data")
            return self._create_synthetic_banks()
        
        # Extract unique banks
        unique_banks = crar_latest[bank_col].unique()
        
        for bank_name in unique_banks:
            bank_data = crar_latest[crar_latest[bank_col] == bank_name].iloc[0]
            
            # Extract CRAR (try different column names)
            crar_value = None
            for crar_col in ['CRAR', 'Capital Adequacy', 'CAR', 'crar']:
                if crar_col in bank_data.index:
                    crar_value = float(bank_data[crar_col])
                    break
            
            if crar_value is None:
                crar_value = 12.0  # Default
            
            # Calculate capital and RWA from CRAR
            # CRAR = (Capital / RWA) * 100
            # Assume RWA as a base value, then derive capital
            assumed_rwa = 10000.0  # Base scale
            capital = (crar_value / 100.0) * assumed_rwa
            
            # Get NPA ratio if available
            npa_ratio = 0.0
            if self.bank_npa_df is not None:
                npa_latest = self.get_latest_year_data(self.bank_npa_df)
                if bank_col in npa_latest.columns:
                    bank_npa = npa_latest[npa_latest[bank_col] == bank_name]
                    if not bank_npa.empty:
                        # Try different NPA column names
                        for npa_col in ['Gross NPA', 'NPA', 'GrossNPA', 'npa_ratio']:
                            if npa_col in bank_npa.columns:
                                npa_ratio = float(bank_npa.iloc[0][npa_col])
                                break
            
            # Calculate liquidity from maturity profile (if available)
            liquidity = capital * 0.2  # Default: 20% of capital
            if self.bank_maturity_df is not None:
                maturity_latest = self.get_latest_year_data(self.bank_maturity_df)
                if bank_col in maturity_latest.columns:
                    bank_maturity = maturity_latest[maturity_latest[bank_col] == bank_name]
                    if not bank_maturity.empty:
                        # Sum short-term assets (high liquidity)
                        for col in ['1-14 days', '15-28 days', 'short_term']:
                            if col in bank_maturity.columns:
                                liquidity += float(bank_maturity.iloc[0][col])
            
            # Create BankAgent
            bank_id = bank_name.replace(' ', '_').upper()
            bank_agent = BankAgent(
                agent_id=bank_id,
                initial_capital=capital,
                initial_assets=assumed_rwa,
                initial_liquidity=max(liquidity, capital * 0.1),  # At least 10% of capital
                initial_npa_ratio=npa_ratio,
                initial_crar=crar_value,
                regulatory_min_crar=9.0
            )
            
            banks[bank_id] = bank_agent
            logger.info(f"Created {bank_id}: Capital={capital:.2f}, CRAR={crar_value:.2f}%, NPA={npa_ratio:.2f}%")
        
        return banks
    
    def _create_synthetic_banks(self, num_banks: int = 10) -> Dict[str, BankAgent]:
        """
        Fallback: Create synthetic banks if real data is unavailable.
        """
        banks = {}
        
        for i in range(num_banks):
            bank_id = f"BANK_{i+1}"
            
            # Vary parameters to create heterogeneity
            capital = 1000 + i * 300 + np.random.uniform(-200, 200)
            rwa = capital * (8 + np.random.uniform(2, 6))
            crar = (capital / rwa) * 100
            npa_ratio = 2.0 + np.random.uniform(0, 8)
            liquidity = capital * np.random.uniform(0.15, 0.30)
            
            bank_agent = BankAgent(
                agent_id=bank_id,
                initial_capital=capital,
                initial_assets=rwa,
                initial_liquidity=liquidity,
                initial_npa_ratio=npa_ratio,
                initial_crar=crar
            )
            
            banks[bank_id] = bank_agent
        
        logger.info(f"Created {num_banks} synthetic banks")
        return banks
    
    def create_sector_agents(self) -> Dict[str, SectorAgent]:
        """
        Create SectorAgent instances from bank_sensitive_sector.csv.
        
        Returns:
            Dict mapping sector_id -> SectorAgent
        """
        sectors = {}
        
        if self.bank_sector_df is None:
            logger.warning("No sector exposure data. Creating default sectors.")
            return self._create_default_sectors()
        
        # Identify sector columns (exclude Bank, Year, etc.)
        sector_latest = self.get_latest_year_data(self.bank_sector_df)
        
        # Common non-sector columns
        exclude_cols = ['Bank', 'Bank Name', 'Year', 'Total', 'bank_name', 'BankName']
        sector_columns = [col for col in sector_latest.columns if col not in exclude_cols]
        
        # Create a sector agent for each sector column
        for sector_name in sector_columns:
            sector_id = f"SECTOR_{sector_name.replace(' ', '_').upper()}"
            
            # Calculate average exposure to gauge sector size
            avg_exposure = sector_latest[sector_name].mean()
            
            # Higher exposure sectors are assumed to be "larger" (more debt)
            # Health inversely related to size (larger sectors more fragile)
            health = 0.9 - (avg_exposure / (sector_latest[sector_columns].sum(axis=1).mean() + 1e-6)) * 0.3
            health = max(0.5, min(1.0, health))
            
            sector_agent = SectorAgent(
                agent_id=sector_id,
                sector_name=sector_name,
                initial_health=health,
                base_volatility=0.1
            )
            
            sectors[sector_id] = sector_agent
            logger.info(f"Created {sector_id}: Health={health:.2f}")
        
        return sectors
    
    def _create_default_sectors(self) -> Dict[str, SectorAgent]:
        """
        Fallback: Create default sector agents.
        """
        default_sectors = {
            'Real Estate': 0.75,
            'Capital Market': 0.80,
            'Commodities': 0.70,
            'Infrastructure': 0.78,
            'Aviation': 0.65
        }
        
        sectors = {}
        for sector_name, health in default_sectors.items():
            sector_id = f"SECTOR_{sector_name.replace(' ', '_').upper()}"
            sectors[sector_id] = SectorAgent(
                agent_id=sector_id,
                sector_name=sector_name,
                initial_health=health
            )
        
        logger.info(f"Created {len(sectors)} default sectors")
        return sectors
    
    def create_bank_sector_exposures(
        self,
        banks: Dict[str, BankAgent],
        sectors: Dict[str, SectorAgent]
    ) -> List[Tuple[str, str, float]]:
        """
        Create bank -> sector exposure edges from bank_sensitive_sector.csv.
        
        Returns:
            List of (bank_id, sector_id, exposure_amount) tuples
        """
        exposures = []
        
        if self.bank_sector_df is None:
            logger.warning("No exposure data. Creating random exposures.")
            return self._create_random_exposures(banks, sectors)
        
        sector_latest = self.get_latest_year_data(self.bank_sector_df)
        
        # Identify bank column
        possible_bank_cols = ['Bank', 'Bank Name', 'BankName', 'bank_name']
        bank_col = None
        for col in possible_bank_cols:
            if col in sector_latest.columns:
                bank_col = col
                break
        
        if bank_col is None:
            logger.warning("Could not find bank column in sector exposure data")
            return self._create_random_exposures(banks, sectors)
        
        # Map sector names to sector_ids
        sector_name_to_id = {
            s.sector_name: sid for sid, s in sectors.items()
        }
        
        for _, row in sector_latest.iterrows():
            bank_name = row[bank_col]
            bank_id = bank_name.replace(' ', '_').upper()
            
            if bank_id not in banks:
                continue  # Skip if bank not in our system
            
            # For each sector column
            for sector_name, sector_id in sector_name_to_id.items():
                if sector_name in row.index:
                    exposure = float(row[sector_name])
                    
                    if exposure > 0:  # Only create edge if positive exposure
                        exposures.append((bank_id, sector_id, exposure))
        
        logger.info(f"Created {len(exposures)} bank-sector exposures from data")
        return exposures
    
    def _create_random_exposures(
        self,
        banks: Dict[str, BankAgent],
        sectors: Dict[str, SectorAgent]
    ) -> List[Tuple[str, str, float]]:
        """
        Fallback: Create random exposures.
        """
        exposures = []
        
        for bank_id, bank in banks.items():
            # Each bank lends to 2-4 random sectors
            num_sectors = np.random.randint(2, min(5, len(sectors) + 1))
            selected_sectors = np.random.choice(list(sectors.keys()), size=num_sectors, replace=False)
            
            for sector_id in selected_sectors:
                # Exposure is a fraction of bank's capital
                exposure = bank.capital * np.random.uniform(1.5, 3.0)
                exposures.append((bank_id, sector_id, exposure))
        
        logger.info(f"Created {len(exposures)} random bank-sector exposures")
        return exposures
    
    def create_interbank_exposures(self, banks: Dict[str, BankAgent]) -> List[Tuple[str, str, float]]:
        """
        Create interbank lending network (stylized).
        Uses a random network model with preferential attachment.
        
        Returns:
            List of (creditor_bank_id, debtor_bank_id, exposure) tuples
        """
        exposures = []
        bank_ids = list(banks.keys())
        
        if len(bank_ids) < 2:
            return exposures
        
        # Create a sparse interbank network
        # Larger banks lend to smaller banks
        sorted_banks = sorted(bank_ids, key=lambda b: banks[b].capital, reverse=True)
        
        for i, creditor_id in enumerate(sorted_banks[:-1]):
            # Each bank lends to 1-3 other banks
            num_debtors = min(3, len(sorted_banks) - i - 1)
            
            for j in range(num_debtors):
                debtor_id = sorted_banks[i + j + 1]
                
                # Exposure is a fraction of creditor's interbank limit
                creditor = banks[creditor_id]
                exposure = creditor.interbank_limit * np.random.uniform(0.2, 0.5)
                
                exposures.append((creditor_id, debtor_id, exposure))
        
        logger.info(f"Created {len(exposures)} interbank exposures")
        return exposures
    
    def get_global_liquidity(self) -> float:
        """
        Calculate global liquidity from reverse_repo.csv.
        
        Returns:
            Normalized liquidity value [0, 1]
        """
        if self.reverse_repo_df is None:
            return 1.0  # Default: ample liquidity
        
        repo_latest = self.get_latest_year_data(self.reverse_repo_df)
        
        # Try to find the reverse repo amount column
        for col in ['Amount', 'Volume', 'Reverse Repo', 'reverse_repo']:
            if col in repo_latest.columns:
                avg_amount = repo_latest[col].mean()
                
                # Normalize (very rough heuristic)
                # Higher reverse repo usage = tighter liquidity
                # Assuming typical range is 0 to 100,000 (depends on your data scale)
                normalized = 1.0 - (avg_amount / 100000.0)
                return max(0.3, min(1.0, normalized))
        
        return 1.0
    
    def build_ecosystem(self, config: SimulationConfig) -> FinancialEcosystem:
        """
        Complete pipeline: Load data and construct a fully initialized FinancialEcosystem.
        
        Returns:
            FinancialEcosystem ready to simulate
        """
        logger.info("=" * 60)
        logger.info("Building Financial Ecosystem from Real Data")
        logger.info("=" * 60)
        
        # Load data
        self.load_all_data()
        
        # Create ecosystem
        ecosystem = FinancialEcosystem(config)
        
        # Create agents
        banks = self.create_bank_agents()
        sectors = self.create_sector_agents()
        
        # Add banks
        for bank in banks.values():
            ecosystem.add_agent(bank)
        
        # Add sectors
        for sector in sectors.values():
            ecosystem.add_agent(sector)
        
        # Add CCP (Central Counterparty)
        total_bank_capital = sum(b.capital for b in banks.values())
        ccp = CCPAgent(
            agent_id="CCP_MAIN",
            initial_default_fund=total_bank_capital * 0.05,  # 5% of total capital
            initial_margin_requirement=10.0
        )
        ecosystem.add_agent(ccp)
        
        # Add Regulator
        base_rate = 6.0  # Default repo rate
        regulator = RegulatorAgent("RBI", base_repo_rate=base_rate, min_crar=9.0)
        ecosystem.add_agent(regulator)
        
        # Create exposures
        bank_sector_exposures = self.create_bank_sector_exposures(banks, sectors)
        for creditor, debtor, amount in bank_sector_exposures:
            ecosystem.add_exposure(creditor, debtor, amount, edge_type="sector_loan")
        
        interbank_exposures = self.create_interbank_exposures(banks)
        for creditor, debtor, amount in interbank_exposures:
            ecosystem.add_exposure(creditor, debtor, amount, edge_type="interbank_loan")
        
        # Set global liquidity
        global_liquidity = self.get_global_liquidity()
        ecosystem.global_state['system_liquidity'] = global_liquidity
        
        logger.info("=" * 60)
        logger.info(f"Ecosystem built successfully!")
        logger.info(f"  Banks: {len(banks)}")
        logger.info(f"  Sectors: {len(sectors)}")
        logger.info(f"  Bank-Sector Exposures: {len(bank_sector_exposures)}")
        logger.info(f"  Interbank Exposures: {len(interbank_exposures)}")
        logger.info(f"  Global Liquidity: {global_liquidity:.2f}")
        logger.info("=" * 60)
        
        return ecosystem


# Convenience function
def load_ecosystem_from_data(
    data_dir: str,
    max_timesteps: int = 100,
    enable_shocks: bool = True,
    random_seed: Optional[int] = None
) -> FinancialEcosystem:
    """
    One-liner to load a complete ecosystem from CSV data.
    
    Example:
        ecosystem = load_ecosystem_from_data("backend/ccp_ml/data", max_timesteps=50)
        snapshots = ecosystem.run()
    """
    config = SimulationConfig(
        max_timesteps=max_timesteps,
        enable_shocks=enable_shocks,
        random_seed=random_seed
    )
    
    loader = InitialStateLoader(Path(data_dir))
    return loader.build_ecosystem(config)
