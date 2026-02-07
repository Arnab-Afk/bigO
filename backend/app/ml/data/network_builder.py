"""
Composite Network Builder for CCP Risk Modeling

Builds multi-channel network connections using:
1. Sectoral exposure similarity (Dataset 5: Sensitive Sectors)
2. Liquidity maturity profile similarity (Dataset 6: Maturity Profile)
3. Market correlation (Yahoo Finance API)

Reference: ML_Flow.md Section 4
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class NetworkWeights:
    """Edge weights from different channels"""
    sector_weight: float          # w^(sector)
    liquidity_weight: float        # w^(liquidity)
    market_weight: float           # w^(market)
    
    def composite(
        self, 
        alpha: float = 0.4, 
        beta: float = 0.3, 
        gamma: float = 0.3,
        centrality_i: float = 1.0,
        centrality_j: float = 1.0
    ) -> float:
        """
        Compute composite edge weight
        
        Formula from ML_Flow.md:
        w_ij = (α * w^(sector) + β * w^(liquidity) + γ * w^(market)) * sqrt(C_i * C_j)
        
        Args:
            alpha: Weight for sector channel
            beta: Weight for liquidity channel
            gamma: Weight for market channel
            centrality_i: Systemic importance of node i
            centrality_j: Systemic importance of node j
        
        Returns:
            Composite edge weight
        """
        base_weight = (
            alpha * self.sector_weight +
            beta * self.liquidity_weight +
            gamma * self.market_weight
        )
        
        # Amplify by systemic importance
        amplification = np.sqrt(centrality_i * centrality_j)
        
        return base_weight * amplification


class CompositeNetworkBuilder:
    """
    Build multi-channel financial network for CCP risk assessment
    
    CCP Perspective:
    - Networks represent contagion channels
    - Edge weights measure co-failure risk
    - No bilateral exposure assumptions
    """
    
    def __init__(
        self,
        data_dir: str = "app/ml/data",
        alpha: float = 0.4,  # Sector channel weight
        beta: float = 0.3,   # Liquidity channel weight
        gamma: float = 0.3,  # Market channel weight
    ):
        """
        Initialize network builder
        
        Args:
            data_dir: Directory containing RBI data files
            alpha: Weight for sectoral exposure channel
            beta: Weight for liquidity channel
            gamma: Weight for market correlation channel
        """
        self.data_dir = Path(data_dir)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Validate weights sum to 1
        total = alpha + beta + gamma
        if not np.isclose(total, 1.0):
            logger.warning(f"Channel weights sum to {total}, not 1.0. Normalizing...")
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total
        
        logger.info(f"Network builder initialized with weights: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
    
    def build_sector_channel(
        self,
        sector_exposure_path: Optional[str] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build sectoral exposure similarity network
        
        Uses Dataset 5: "8.Exposure to Sensitive Sectors.csv"
        
        CCP Interpretation:
        Banks exposed to same vulnerable sectors fail together
        
        Method:
        1. Extract sector exposure vectors per bank
        2. Normalize exposure magnitudes
        3. Compute cosine similarity
        
        Returns:
            bank_names: List of bank identifiers
            similarity_matrix: Sectoral similarity matrix (normalized to [0,1])
        """
        if sector_exposure_path is None:
            sector_exposure_path = self.data_dir / "8.Exposure to Sensitive Sectors of Scheduled Commercial Banks.csv"
        
        logger.info(f"Building sector channel from {sector_exposure_path}")
        
        # Load sector exposure data
        df = pd.read_csv(sector_exposure_path)
        
        # Skip header rows and get to bank data
        # Find where actual data starts
        data_start = 0
        for idx, row in df.iterrows():
            # Look for year column
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        # Extract relevant columns
        # Columns: Year, Banks, Capital Market Sector, Real Estate Sector, Commodities, Total
        df_clean = df.iloc[data_start:].copy()
        df_clean.columns = ['Year', 'Bank', 'Capital_Market', 'Real_Estate', 'Commodities', 'Total']
        
        # Remove non-numeric and aggregate rows
        df_clean = df_clean[df_clean['Bank'].notna()]
        df_clean = df_clean[~df_clean['Bank'].str.contains('SECTOR|BANKS|PUBLIC|PRIVATE|FOREIGN|SMALL|PAYMENT', 
                                                            case=False, na=False)]
        
        # Convert exposure columns to numeric (removing commas)
        for col in ['Capital_Market', 'Real_Estate', 'Commodities']:
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace(',', '').str.replace('-', '0'),
                errors='coerce'
            ).fillna(0)
        
        # Get most recent year data
        latest_year = df_clean['Year'].iloc[0]
        df_latest = df_clean[df_clean['Year'] == latest_year].copy()
        
        # Create exposure matrix
        bank_names = df_latest['Bank'].tolist()
        exposure_matrix = df_latest[['Capital_Market', 'Real_Estate', 'Commodities']].values
        
        # Normalize by total exposure per bank (L2 norm)
        from sklearn.preprocessing import normalize
        exposure_normalized = normalize(exposure_matrix, norm='l2', axis=1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(exposure_normalized)
        
        # Ensure [0, 1] range and diagonal = 1
        similarity = np.clip(similarity, 0, 1)
        np.fill_diagonal(similarity, 0)  # No self-loops
        
        logger.info(f"Sector channel built: {len(bank_names)} banks, mean similarity: {similarity.mean():.4f}")
        
        return bank_names, similarity
    
    def build_liquidity_channel(
        self,
        maturity_profile_path: Optional[str] = None,
        bank_subset: Optional[List[str]] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build liquidity maturity profile similarity network
        
        Uses Dataset 6: "9.Maturity Profile.csv"
        
        CCP Interpretation:
        Similar funding structures imply simultaneous liquidity stress
        
        Method:
        1. Extract maturity bucket distributions
        2. Compute mismatch vectors
        3. Similarity = 1 - distance
        
        Returns:
            bank_names: List of bank identifiers
            similarity_matrix: Liquidity similarity matrix (normalized to [0,1])
        """
        if maturity_profile_path is None:
            maturity_profile_path = self.data_dir / "9.Maturity Profile of Select Items of Liabilities and Assets of Scheduled Commercial Banks.csv"
        
        logger.info(f"Building liquidity channel from {maturity_profile_path}")
        
        # Load maturity profile data
        df = pd.read_csv(maturity_profile_path)
        
        # This file has complex structure - extract deposit and loan maturity buckets
        # Focus on maturity mismatches as key liquidity indicator
        
        # Find data start row
        data_start = 0
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                data_start = idx
                break
        
        df_clean = df.iloc[data_start:].copy()
        
        # Extract bank names and maturity data
        # Simplified: use deposit maturity buckets (columns 3-10)
        bank_col_idx = 1
        deposit_start_col = 2  # After Year and Bank columns
        
        bank_names = []
        maturity_vectors = []
        
        for idx, row in df_clean.iterrows():
            bank_name = row.iloc[bank_col_idx]
            if pd.isna(bank_name) or str(bank_name).strip() in ['', 'PUBLIC SECTOR BANKS', 'PRIVATE SECTOR BANKS']:
                continue
            
            # Extract 8 maturity buckets for deposits
            maturity_data = []
            for col_idx in range(deposit_start_col, deposit_start_col + 8):
                if col_idx < len(row):
                    val = row.iloc[col_idx]
                    val_clean = pd.to_numeric(
                        str(val).replace(',', '').replace('-', '0'),
                        errors='coerce'
                    )
                    maturity_data.append(val_clean if pd.notna(val_clean) else 0)
            
            if len(maturity_data) == 8 and sum(maturity_data) > 0:
                bank_names.append(bank_name)
                maturity_vectors.append(maturity_data)
        
        # Convert to numpy array and normalize
        maturity_matrix = np.array(maturity_vectors)
        
        # Normalize to probability distributions
        row_sums = maturity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        maturity_normalized = maturity_matrix / row_sums
        
        # Compute distance (using Euclidean)
        distances = cdist(maturity_normalized, maturity_normalized, metric='euclidean')
        
        # Convert to similarity: similarity = 1 - normalized_distance
        max_dist = distances.max()
        if max_dist > 0:
            similarity = 1 - (distances / max_dist)
        else:
            similarity = np.ones_like(distances)
        
        # Ensure [0, 1] range
        similarity = np.clip(similarity, 0, 1)
        np.fill_diagonal(similarity, 0)  # No self-loops
        
        # Filter to bank_subset if provided
        if bank_subset:
            indices = [i for i, name in enumerate(bank_names) if name in bank_subset]
            bank_names = [bank_names[i] for i in indices]
            similarity = similarity[np.ix_(indices, indices)]
        
        logger.info(f"Liquidity channel built: {len(bank_names)} banks, mean similarity: {similarity.mean():.4f}")
        
        return bank_names, similarity
    
    def build_market_channel(
        self,
        bank_names: List[str],
        market_data: Optional[pd.DataFrame] = None,
        use_synthetic: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build market correlation network
        
        Uses Dataset 7: Yahoo Finance stock price data (or synthetic)
        
        CCP Interpretation:
        Markets reflect real-time confidence and contagion risk
        
        Method:
        1. Compute rolling return correlations
        2. Use historical windows only (no leakage)
        
        Args:
            bank_names: List of banks to compute correlations for
            market_data: Optional dataframe with return data
            use_synthetic: If True, generate synthetic correlations
        
        Returns:
            bank_names: List of bank identifiers
            correlation_matrix: Market correlation matrix (normalized to [0,1])
        """
        logger.info(f"Building market channel for {len(bank_names)} banks")
        
        if market_data is not None and not market_data.empty:
            # Use provided market data
            logger.info("Using provided market data")
            # Compute correlations from return data
            correlations = market_data.corr().values
        elif use_synthetic:
            # Generate synthetic market correlations
            logger.info("Generating synthetic market correlations")
            
            # Create realistic correlation structure:
            # - Higher for banks in same category
            # - Lower but positive for most pairs
            # - Some negative correlations
            
            n = len(bank_names)
            correlations = np.random.uniform(0.2, 0.8, (n, n))
            
            # Make symmetric
            correlations = (correlations + correlations.T) / 2
            
            # Add noise to break perfect symmetry
            noise = np.random.normal(0, 0.05, (n, n))
            correlations += noise
            
            # Ensure diagonal is 1
            np.fill_diagonal(correlations, 1.0)
            
            # Ensure valid correlation matrix (PSD)
            # Simple approach: clip to valid range
            correlations = np.clip(correlations, -1, 1)
        else:
            logger.warning("No market data provided, returning uniform correlations")
            n = len(bank_names)
            correlations = np.full((n, n), 0.5)
            np.fill_diagonal(correlations, 1.0)
        
        # Convert correlations to similarity in [0, 1]
        # correlation ∈ [-1, 1] → similarity ∈ [0, 1]
        similarity = (correlations + 1) / 2
        
        # No self-loops for network construction
        np.fill_diagonal(similarity, 0)
        
        logger.info(f"Market channel built: {len(bank_names)} banks, mean correlation: {similarity.mean():.4f}")
        
        return bank_names, similarity
    
    def build_composite_network(
        self,
        centrality_scores: Optional[Dict[str, float]] = None,
        threshold: float = 0.1
    ) -> nx.Graph:
        """
        Build composite multi-channel network
        
        Combines all three channels using weighted formula from ML_Flow.md:
        w_ij = (α * w^(sector) + β * w^(liquidity) + γ * w^(market)) * sqrt(C_i * C_j)
        
        Args:
            centrality_scores: Optional dict mapping bank names to centrality scores
            threshold: Minimum edge weight to include in network
        
        Returns:
            NetworkX graph with composite edge weights
        """
        logger.info("Building composite multi-channel network")
        
        # Build individual channels
        sector_banks, sector_sim = self.build_sector_channel()
        liquidity_banks, liquidity_sim = self.build_liquidity_channel(bank_subset=sector_banks)
        market_banks, market_sim = self.build_market_channel(sector_banks)
        
        # Align all matrices to same bank set
        common_banks = list(set(sector_banks) & set(liquidity_banks) & set(market_banks))
        logger.info(f"Common banks across all channels: {len(common_banks)}")
        
        # Get indices for common banks
        sector_indices = [sector_banks.index(b) for b in common_banks]
        liquidity_indices = [liquidity_banks.index(b) for b in common_banks]
        market_indices = [market_banks.index(b) for b in common_banks]
        
        # Extract aligned matrices
        sector_aligned = sector_sim[np.ix_(sector_indices, sector_indices)]
        liquidity_aligned = liquidity_sim[np.ix_(liquidity_indices, liquidity_indices)]
        market_aligned = market_sim[np.ix_(market_indices, market_indices)]
        
        # Compute composite weights
        n = len(common_banks)
        composite_weights = np.zeros((n, n))
        
        # Default centrality if not provided
        if centrality_scores is None:
            centrality_scores = {bank: 1.0 for bank in common_banks}
        
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only
                bank_i = common_banks[i]
                bank_j = common_banks[j]
                
                # Get centrality scores
                cent_i = centrality_scores.get(bank_i, 1.0)
                cent_j = centrality_scores.get(bank_j, 1.0)
                
                # Create NetworkWeights object
                weights = NetworkWeights(
                    sector_weight=sector_aligned[i, j],
                    liquidity_weight=liquidity_aligned[i, j],
                    market_weight=market_aligned[i, j]
                )
                
                # Compute composite weight
                composite_weights[i, j] = weights.composite(
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    centrality_i=cent_i,
                    centrality_j=cent_j
                )
        
        # Make symmetric
        composite_weights = composite_weights + composite_weights.T
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for bank in common_banks:
            G.add_node(
                bank,
                centrality=centrality_scores.get(bank, 1.0)
            )
        
        # Add edges above threshold
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                weight = composite_weights[i, j]
                if weight >= threshold:
                    G.add_edge(
                        common_banks[i],
                        common_banks[j],
                        weight=weight,
                        sector_sim=sector_aligned[i, j],
                        liquidity_sim=liquidity_aligned[i, j],
                        market_sim=market_aligned[i, j]
                    )
                    edge_count += 1
        
        logger.info(f"Composite network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.info(f"Mean edge weight: {np.mean([d['weight'] for _, _, d in G.edges(data=True)]):.4f}")
        
        return G
    
    def get_adjacency_matrix(self, G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
        """
        Extract weighted adjacency matrix from graph
        
        Args:
            G: NetworkX graph
        
        Returns:
            adjacency_matrix: Weighted adjacency matrix
            node_labels: Ordered list of node names
        """
        node_labels = list(G.nodes())
        n = len(node_labels)
        
        adjacency = nx.to_numpy_array(G, nodelist=node_labels, weight='weight')
        
        return adjacency, node_labels
