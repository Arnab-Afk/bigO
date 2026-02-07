"""
Network Builder Module

Constructs the interdependence network according to ML_Flow.md specification.

Edge Types (all inferred, no bilateral exposures):
1. Sector similarity edges - Banks with similar sensitive sector exposures
2. Liquidity contagion edges - Banks with similar maturity profiles
3. Market correlation edges - Banks with correlated stock returns

Composite Weight: w_ij = α·sector_sim + β·liquidity_sim + γ·market_corr
Default: α=0.3, β=0.4, γ=0.3
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from .data_loader import DatasetContainer

logger = logging.getLogger(__name__)


@dataclass
class NetworkEdge:
    """Represents an edge in the interdependence network"""
    source: str
    target: str
    weight: float
    sector_similarity: float = 0.0
    liquidity_similarity: float = 0.0
    market_correlation: float = 0.0


class NetworkBuilder:
    """
    Builds the bank interdependence network for CCP risk modeling.
    
    Key insight from ML_Flow.md: We don't have bilateral exposure data,
    so we infer connections through:
    - Similar risk exposures (sector)
    - Similar liquidity profiles (maturity)
    - Market correlations (stock prices)
    """
    
    def __init__(
        self,
        sector_weight: float = 0.3,
        liquidity_weight: float = 0.4,
        market_weight: float = 0.3,
        edge_threshold: float = 0.1  # Lowered from 0.3 to create more edges
    ):
        """
        Initialize network builder.
        
        Args:
            sector_weight: Weight for sector similarity (α)
            liquidity_weight: Weight for liquidity similarity (β)
            market_weight: Weight for market correlation (γ)
            edge_threshold: Minimum composite weight to create edge
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required. Install with: pip install networkx")
        
        self.sector_weight = sector_weight
        self.liquidity_weight = liquidity_weight
        self.market_weight = market_weight
        self.edge_threshold = edge_threshold
        
        self.graph = None
        self.edges = []
    
    def build_network(
        self,
        data: DatasetContainer,
        year: int = None,
        market_data: Dict = None
    ) -> 'nx.DiGraph':
        """
        Build the complete interdependence network.
        
        Args:
            data: DatasetContainer with all loaded data
            year: Specific year to build network for
            market_data: Optional pre-computed market correlations
            
        Returns:
            NetworkX DiGraph with bank nodes and weighted edges
        """
        # Get list of banks
        banks = data.banks if data.banks else []
        if not banks and data.ml_ready is not None:
            banks = data.ml_ready['bank_name'].unique().tolist()
        
        if len(banks) < 2:
            logger.warning("Need at least 2 banks to build network")
            return nx.DiGraph()
        
        # Initialize graph
        self.graph = nx.DiGraph()
        for bank in banks:
            self.graph.add_node(bank)
        
        # Compute each similarity type
        sector_sim = self._compute_sector_similarity(data, year)
        liquidity_sim = self._compute_liquidity_similarity(data, year)
        market_corr = self._compute_market_correlation(market_data)
        
        # Create edges
        self.edges = []
        for i, bank_i in enumerate(banks):
            for j, bank_j in enumerate(banks):
                if i >= j:  # Skip self-loops and duplicate pairs
                    continue
                
                # Get similarity scores
                key = (bank_i, bank_j)
                s_sim = sector_sim.get(key, 0.0)
                l_sim = liquidity_sim.get(key, 0.0)
                m_corr = market_corr.get(key, 0.0)
                
                # Compute composite weight
                weight = (
                    self.sector_weight * s_sim +
                    self.liquidity_weight * l_sim +
                    self.market_weight * m_corr
                )
                
                # Add edge if above threshold
                if weight >= self.edge_threshold:
                    edge = NetworkEdge(
                        source=bank_i,
                        target=bank_j,
                        weight=weight,
                        sector_similarity=s_sim,
                        liquidity_similarity=l_sim,
                        market_correlation=m_corr
                    )
                    self.edges.append(edge)
                    
                    # Add bidirectional edges (undirected relationship)
                    self.graph.add_edge(bank_i, bank_j, weight=weight, 
                                       sector_sim=s_sim, liquidity_sim=l_sim, market_corr=m_corr)
                    self.graph.add_edge(bank_j, bank_i, weight=weight,
                                       sector_sim=s_sim, liquidity_sim=l_sim, market_corr=m_corr)
        
        logger.info(f"Built network: {len(banks)} nodes, {len(self.edges)} edges")
        return self.graph
    
    def _compute_sector_similarity(
        self, 
        data: DatasetContainer, 
        year: int = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute sector similarity between bank pairs.
        
        Banks with similar exposure profiles to sensitive sectors
        (capital market, real estate, commodities) are more connected.
        """
        if data.sector_exposures is None:
            return {}
        
        df = data.sector_exposures.copy()
        
        # Filter by year if specified
        if year is not None:
            df = df[df['year'] == year]
        else:
            # Use latest year
            df = df[df['year'] == df['year'].max()]
        
        # Create exposure vectors
        exposure_cols = ['capital_market_pct', 'real_estate_pct', 'commodities_pct']
        exposure_cols = [c for c in exposure_cols if c in df.columns]
        
        if not exposure_cols:
            return {}
        
        # Fill NaN with 0
        df[exposure_cols] = df[exposure_cols].fillna(0)
        
        # Compute pairwise cosine similarity
        similarity = {}
        banks = df['bank_name'].unique()
        
        for i, bank_i in enumerate(banks):
            vec_i = df[df['bank_name'] == bank_i][exposure_cols].values
            if len(vec_i) == 0:
                continue
            vec_i = vec_i[0]
            
            for j, bank_j in enumerate(banks):
                if i >= j:
                    continue
                
                vec_j = df[df['bank_name'] == bank_j][exposure_cols].values
                if len(vec_j) == 0:
                    continue
                vec_j = vec_j[0]
                
                # Cosine similarity
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    similarity[(bank_i, bank_j)] = max(0, sim)  # Clip negative
                    similarity[(bank_j, bank_i)] = max(0, sim)
        
        return similarity
    
    def _compute_liquidity_similarity(
        self, 
        data: DatasetContainer, 
        year: int = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute liquidity profile similarity.
        
        Banks with similar maturity mismatch profiles face
        correlated liquidity stress under market-wide events.
        """
        if data.maturity_profile is None:
            # Fall back to ml_ready liquidity features
            if data.ml_ready is None:
                return {}
            
            df = data.ml_ready.copy()
            liquidity_cols = ['liquidity_buffer', 'stress_level']
            liquidity_cols = [c for c in liquidity_cols if c in df.columns]
            
            if not liquidity_cols:
                return {}
            
            df[liquidity_cols] = df[liquidity_cols].fillna(0)
            
            similarity = {}
            banks = df['bank_name'].unique()
            
            for i, bank_i in enumerate(banks):
                vec_i = df[df['bank_name'] == bank_i][liquidity_cols].values
                if len(vec_i) == 0:
                    continue
                vec_i = vec_i[0]
                
                for j, bank_j in enumerate(banks):
                    if i >= j:
                        continue
                    
                    vec_j = df[df['bank_name'] == bank_j][liquidity_cols].values
                    if len(vec_j) == 0:
                        continue
                    vec_j = vec_j[0]
                    
                    # Euclidean distance converted to similarity
                    dist = np.linalg.norm(vec_i - vec_j)
                    sim = 1 / (1 + dist)  # Convert distance to similarity
                    similarity[(bank_i, bank_j)] = sim
                    similarity[(bank_j, bank_i)] = sim
            
            return similarity
        
        # Use maturity profile data if available
        df = data.maturity_profile.copy()
        
        # TODO: Parse maturity profile columns when data format is confirmed
        return {}
    
    def _compute_market_correlation(
        self, 
        market_data: Dict = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute market-implied correlations from stock returns.
        
        High correlation in returns suggests common exposures
        and potential for synchronized distress.
        """
        if market_data is None or 'correlation' not in market_data:
            return {}
        
        corr_matrix = market_data['correlation']
        
        # Convert correlation matrix to pairwise dict
        similarity = {}
        
        for ticker_i in corr_matrix.index:
            for ticker_j in corr_matrix.columns:
                if ticker_i >= ticker_j:
                    continue
                
                corr = corr_matrix.loc[ticker_i, ticker_j]
                if not np.isnan(corr):
                    # Map tickers to bank names (simplified)
                    bank_i = ticker_i.replace('.NS', '')
                    bank_j = ticker_j.replace('.NS', '')
                    
                    similarity[(bank_i, bank_j)] = max(0, corr)
                    similarity[(bank_j, bank_i)] = max(0, corr)
        
        return similarity
    
    def compute_network_metrics(self) -> pd.DataFrame:
        """
        Compute network centrality metrics for all nodes.
        
        Returns DataFrame with:
        - degree_centrality: Local importance
        - betweenness_centrality: Bridge role in network
        - eigenvector_centrality: Importance of neighbors
        - pagerank: Google-style influence measure
        - clustering: Local density
        """
        if self.graph is None or len(self.graph) == 0:
            return pd.DataFrame()
        
        metrics = pd.DataFrame(index=list(self.graph.nodes()))
        
        # Degree centrality
        metrics['degree_centrality'] = pd.Series(nx.degree_centrality(self.graph))
        metrics['in_degree'] = pd.Series(dict(self.graph.in_degree()))
        metrics['out_degree'] = pd.Series(dict(self.graph.out_degree()))
        
        # Betweenness centrality
        metrics['betweenness_centrality'] = pd.Series(
            nx.betweenness_centrality(self.graph, weight='weight')
        )
        
        # Eigenvector centrality (with convergence handling)
        try:
            metrics['eigenvector_centrality'] = pd.Series(
                nx.eigenvector_centrality(self.graph, max_iter=500, weight='weight')
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality did not converge")
            metrics['eigenvector_centrality'] = 0.0
        
        # PageRank
        metrics['pagerank'] = pd.Series(
            nx.pagerank(self.graph, weight='weight')
        )
        
        # Clustering coefficient
        metrics['clustering'] = pd.Series(nx.clustering(self.graph))
        
        return metrics.reset_index().rename(columns={'index': 'bank_name'})
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get weighted adjacency matrix"""
        if self.graph is None:
            return np.array([])
        return nx.to_numpy_array(self.graph, weight='weight')
    
    def get_laplacian_matrix(self) -> np.ndarray:
        """Get graph Laplacian matrix for spectral analysis"""
        if self.graph is None:
            return np.array([])
        # Laplacian requires undirected graph
        undirected = self.graph.to_undirected()
        return nx.laplacian_matrix(undirected, weight='weight').toarray()
    
    def find_communities(self) -> Dict[str, int]:
        """
        Detect communities in the network.
        
        Banks in the same community share similar risk profiles
        and may fail together.
        """
        if self.graph is None or len(self.graph) == 0:
            return {}
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(undirected))
            
            # Map nodes to community IDs
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            
            return community_map
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}
    
    def export_to_dict(self) -> Dict:
        """Export network to dictionary format"""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'weight': e.weight,
                    'sector_similarity': e.sector_similarity,
                    'liquidity_similarity': e.liquidity_similarity,
                    'market_correlation': e.market_correlation
                }
                for e in self.edges
            ],
            'config': {
                'sector_weight': self.sector_weight,
                'liquidity_weight': self.liquidity_weight,
                'market_weight': self.market_weight,
                'edge_threshold': self.edge_threshold
            }
        }


if __name__ == "__main__":
    # Test network building
    logging.basicConfig(level=logging.INFO)
    
    from .data_loader import load_data
    
    data = load_data()
    builder = NetworkBuilder()
    graph = builder.build_network(data)
    
    metrics = builder.compute_network_metrics()
    print(f"\nNetwork metrics:\n{metrics}")
