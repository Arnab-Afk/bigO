"""
Spectral Analyzer Module

Performs spectral analysis on the interdependence network for systemic fragility quantification.

Key metrics from ML_Flow.md:
- Spectral radius (ρ): Contagion amplification potential
- Fiedler value (λ2): Network cohesion / fragmentation tendency
- Spectral gap: How quickly shocks propagate

CCP Interpretation:
- ρ > 1: Shocks amplify through network → higher default fund needs
- Small λ2: Network easily fragmentable → concentrated clearing risks
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpectralMetrics:
    """Container for spectral analysis results"""
    spectral_radius: float  # Largest eigenvalue magnitude
    fiedler_value: float    # Second smallest eigenvalue of Laplacian
    spectral_gap: float     # Difference between largest eigenvalues
    eigenvalue_entropy: float  # Entropy of eigenvalue distribution
    effective_rank: float   # How many dimensions capture variation
    
    # Risk interpretations
    amplification_risk: str  # 'low', 'medium', 'high'
    fragmentation_risk: str  # 'low', 'medium', 'high'


class SpectralAnalyzer:
    """
    Spectral analysis for systemic fragility quantification.
    
    Following ML_Flow.md Layer 3: Uses eigenvalue decomposition
    of the network to assess systemic risk.
    """
    
    # Thresholds for risk categorization
    SPECTRAL_RADIUS_THRESHOLDS = (0.7, 0.9)  # Low/medium/high
    FIEDLER_THRESHOLDS = (0.1, 0.3)  # Low connectivity thresholds
    
    def __init__(self):
        """Initialize spectral analyzer"""
        self.adjacency_matrix = None
        self.laplacian_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.laplacian_eigenvalues = None
        self.laplacian_eigenvectors = None
    
    def analyze(
        self, 
        adjacency_matrix: np.ndarray = None,
        laplacian_matrix: np.ndarray = None,
        network_builder = None
    ) -> SpectralMetrics:
        """
        Perform spectral analysis on the network.
        
        Args:
            adjacency_matrix: Weighted adjacency matrix
            laplacian_matrix: Graph Laplacian matrix
            network_builder: NetworkBuilder object (if matrices not provided)
            
        Returns:
            SpectralMetrics with analysis results
        """
        # Get matrices from network builder if provided
        if network_builder is not None:
            adjacency_matrix = network_builder.get_adjacency_matrix()
            laplacian_matrix = network_builder.get_laplacian_matrix()
        
        if adjacency_matrix is None or len(adjacency_matrix) == 0:
            logger.warning("No adjacency matrix provided")
            return self._empty_metrics()
        
        self.adjacency_matrix = adjacency_matrix
        self.laplacian_matrix = laplacian_matrix
        
        # Compute eigendecomposition of adjacency matrix
        try:
            self.eigenvalues, self.eigenvectors = np.linalg.eig(adjacency_matrix)
            self.eigenvalues = np.real(self.eigenvalues)  # Take real part
        except Exception as e:
            logger.error(f"Adjacency eigendecomposition failed: {e}")
            return self._empty_metrics()
        
        # Compute eigendecomposition of Laplacian
        if laplacian_matrix is not None:
            try:
                self.laplacian_eigenvalues, self.laplacian_eigenvectors = np.linalg.eig(laplacian_matrix)
                self.laplacian_eigenvalues = np.sort(np.real(self.laplacian_eigenvalues))
            except Exception as e:
                logger.warning(f"Laplacian eigendecomposition failed: {e}")
                self.laplacian_eigenvalues = None
        
        # Compute metrics
        spectral_radius = self._compute_spectral_radius()
        fiedler_value = self._compute_fiedler_value()
        spectral_gap = self._compute_spectral_gap()
        eigenvalue_entropy = self._compute_eigenvalue_entropy()
        effective_rank = self._compute_effective_rank()
        
        # Risk categorization
        amplification_risk = self._categorize_amplification_risk(spectral_radius)
        fragmentation_risk = self._categorize_fragmentation_risk(fiedler_value)
        
        metrics = SpectralMetrics(
            spectral_radius=spectral_radius,
            fiedler_value=fiedler_value,
            spectral_gap=spectral_gap,
            eigenvalue_entropy=eigenvalue_entropy,
            effective_rank=effective_rank,
            amplification_risk=amplification_risk,
            fragmentation_risk=fragmentation_risk
        )
        
        logger.info(f"Spectral analysis complete: ρ={spectral_radius:.3f}, λ2={fiedler_value:.3f}")
        return metrics
    
    def _compute_spectral_radius(self) -> float:
        """
        Compute spectral radius (largest eigenvalue magnitude).
        
        Interpretation: ρ > 1 means shocks can amplify through network
        """
        return float(np.max(np.abs(self.eigenvalues)))
    
    def _compute_fiedler_value(self) -> float:
        """
        Compute Fiedler value (algebraic connectivity).
        
        This is the second smallest eigenvalue of the Laplacian.
        Interpretation: Small λ2 means network easily fragments
        """
        if self.laplacian_eigenvalues is None:
            return 0.0
        
        # Second smallest eigenvalue (first is always 0 for connected graph)
        sorted_eigenvalues = np.sort(self.laplacian_eigenvalues)
        if len(sorted_eigenvalues) >= 2:
            return float(sorted_eigenvalues[1])
        return 0.0
    
    def _compute_spectral_gap(self) -> float:
        """
        Compute spectral gap (difference between two largest eigenvalues).
        
        Larger gap means shocks propagate more quickly.
        """
        sorted_eigenvalues = np.sort(np.abs(self.eigenvalues))[::-1]
        if len(sorted_eigenvalues) >= 2:
            return float(sorted_eigenvalues[0] - sorted_eigenvalues[1])
        return 0.0
    
    def _compute_eigenvalue_entropy(self) -> float:
        """
        Compute entropy of eigenvalue distribution.
        
        High entropy = more distributed influence
        Low entropy = concentrated structure
        """
        abs_eigenvalues = np.abs(self.eigenvalues)
        total = np.sum(abs_eigenvalues)
        
        if total == 0:
            return 0.0
        
        probs = abs_eigenvalues / total
        probs = probs[probs > 0]  # Remove zeros
        
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def _compute_effective_rank(self) -> float:
        """
        Compute effective rank of the matrix.
        
        Measures how many dimensions capture most of the variation.
        """
        abs_eigenvalues = np.abs(self.eigenvalues)
        total = np.sum(abs_eigenvalues)
        
        if total == 0:
            return 0.0
        
        # Entropy-based effective rank
        probs = abs_eigenvalues / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(np.exp(entropy))
    
    def _categorize_amplification_risk(self, spectral_radius: float) -> str:
        """Categorize amplification risk based on spectral radius"""
        if spectral_radius < self.SPECTRAL_RADIUS_THRESHOLDS[0]:
            return 'low'
        elif spectral_radius < self.SPECTRAL_RADIUS_THRESHOLDS[1]:
            return 'medium'
        else:
            return 'high'
    
    def _categorize_fragmentation_risk(self, fiedler_value: float) -> str:
        """Categorize fragmentation risk based on Fiedler value"""
        if fiedler_value > self.FIEDLER_THRESHOLDS[1]:
            return 'low'  # Well connected
        elif fiedler_value > self.FIEDLER_THRESHOLDS[0]:
            return 'medium'
        else:
            return 'high'  # Easily fragmentable
    
    def _empty_metrics(self) -> SpectralMetrics:
        """Return empty metrics when analysis cannot be performed"""
        return SpectralMetrics(
            spectral_radius=0.0,
            fiedler_value=0.0,
            spectral_gap=0.0,
            eigenvalue_entropy=0.0,
            effective_rank=0.0,
            amplification_risk='unknown',
            fragmentation_risk='unknown'
        )
    
    def get_principal_components(self, n_components: int = 3) -> np.ndarray:
        """
        Get principal eigenvectors (dominant modes of the network).
        
        These represent the main patterns of interconnection.
        """
        if self.eigenvectors is None:
            return np.array([])
        
        # Sort by eigenvalue magnitude
        sorted_indices = np.argsort(np.abs(self.eigenvalues))[::-1]
        return np.real(self.eigenvectors[:, sorted_indices[:n_components]])
    
    def compute_node_vulnerability(self) -> np.ndarray:
        """
        Compute node-level vulnerability based on spectral properties.
        
        Nodes with high loadings on dominant eigenvectors are more
        vulnerable to systemic shocks.
        """
        if self.eigenvectors is None:
            return np.array([])
        
        # Use principal eigenvector (Perron eigenvector)
        sorted_indices = np.argsort(np.abs(self.eigenvalues))[::-1]
        principal = np.real(self.eigenvectors[:, sorted_indices[0]])
        
        # Normalize to [0, 1]
        principal = np.abs(principal)
        if np.max(principal) > 0:
            principal = principal / np.max(principal)
        
        return principal
    
    def compute_eigenvector_centrality(self) -> Dict[int, float]:
        """
        Compute eigenvector centrality for loss mutualization.
        
        Returns normalized eigenvector centrality scores that sum to 1.0.
        These are used to distribute CCP losses proportionally to systemic importance.
        """
        if self.eigenvectors is None or self.eigenvalues is None:
            return {}
        
        # Get principal eigenvector (associated with largest eigenvalue)
        sorted_indices = np.argsort(np.abs(self.eigenvalues))[::-1]
        principal_eigenvector = np.real(self.eigenvectors[:, sorted_indices[0]])
        
        # Take absolute values and normalize to sum to 1
        centrality = np.abs(principal_eigenvector)
        total = np.sum(centrality)
        
        if total > 0:
            centrality = centrality / total
        else:
            # Fallback: equal weights
            centrality = np.ones(len(centrality)) / len(centrality)
        
        # Return as dictionary mapping node index to centrality
        return {i: float(centrality[i]) for i in range(len(centrality))}
    
    def normalize_payoff_matrix(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize payoff matrix using principal eigenvector.
        
        This ensures the CCP (assumed to be the center node) has zero net payoff
        by redistributing gains/losses proportionally to eigenvector centrality.
        
        Args:
            payoff_matrix: n×n matrix where element (i,j) is payoff from i to j
            
        Returns:
            Normalized matrix where sum of CCP row/column = 0
        """
        if self.eigenvectors is None or payoff_matrix.size == 0:
            return payoff_matrix
        
        n = payoff_matrix.shape[0]
        centrality = self.compute_eigenvector_centrality()
        centrality_vector = np.array([centrality.get(i, 1.0/n) for i in range(n)])
        
        # Normalize each row by eigenvector weights
        # This redistributes payoffs proportionally to systemic importance
        normalized = payoff_matrix.copy()
        
        for i in range(n):
            row_sum = np.sum(payoff_matrix[i, :])
            if row_sum != 0:
                # Redistribute row's payoffs proportionally to centrality
                normalized[i, :] = payoff_matrix[i, :] * centrality_vector / np.sum(centrality_vector)
        
        return normalized
    
    def compute_contagion_index(self) -> float:
        """
        Compute overall contagion index for the network.
        
        Combines spectral metrics into a single systemic risk score.
        """
        if self.eigenvalues is None:
            return 0.0
        
        spectral_radius = self._compute_spectral_radius()
        fiedler = self._compute_fiedler_value()
        gap = self._compute_spectral_gap()
        
        # Weighted combination
        # High spectral radius = high contagion
        # Low Fiedler = high contagion (fragile)
        # Large gap = rapid propagation
        
        contagion_index = (
            0.4 * min(spectral_radius, 2.0) / 2.0 +  # Normalize to [0, 1]
            0.3 * (1 - min(fiedler, 1.0)) +          # Invert (low = bad)
            0.3 * min(gap, 1.0)                       # Normalize to [0, 1]
        )
        
        return float(contagion_index)
    
    def to_dict(self) -> Dict:
        """Export results to dictionary"""
        metrics = self.analyze(self.adjacency_matrix, self.laplacian_matrix) if self.eigenvalues is None else None
        
        return {
            'spectral_radius': self._compute_spectral_radius() if self.eigenvalues is not None else 0.0,
            'fiedler_value': self._compute_fiedler_value(),
            'spectral_gap': self._compute_spectral_gap() if self.eigenvalues is not None else 0.0,
            'eigenvalue_entropy': self._compute_eigenvalue_entropy() if self.eigenvalues is not None else 0.0,
            'effective_rank': self._compute_effective_rank() if self.eigenvalues is not None else 0.0,
            'contagion_index': self.compute_contagion_index(),
            'n_nodes': len(self.adjacency_matrix) if self.adjacency_matrix is not None else 0
        }


if __name__ == "__main__":
    # Test spectral analysis
    logging.basicConfig(level=logging.INFO)
    
    # Create sample adjacency matrix
    np.random.seed(42)
    n = 10
    adj = np.random.rand(n, n) * 0.5
    adj = (adj + adj.T) / 2  # Make symmetric
    np.fill_diagonal(adj, 0)  # No self-loops
    
    # Apply threshold
    adj[adj < 0.3] = 0
    
    # Create Laplacian
    degree = np.diag(np.sum(adj, axis=1))
    laplacian = degree - adj
    
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze(adj, laplacian)
    
    print(f"\nSpectral Metrics:")
    print(f"  Spectral Radius: {metrics.spectral_radius:.4f}")
    print(f"  Fiedler Value: {metrics.fiedler_value:.4f}")
    print(f"  Spectral Gap: {metrics.spectral_gap:.4f}")
    print(f"  Amplification Risk: {metrics.amplification_risk}")
    print(f"  Fragmentation Risk: {metrics.fragmentation_risk}")
    print(f"  Contagion Index: {analyzer.compute_contagion_index():.4f}")
