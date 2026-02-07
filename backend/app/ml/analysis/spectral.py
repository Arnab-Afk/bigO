"""
Spectral Analysis Module for Systemic Fragility Quantification

Implements spectral methods for analyzing network stability and contagion amplification.

Based on ML_Flow.md Section 5: Systemic Fragility & Spectral Analysis

Key metrics:
- Largest eigenvalue (λ_max): System stability indicator
- Eigenvector centrality: Systemically important nodes
- Spectral gap: Resilience measure
- Network density and clustering
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs

logger = logging.getLogger(__name__)


@dataclass
class SpectralMetrics:
    """Systemic fragility indicators from spectral analysis"""
    
    # Eigenvalue metrics
    largest_eigenvalue: float       # λ_max - amplification factor
    second_eigenvalue: float        # λ_2 - secondary mode
    spectral_gap: float            # λ_max - λ_2 - resilience
    spectral_radius: float         # Max absolute eigenvalue
    
    # Network structural metrics
    density: float                  # Edge density
    avg_clustering: float          # Clustering coefficient
    avg_degree: float              # Mean degree
    
    # Node-level metrics
    eigenvector_centrality: Dict[str, float]  # Systemic importance
    
    # Risk interpretation
    fragility_index: float         # Composite fragility score [0, 1]
    risk_level: str                # LOW, MEDIUM, HIGH, CRITICAL
    
    def __str__(self) -> str:
        return f"""Systemic Fragility Analysis
{'='*50}
Largest Eigenvalue (λ_max): {self.largest_eigenvalue:.4f}
  → Amplification Factor: {self.largest_eigenvalue:.2f}x
  → Interpretation: {"HIGH RISK" if self.largest_eigenvalue > 5 else "MODERATE" if self.largest_eigenvalue > 2 else "LOW"}

Spectral Gap (λ_max - λ_2): {self.spectral_gap:.4f}
  → Resilience: {"LOW" if self.spectral_gap < 1 else "MODERATE" if self.spectral_gap < 3 else "HIGH"}

Network Structure:
  - Density: {self.density:.4f}
  - Avg Clustering: {self.avg_clustering:.4f}
  - Avg Degree: {self.avg_degree:.2f}

Overall Fragility: {self.fragility_index:.2%} ({self.risk_level})
{'='*50}"""


class SpectralAnalyzer:
    """
    Spectral analysis for financial network stability
    
    CCP Perspective:
    - High λ_max indicates strong contagion amplification
    - Eigenvector centrality identifies critical institutions
    - Spectral gap measures system resilience
    """
    
    def __init__(self):
        """Initialize spectral analyzer"""
        self.metrics: Optional[SpectralMetrics] = None
    
    def analyze_network(
        self,
        adjacency_matrix: np.ndarray,
        node_labels: Optional[List[str]] = None
    ) -> SpectralMetrics:
        """
        Perform comprehensive spectral analysis
        
        Args:
            adjacency_matrix: Weighted adjacency matrix (n x n)
            node_labels: Optional list of node names
        
        Returns:
            SpectralMetrics object with all computed metrics
        """
        n = adjacency_matrix.shape[0]
        
        if node_labels is None:
            node_labels = [f"Node_{i}" for i in range(n)]
        
        logger.info(f"Performing spectral analysis on {n}x{n} network")
        
        # 1. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self._compute_spectrum(adjacency_matrix)
        
        # Sort by magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Extract key eigenvalues
        lambda_max = float(np.abs(eigenvalues[0]))
        lambda_2 = float(np.abs(eigenvalues[1])) if n > 1 else 0.0
        spectral_gap = lambda_max - lambda_2
        spectral_radius = lambda_max
        
        logger.info(f"λ_max = {lambda_max:.4f}, λ_2 = {lambda_2:.4f}, gap = {spectral_gap:.4f}")
        
        # 2. Compute eigenvector centrality
        # Use leading eigenvector
        leading_eigenvector = np.abs(eigenvectors[:, 0])
        
        # Normalize to [0, 1]
        if leading_eigenvector.max() > 0:
            leading_eigenvector = leading_eigenvector / leading_eigenvector.max()
        
        eigenvector_cent = {
            node_labels[i]: float(leading_eigenvector[i])
            for i in range(n)
        }
        
        # 3. Compute network structural metrics
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Relabel nodes
        mapping = {i: node_labels[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        
        density = nx.density(G)
        
        try:
            avg_clustering = nx.average_clustering(G, weight='weight')
        except:
            avg_clustering = 0.0
        
        avg_degree = np.mean([d for _, d in G.degree(weight='weight')])
        
        # 4. Compute fragility index
        fragility_index = self._compute_fragility_index(
            lambda_max=lambda_max,
            density=density,
            avg_clustering=avg_clustering,
            spectral_gap=spectral_gap
        )
        
        # 5. Determine risk level
        risk_level = self._classify_risk_level(fragility_index, lambda_max)
        
        # Create metrics object
        metrics = SpectralMetrics(
            largest_eigenvalue=lambda_max,
            second_eigenvalue=lambda_2,
            spectral_gap=spectral_gap,
            spectral_radius=spectral_radius,
            density=density,
            avg_clustering=avg_clustering,
            avg_degree=avg_degree,
            eigenvector_centrality=eigenvector_cent,
            fragility_index=fragility_index,
            risk_level=risk_level
        )
        
        self.metrics = metrics
        
        return metrics
    
    def _compute_spectrum(
        self,
        adjacency_matrix: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors
        
        Args:
            adjacency_matrix: Adjacency matrix
            k: Number of eigenvalues to compute (None = all)
        
        Returns:
            eigenvalues: Array of eigenvalues
            eigenvectors: Matrix of eigenvectors (column vectors)
        """
        n = adjacency_matrix.shape[0]
        
        if k is not None and k < n:
            # Sparse eigenvalue computation (for large networks)
            try:
                eigenvalues, eigenvectors = eigs(
                    adjacency_matrix,
                    k=min(k, n-1),
                    which='LM'  # Largest magnitude
                )
                # Convert to real (should already be real for symmetric)
                eigenvalues = np.real(eigenvalues)
                eigenvectors = np.real(eigenvectors)
            except Exception as e:
                logger.warning(f"Sparse eigenvalue computation failed: {e}. Falling back to dense.")
                eigenvalues, eigenvectors = eigh(adjacency_matrix)
        else:
            # Dense eigenvalue computation
            eigenvalues, eigenvectors = eigh(adjacency_matrix)
        
        return eigenvalues, eigenvectors
    
    def _compute_fragility_index(
        self,
        lambda_max: float,
        density: float,
        avg_clustering: float,
        spectral_gap: float
    ) -> float:
        """
        Compute composite fragility index [0, 1]
        
        Higher values indicate greater systemic fragility
        
        Formula:
        fragility = w1 * λ_norm + w2 * density + w3 * clustering - w4 * gap_norm
        
        Args:
            lambda_max: Largest eigenvalue
            density: Network density
            avg_clustering: Average clustering coefficient
            spectral_gap: Spectral gap
        
        Returns:
            Fragility index in [0, 1]
        """
        # Normalize components
        # λ_max normalization: typical range [1, 10], normalize to [0, 1]
        lambda_norm = np.clip(lambda_max / 10, 0, 1)
        
        # Density already in [0, 1]
        
        # Clustering already in [0, 1]
        
        # Spectral gap normalization: typical range [0, 5], inverse
        gap_norm = 1 - np.clip(spectral_gap / 5, 0, 1)
        
        # Weights
        w1 = 0.4  # λ_max is most important
        w2 = 0.2  # Density
        w3 = 0.1  # Clustering
        w4 = 0.3  # Spectral gap (resilience, inverse effect)
        
        fragility = (
            w1 * lambda_norm +
            w2 * density +
            w3 * avg_clustering +
            w4 * gap_norm
        )
        
        # Ensure [0, 1]
        fragility = np.clip(fragility, 0, 1)
        
        return float(fragility)
    
    def _classify_risk_level(
        self,
        fragility_index: float,
        lambda_max: float
    ) -> str:
        """
        Classify systemic risk level
        
        Args:
            fragility_index: Composite fragility score
            lambda_max: Largest eigenvalue
        
        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        # Critical thresholds
        if lambda_max > 8 or fragility_index > 0.8:
            return "CRITICAL"
        elif lambda_max > 5 or fragility_index > 0.6:
            return "HIGH"
        elif lambda_max > 3 or fragility_index > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_critical_nodes(
        self,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most systemically important nodes
        
        Args:
            top_k: Number of nodes to return
        
        Returns:
            List of (node_name, centrality_score) tuples, sorted descending
        """
        if self.metrics is None:
            raise ValueError("Must run analyze_network() first")
        
        # Sort by eigenvector centrality
        sorted_nodes = sorted(
            self.metrics.eigenvector_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:top_k]
    
    def compute_stress_amplification(
        self,
        adjacency_matrix: np.ndarray,
        initial_shock: np.ndarray
    ) -> np.ndarray:
        """
        Compute propagated stress through network
        
        Uses spectral decomposition to model contagion:
        final_stress = (I - ρA)^(-1) * initial_shock
        
        Where ρ = 0.9 / λ_max ensures convergence
        
        Args:
            adjacency_matrix: Weighted adjacency matrix
            initial_shock: Initial stress vector
        
        Returns:
            Amplified stress vector
        """
        n = adjacency_matrix.shape[0]
        
        # Get λ_max
        eigenvalues, _ = self._compute_spectrum(adjacency_matrix, k=1)
        lambda_max = float(np.abs(eigenvalues[0]))
        
        # Choose damping factor to ensure convergence
        rho = 0.9 / max(lambda_max, 1e-6)
        
        logger.info(f"Computing stress amplification with ρ={rho:.4f}")
        
        # Compute (I - ρA)
        I = np.eye(n)
        propagation_matrix = I - rho * adjacency_matrix
        
        # Solve linear system
        try:
            final_stress = np.linalg.solve(propagation_matrix, initial_shock)
        except np.linalg.LinAlgError:
            logger.warning("Linear system singular, using pseudo-inverse")
            final_stress = np.linalg.lstsq(propagation_matrix, initial_shock, rcond=None)[0]
        
        # Amplification ratio
        initial_magnitude = np.linalg.norm(initial_shock)
        final_magnitude = np.linalg.norm(final_stress)
        
        if initial_magnitude > 0:
            amplification = final_magnitude / initial_magnitude
            logger.info(f"Stress amplification factor: {amplification:.2f}x")
        
        return final_stress
    
    def monte_carlo_cascade_risk(
        self,
        adjacency_matrix: np.ndarray,
        node_default_probs: Dict[str, float],
        node_labels: List[str],
        n_simulations: int = 1000,
        threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation of cascade defaults
        
        Estimates probability that each node triggers a cascade
        
        Args:
            adjacency_matrix: Weighted adjacency matrix
            node_default_probs: Default probability for each node
            node_labels: Node names
            n_simulations: Number of simulations
            threshold: Stress threshold for triggering default
        
        Returns:
            Dict mapping node names to cascade trigger probabilities
        """
        logger.info(f"Running {n_simulations} Monte Carlo cascade simulations")
        
        n = len(node_labels)
        cascade_counts = {node: 0 for node in node_labels}
        
        for sim in range(n_simulations):
            # Sample initial defaults
            defaults = np.zeros(n)
            for i, node in enumerate(node_labels):
                prob = node_default_probs.get(node, 0.0)
                if np.random.random() < prob:
                    defaults[i] = 1.0
            
            # Skip if no defaults
            if defaults.sum() == 0:
                continue
            
            # Propagate stress
            initial_shock = defaults.copy()
            final_stress = self.compute_stress_amplification(
                adjacency_matrix,
                initial_shock
            )
            
            # Count cascades (stress exceeds threshold)
            cascaded = (final_stress > threshold).astype(int)
            cascade_size = cascaded.sum()
            
            # Attribute cascade to initially defaulted node(s)
            if cascade_size > defaults.sum():
                for i in range(n):
                    if defaults[i] > 0:
                        cascade_counts[node_labels[i]] += 1
        
        # Convert to probabilities
        cascade_probs = {
            node: count / n_simulations
            for node, count in cascade_counts.items()
        }
        
        return cascade_probs


def analyze_systemic_fragility(
    network: nx.Graph,
    verbose: bool = True
) -> SpectralMetrics:
    """
    Convenience function for network fragility analysis
    
    Args:
        network: NetworkX graph
        verbose: Print analysis results
    
    Returns:
        SpectralMetrics object
    """
    # Convert to adjacency matrix
    node_labels = list(network.nodes())
    adjacency = nx.to_numpy_array(network, nodelist=node_labels, weight='weight')
    
    # Analyze
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze_network(adjacency, node_labels)
    
    if verbose:
        print(metrics)
        print("\nTop 10 Systemically Important Nodes:")
        for i, (node, score) in enumerate(analyzer.get_critical_nodes(10), 1):
            print(f"  {i:2d}. {node}: {score:.4f}")
    
    return metrics
