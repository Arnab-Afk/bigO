"""
Network Analysis Engine

Implements centrality measures, contagion path analysis, and systemic risk metrics.
Based on Technical Documentation Section 6.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import networkx as nx
import numpy as np


@dataclass
class NetworkMetrics:
    """Container for network-level statistics"""
    node_count: int
    edge_count: int
    density: float
    average_clustering: float
    average_path_length: Optional[float]
    diameter: Optional[int]
    
    # Systemic risk indicators
    concentration_index: float  # HHI
    interconnectedness_score: float
    complexity_score: float


@dataclass
class NodeCentrality:
    """Centrality metrics for a single node"""
    node_id: UUID
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    katz_centrality: float
    closeness_centrality: float
    
    @property
    def systemic_importance(self) -> float:
        """Aggregate systemic importance score"""
        return (
            0.2 * self.degree_centrality +
            0.3 * self.betweenness_centrality +
            0.2 * self.eigenvector_centrality +
            0.3 * self.pagerank
        )


@dataclass
class ContagionPath:
    """A path through which contagion can propagate"""
    path: List[UUID]
    probability: float
    total_exposure: float
    path_length: int
    
    @property
    def risk_score(self) -> float:
        """Overall risk score for this path"""
        return self.probability * self.total_exposure / (1 + self.path_length)


class NetworkAnalyzer:
    """
    Network analysis and centrality computation
    
    Analyzes the financial network graph to identify critical nodes,
    bottlenecks, and contagion pathways.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Args:
            graph: NetworkX directed graph representing financial network
        """
        self.graph = graph
        self._centrality_cache: Dict[str, Dict] = {}
    
    def compute_all_centralities(self) -> Dict[UUID, NodeCentrality]:
        """
        Compute comprehensive centrality metrics for all nodes
        
        Returns:
            Dictionary mapping node ID to centrality measures
        """
        results = {}
        
        # Compute each centrality measure
        degree = self._compute_degree_centrality()
        betweenness = self._compute_betweenness_centrality()
        eigenvector = self._compute_eigenvector_centrality()
        pagerank = self._compute_pagerank()
        katz = self._compute_katz_centrality()
        closeness = self._compute_closeness_centrality()
        
        # Combine into NodeCentrality objects
        for node in self.graph.nodes():
            results[node] = NodeCentrality(
                node_id=node,
                degree_centrality=degree.get(node, 0.0),
                betweenness_centrality=betweenness.get(node, 0.0),
                eigenvector_centrality=eigenvector.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                katz_centrality=katz.get(node, 0.0),
                closeness_centrality=closeness.get(node, 0.0),
            )
        
        return results
    
    def _compute_degree_centrality(self) -> Dict[UUID, float]:
        """
        Degree centrality: d(v) = deg(v) / (n-1)
        Measures direct connectivity
        """
        if 'degree' not in self._centrality_cache:
            self._centrality_cache['degree'] = nx.degree_centrality(self.graph)
        return self._centrality_cache['degree']
    
    def _compute_betweenness_centrality(self) -> Dict[UUID, float]:
        """
        Betweenness centrality: g(v) = Σ σ(s,t|v) / σ(s,t)
        Identifies critical intermediary nodes (bridges)
        """
        if 'betweenness' not in self._centrality_cache:
            self._centrality_cache['betweenness'] = nx.betweenness_centrality(
                self.graph,
                weight='exposure_magnitude',
                normalized=True
            )
        return self._centrality_cache['betweenness']
    
    def _compute_eigenvector_centrality(self) -> Dict[UUID, float]:
        """
        Eigenvector centrality: influence through connections
        """
        if 'eigenvector' not in self._centrality_cache:
            try:
                self._centrality_cache['eigenvector'] = nx.eigenvector_centrality(
                    self.graph,
                    max_iter=1000,
                    weight='contagion_probability'
                )
            except nx.PowerIterationFailedConvergence:
                # Fallback to degree if eigenvector fails
                self._centrality_cache['eigenvector'] = self._compute_degree_centrality()
        return self._centrality_cache['eigenvector']
    
    def _compute_pagerank(self, damping: float = 0.85) -> Dict[UUID, float]:
        """
        PageRank: Risk-flow importance based on exposure network
        """
        if 'pagerank' not in self._centrality_cache:
            self._centrality_cache['pagerank'] = nx.pagerank(
                self.graph,
                alpha=damping,
                weight='contagion_probability'
            )
        return self._centrality_cache['pagerank']
    
    def _compute_katz_centrality(self, alpha: float = 0.1) -> Dict[UUID, float]:
        """
        Katz centrality: Generalization of eigenvector centrality
        """
        if 'katz' not in self._centrality_cache:
            try:
                self._centrality_cache['katz'] = nx.katz_centrality(
                    self.graph,
                    alpha=alpha,
                    beta=1.0,
                    normalized=True
                )
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                # Fallback
                self._centrality_cache['katz'] = self._compute_degree_centrality()
        return self._centrality_cache['katz']
    
    def _compute_closeness_centrality(self) -> Dict[UUID, float]:
        """
        Closeness centrality: Average distance to all other nodes
        """
        if 'closeness' not in self._centrality_cache:
            self._centrality_cache['closeness'] = nx.closeness_centrality(
                self.graph,
                distance='weight'
            )
        return self._centrality_cache['closeness']
    
    def compute_network_metrics(self) -> NetworkMetrics:
        """
        Compute aggregate network statistics
        
        Returns:
            NetworkMetrics object with network-level indicators
        """
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        
        # Basic metrics
        density = nx.density(self.graph)
        
        # Clustering (convert to undirected for clustering)
        undirected = self.graph.to_undirected()
        avg_clustering = nx.average_clustering(undirected)
        
        # Path metrics (only for connected components)
        try:
            if nx.is_weakly_connected(self.graph):
                avg_path_length = nx.average_shortest_path_length(self.graph)
                diameter = nx.diameter(self.graph)
            else:
                avg_path_length = None
                diameter = None
        except nx.NetworkXError:
            avg_path_length = None
            diameter = None
        
        # Systemic risk metrics
        concentration = self._compute_concentration_index()
        interconnectedness = self._compute_interconnectedness_score()
        complexity = self._compute_complexity_score()
        
        return NetworkMetrics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            average_clustering=avg_clustering,
            average_path_length=avg_path_length,
            diameter=diameter,
            concentration_index=concentration,
            interconnectedness_score=interconnectedness,
            complexity_score=complexity,
        )
    
    def _compute_concentration_index(self) -> float:
        """
        Herfindahl-Hirschman Index of exposure concentration
        """
        if self.graph.number_of_edges() == 0:
            return 0.0
        
        total_exposure = sum(
            data.get('exposure_magnitude', 0)
            for _, _, data in self.graph.edges(data=True)
        )
        
        if total_exposure == 0:
            return 0.0
        
        hhi = sum(
            (data.get('exposure_magnitude', 0) / total_exposure) ** 2
            for _, _, data in self.graph.edges(data=True)
        )
        
        return hhi
    
    def _compute_interconnectedness_score(self) -> float:
        """
        Combination of density and clustering
        """
        density = nx.density(self.graph)
        undirected = self.graph.to_undirected()
        avg_clustering = nx.average_clustering(undirected)
        
        return (density + avg_clustering) / 2
    
    def _compute_complexity_score(self) -> float:
        """
        Complexity based on path lengths and structure
        """
        try:
            if nx.is_weakly_connected(self.graph):
                avg_path = nx.average_shortest_path_length(self.graph)
                # Normalize to 0-1 scale
                return min(1.0, avg_path / 10.0)
            else:
                # For disconnected graphs, use component analysis
                components = list(nx.weakly_connected_components(self.graph))
                return len(components) / self.graph.number_of_nodes()
        except nx.NetworkXError:
            return 0.0
    
    def find_critical_paths(
        self,
        source: UUID,
        threshold: float = 0.5,
        max_length: int = 5
    ) -> List[ContagionPath]:
        """
        Find paths through which contagion can propagate with probability above threshold
        
        Args:
            source: Starting node
            threshold: Minimum contagion probability
            max_length: Maximum path length to consider
        
        Returns:
            List of critical contagion paths
        """
        critical_paths = []
        
        # BFS with probability tracking
        queue = [(source, [source], 1.0)]
        visited: Set[UUID] = set()
        
        while queue:
            node, path, prob = queue.pop(0)
            
            # Skip if path too long
            if len(path) > max_length:
                continue
            
            # Mark as visited for this path
            if node in visited and node != source:
                continue
            
            if node != source:
                visited.add(node)
            
            # Explore neighbors
            if node in self.graph:
                for neighbor in self.graph.successors(node):
                    if neighbor not in path:  # Avoid cycles
                        edge_data = self.graph[node][neighbor]
                        contagion_prob = edge_data.get('contagion_probability', 0.0)
                        new_prob = prob * contagion_prob
                        
                        if new_prob >= threshold:
                            new_path = path + [neighbor]
                            total_exposure = self._sum_path_exposures(new_path)
                            
                            critical_paths.append(
                                ContagionPath(
                                    path=new_path,
                                    probability=new_prob,
                                    total_exposure=total_exposure,
                                    path_length=len(new_path) - 1,
                                )
                            )
                            
                            queue.append((neighbor, new_path, new_prob))
        
        # Sort by risk score
        critical_paths.sort(key=lambda x: x.risk_score, reverse=True)
        return critical_paths
    
    def _sum_path_exposures(self, path: List[UUID]) -> float:
        """Sum exposure values along a path"""
        total = 0.0
        for i in range(len(path) - 1):
            if path[i] in self.graph and path[i + 1] in self.graph[path[i]]:
                edge_data = self.graph[path[i]][path[i + 1]]
                total += edge_data.get('exposure_magnitude', 0.0)
        return total
    
    def identify_bottlenecks(self, top_k: int = 10) -> List[Tuple[UUID, float]]:
        """
        Find nodes whose failure would most disrupt network flow
        
        Args:
            top_k: Number of top bottlenecks to return
        
        Returns:
            List of (node_id, impact_score) tuples
        """
        bottlenecks = []
        
        # Compute baseline efficiency
        baseline_efficiency = self._compute_network_efficiency()
        
        for node in self.graph.nodes():
            # Create network without this node
            reduced_graph = self.graph.copy()
            reduced_graph.remove_node(node)
            
            # Compute reduced efficiency
            reduced_efficiency = self._compute_network_efficiency(reduced_graph)
            
            # Impact score
            if baseline_efficiency > 0:
                impact = (baseline_efficiency - reduced_efficiency) / baseline_efficiency
            else:
                impact = 0.0
            
            # Get exposure concentration at this node
            in_degree = self.graph.in_degree(node, weight='exposure_magnitude')
            out_degree = self.graph.out_degree(node, weight='exposure_magnitude')
            exposure_concentration = in_degree + out_degree
            
            # Combined impact score
            combined_score = 0.6 * impact + 0.4 * (exposure_concentration / 1000000)
            
            bottlenecks.append((node, combined_score))
        
        # Sort and return top k
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks[:top_k]
    
    def _compute_network_efficiency(self, graph: Optional[nx.DiGraph] = None) -> float:
        """
        Compute global efficiency of the network
        """
        if graph is None:
            graph = self.graph
        
        if graph.number_of_nodes() < 2:
            return 0.0
        
        try:
            efficiency = nx.global_efficiency(graph)
            return efficiency
        except:
            return 0.0
    
    def clear_cache(self):
        """Clear cached centrality computations"""
        self._centrality_cache.clear()


def build_network_graph(
    institutions: List[Dict],
    exposures: List[Dict]
) -> nx.DiGraph:
    """
    Build NetworkX graph from institution and exposure data
    
    Args:
        institutions: List of institution dictionaries
        exposures: List of exposure dictionaries
    
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for inst in institutions:
        G.add_node(
            inst['id'],
            name=inst.get('name', ''),
            type=inst.get('type', ''),
            tier=inst.get('tier', ''),
        )
    
    # Add edges
    for exp in exposures:
        G.add_edge(
            exp['source_institution_id'],
            exp['target_institution_id'],
            exposure_type=exp.get('exposure_type', ''),
            exposure_magnitude=float(exp.get('gross_exposure', 0)),
            contagion_probability=float(exp.get('contagion_probability', 0)),
            recovery_rate=float(exp.get('recovery_rate', 0.55)),
            settlement_urgency=float(exp.get('settlement_urgency', 0.5)),
        )
    
    return G
