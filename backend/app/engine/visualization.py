"""
Network Visualization Utilities
================================
Helper functions for preparing ABM network data for frontend visualization.

Supports:
- D3.js force-directed graphs
- Cytoscape.js
- React Force Graph
- Gephi/Cytoscape (via export functions)
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class NodeVisual:
    """Visual properties for a network node"""
    id: str
    label: str
    type: str
    color: str
    size: float
    x: Optional[float] = None
    y: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'color': self.color,
            'size': self.size,
            'x': self.x,
            'y': self.y
        }


@dataclass
class EdgeVisual:
    """Visual properties for a network edge"""
    source: str
    target: str
    weight: float
    color: str
    width: float
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'color': self.color,
            'width': self.width,
            'label': self.label
        }


class NetworkVisualizer:
    """
    Converts simulation network state to frontend-friendly formats.
    """
    
    # Color schemes for different node types
    NODE_COLORS = {
        'bank': '#3498db',      # Blue
        'sector': '#e74c3c',    # Red
        'ccp': '#2ecc71',       # Green
        'regulator': '#f39c12'  # Orange
    }
    
    # Color mapping for node health (gradient)
    HEALTH_COLORS = {
        'critical': '#c0392b',   # Dark Red
        'poor': '#e74c3c',       # Red
        'fair': '#f39c12',       # Orange
        'good': '#2ecc71',       # Green
        'excellent': '#27ae60'   # Dark Green
    }
    
    @staticmethod
    def get_health_color(health_score: float) -> str:
        """Map health score [0, 1] to color"""
        if health_score < 0.2:
            return NetworkVisualizer.HEALTH_COLORS['critical']
        elif health_score < 0.4:
            return NetworkVisualizer.HEALTH_COLORS['poor']
        elif health_score < 0.6:
            return NetworkVisualizer.HEALTH_COLORS['fair']
        elif health_score < 0.8:
            return NetworkVisualizer.HEALTH_COLORS['good']
        else:
            return NetworkVisualizer.HEALTH_COLORS['excellent']
    
    @staticmethod
    def convert_to_d3(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert simulation snapshot to D3.js format.
        
        D3 expects:
        {
            "nodes": [{"id": "...", "group": "...", ...}],
            "links": [{"source": "...", "target": "...", "value": ...}]
        }
        """
        nodes = []
        links = []
        
        # Process nodes
        for node in snapshot.get('network_state', {}).get('nodes', []):
            agent_state = snapshot.get('agent_states', {}).get(node['id'], {})
            health = agent_state.get('health', 0.5)
            
            nodes.append({
                'id': node['id'],
                'group': node['type'],
                'health': health,
                'color': NetworkVisualizer.get_health_color(health),
                'alive': node.get('alive', True),
                'size': 10 + health * 20,  # Size based on health
                **agent_state  # Include all agent-specific data
            })
        
        # Process edges
        for edge in snapshot.get('network_state', {}).get('edges', []):
            weight = edge.get('weight', 0)
            
            links.append({
                'source': edge['source'],
                'target': edge['target'],
                'value': weight,
                'type': edge.get('type', 'loan'),
                'width': max(1, min(10, weight / 1000))  # Scale width
            })
        
        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'timestep': snapshot.get('timestep', 0),
                'global_metrics': snapshot.get('global_metrics', {})
            }
        }
    
    @staticmethod
    def convert_to_cytoscape(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert simulation snapshot to Cytoscape.js format.
        
        Cytoscape expects an array of elements with 'data' and optional 'classes'.
        """
        elements = []
        
        # Nodes
        for node in snapshot.get('network_state', {}).get('nodes', []):
            agent_state = snapshot.get('agent_states', {}).get(node['id'], {})
            health = agent_state.get('health', 0.5)
            
            classes = [node['type']]
            if not node.get('alive', True):
                classes.append('dead')
            elif health < 0.3:
                classes.append('critical')
            
            elements.append({
                'data': {
                    'id': node['id'],
                    'label': node['id'].replace('_', ' '),
                    'type': node['type'],
                    'health': health,
                    **agent_state
                },
                'classes': ' '.join(classes)
            })
        
        # Edges
        for edge in snapshot.get('network_state', {}).get('edges', []):
            elements.append({
                'data': {
                    'id': f"{edge['source']}-{edge['target']}",
                    'source': edge['source'],
                    'target': edge['target'],
                    'weight': edge.get('weight', 0),
                    'type': edge.get('type', 'loan')
                }
            })
        
        return elements
    
    @staticmethod
    def create_time_series(history: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """
        Extract time series data for a specific metric across all timesteps.
        
        Useful for dashboard charts.
        
        Args:
            history: List of snapshots
            metric: Metric to extract (e.g., 'survival_rate', 'avg_crar')
        
        Returns:
            Dict with timesteps and values
        """
        timesteps = []
        values = []
        
        for snapshot in history:
            timesteps.append(snapshot.get('timestep', 0))
            
            # Try to get from global_metrics
            value = snapshot.get('global_metrics', {}).get(metric)
            
            if value is not None:
                values.append(value)
            else:
                values.append(None)
        
        return {
            'metric': metric,
            'timesteps': timesteps,
            'values': values
        }
    
    @staticmethod
    def create_agent_time_series(
        history: List[Dict[str, Any]],
        agent_id: str,
        metric: str
    ) -> Dict[str, Any]:
        """
        Extract time series for a specific agent's metric.
        
        Example: Track BANK_1's CRAR over time.
        """
        timesteps = []
        values = []
        
        for snapshot in history:
            timesteps.append(snapshot.get('timestep', 0))
            
            agent_state = snapshot.get('agent_states', {}).get(agent_id, {})
            value = agent_state.get(metric)
            
            values.append(value)
        
        return {
            'agent_id': agent_id,
            'metric': metric,
            'timesteps': timesteps,
            'values': values
        }
    
    @staticmethod
    def compute_network_metrics(snapshot: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute topological metrics for visualization insights.
        
        Returns:
            Dict of network centrality and structure metrics
        """
        import networkx as nx
        
        # Reconstruct networkx graph from snapshot
        G = nx.DiGraph()
        
        for node in snapshot.get('network_state', {}).get('nodes', []):
            G.add_node(node['id'], **node)
        
        for edge in snapshot.get('network_state', {}).get('edges', []):
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))
        
        metrics = {}
        
        if G.number_of_nodes() > 0:
            # Degree centrality
            degree_cent = nx.degree_centrality(G)
            metrics['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
            metrics['max_degree_centrality'] = max(degree_cent.values())
            
            # Betweenness centrality (expensive for large graphs)
            if G.number_of_nodes() < 100:
                between_cent = nx.betweenness_centrality(G)
                metrics['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
            
            # Clustering coefficient (for undirected version)
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
            metrics['avg_clustering'] = np.mean(list(clustering.values()))
            
            # Density
            metrics['density'] = nx.density(G)
            
            # Number of connected components
            metrics['num_weakly_connected_components'] = nx.number_weakly_connected_components(G)
        
        return metrics
    
    @staticmethod
    def create_heatmap_data(
        history: List[Dict[str, Any]],
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create heatmap data showing agent health over time.
        
        Useful for dashboard matrix visualization.
        
        Returns:
            {
                'agents': [agent_id, ...],
                'timesteps': [0, 1, 2, ...],
                'matrix': [[health_t0_a0, health_t0_a1, ...], [health_t1_a0, ...]]
            }
        """
        if not history:
            return {'agents': [], 'timesteps': [], 'matrix': []}
        
        # Determine agent list
        if agent_ids is None:
            # Use all agents from first snapshot
            first_snapshot = history[0]
            agent_ids = list(first_snapshot.get('agent_states', {}).keys())
        
        timesteps = []
        matrix = []
        
        for snapshot in history:
            timesteps.append(snapshot.get('timestep', 0))
            
            row = []
            agent_states = snapshot.get('agent_states', {})
            
            for agent_id in agent_ids:
                health = agent_states.get(agent_id, {}).get('health', 0.0)
                row.append(health)
            
            matrix.append(row)
        
        return {
            'agents': agent_ids,
            'timesteps': timesteps,
            'matrix': matrix
        }
    
    @staticmethod
    def identify_critical_nodes(snapshot: Dict[str, Any], threshold: float = 0.3) -> List[str]:
        """
        Identify nodes in critical condition (health < threshold).
        
        Returns:
            List of agent IDs in distress
        """
        critical_nodes = []
        
        agent_states = snapshot.get('agent_states', {})
        
        for agent_id, state in agent_states.items():
            health = state.get('health', 1.0)
            alive = state.get('alive', True)
            
            if not alive or health < threshold:
                critical_nodes.append(agent_id)
        
        return critical_nodes
    
    @staticmethod
    def compute_cascade_tree(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the contagion cascade: which bank failures triggered others?
        
        Returns:
            Tree structure showing cascade propagation
        """
        cascade = {
            'root_failures': [],
            'cascades': []
        }
        
        previous_failed = set()
        
        for snapshot in history:
            current_failed = set()
            
            agent_states = snapshot.get('agent_states', {})
            for agent_id, state in agent_states.items():
                if not state.get('alive', True):
                    current_failed.add(agent_id)
            
            # New failures in this timestep
            new_failures = current_failed - previous_failed
            
            if new_failures:
                cascade['cascades'].append({
                    'timestep': snapshot.get('timestep', 0),
                    'new_failures': list(new_failures),
                    'total_failures': len(current_failed)
                })
            
            previous_failed = current_failed
        
        return cascade


# Example usage functions

def prepare_dashboard_data(simulation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepare a complete data package for a real-time dashboard.
    
    Returns all necessary visualizations and metrics.
    """
    if not simulation_history:
        return {}
    
    latest_snapshot = simulation_history[-1]
    
    return {
        # Network visualization
        'd3_network': NetworkVisualizer.convert_to_d3(latest_snapshot),
        'cytoscape_network': NetworkVisualizer.convert_to_cytoscape(latest_snapshot),
        
        # Time series
        'survival_rate_ts': NetworkVisualizer.create_time_series(simulation_history, 'survival_rate'),
        'avg_crar_ts': NetworkVisualizer.create_time_series(simulation_history, 'avg_crar'),
        'system_npa_ts': NetworkVisualizer.create_time_series(simulation_history, 'avg_npa'),
        
        # Heatmap
        'health_heatmap': NetworkVisualizer.create_heatmap_data(simulation_history),
        
        # Alerts
        'critical_nodes': NetworkVisualizer.identify_critical_nodes(latest_snapshot),
        
        # Network metrics
        'network_metrics': NetworkVisualizer.compute_network_metrics(latest_snapshot),
        
        # Cascade analysis
        'cascade_tree': NetworkVisualizer.compute_cascade_tree(simulation_history),
        
        # Current state
        'current_timestep': latest_snapshot.get('timestep', 0),
        'global_metrics': latest_snapshot.get('global_metrics', {})
    }
