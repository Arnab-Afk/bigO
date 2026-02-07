"""
Graph Generation Module

Creates various visualizations for CCP risk analysis:
- Network graphs
- Risk distribution charts
- Time series plots
- Spectral analysis plots
"""

import io
import base64
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


class GraphGenerator:
    """Generates visualization graphs for CCP risk analysis"""
    
    def __init__(self):
        self.figure_cache = {}
    
    def generate_network_graph(self, network_builder, 
                              highlight_nodes: Optional[List[str]] = None,
                              format: str = 'plotly') -> Dict:
        """Generate interactive network graph"""
        if not NETWORKX_AVAILABLE or not network_builder or not network_builder.graph:
            return {"error": "Network not available"}
        
        if format == 'plotly' and PLOTLY_AVAILABLE:
            return self._plotly_network_graph(network_builder, highlight_nodes)
        else:
            return self._matplotlib_network_graph(network_builder, highlight_nodes)
    
    def _plotly_network_graph(self, network_builder, highlight_nodes=None) -> Dict:
        """Create interactive Plotly network graph"""
        G = network_builder.graph
        
        # Get layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        metrics = network_builder.compute_network_metrics()
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get node metrics
            node_metric = metrics[metrics['bank_name'] == node]
            if not node_metric.empty:
                degree = node_metric.iloc[0].get('degree_centrality', 0.5)
                pagerank = node_metric.iloc[0].get('pagerank', 0.01)
                
                # Color by centrality
                node_color.append(degree)
                # Size by pagerank
                node_size.append(max(10, pagerank * 1000))
            else:
                node_color.append(0.5)
                node_size.append(10)
        
        # Highlight specific nodes
        if highlight_nodes:
            for i, node in enumerate(G.nodes()):
                if node in highlight_nodes:
                    node_color[i] = 1.0
                    node_size[i] *= 1.5
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Centrality',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Banking Network Topology',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           height=700
                       ))
        
        return {
            "type": "plotly",
            "data": fig.to_json()
        }
    
    def _matplotlib_network_graph(self, network_builder, highlight_nodes=None) -> Dict:
        """Create static matplotlib network graph"""
        G = network_builder.graph
        
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Get metrics for node coloring
        metrics = network_builder.compute_network_metrics()
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_metric = metrics[metrics['bank_name'] == node]
            if not node_metric.empty:
                degree = node_metric.iloc[0].get('degree_centrality', 0.5)
                pagerank = node_metric.iloc[0].get('pagerank', 0.01)
                node_colors.append(degree)
                node_sizes.append(max(100, pagerank * 5000))
            else:
                node_colors.append(0.5)
                node_sizes.append(100)
        
        # Draw network
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              cmap='YlOrRd',
                              vmin=0, vmax=1,
                              ax=ax)
        
        # Highlight nodes
        if highlight_nodes:
            highlight_pos = {k: v for k, v in pos.items() if k in highlight_nodes}
            nx.draw_networkx_nodes(G, highlight_pos,
                                  nodelist=highlight_nodes,
                                  node_color='red',
                                  node_size=[s*1.5 for s in node_sizes],
                                  ax=ax)
        
        ax.set_title('Banking Network Topology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "type": "image",
            "format": "png",
            "data": img_base64
        }
    
    def generate_risk_distribution(self, risk_scores: pd.DataFrame, 
                                   format: str = 'plotly') -> Dict:
        """Generate risk distribution charts"""
        if format == 'plotly' and PLOTLY_AVAILABLE:
            return self._plotly_risk_distribution(risk_scores)
        else:
            return self._matplotlib_risk_distribution(risk_scores)
    
    def _plotly_risk_distribution(self, risk_scores: pd.DataFrame) -> Dict:
        """Create Plotly risk distribution"""
        if 'stress_level' not in risk_scores.columns:
            return {"error": "No stress_level data"}
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stress Level Distribution', 'Capital Ratio Distribution',
                          'Risk Categories', 'Default Probability'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'pie'}, {'type': 'histogram'}]]
        )
        
        # Stress level histogram
        fig.add_trace(
            go.Histogram(x=risk_scores['stress_level'], name='Stress',
                        marker_color='rgba(255, 100, 100, 0.7)',
                        nbinsx=20),
            row=1, col=1
        )
        
        # Capital ratio histogram
        if 'capital_ratio' in risk_scores.columns:
            fig.add_trace(
                go.Histogram(x=risk_scores['capital_ratio'], name='Capital',
                            marker_color='rgba(100, 200, 100, 0.7)',
                            nbinsx=20),
                row=1, col=2
            )
        
        # Risk categories pie
        risk_categories = {
            'Low': (risk_scores['stress_level'] <= 0.3).sum(),
            'Medium': ((risk_scores['stress_level'] > 0.3) & 
                      (risk_scores['stress_level'] <= 0.7)).sum(),
            'High': (risk_scores['stress_level'] > 0.7).sum()
        }
        
        fig.add_trace(
            go.Pie(labels=list(risk_categories.keys()),
                  values=list(risk_categories.values()),
                  marker=dict(colors=['green', 'yellow', 'red'])),
            row=2, col=1
        )
        
        # Default probability
        if 'default_probability' in risk_scores.columns:
            fig.add_trace(
                go.Histogram(x=risk_scores['default_probability'], name='Default Prob',
                            marker_color='rgba(255, 150, 0, 0.7)',
                            nbinsx=20),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Risk Analysis Dashboard")
        
        return {
            "type": "plotly",
            "data": fig.to_json()
        }
    
    def _matplotlib_risk_distribution(self, risk_scores: pd.DataFrame) -> Dict:
        """Create matplotlib risk distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Stress level distribution
        if 'stress_level' in risk_scores.columns:
            axes[0, 0].hist(risk_scores['stress_level'], bins=20, 
                          color='coral', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Stress Level Distribution')
            axes[0, 0].set_xlabel('Stress Level')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(0.7, color='red', linestyle='--', label='High Risk Threshold')
            axes[0, 0].legend()
        
        # Capital ratio distribution
        if 'capital_ratio' in risk_scores.columns:
            axes[0, 1].hist(risk_scores['capital_ratio'], bins=20,
                          color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Capital Ratio Distribution')
            axes[0, 1].set_xlabel('Capital Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0.09, color='red', linestyle='--', label='Minimum Requirement')
            axes[0, 1].legend()
        
        # Risk categories pie chart
        if 'stress_level' in risk_scores.columns:
            risk_categories = [
                (risk_scores['stress_level'] <= 0.3).sum(),
                ((risk_scores['stress_level'] > 0.3) & (risk_scores['stress_level'] <= 0.7)).sum(),
                (risk_scores['stress_level'] > 0.7).sum()
            ]
            axes[1, 0].pie(risk_categories, labels=['Low', 'Medium', 'High'],
                         colors=['green', 'yellow', 'red'], autopct='%1.1f%%')
            axes[1, 0].set_title('Risk Distribution by Category')
        
        # Default probability
        if 'default_probability' in risk_scores.columns:
            axes[1, 1].hist(risk_scores['default_probability'], bins=20,
                          color='orange', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Default Probability Distribution')
            axes[1, 1].set_xlabel('Default Probability')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "type": "image",
            "format": "png",
            "data": img_base64
        }
    
    def generate_time_series(self, history: List[Dict], format: str = 'plotly') -> Dict:
        """Generate time series plots from simulation history"""
        if not history:
            return {"error": "No history data"}
        
        if format == 'plotly' and PLOTLY_AVAILABLE:
            return self._plotly_time_series(history)
        else:
            return self._matplotlib_time_series(history)
    
    def _plotly_time_series(self, history: List[Dict]) -> Dict:
        """Create Plotly time series"""
        timesteps = [h['timestep'] for h in history]
        defaults = [h['default_count'] for h in history]
        avg_stress = [h['total_stress'] for h in history]
        avg_capital = [h['average_capital_ratio'] for h in history]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Default Count Over Time', 'Average Stress Level', 'Average Capital Ratio'),
            vertical_spacing=0.1
        )
        
        # Defaults
        fig.add_trace(
            go.Scatter(x=timesteps, y=defaults, mode='lines+markers',
                      name='Defaults', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Stress
        fig.add_trace(
            go.Scatter(x=timesteps, y=avg_stress, mode='lines+markers',
                      name='Stress', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # Capital
        fig.add_trace(
            go.Scatter(x=timesteps, y=avg_capital, mode='lines+markers',
                      name='Capital', line=dict(color='green', width=2)),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Timestep", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Level", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
        
        fig.update_layout(height=900, showlegend=False, title_text="Simulation Time Series")
        
        return {
            "type": "plotly",
            "data": fig.to_json()
        }
    
    def _matplotlib_time_series(self, history: List[Dict]) -> Dict:
        """Create matplotlib time series"""
        timesteps = [h['timestep'] for h in history]
        defaults = [h['default_count'] for h in history]
        avg_stress = [h['total_stress'] for h in history]
        avg_capital = [h['average_capital_ratio'] for h in history]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Defaults
        axes[0].plot(timesteps, defaults, 'ro-', linewidth=2, markersize=4)
        axes[0].set_ylabel('Default Count')
        axes[0].set_title('Default Count Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Stress
        axes[1].plot(timesteps, avg_stress, 'o-', color='orange', linewidth=2, markersize=4)
        axes[1].set_ylabel('Average Stress Level')
        axes[1].set_title('System Stress Evolution')
        axes[1].grid(True, alpha=0.3)
        
        # Capital
        axes[2].plot(timesteps, avg_capital, 'go-', linewidth=2, markersize=4)
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Average Capital Ratio')
        axes[2].set_title('Capital Adequacy Over Time')
        axes[2].axhline(0.09, color='red', linestyle='--', alpha=0.5, label='Min Requirement')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Simulation Time Series', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "type": "image",
            "format": "png",
            "data": img_base64
        }
    
    def generate_spectral_analysis(self, spectral_results: Dict, 
                                   eigenvalues: Optional[np.ndarray] = None,
                                   format: str = 'plotly') -> Dict:
        """Generate spectral analysis plots"""
        if format == 'plotly' and PLOTLY_AVAILABLE and eigenvalues is not None:
            return self._plotly_spectral(spectral_results, eigenvalues)
        else:
            return self._matplotlib_spectral(spectral_results, eigenvalues)
    
    def _plotly_spectral(self, spectral_results: Dict, eigenvalues: np.ndarray) -> Dict:
        """Create Plotly spectral analysis"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Eigenvalue Spectrum', 'Key Metrics')
        )
        
        # Eigenvalue spectrum
        if eigenvalues is not None:
            fig.add_trace(
                go.Scatter(x=list(range(len(eigenvalues))), y=eigenvalues,
                          mode='markers',
                          marker=dict(size=8, color=eigenvalues, colorscale='Viridis'),
                          name='Eigenvalues'),
                row=1, col=1
            )
        
        # Key metrics bar chart
        metrics = {
            'Spectral Radius': spectral_results.get('spectral_radius', 0),
            'Fiedler Value': spectral_results.get('fiedler_value', 0),
            'Spectral Gap': spectral_results.get('spectral_gap', 0),
            'Contagion Index': spectral_results.get('contagion_index', 0)
        }
        
        fig.add_trace(
            go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                  marker_color=['red', 'blue', 'green', 'orange']),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Index", row=1, col=1)
        fig.update_yaxes(title_text="Eigenvalue", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=False, title_text="Spectral Analysis")
        
        return {
            "type": "plotly",
            "data": fig.to_json()
        }
    
    def _matplotlib_spectral(self, spectral_results: Dict, eigenvalues: Optional[np.ndarray]) -> Dict:
        """Create matplotlib spectral analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Eigenvalue spectrum
        if eigenvalues is not None:
            axes[0].scatter(range(len(eigenvalues)), eigenvalues, 
                          c=eigenvalues, cmap='viridis', s=50, alpha=0.7)
            axes[0].set_xlabel('Index')
            axes[0].set_ylabel('Eigenvalue')
            axes[0].set_title('Eigenvalue Spectrum')
            axes[0].grid(True, alpha=0.3)
        
        # Key metrics
        metrics = {
            'Spectral\nRadius': spectral_results.get('spectral_radius', 0),
            'Fiedler\nValue': spectral_results.get('fiedler_value', 0),
            'Spectral\nGap': spectral_results.get('spectral_gap', 0),
            'Contagion\nIndex': spectral_results.get('contagion_index', 0)
        }
        
        bars = axes[1].bar(metrics.keys(), metrics.values(),
                          color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Key Spectral Metrics')
        axes[1].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Spectral Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "type": "image",
            "format": "png",
            "data": img_base64
        }
