"""
Real-time Simulation Engine

Enables progressive simulation with timestep-based execution and live updates.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


@dataclass
class SimulationStep:
    """Represents one timestep in the simulation"""
    timestep: int
    timestamp: str
    bank_states: List[Dict]
    network_metrics: Dict
    spectral_metrics: Dict
    risk_distribution: Dict
    default_count: int
    total_stress: float
    average_capital_ratio: float


class RealtimeSimulation:
    """Manages real-time progressive simulation"""
    
    def __init__(self, data_loader, feature_engineer, network_builder, 
                 spectral_analyzer, risk_model, ccp_engine):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.network_builder = network_builder
        self.spectral_analyzer = spectral_analyzer
        self.risk_model = risk_model
        self.ccp_engine = ccp_engine
        
        self.is_running = False
        self.current_timestep = 0
        self.max_timesteps = 100
        self.history: List[SimulationStep] = []
        self.initial_features = None
        
    def reset(self, max_timesteps: int = 100):
        """Reset simulation to initial state"""
        self.is_running = False
        self.current_timestep = 0
        self.max_timesteps = max_timesteps
        self.history = []
        logger.info(f"Simulation reset with {max_timesteps} timesteps")
        
    def initialize(self, features: pd.DataFrame):
        """Initialize simulation with base features"""
        self.initial_features = features.copy()
        self.current_features = features.copy()
        self.reset()
        logger.info(f"Simulation initialized with {len(features)} banks")
        
    def step(self, shock_config: Optional[Dict] = None) -> SimulationStep:
        """Execute one simulation timestep"""
        if self.current_timestep >= self.max_timesteps:
            logger.warning("Simulation already at max timesteps")
            return None
            
        self.current_timestep += 1
        
        # Apply shock if provided
        if shock_config:
            self._apply_shock(shock_config)
        
        # Propagate contagion through network
        self._propagate_contagion()
        
        # Update bank states based on network effects
        self._update_bank_states()
        
        # Capture current state
        step_data = self._capture_state()
        self.history.append(step_data)
        
        logger.info(f"Timestep {self.current_timestep}: {step_data.default_count} defaults, "
                   f"avg stress: {step_data.total_stress:.3f}")
        
        return step_data
    
    def run_multiple_steps(self, n_steps: int, 
                          shock_config: Optional[Dict] = None) -> List[SimulationStep]:
        """Run multiple simulation steps"""
        steps = []
        for _ in range(n_steps):
            if self.current_timestep >= self.max_timesteps:
                break
            step = self.step(shock_config if _ == 0 else None)
            if step:
                steps.append(step)
        return steps
    
    async def run_async(self, n_steps: int, 
                       shock_config: Optional[Dict] = None,
                       callback=None) -> List[SimulationStep]:
        """Run simulation asynchronously with callbacks"""
        self.is_running = True
        steps = []
        
        for i in range(n_steps):
            if not self.is_running or self.current_timestep >= self.max_timesteps:
                break
                
            step = self.step(shock_config if i == 0 else None)
            if step:
                steps.append(step)
                if callback:
                    await callback(step)
                await asyncio.sleep(0.1)  # Small delay for real-time feel
        
        self.is_running = False
        return steps
    
    def stop(self):
        """Stop running simulation"""
        self.is_running = False
        logger.info("Simulation stopped")
    
    def _apply_shock(self, config: Dict):
        """Apply shock to banks"""
        shock_type = config.get('type', 'capital')
        magnitude = config.get('magnitude', 0.2)
        target_banks = config.get('target_banks', None)
        
        if shock_type == 'capital':
            if 'capital_ratio' in self.current_features.columns:
                if target_banks:
                    mask = self.current_features['bank_name'].isin(target_banks)
                    self.current_features.loc[mask, 'capital_ratio'] *= (1 - magnitude)
                else:
                    self.current_features['capital_ratio'] *= (1 - magnitude)
        
        elif shock_type == 'liquidity':
            if 'liquidity_buffer' in self.current_features.columns:
                if target_banks:
                    mask = self.current_features['bank_name'].isin(target_banks)
                    self.current_features.loc[mask, 'liquidity_buffer'] *= (1 - magnitude)
                else:
                    self.current_features['liquidity_buffer'] *= (1 - magnitude)
        
        elif shock_type == 'stress':
            if 'stress_level' in self.current_features.columns:
                if target_banks:
                    mask = self.current_features['bank_name'].isin(target_banks)
                    self.current_features.loc[mask, 'stress_level'] += magnitude
                else:
                    self.current_features['stress_level'] += magnitude
                self.current_features['stress_level'] = self.current_features['stress_level'].clip(0, 1)
    
    def _propagate_contagion(self):
        """Propagate stress through network connections"""
        if not self.network_builder or not self.network_builder.graph:
            return
        
        # Get network adjacency
        graph = self.network_builder.graph
        
        # Update neighbor stress metrics
        for bank in self.current_features['bank_name']:
            if bank in graph:
                neighbors = list(graph.neighbors(bank))
                if neighbors:
                    neighbor_mask = self.current_features['bank_name'].isin(neighbors)
                    neighbor_stress = self.current_features.loc[neighbor_mask, 'stress_level']
                    
                    # Update neighbor metrics
                    idx = self.current_features[self.current_features['bank_name'] == bank].index
                    if len(idx) > 0 and not neighbor_stress.empty:
                        self.current_features.loc[idx, 'neighbor_avg_stress'] = neighbor_stress.mean()
                        self.current_features.loc[idx, 'neighbor_max_stress'] = neighbor_stress.max()
    
    def _update_bank_states(self):
        """Update bank states based on current conditions"""
        # Increase stress based on neighbor effects
        if 'neighbor_avg_stress' in self.current_features.columns:
            contagion_factor = 0.1  # 10% contagion effect
            self.current_features['stress_level'] += (
                self.current_features['neighbor_avg_stress'] * contagion_factor
            )
            self.current_features['stress_level'] = self.current_features['stress_level'].clip(0, 1)
        
        # Update default status
        if 'capital_ratio' in self.current_features.columns:
            # Banks default if capital ratio drops below threshold
            default_threshold = 0.09  # 9% minimum capital
            self.current_features['defaulted'] = (
                self.current_features['capital_ratio'] < default_threshold
            ).astype(int)
        
        # Decrease capital due to stress
        if 'stress_level' in self.current_features.columns and 'capital_ratio' in self.current_features.columns:
            stress_impact = 0.01  # 1% capital erosion per stress unit
            self.current_features['capital_ratio'] -= (
                self.current_features['stress_level'] * stress_impact
            )
            self.current_features['capital_ratio'] = self.current_features['capital_ratio'].clip(0, 1)
    
    def _capture_state(self) -> SimulationStep:
        """Capture current simulation state"""
        # Bank states
        bank_states = self.current_features[[
            'bank_name', 'capital_ratio', 'stress_level', 'defaulted'
        ]].to_dict('records') if all(col in self.current_features.columns 
                                     for col in ['bank_name', 'capital_ratio', 'stress_level', 'defaulted']) else []
        
        # Network metrics
        network_metrics = {}
        if self.network_builder and self.network_builder.graph:
            network_metrics = {
                'num_nodes': self.network_builder.graph.number_of_nodes(),
                'num_edges': self.network_builder.graph.number_of_edges(),
                'density': nx.density(self.network_builder.graph) if hasattr(nx, 'density') else 0
            }
        
        # Spectral metrics
        spectral_metrics = {}
        if self.spectral_analyzer and self.network_builder:
            try:
                results = self.spectral_analyzer.analyze(network_builder=self.network_builder)
                spectral_metrics = {
                    'spectral_radius': results.spectral_radius,
                    'fiedler_value': results.fiedler_value,
                    'contagion_index': self.spectral_analyzer.compute_contagion_index()
                }
            except:
                pass
        
        # Risk distribution
        risk_distribution = {}
        if 'stress_level' in self.current_features.columns:
            risk_distribution = {
                'low': int((self.current_features['stress_level'] <= 0.3).sum()),
                'medium': int(((self.current_features['stress_level'] > 0.3) & 
                              (self.current_features['stress_level'] <= 0.7)).sum()),
                'high': int((self.current_features['stress_level'] > 0.7).sum())
            }
        
        # Aggregate metrics
        default_count = int(self.current_features.get('defaulted', pd.Series([0])).sum())
        total_stress = float(self.current_features.get('stress_level', pd.Series([0])).mean())
        avg_capital = float(self.current_features.get('capital_ratio', pd.Series([0])).mean())
        
        return SimulationStep(
            timestep=self.current_timestep,
            timestamp=datetime.now().isoformat(),
            bank_states=bank_states,
            network_metrics=network_metrics,
            spectral_metrics=spectral_metrics,
            risk_distribution=risk_distribution,
            default_count=default_count,
            total_stress=total_stress,
            average_capital_ratio=avg_capital
        )
    
    def get_history(self) -> List[Dict]:
        """Get full simulation history"""
        return [asdict(step) for step in self.history]
    
    def get_current_state(self) -> Optional[SimulationStep]:
        """Get most recent simulation state"""
        return self.history[-1] if self.history else None
