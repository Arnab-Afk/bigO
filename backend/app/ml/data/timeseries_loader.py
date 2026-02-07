"""
Time Series Data Loader for LSTM Forecasting

Extracts sequences from simulation history for time series forecasting.
"""

import logging
from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np

from app.engine.simulation import TimestepState, SimulationState
from app.engine.game_theory import AgentState

logger = logging.getLogger(__name__)


class TimeSeriesLoader:
    """
    Extract time series sequences from simulation history
    
    For LSTM forecasting of institution states
    """
    
    def __init__(
        self,
        window_size: int = 20,
        forecast_horizon: int = 10,
    ):
        """
        Args:
            window_size: Length of historical input sequence
            forecast_horizon: Number of future timesteps to predict
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
    
    def extract_sequences_from_simulation(
        self,
        sim_state: SimulationState,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[UUID]]:
        """
        Extract sequences from a single simulation
        
        Args:
            sim_state: Complete simulation state
        
        Returns:
            Tuple of (input_sequences, target_sequences, institution_ids)
        """
        input_sequences = []
        target_sequences = []
        institution_ids = []
        
        # Get all timesteps
        timesteps = sim_state.timesteps
        
        if len(timesteps) < self.window_size + self.forecast_horizon:
            logger.warning(
                f"Simulation too short: {len(timesteps)} timesteps, "
                f"need {self.window_size + self.forecast_horizon}"
            )
            return [], [], []
        
        # Extract sequences for each institution
        for inst_id in timesteps[0].agent_states.keys():
            # Sliding window over simulation history
            for t in range(len(timesteps) - self.window_size - self.forecast_horizon):
                # Input: window_size historical states
                input_seq = []
                for i in range(t, t + self.window_size):
                    if inst_id in timesteps[i].agent_states:
                        state_vector = self._state_to_vector(
                            timesteps[i].agent_states[inst_id]
                        )
                        input_seq.append(state_vector)
                    else:
                        # Institution defaulted, use zero vector
                        input_seq.append(np.zeros(7))
                
                # Target: forecast_horizon future states
                target_seq = []
                for i in range(t + self.window_size, t + self.window_size + self.forecast_horizon):
                    if inst_id in timesteps[i].agent_states:
                        state_vector = self._state_to_vector(
                            timesteps[i].agent_states[inst_id]
                        )
                        target_seq.append(state_vector)
                    else:
                        target_seq.append(np.zeros(7))
                
                input_sequences.append(np.array(input_seq))
                target_sequences.append(np.array(target_seq))
                institution_ids.append(inst_id)
        
        return input_sequences, target_sequences, institution_ids
    
    def extract_sequences_from_multiple_simulations(
        self,
        sim_states: List[SimulationState],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract sequences from multiple simulations
        
        Args:
            sim_states: List of simulation states
        
        Returns:
            Tuple of (all_input_sequences, all_target_sequences)
        """
        all_input_sequences = []
        all_target_sequences = []
        
        for sim_state in sim_states:
            input_seqs, target_seqs, _ = self.extract_sequences_from_simulation(sim_state)
            all_input_sequences.extend(input_seqs)
            all_target_sequences.extend(target_seqs)
        
        logger.info(
            f"Extracted {len(all_input_sequences)} sequences from "
            f"{len(sim_states)} simulations"
        )
        
        return all_input_sequences, all_target_sequences
    
    def _state_to_vector(self, state: AgentState) -> np.ndarray:
        """
        Convert AgentState to feature vector for LSTM
        
        Features: [capital_ratio, liquidity_buffer, credit_exposure,
                   default_probability, stress_level, risk_appetite,
                   ml_prediction_confidence]
        """
        return np.array([
            state.capital_ratio,
            state.liquidity_buffer,
            state.credit_exposure / 1000.0,  # Normalize
            state.default_probability,
            state.stress_level,
            state.risk_appetite,
            state.ml_prediction_confidence,
        ], dtype=np.float32)
    
    @staticmethod
    def normalize_sequences(
        sequences: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Normalize sequences using z-score normalization
        
        Args:
            sequences: List of sequence arrays
        
        Returns:
            Tuple of (normalized_sequences, mean, std)
        """
        # Stack all sequences
        all_data = np.concatenate(sequences, axis=0)
        
        # Compute statistics
        mean = all_data.mean(axis=0, keepdims=True)
        std = all_data.std(axis=0, keepdims=True) + 1e-8
        
        # Normalize
        normalized_sequences = [
            (seq - mean) / std for seq in sequences
        ]
        
        return normalized_sequences, mean, std
