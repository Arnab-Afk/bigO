"""
Bayesian Belief Update Mechanism

Implements incomplete information handling and belief updates.
Based on Technical Documentation Section 5.3.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np


@dataclass
class Signal:
    """Information signal about an institution"""
    source_id: UUID
    target_id: UUID
    signal_type: str  # 'rating_change', 'price_movement', 'default', etc.
    value: float
    reliability: float  # 0-1 scale
    timestamp: int


@dataclass
class BeliefDistribution:
    """Probability distribution over institution types/states"""
    institution_id: UUID
    beliefs: Dict[str, float]  # state -> probability
    
    def entropy(self) -> float:
        """Calculate entropy of belief distribution"""
        entropy = 0.0
        for prob in self.beliefs.values():
            if prob > 0:
                entropy -= prob * np.log(prob)
        return entropy
    
    def most_likely_state(self) -> str:
        """Return most likely state"""
        return max(self.beliefs.items(), key=lambda x: x[1])[0]


class SignalProcessor:
    """
    Process and generate signals about institution states
    """
    
    def __init__(self, signal_precision: float = 0.8):
        """
        Args:
            signal_precision: Accuracy of signals (0-1)
        """
        self.precision = signal_precision
    
    def generate_signal(
        self,
        true_state: str,
        observed_by: UUID,
        about: UUID,
        signal_type: str = "state_observation",
        noise_level: float = 0.2
    ) -> Signal:
        """
        Generate a noisy signal about institution's true state
        
        Args:
            true_state: Actual state of the institution
            observed_by: Institution making the observation
            about: Institution being observed
            signal_type: Type of signal
            noise_level: Amount of noise (0-1)
        
        Returns:
            Signal object
        """
        # With probability = precision, signal is accurate
        if np.random.random() < self.precision:
            # Accurate signal
            value = self._state_to_value(true_state)
            noise = np.random.normal(0, noise_level * 0.1)
        else:
            # Noisy signal
            value = np.random.uniform(0, 1)
            noise = np.random.normal(0, noise_level)
        
        return Signal(
            source_id=observed_by,
            target_id=about,
            signal_type=signal_type,
            value=value + noise,
            reliability=self.precision * (1 - noise_level),
            timestamp=0,
        )
    
    def _state_to_value(self, state: str) -> float:
        """Convert state to numerical value"""
        state_map = {
            "healthy": 0.8,
            "stressed": 0.5,
            "distressed": 0.2,
            "defaulted": 0.0,
        }
        return state_map.get(state, 0.5)
    
    def extract_signals_from_market(
        self,
        institution_states: Dict[UUID, Dict],
        market_data: Dict
    ) -> List[Signal]:
        """
        Extract observable signals from market data
        
        Args:
            institution_states: Current states of institutions
            market_data: Market prices, spreads, etc.
        
        Returns:
            List of generated signals
        """
        signals = []
        
        for inst_id, state in institution_states.items():
            # Credit spread signal
            if inst_id in market_data.get("credit_spreads", {}):
                spread = market_data["credit_spreads"][inst_id]
                signals.append(
                    Signal(
                        source_id=UUID(int=0),  # Market signal
                        target_id=inst_id,
                        signal_type="credit_spread",
                        value=spread,
                        reliability=0.7,
                        timestamp=0,
                    )
                )
            
            # Stock price signal
            if inst_id in market_data.get("stock_prices", {}):
                price_change = market_data["stock_prices"][inst_id]
                signals.append(
                    Signal(
                        source_id=UUID(int=0),
                        target_id=inst_id,
                        signal_type="stock_price",
                        value=price_change,
                        reliability=0.6,
                        timestamp=0,
                    )
                )
        
        return signals


class BayesianBeliefUpdater:
    """
    Bayesian belief update system for incomplete information games
    
    Implements: P(θ|s) ∝ P(s|θ) × P(θ)
    """
    
    def __init__(self, prior_beliefs: Optional[Dict[UUID, BeliefDistribution]] = None):
        """
        Args:
            prior_beliefs: Initial belief distributions
        """
        self.beliefs = prior_beliefs or {}
        self.signal_history: List[Signal] = []
    
    def initialize_beliefs(
        self,
        institutions: List[UUID],
        default_distribution: Optional[Dict[str, float]] = None
    ):
        """
        Initialize uniform or custom prior beliefs
        
        Args:
            institutions: List of institution IDs
            default_distribution: Default belief distribution
        """
        if default_distribution is None:
            # Uniform prior over states
            default_distribution = {
                "healthy": 0.70,
                "stressed": 0.20,
                "distressed": 0.08,
                "defaulted": 0.02,
            }
        
        for inst_id in institutions:
            self.beliefs[inst_id] = BeliefDistribution(
                institution_id=inst_id,
                beliefs=default_distribution.copy()
            )
    
    def update_belief(
        self,
        institution_id: UUID,
        signal: Signal,
        likelihood_model: Optional[Dict[str, Dict[str, float]]] = None
    ) -> BeliefDistribution:
        """
        Update belief about institution given new signal using Bayes' rule
        
        P(state|signal) ∝ P(signal|state) × P(state)
        
        Args:
            institution_id: Institution being updated
            signal: New information signal
            likelihood_model: P(signal|state) mapping
        
        Returns:
            Updated belief distribution
        """
        if institution_id not in self.beliefs:
            # Initialize if not exists
            self.initialize_beliefs([institution_id])
        
        prior = self.beliefs[institution_id]
        
        # Default likelihood model
        if likelihood_model is None:
            likelihood_model = self._default_likelihood_model(signal)
        
        # Bayesian update
        posterior = {}
        normalizer = 0.0
        
        for state, prior_prob in prior.beliefs.items():
            # Likelihood P(signal|state)
            likelihood = self._compute_likelihood(signal, state, likelihood_model)
            
            # Posterior ∝ likelihood × prior
            posterior[state] = likelihood * prior_prob
            normalizer += posterior[state]
        
        # Normalize
        if normalizer > 0:
            for state in posterior:
                posterior[state] /= normalizer
        else:
            # Fallback to prior if normalization fails
            posterior = prior.beliefs.copy()
        
        # Update stored beliefs
        self.beliefs[institution_id] = BeliefDistribution(
            institution_id=institution_id,
            beliefs=posterior
        )
        
        # Record signal
        self.signal_history.append(signal)
        
        return self.beliefs[institution_id]
    
    def _compute_likelihood(
        self,
        signal: Signal,
        state: str,
        likelihood_model: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute P(signal|state)
        """
        if signal.signal_type in likelihood_model:
            if state in likelihood_model[signal.signal_type]:
                # Use model
                expected_value = likelihood_model[signal.signal_type][state]
                
                # Gaussian likelihood around expected value
                variance = (1 - signal.reliability) ** 2
                likelihood = np.exp(
                    -((signal.value - expected_value) ** 2) / (2 * variance)
                )
                return float(likelihood)
        
        # Default uniform likelihood
        return 0.25
    
    def _default_likelihood_model(self, signal: Signal) -> Dict[str, Dict[str, float]]:
        """
        Create default likelihood model based on signal type
        """
        if signal.signal_type == "credit_spread":
            return {
                "credit_spread": {
                    "healthy": 0.01,      # Low spread
                    "stressed": 0.03,     # Medium spread
                    "distressed": 0.07,   # High spread
                    "defaulted": 0.20,    # Very high spread
                }
            }
        elif signal.signal_type == "stock_price":
            return {
                "stock_price": {
                    "healthy": 0.10,      # Positive return
                    "stressed": 0.00,     # Flat
                    "distressed": -0.15,  # Negative return
                    "defaulted": -0.50,   # Large negative
                }
            }
        else:
            # Generic state observation
            return {
                signal.signal_type: {
                    "healthy": 0.8,
                    "stressed": 0.5,
                    "distressed": 0.2,
                    "defaulted": 0.0,
                }
            }
    
    def update_from_batch(
        self,
        signals: List[Signal]
    ) -> Dict[UUID, BeliefDistribution]:
        """
        Update beliefs from multiple signals
        
        Args:
            signals: List of signals to process
        
        Returns:
            Updated belief distributions
        """
        updated_beliefs = {}
        
        for signal in signals:
            belief = self.update_belief(signal.target_id, signal)
            updated_beliefs[signal.target_id] = belief
        
        return updated_beliefs
    
    def get_expected_default_probability(self, institution_id: UUID) -> float:
        """
        Compute expected default probability from belief distribution
        """
        if institution_id not in self.beliefs:
            return 0.0
        
        belief = self.beliefs[institution_id]
        
        # Weight states by default likelihood
        state_to_pd = {
            "healthy": 0.01,
            "stressed": 0.05,
            "distressed": 0.20,
            "defaulted": 1.00,
        }
        
        expected_pd = sum(
            belief.beliefs.get(state, 0.0) * pd
            for state, pd in state_to_pd.items()
        )
        
        return expected_pd
    
    def compute_information_entropy(self) -> Dict[UUID, float]:
        """
        Compute entropy for each institution's belief distribution
        
        High entropy = high uncertainty
        """
        return {
            inst_id: belief.entropy()
            for inst_id, belief in self.beliefs.items()
        }
    
    def detect_belief_cascades(
        self,
        threshold: float = 0.5
    ) -> List[Tuple[UUID, str]]:
        """
        Detect institutions where beliefs have shifted dramatically
        
        Args:
            threshold: Minimum probability shift to detect
        
        Returns:
            List of (institution_id, new_likely_state) tuples
        """
        cascades = []
        
        for inst_id, belief in self.beliefs.items():
            most_likely = belief.most_likely_state()
            prob = belief.beliefs[most_likely]
            
            # If strong belief in worse state
            if most_likely in ["distressed", "defaulted"] and prob > threshold:
                cascades.append((inst_id, most_likely))
        
        return cascades
    
    def get_correlation_network(self) -> Dict[Tuple[UUID, UUID], float]:
        """
        Compute belief correlation between institutions
        
        Returns:
            Dictionary of (inst_i, inst_j) -> correlation
        """
        correlations = {}
        institutions = list(self.beliefs.keys())
        
        for i, inst_i in enumerate(institutions):
            for inst_j in institutions[i+1:]:
                # Compute correlation based on belief similarity
                belief_i = self.beliefs[inst_i].beliefs
                belief_j = self.beliefs[inst_j].beliefs
                
                # Cosine similarity
                dot_product = sum(
                    belief_i.get(state, 0) * belief_j.get(state, 0)
                    for state in belief_i.keys()
                )
                
                norm_i = np.sqrt(sum(p ** 2 for p in belief_i.values()))
                norm_j = np.sqrt(sum(p ** 2 for p in belief_j.values()))
                
                if norm_i > 0 and norm_j > 0:
                    correlation = dot_product / (norm_i * norm_j)
                    correlations[(inst_i, inst_j)] = correlation
        
        return correlations
