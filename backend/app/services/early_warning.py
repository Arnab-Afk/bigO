"""
Early Warning Service

Provides real-time early warning system for financial institutions
using LSTM forecasts to detect deterioration trends.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from app.ml.models.state_forecaster import StateForecastLSTM

logger = logging.getLogger(__name__)


@dataclass
class RiskAlert:
    """Risk alert for an institution"""
    institution_id: str
    alert_type: str  # 'capital_deterioration', 'default_risk', 'liquidity_crisis'
    severity: str  # 'low', 'medium', 'high', 'critical'
    risk_score: float
    message: str
    forecasted_capital_ratio: List[float]
    forecasted_default_prob: List[float]
    time_to_default: Optional[int]  # Estimated timesteps until default


class EarlyWarningService:
    """
    Early Warning Service for Financial Institutions

    Uses LSTM forecasts to:
    - Predict future states 10 timesteps ahead
    - Detect deterioration trends
    - Generate risk alerts
    - Estimate time to default
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        risk_threshold: float = 0.6,
        capital_warning_threshold: float = 0.09,  # 9% CRAR
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_path: Path to trained LSTM model
            risk_threshold: Threshold for generating alerts
            capital_warning_threshold: Minimum acceptable capital ratio
            device: torch device
        """
        self.risk_threshold = risk_threshold
        self.capital_warning_threshold = capital_warning_threshold

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load model if path provided
        self.model = None
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning("No model loaded - early warning system will use heuristics")

    def load_model(self, model_path: Path) -> None:
        """Load trained LSTM model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})

        self.model = StateForecastLSTM(
            input_dim=config.get('input_dim', 7),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            forecast_horizon=config.get('forecast_horizon', 10),
            dropout=config.get('dropout', 0.3),
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"LSTM model loaded from {model_path}")

    def predict_future_states(
        self,
        historical_sequence: np.ndarray,
        num_steps: int = 10,
    ) -> np.ndarray:
        """
        Predict future states using LSTM

        Args:
            historical_sequence: Historical state sequence [seq_len, input_dim]
            num_steps: Number of timesteps to predict

        Returns:
            Forecasted states [num_steps, input_dim]
        """
        if self.model is None:
            # Fallback to simple linear extrapolation
            return self._heuristic_forecast(historical_sequence, num_steps)

        # Prepare input
        # historical_sequence shape: [seq_len, input_dim]
        # Model expects: [batch_size, seq_len, input_dim]
        input_seq = torch.FloatTensor(historical_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            forecast = self.model.predict_multi_step(input_seq, num_steps)

        # Shape: [1, num_steps, input_dim] -> [num_steps, input_dim]
        forecast = forecast.squeeze(0).cpu().numpy()

        return forecast

    def _heuristic_forecast(
        self,
        historical_sequence: np.ndarray,
        num_steps: int,
    ) -> np.ndarray:
        """
        Simple heuristic forecast using linear extrapolation

        Args:
            historical_sequence: Historical state sequence [seq_len, input_dim]
            num_steps: Number of timesteps to predict

        Returns:
            Forecasted states [num_steps, input_dim]
        """
        seq_len, input_dim = historical_sequence.shape

        if seq_len < 2:
            # Just repeat last state
            return np.tile(historical_sequence[-1], (num_steps, 1))

        # Linear trend for each feature
        forecasts = []
        for step in range(1, num_steps + 1):
            # Simple linear extrapolation
            trend = historical_sequence[-1] - historical_sequence[0]
            trend_per_step = trend / seq_len

            forecast_state = historical_sequence[-1] + trend_per_step * step

            # Clamp to valid ranges
            forecast_state = np.clip(forecast_state, 0.0, 1.0)

            forecasts.append(forecast_state)

        return np.array(forecasts)

    def detect_deterioration_trend(
        self,
        historical_sequence: np.ndarray,
        forecast: np.ndarray,
    ) -> Dict[str, bool]:
        """
        Detect deterioration trends in forecasted states

        Args:
            historical_sequence: Historical state sequence [seq_len, input_dim]
            forecast: Forecasted states [num_steps, input_dim]

        Returns:
            Dictionary with deterioration indicators
        """
        # State features indices
        # [capital_ratio, liquidity_buffer, credit_exposure,
        #  default_probability, stress_level, risk_appetite, ml_conf]

        current_capital = historical_sequence[-1, 0]
        forecast_capital = forecast[:, 0]

        current_liquidity = historical_sequence[-1, 1]
        forecast_liquidity = forecast[:, 1]

        current_default_prob = historical_sequence[-1, 3]
        forecast_default_prob = forecast[:, 3]

        current_stress = historical_sequence[-1, 4]
        forecast_stress = forecast[:, 4]

        # Check for declining trends
        capital_declining = forecast_capital[-1] < current_capital
        liquidity_declining = forecast_liquidity[-1] < current_liquidity
        default_prob_rising = forecast_default_prob[-1] > current_default_prob
        stress_increasing = forecast_stress[-1] > current_stress

        # Check if CRAR falls below threshold
        crar_below_threshold = np.any(forecast_capital < self.capital_warning_threshold)

        return {
            'capital_declining': capital_declining,
            'liquidity_declining': liquidity_declining,
            'default_prob_rising': default_prob_rising,
            'stress_increasing': stress_increasing,
            'crar_below_threshold': crar_below_threshold,
        }

    def generate_risk_alerts(
        self,
        institution_id: str,
        historical_sequence: np.ndarray,
        forecast: Optional[np.ndarray] = None,
    ) -> List[RiskAlert]:
        """
        Generate risk alerts for an institution

        Args:
            institution_id: Institution identifier
            historical_sequence: Historical state sequence
            forecast: Pre-computed forecast (will compute if None)

        Returns:
            List of risk alerts
        """
        if forecast is None:
            forecast = self.predict_future_states(historical_sequence)

        trends = self.detect_deterioration_trend(historical_sequence, forecast)

        # Calculate risk score
        risk_score = (
            float(trends['capital_declining']) * 0.3 +
            float(trends['liquidity_declining']) * 0.2 +
            float(trends['default_prob_rising']) * 0.3 +
            float(trends['stress_increasing']) * 0.1 +
            float(trends['crar_below_threshold']) * 0.1
        )

        alerts = []

        # Generate alerts based on risk score and trends
        if risk_score > self.risk_threshold:
            severity = 'critical' if risk_score > 0.8 else 'high' if risk_score > 0.7 else 'medium'

            # Determine alert type
            if trends['crar_below_threshold']:
                alert_type = 'capital_deterioration'
                message = f"Capital ratio forecasted to fall below regulatory minimum ({self.capital_warning_threshold * 100:.1f}%)"
            elif trends['default_prob_rising']:
                alert_type = 'default_risk'
                message = "Default probability trending upward significantly"
            elif trends['liquidity_declining']:
                alert_type = 'liquidity_crisis'
                message = "Liquidity buffer declining rapidly"
            else:
                alert_type = 'general_deterioration'
                message = "Overall financial health deteriorating"

            # Estimate time to default
            time_to_default = self.compute_time_to_default(forecast)

            alerts.append(RiskAlert(
                institution_id=institution_id,
                alert_type=alert_type,
                severity=severity,
                risk_score=risk_score,
                message=message,
                forecasted_capital_ratio=forecast[:, 0].tolist(),
                forecasted_default_prob=forecast[:, 3].tolist(),
                time_to_default=time_to_default,
            ))

        return alerts

    def compute_time_to_default(
        self,
        forecast: np.ndarray,
        default_capital_threshold: float = 0.0,
    ) -> Optional[int]:
        """
        Estimate time to default based on forecast

        Args:
            forecast: Forecasted states [num_steps, input_dim]
            default_capital_threshold: Capital threshold for default

        Returns:
            Number of timesteps until default (None if not forecasted)
        """
        # Check when capital ratio falls below threshold
        forecast_capital = forecast[:, 0]

        # Find first timestep where capital falls below threshold
        default_timesteps = np.where(forecast_capital <= default_capital_threshold)[0]

        if len(default_timesteps) > 0:
            return int(default_timesteps[0]) + 1  # +1 for 1-indexed timesteps
        else:
            # Check if trending toward default
            if len(forecast_capital) >= 2:
                final_trend = forecast_capital[-1] - forecast_capital[0]
                if final_trend < -0.05:  # Significant decline
                    # Extrapolate to estimate when it would hit 0
                    decline_rate = final_trend / len(forecast_capital)
                    if decline_rate < 0:
                        timesteps_to_zero = int(forecast_capital[-1] / abs(decline_rate))
                        return len(forecast_capital) + timesteps_to_zero

        return None

    def check_institutions(
        self,
        institutions: Dict[str, np.ndarray],
    ) -> Dict[str, List[RiskAlert]]:
        """
        Check multiple institutions for early warning signals

        Args:
            institutions: Dict mapping institution_id to historical_sequence

        Returns:
            Dict mapping institution_id to list of alerts
        """
        all_alerts = {}

        for institution_id, historical_sequence in institutions.items():
            try:
                alerts = self.generate_risk_alerts(institution_id, historical_sequence)
                if alerts:
                    all_alerts[institution_id] = alerts
            except Exception as e:
                logger.error(f"Error checking institution {institution_id}: {e}")

        return all_alerts
