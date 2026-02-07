"""
CCP Risk Model Module

Participant-level risk estimation for CCP perspective.

ML_Flow.md Layer 1: Participant Risk Estimation
- XGBoost/GNN for default probability
- Features: Capital, liquidity, credit, network position
- Output: P(default | features) per participant

CCP Interpretation:
- High P(default) → Higher initial margin requirement
- Clustered high-risk banks → Stress testing scenarios
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, average_precision_score,
        classification_report, confusion_matrix
    )
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    XGBOOST = "xgboost"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass
class ModelResult:
    """Container for model results"""
    predictions: np.ndarray
    probabilities: np.ndarray
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]


class CCPRiskModel:
    """
    CCP-centric risk model for default probability estimation.
    
    Key perspective from ML_Flow.md:
    - We're modeling as a risk ABSORBER, not profit optimizer
    - Focus on tail risk and conservative estimates
    - Explainability is critical for regulatory acceptance
    """
    
    # Feature categories with CCP relevance
    FEATURE_GROUPS = {
        'capital': ['capital_ratio', 'crar_tier1', 'crar_total', 'leverage'],
        'liquidity': ['liquidity_buffer', 'stress_level'],
        'credit': ['gross_npa', 'net_npa', 'npa_growth_rate', 'default_probability_prior'],
        'network': ['degree_centrality', 'betweenness_centrality', 'pagerank', 'network_influence'],
        'sector': ['capital_market_exposure', 'real_estate_exposure', 'sector_concentration'],
        'derived': ['capital_stress_ratio', 'leverage_risk', 'composite_risk']
    }
    
    def __init__(
        self,
        model_type: ModelType = ModelType.GRADIENT_BOOSTING,
        calibrate: bool = True,
        conservative_threshold: float = 0.3
    ):
        """
        Initialize CCP risk model.
        
        Args:
            model_type: Type of model to use
            calibrate: Whether to calibrate probabilities
            conservative_threshold: Threshold for binary classification
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.calibrate = calibrate
        self.conservative_threshold = conservative_threshold
        
        self.model = None
        self.calibrated_model = None
        self.feature_names = []
        self.is_fitted = False
    
    def _create_model(self):
        """Create the underlying model"""
        if self.model_type == ModelType.XGBOOST:
            if not XGB_AVAILABLE:
                logger.warning("XGBoost not available, falling back to GradientBoosting")
                self.model_type = ModelType.GRADIENT_BOOSTING
            else:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='auc'
                )
        
        if self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
        
        if self.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=500,
                class_weight='balanced',
                random_state=42
            )
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Train the risk model.
        
        Args:
            X: Feature matrix
            y: Target variable (1 = default, 0 = no default)
            feature_names: Names of features
            
        Returns:
            Dictionary of training metrics
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to numpy if needed
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        
        # Handle missing values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_arr, y_arr)
        
        # Calibrate if requested
        if self.calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            try:
                self.calibrated_model.fit(X_arr, y_arr)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                self.calibrated_model = None
        
        self.is_fitted = True
        
        # Compute training metrics
        train_probs = self.predict_proba(X)[:, 1]
        metrics = {
            'train_auc': roc_auc_score(y_arr, train_probs),
            'train_ap': average_precision_score(y_arr, train_probs),
            'n_samples': len(y),
            'n_positive': int(np.sum(y_arr)),
            'positive_rate': float(np.mean(y_arr))
        }
        
        logger.info(f"Model trained: AUC={metrics['train_auc']:.4f}")
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of [P(no_default), P(default)] for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_arr = X.values if hasattr(X, 'values') else X
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_arr)
        return self.model.predict_proba(X_arr)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary default outcome.
        
        Uses conservative threshold for CCP perspective.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.conservative_threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            threshold: Classification threshold (default: conservative_threshold)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if threshold is None:
            threshold = self.conservative_threshold
        
        y_arr = y.values if hasattr(y, 'values') else y
        
        probs = self.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_arr, probs),
            'average_precision': average_precision_score(y_arr, probs),
            'threshold': threshold
        }
        
        # Add classification report
        conf_matrix = confusion_matrix(y_arr, preds)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        # CCP-specific metrics
        # For CCP, false negatives (missed defaults) are very costly
        metrics['miss_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for explainability.
        
        Returns dictionary mapping feature name to importance score.
        """
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        bank_name: str = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction for CCP decision-making.
        
        Args:
            X_single: Single sample features
            bank_name: Name of the bank for context
            
        Returns:
            Dictionary with prediction explanation
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        prob = self.predict_proba(X_single)[0, 1]
        pred = int(prob >= self.conservative_threshold)
        
        # Get feature contributions
        importance = self.get_feature_importance()
        
        # Get actual feature values
        feature_values = X_single.iloc[0].to_dict() if hasattr(X_single, 'iloc') else {}
        
        # Identify top risk factors
        top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        explanation = {
            'bank_name': bank_name,
            'default_probability': float(prob),
            'predicted_default': bool(pred),
            'risk_level': 'high' if prob > 0.6 else 'medium' if prob > 0.3 else 'low',
            'top_risk_factors': [
                {
                    'feature': f,
                    'importance': float(imp),
                    'value': float(feature_values.get(f, 0))
                }
                for f, imp in top_factors
            ],
            'ccp_recommendation': self._generate_recommendation(prob, feature_values)
        }
        
        return explanation
    
    def _generate_recommendation(
        self, 
        default_prob: float, 
        features: Dict
    ) -> str:
        """Generate CCP-specific recommendation based on prediction"""
        if default_prob > 0.7:
            return "CRITICAL: Consider enhanced IM requirements and stress buffer"
        elif default_prob > 0.5:
            return "HIGH RISK: Increase margin call frequency and monitor closely"
        elif default_prob > 0.3:
            return "ELEVATED: Standard monitoring with enhanced reporting"
        else:
            return "NORMAL: Standard margin requirements apply"
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with CV metrics
        """
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        model = self._create_model()
        
        auc_scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring='roc_auc')
        
        return {
            'cv_auc_mean': float(np.mean(auc_scores)),
            'cv_auc_std': float(np.std(auc_scores)),
            'cv_auc_scores': auc_scores.tolist()
        }
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'conservative_threshold': self.conservative_threshold,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CCPRiskModel':
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        model = cls(
            model_type=save_dict['model_type'],
            conservative_threshold=save_dict['conservative_threshold']
        )
        model.model = save_dict['model']
        model.calibrated_model = save_dict['calibrated_model']
        model.feature_names = save_dict['feature_names']
        model.is_fitted = save_dict['is_fitted']
        
        return model


def select_features(
    features: pd.DataFrame,
    target_col: str = 'defaulted'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features for modeling.
    
    Separates feature matrix from target and removes non-predictive columns.
    """
    # Columns to exclude
    exclude_cols = ['institution_id', 'bank_name', 'timestamp', target_col]
    exclude_cols = [c for c in exclude_cols if c in features.columns]
    
    # Get target
    y = features[target_col] if target_col in features.columns else pd.Series([0] * len(features))
    
    # Get features
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    X = features[feature_cols]
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    return X, y


if __name__ == "__main__":
    # Test risk model
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'capital_ratio': np.random.uniform(0.08, 0.25, n_samples),
        'liquidity_buffer': np.random.uniform(0.1, 0.5, n_samples),
        'npa_ratio': np.random.uniform(0.01, 0.15, n_samples),
        'degree_centrality': np.random.uniform(0, 1, n_samples),
        'stress_level': np.random.uniform(0, 1, n_samples),
    })
    
    # Create target (default probability increases with NPA and stress, decreases with capital)
    default_prob = (
        0.2 + 
        0.3 * X['npa_ratio'] / 0.15 +
        0.3 * X['stress_level'] -
        0.2 * X['capital_ratio'] / 0.25
    )
    y = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Train model
    model = CCPRiskModel(model_type=ModelType.GRADIENT_BOOSTING)
    train_metrics = model.fit(X, y)
    print(f"\nTraining metrics: {train_metrics}")
    
    # Cross-validate
    cv_metrics = model.cross_validate(X, y)
    print(f"\nCV metrics: {cv_metrics}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    # Explain a prediction
    explanation = model.explain_prediction(X.iloc[[0]], bank_name="Test Bank")
    print(f"\nExplanation:\n{explanation}")
