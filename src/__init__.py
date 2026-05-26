"""
London Weather Temperature Prediction - Modular Codebase

This package contains modular functions for:
- Data processing and temporal splits
- Feature engineering (v1 and v2)
- Model training (Baseline, Ridge, Random Forest, XGBoost)
- Model evaluation and confidence intervals
- Visualization generation

Author: Ismael Alejandro Carreño Diaz
Date: May 2026
"""

__version__ = "1.0.0"
__author__ = "Ismael Alejandro Carreño Diaz"

# Optional: Import main functions for easier access
from .data_processing import load_and_clean_data, temporal_split
from .feature_engineering import create_features_v1, create_features_v2
from .models import train_all_models, train_random_forest
from .evaluation import calculate_metrics, bootstrap_confidence_intervals
from .visualization import (
    plot_distributions,
    plot_feature_importance,
    plot_model_comparison
)

__all__ = [
    'load_and_clean_data',
    'temporal_split',
    'create_features_v1',
    'create_features_v2',
    'train_all_models',
    'train_random_forest',
    'calculate_metrics',
    'bootstrap_confidence_intervals',
    'plot_distributions',
    'plot_feature_importance',
    'plot_model_comparison'
]