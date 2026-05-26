"""
Model Training Module

Functions for training baseline and ML models:
- Seasonal Naive baseline
- Random Forest
- XGBoost
- Ridge Regression
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
try:
    from xgboost import XGBRegressor
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    XGBRegressor = None

from typing import Dict, Tuple, Any
import joblib


def train_seasonal_naive_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Train Seasonal Naive baseline.
    
    Predicts temperature as the average temperature of the same season
    from the previous year.
    
    Parameters
    ----------
    train : pd.DataFrame
        Training set with 'date' and 'mean_temp'
    val : pd.DataFrame
        Validation set
    test : pd.DataFrame
        Test set
        
    Returns
    -------
    dict
        Dictionary with predictions and seasonal averages
    """
    # Helper function to assign season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Autumn'
    
    # Add season to train/val/test (create copies to avoid SettingWithCopyWarning)
    train_copy = train.copy()
    val_copy = val.copy()
    test_copy = test.copy()
    
    train_copy['month'] = train_copy['date'].dt.month
    val_copy['month'] = val_copy['date'].dt.month
    test_copy['month'] = test_copy['date'].dt.month
    
    train_copy['season'] = train_copy['month'].apply(get_season)
    val_copy['season'] = val_copy['month'].apply(get_season)
    test_copy['season'] = test_copy['month'].apply(get_season)
    
    # Calculate seasonal averages from training set
    seasonal_avg = train_copy.groupby('season')['mean_temp'].mean().to_dict()
    
    # Predict using seasonal averages
    val_pred = val_copy['season'].map(seasonal_avg).values
    test_pred = test_copy['season'].map(seasonal_avg).values
    
    return {
        'model_type': 'Seasonal Naive',
        'seasonal_averages': seasonal_avg,
        'val_predictions': val_pred,
        'test_predictions': test_pred
    }


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    max_features: str = 'sqrt',
    random_state: int = 42,
    **kwargs
) -> RandomForestRegressor:
    """
    Train Random Forest model with EXACT hyperparameters from original notebook.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int, default=200
        Number of trees (OPTIMIZED from original notebook)
    max_depth : int, default=10
        Maximum tree depth (OPTIMIZED from original notebook)
    min_samples_split : int, default=10
        Minimum samples to split node
    min_samples_leaf : int, default=4
        Minimum samples in leaf
    max_features : str, default='sqrt'
        Number of features for best split
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    RandomForestRegressor
        Trained model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        **kwargs
    )
    
    print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    model.fit(X_train, y_train)
    print("✅ Random Forest trained")
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Train XGBoost model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int, default=100
        Number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.1
        Learning rate (eta)
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    XGBRegressor
        Trained model
    """
    if XGBRegressor is None:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        **kwargs
    )
    
    print(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
    model.fit(X_train, y_train)
    print("✅ XGBoost trained")
    
    return model


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    **kwargs
) -> Ridge:
    """
    Train Ridge Regression model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    alpha : float, default=1.0
        Regularization strength
        
    Returns
    -------
    Ridge
        Trained model
    """
    model = Ridge(alpha=alpha, **kwargs)
    
    print(f"Training Ridge Regression (alpha={alpha})...")
    model.fit(X_train, y_train)
    print("✅ Ridge Regression trained")
    
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_xgb: bool = True
) -> Dict[str, Any]:
    """
    Train all ML models at once with EXACT hyperparameters from original notebook.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    train_xgb : bool, default=True
        Whether to train XGBoost (set False if not installed)
        
    Returns
    -------
    dict
        Dictionary with all trained models
    """
    models = {}
    
    print("=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    # Random Forest (EXACT hyperparameters from original notebook)
    models['random_forest'] = train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt'
    )
    
    # XGBoost
    if train_xgb and XGBRegressor is not None:
        models['xgboost'] = train_xgboost(
            X_train, y_train,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    else:
        print("⚠️ Skipping XGBoost (not installed or train_xgb=False)")
    
    # Ridge
    models['ridge'] = train_ridge(
        X_train, y_train,
        alpha=1.0
    )
    
    print("=" * 80)
    print(f"✅ All models trained ({len(models)} total)")
    print("=" * 80)
    
    return models


def save_model(model: Any, filepath: str, add_timestamp: bool = False) -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : Any
        Trained sklearn/xgboost model
    filepath : str
        Path to save model (e.g., 'models/random_forest_v1.pkl')
    add_timestamp : bool, default=False
        If True, adds timestamp to filename for versioning
    """
    import os
    from datetime import datetime
    
    if add_timestamp:
        # Extract path and extension
        base, ext = os.path.splitext(filepath)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"{base}_{timestamp}{ext}"
    
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved model
        
    Returns
    -------
    Any
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model


# Example usage
if __name__ == "__main__":
    # This block runs only if you execute this file directly (for testing)
    from data_processing import prepare_datasets
    from feature_engineering import create_features_v1
    
    # Load data
    train, val, test = prepare_datasets('../london_weather.csv')
    
    # Create features
    X_train = create_features_v1(train)
    y_train = train['mean_temp']
    
    # Train models
    models = train_all_models(X_train, y_train, train_xgb=True)
    
    # Save best model
    save_model(models['random_forest'], '../models/random_forest_v1_test.pkl')
    
    print("\n✅ Model training complete!")