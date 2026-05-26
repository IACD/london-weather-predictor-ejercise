"""
Feature Engineering Module

Functions for creating features from raw temporal data:
- v1: Without temporal leakage (for production)
- v2: With temporal leakage (for comparison only)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features WITHOUT temporal leakage (production version).
    
    EXACT ORDER from original notebook to ensure reproducibility.
    
    Features:
    - Meteorological: cloud_cover, sunshine, global_radiation, 
                     precipitation, pressure, snow_depth
    - Temporal: year_normalized, day_of_year
    - Seasonal: season_spring, season_summer, season_winter
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' column
        
    Returns
    -------
    pd.DataFrame
        Features dataframe (11 columns) in EXACT order
    """
    df_feat = df.copy()
    
    # Extract temporal features
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    
    # Normalize year (0 to 1)
    df_feat['year_normalized'] = (
        (df_feat['year'] - df_feat['year'].min()) / 
        (df_feat['year'].max() - df_feat['year'].min())
    )
    
    # Create season dummy variables
    # IMPORTANT: Match exact logic from original notebook
    df_feat['season_winter'] = (df_feat['month'].isin([12, 1, 2])).astype(int)
    df_feat['season_spring'] = (df_feat['month'].isin([3, 4, 5])).astype(int)
    df_feat['season_summer'] = (df_feat['month'].isin([6, 7, 8])).astype(int)
    # season_autumn is implicit (all 0s)
    
    # Select features in EXACT ORDER from original notebook
    features_v1 = [
        # Meteorological variables (order matters for reproducibility)
        'cloud_cover',
        'sunshine',
        'global_radiation',
        'precipitation',      # Position 4
        'pressure',           # Position 5
        'snow_depth',
        
        # Temporal variables
        'year_normalized',    # Position 7
        'day_of_year',        # Position 8
        
        # Seasonal dummies (alphabetical order: spring, summer, winter)
        'season_spring',      # Position 9
        'season_summer',      # Position 10
        'season_winter'       # Position 11
    ]
    
    X = df_feat[features_v1].copy()
    
    return X


def create_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features WITH temporal leakage (for comparison only).
    
    ⚠️ WARNING: Includes max_temp and min_temp which have high correlation
    with mean_temp (the target). This creates data leakage.
    
    Use ONLY for benchmarking upper bound performance.
    DO NOT use for production.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date', 'max_temp', 'min_temp' columns
        
    Returns
    -------
    pd.DataFrame
        Features dataframe (13 columns)
    """
    # Start with v1 features
    X_v1 = create_features_v1(df)
    
    # Add leakage features
    X_v2 = X_v1.copy()
    X_v2['max_temp'] = df['max_temp'].values
    X_v2['min_temp'] = df['min_temp'].values
    
    return X_v2


def get_feature_names_v1() -> list:
    """
    Get list of feature names for v1 (without leakage) in exact order.
    
    Returns
    -------
    list
        List of 11 feature names in exact order from original notebook
    """
    return [
        'cloud_cover',
        'sunshine',
        'global_radiation',
        'precipitation',
        'pressure',
        'snow_depth',
        'year_normalized',
        'day_of_year',
        'season_spring',
        'season_summer',
        'season_winter'
    ]


def get_feature_names_v2() -> list:
    """
    Get list of feature names for v2 (with leakage).
    
    Returns
    -------
    list
        List of 13 feature names
    """
    return get_feature_names_v1() + ['max_temp', 'min_temp']


def prepare_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    version: str = 'v1'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Full pipeline: create features for train/val/test and separate X/y.
    
    Parameters
    ----------
    train : pd.DataFrame
        Training set with 'date' and 'mean_temp'
    val : pd.DataFrame
        Validation set
    test : pd.DataFrame
        Test set
    version : str, default='v1'
        Feature version: 'v1' (no leakage) or 'v2' (with leakage)
        
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Select feature creation function
    if version == 'v1':
        feature_fn = create_features_v1
    elif version == 'v2':
        feature_fn = create_features_v2
    else:
        raise ValueError(f"Invalid version: {version}. Use 'v1' or 'v2'")
    
    # Create features
    X_train = feature_fn(train)
    X_val = feature_fn(val)
    X_test = feature_fn(test)
    
    # Extract target
    y_train = train['mean_temp'].copy()
    y_val = val['mean_temp'].copy()
    y_test = test['mean_temp'].copy()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage
if __name__ == "__main__":
    # This block runs only if you execute this file directly (for testing)
    from data_processing import prepare_datasets
    
    # Load data
    train, val, test = prepare_datasets('../london_weather.csv')
    
    # Create features v1
    X_train_v1, X_val_v1, X_test_v1, y_train, y_val, y_test = prepare_features(
        train, val, test, version='v1'
    )
    
    print("\n✅ Feature engineering v1 complete!")
    print(f"X_train shape: {X_train_v1.shape}")
    print(f"Features: {list(X_train_v1.columns)}")
    
    # Create features v2
    X_train_v2, X_val_v2, X_test_v2, _, _, _ = prepare_features(
        train, val, test, version='v2'
    )
    
    print("\n✅ Feature engineering v2 complete!")
    print(f"X_train shape: {X_train_v2.shape}")
    print(f"Features: {list(X_train_v2.columns)}")