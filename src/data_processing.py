"""
Data Processing Module

Functions for loading, cleaning, and splitting the London Weather dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load London Weather CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to london_weather.csv
        
    Returns
    -------
    pd.DataFrame
        Raw dataframe with date parsed as datetime
    """
    df = pd.read_csv(filepath)
    
    # Parse date column (format: YYYYMMDD as integer → datetime)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data: handle missing values following EXACT notebook logic.
    
    Strategy:
    1. Drop rows where mean_temp (target) is NaN (~36 rows)
    2. Impute snow_depth NaN with 0 (assume 'no snow')
    3. Forward/backward fill for remaining NaN
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data()
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering
    """
    df_clean = df.copy()
    
    # Sort by date (CRITICAL for forward/backward fill)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    # Step 1: Drop rows where target (mean_temp) is NaN
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(subset=['mean_temp'])
    rows_removed = rows_before - len(df_clean)
    print(f"  - mean_temp NaN eliminados: {rows_removed} filas ({rows_removed/rows_before*100:.2f}%)")
    
    # Step 2: Impute snow_depth NaN with 0 (assume 'no snow')
    snow_missing = df_clean['snow_depth'].isnull().sum()
    df_clean['snow_depth'] = df_clean['snow_depth'].fillna(0)
    print(f"  - snow_depth: {snow_missing} NaN → 0 (asumido 'sin nieve')")
    
    # Step 3: Forward/Backward fill for remaining NaN
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.ffill()   # Forward fill (propagate last valid value)
    df_clean = df_clean.bfill()   # Backward fill (for beginning of series)
    missing_after = df_clean.isnull().sum().sum()
    print(f"  - Otras variables: {missing_before - missing_after} NaN imputados con forward/backward fill")
    
    # Verify no NaN remain
    if df_clean.isnull().sum().sum() == 0:
        print("✓ No quedan valores faltantes en el dataset")
    else:
        print(f"⚠ Aún quedan {df_clean.isnull().sum().sum()} valores faltantes")
        print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
    
    # Ensure numeric columns are float
    numeric_cols = [
        'cloud_cover', 'sunshine', 'global_radiation',
        'max_temp', 'mean_temp', 'min_temp',
        'precipitation', 'pressure', 'snow_depth'
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove duplicates by date (if any)
    df_clean = df_clean.drop_duplicates(subset=['date'], keep='first')
    
    return df_clean


def temporal_split(
    df: pd.DataFrame,
    train_pct: float = 0.60,
    val_pct: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally into Train/Validation/Test using percentages.
    
    IMPORTANT: Uses ROW-BASED SPLIT (not fixed dates) to match original notebook.
    
    Strategy:
    - Sort by date (chronological order)
    - Split by row indices using percentages
    - Train: first 60% of rows
    - Val:   next 20% of rows
    - Test:  last 20% of rows
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'date' column (must be sorted by date)
    train_pct : float, default=0.60
        Percentage of data for training set
    val_pct : float, default=0.20
        Percentage of data for validation set
        
    Returns
    -------
    tuple of (train, val, test) DataFrames
        - train: First 60% of rows (1979-2004)
        - val:   Next 20% of rows (2004-2012)
        - test:  Last 20% of rows (2012-2020)
    """
    # Ensure data is sorted chronologically
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end_idx = int(n * train_pct)
    val_end_idx = int(n * (train_pct + val_pct))
    
    # Split by row indices
    train = df.iloc[:train_end_idx].copy()
    val = df.iloc[train_end_idx:val_end_idx].copy()
    test = df.iloc[val_end_idx:].copy()
    
    # Print split info
    print(f"Train set: {train['date'].min()} to {train['date'].max()} (n={len(train)})")
    print(f"Val set:   {val['date'].min()} to {val['date'].max()} (n={len(val)})")
    print(f"Test set:  {test['date'].min()} to {test['date'].max()} (n={len(test)})")
    
    return train, val, test


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Convenience function: load + clean in one step.
    
    Parameters
    ----------
    filepath : str
        Path to london_weather.csv
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for splitting
    """
    print("="*75)
    print("LIMPIEZA Y TRANSFORMACIÓN DE DATOS")
    print("="*75)
    
    df = load_data(filepath)
    print(f"\nDataset original: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    print("\n[1/2] Manejando valores faltantes...")
    df_clean = clean_data(df)
    
    print(f"\n[2/2] Limpieza completada")
    print(f"  Dataset final: {df_clean.shape[0]:,} filas × {df_clean.shape[1]} columnas")
    print(f"  Filas eliminadas: {df.shape[0] - df_clean.shape[0]:,} ({(df.shape[0] - df_clean.shape[0])/df.shape[0]*100:.2f}%)")
    
    return df_clean


def prepare_datasets(
    filepath: str,
    train_pct: float = 0.60,
    val_pct: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load -> clean -> split.
    
    Parameters
    ----------
    filepath : str
        Path to london_weather.csv
    train_pct : float, default=0.60
        Percentage for training set
    val_pct : float, default=0.20
        Percentage for validation set
        
    Returns
    -------
    tuple of (train, val, test) DataFrames
    """
    df = load_and_clean_data(filepath)
    train, val, test = temporal_split(df, train_pct, val_pct)
    return train, val, test


# Example usage (will be called from notebook)
if __name__ == "__main__":
    # This block runs only if you execute this file directly (for testing)
    filepath = '../london_weather.csv'
    train, val, test = prepare_datasets(filepath)
    print("\n✅ Data processing complete!")
    print(f"Train shape: {train.shape}")
    print(f"Val shape:   {val.shape}")
    print(f"Test shape:  {test.shape}")