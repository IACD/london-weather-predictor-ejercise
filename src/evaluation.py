"""
Model Evaluation Module

Functions for calculating metrics, confidence intervals, and error analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, and R² metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary with 'rmse', 'mae', 'r2' keys
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and print metrics.
    
    Parameters
    ----------
    model : Any
        Trained model with .predict() method
    X : pd.DataFrame
        Features
    y : pd.Series
        True target values
    dataset_name : str, default="Test"
        Name of dataset for printing
        
    Returns
    -------
    dict
        Dictionary with metrics
    """
    y_pred = model.predict(X)
    metrics = calculate_metrics(y.values, y_pred)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}°C")
    print(f"  MAE:  {metrics['mae']:.4f}°C")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    return metrics


def bootstrap_confidence_intervals(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate confidence intervals using bootstrap resampling.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    n_iterations : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level (0.95 = 95%)
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Mean, Lower_95CI, Upper_95CI, CI_Width for each metric
    """
    np.random.seed(random_state)
    
    n_samples = len(X_test)
    bootstrap_scores = {
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    print(f"\nCalculating {confidence_level*100:.0f}% confidence intervals (n={n_iterations} iterations)...")
    
    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        # Predict and calculate metrics
        y_pred_boot = model.predict(X_boot)
        metrics_boot = calculate_metrics(y_boot.values, y_pred_boot)
        
        bootstrap_scores['rmse'].append(metrics_boot['rmse'])
        bootstrap_scores['mae'].append(metrics_boot['mae'])
        bootstrap_scores['r2'].append(metrics_boot['r2'])
        
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_iterations} iterations")
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = []
    for metric_name, scores in bootstrap_scores.items():
        mean_val = np.mean(scores)
        lower_ci = np.percentile(scores, lower_percentile)
        upper_ci = np.percentile(scores, upper_percentile)
        ci_width = upper_ci - lower_ci
        
        results.append({
            'Metric': metric_name.upper(),
            'Mean': mean_val,
            'Lower_95CI': lower_ci,
            'Upper_95CI': upper_ci,
            'CI_Width': ci_width
        })
    
    df_results = pd.DataFrame(results)
    
    print("\n✅ Confidence Intervals:")
    print(df_results.to_string(index=False))
    
    return df_results


def analyze_errors_by_season(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series
) -> pd.DataFrame:
    """
    Analyze model performance by season.
    
    Parameters
    ----------
    y_true : pd.Series
        True temperatures
    y_pred : np.ndarray
        Predicted temperatures
    dates : pd.Series
        Dates (datetime)
        
    Returns
    -------
    pd.DataFrame
        Performance metrics by season
    """
    # Create DataFrame
    df_analysis = pd.DataFrame({
        'date': dates.values,
        'y_true': y_true.values,
        'y_pred': y_pred
    })
    
    # Add season
    df_analysis['month'] = pd.to_datetime(df_analysis['date']).dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Autumn'
    
    df_analysis['season'] = df_analysis['month'].apply(get_season)
    
    # Calculate metrics by season
    season_results = []
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        df_season = df_analysis[df_analysis['season'] == season]
        
        if len(df_season) > 0:
            metrics = calculate_metrics(df_season['y_true'].values, df_season['y_pred'].values)
            season_results.append({
                'Season': season,
                'Sample_Count': len(df_season),
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'Analysis_Type': 'Season'
            })
    
    df_season_results = pd.DataFrame(season_results)
    
    print("\n📊 Performance by Season:")
    print(df_season_results.to_string(index=False))
    
    return df_season_results


def analyze_errors_by_temperature(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Analyze model performance by temperature range.
    
    Parameters
    ----------
    y_true : pd.Series
        True temperatures
    y_pred : np.ndarray
        Predicted temperatures
        
    Returns
    -------
    pd.DataFrame
        Performance metrics by temperature range
    """
    # Create DataFrame
    df_analysis = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred
    })
    
    # Define temperature ranges
    def get_temp_range(temp):
        if temp < 5:
            return '< 5°C (Frío)'
        elif temp < 15:
            return '5-15°C (Templado)'
        elif temp < 25:
            return '15-25°C (Cálido)'
        else:
            return '> 25°C (Extremo)'
    
    df_analysis['temp_range'] = df_analysis['y_true'].apply(get_temp_range)
    
    # Calculate metrics by range
    temp_results = []
    for temp_range in ['< 5°C (Frío)', '5-15°C (Templado)', '15-25°C (Cálido)', '> 25°C (Extremo)']:
        df_range = df_analysis[df_analysis['temp_range'] == temp_range]
        
        if len(df_range) > 0:
            metrics = calculate_metrics(df_range['y_true'].values, df_range['y_pred'].values)
            temp_results.append({
                'Temperature_Range': temp_range,
                'Sample_Count': len(df_range),
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'Analysis_Type': 'Temperature_Range'
            })
    
    df_temp_results = pd.DataFrame(temp_results)
    
    print("\n🌡️ Performance by Temperature Range:")
    print(df_temp_results.to_string(index=False))
    
    return df_temp_results


def create_final_analysis_table(
    models_results: Dict[str, Dict[str, float]],
    save_path: str = None
) -> pd.DataFrame:
    """
    Create final analysis table with all models and versions.
    
    Parameters
    ----------
    models_results : dict
        Dictionary with model results
        Format: {
            'model_name': {
                'version': 'v1',
                'rmse_test': 2.784,
                'mae_test': 2.208,
                'r2_test': 0.759,
                'status': 'PRODUCTIVO'
            }
        }
    save_path : str, optional
        Path to save CSV (e.g., 'results/final_analysis.csv')
        
    Returns
    -------
    pd.DataFrame
        Final analysis table
    """
    results_list = []
    
    for model_name, results in models_results.items():
        results_list.append({
            'Model': model_name,
            'Version': results.get('version', 'N/A'),
            'RMSE_Test': results.get('rmse_test', np.nan),
            'MAE_Test': results.get('mae_test', np.nan),
            'R2_Test': results.get('r2_test', np.nan),
            'Status': results.get('status', 'Benchmark')
        })
    
    df_final = pd.DataFrame(results_list)
    
    print("\n📊 Final Model Comparison:")
    print(df_final.to_string(index=False))
    
    if save_path:
        df_final.to_csv(save_path, index=False)
        print(f"\n✅ Results saved to {save_path}")
    
    return df_final


# Example usage
if __name__ == "__main__":
    # This block runs only if you execute this file directly (for testing)
    print("Evaluation module loaded successfully ✅")
    print("\nAvailable functions:")
    print("  - calculate_metrics()")
    print("  - evaluate_model()")
    print("  - bootstrap_confidence_intervals()")
    print("  - analyze_errors_by_season()")
    print("  - analyze_errors_by_temperature()")
    print("  - create_final_analysis_table()")