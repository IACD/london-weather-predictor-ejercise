"""
Visualization Module

Functions for generating all 24 plots for the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_distributions(df: pd.DataFrame, save_path: str = '01_distribuciones.png') -> None:
    """
    Plot distributions of all numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features
    save_path : str
        Path to save figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date-related columns if present
    numeric_cols = [col for col in numeric_cols if col not in ['year_normalized', 'day_of_year']]
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_boxplots_outliers(df: pd.DataFrame, save_path: str = '02_boxplots_outliers.png') -> None:
    """
    Plot boxplots for outlier detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features
    save_path : str
        Path to save figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['year_normalized', 'day_of_year']]
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = '03_correlacion_heatmap.png') -> None:
    """
    Plot correlation heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features
    save_path : str
        Path to save figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_time_series(df: pd.DataFrame, save_path: str = '05_serie_temporal_completa.png') -> None:
    """
    Plot complete time series of mean_temp.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'date' and 'mean_temp'
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(16, 6))
    plt.plot(df['date'], df['mean_temp'], linewidth=0.5, alpha=0.7)
    plt.title('London Mean Temperature (1979-2020)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_temporal_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                         save_path: str = '09_splits_temporales.png') -> None:
    """
    Visualize temporal train/val/test splits.
    
    Parameters
    ----------
    train, val, test : pd.DataFrame
        Train, validation, and test sets with 'date' and 'mean_temp'
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(16, 6))
    
    plt.plot(train['date'], train['mean_temp'], label='Train (1979-2008)', 
             color='blue', linewidth=0.8, alpha=0.7)
    plt.plot(val['date'], val['mean_temp'], label='Validation (2009-2014)', 
             color='orange', linewidth=0.8, alpha=0.7)
    plt.plot(test['date'], test['mean_temp'], label='Test (2015-2020)', 
             color='green', linewidth=0.8, alpha=0.7)
    
    plt.axvline(train['date'].max(), color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(val['date'].max(), color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.title('Temporal Data Splits (60% Train / 20% Val / 20% Test)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_feature_importance(model: Any, feature_names: List[str], 
                            save_path: str = '24_feature_importance_final.png',
                            title: str = 'Feature Importance') -> None:
    """
    Plot feature importance from tree-based model.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    save_path : str
        Path to save figure
    title : str
        Plot title
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️ Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str = '12_ml_predictions_v1.png',
                               title: str = 'Predictions vs Actual') -> None:
    """
    Scatter plot of predictions vs actual values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    save_path : str
        Path to save figure
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.3, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   save_path: str = '13_ml_residuals_v1.png',
                   title: str = 'Residuals Plot') -> None:
    """
    Plot residuals (errors) distribution.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    save_path : str
        Path to save figure
    title : str
        Plot title
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=20)
    axes[0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Residuals (°C)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (°C)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_model_comparison(results_df: pd.DataFrame, 
                          save_path: str = '20_final_comparison_rmse.png') -> None:
    """
    Bar plot comparing RMSE of different models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: Model, RMSE_Test, Status
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by RMSE
    results_sorted = results_df.sort_values('RMSE_Test', ascending=False)
    
    # Color by status
    colors = []
    for status in results_sorted['Status']:
        if 'PRODUCTIVO' in str(status).upper():
            colors.append('green')
        elif 'LEAKAGE' in str(status).upper() or 'BOUND' in str(status).upper():
            colors.append('red')
        else:
            colors.append('gray')
    
    plt.barh(results_sorted['Model'], results_sorted['RMSE_Test'], color=colors, alpha=0.7)
    plt.xlabel('RMSE (°C)', fontsize=12)
    plt.title('Model Comparison - RMSE on Test Set', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (model, rmse) in enumerate(zip(results_sorted['Model'], results_sorted['RMSE_Test'])):
        plt.text(rmse + 0.1, i, f'{rmse:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_confidence_intervals(ci_df: pd.DataFrame, 
                              save_path: str = '21_confidence_intervals.png') -> None:
    """
    Plot confidence intervals for metrics.
    
    Parameters
    ----------
    ci_df : pd.DataFrame
        DataFrame with columns: Metric, Mean, Lower_95CI, Upper_95CI
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ci_df['Metric'].values
    means = ci_df['Mean'].values
    lower = ci_df['Lower_95CI'].values
    upper = ci_df['Upper_95CI'].values
    
    y_pos = np.arange(len(metrics))
    
    ax.errorbar(means, y_pos, xerr=[means - lower, upper - means],
                fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('95% Confidence Intervals (Bootstrap)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_error_by_season(season_df: pd.DataFrame,
                         save_path: str = '22_error_by_season.png') -> None:
    """
    Bar plot of RMSE by season.
    
    Parameters
    ----------
    season_df : pd.DataFrame
        DataFrame with columns: Season, RMSE
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.bar(season_df['Season'], season_df['RMSE'], 
            color=['blue', 'green', 'orange', 'brown'], alpha=0.7, edgecolor='black')
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('RMSE (°C)', fontsize=12)
    plt.title('Model Performance by Season', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (season, rmse) in enumerate(zip(season_df['Season'], season_df['RMSE'])):
        plt.text(i, rmse + 0.05, f'{rmse:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_error_by_temperature(temp_df: pd.DataFrame,
                              save_path: str = '23_error_by_temp_range.png') -> None:
    """
    Bar plot of RMSE by temperature range.
    
    Parameters
    ----------
    temp_df : pd.DataFrame
        DataFrame with columns: Temperature_Range, RMSE
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'green', 'orange', 'red']
    plt.bar(temp_df['Temperature_Range'], temp_df['RMSE'], 
            color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Temperature Range', fontsize=12)
    plt.ylabel('RMSE (°C)', fontsize=12)
    plt.title('Model Performance by Temperature Range', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (temp_range, rmse) in enumerate(zip(temp_df['Temperature_Range'], temp_df['RMSE'])):
        plt.text(i, rmse + 0.2, f'{rmse:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# Example usage
if __name__ == "__main__":
    print("Visualization module loaded successfully ✅")
    print("\nAvailable plotting functions:")
    print("  - plot_distributions()")
    print("  - plot_boxplots_outliers()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_time_series()")
    print("  - plot_temporal_splits()")
    print("  - plot_feature_importance()")
    print("  - plot_predictions_vs_actual()")
    print("  - plot_residuals()")
    print("  - plot_model_comparison()")
    print("  - plot_confidence_intervals()")
    print("  - plot_error_by_season()")
    print("  - plot_error_by_temperature()")