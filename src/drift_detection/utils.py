"""
Utilities and Experiment Orchestration
=======================================
Helper functions and high-level experiment runner.

Key Components:
1. Clinical feature ranges (for drift clipping)
2. Experiment orchestrator (runs full drift pipeline)
3. Results saving/logging
4. Visualization helpers (optional)

Main Entry Point: run_experiment()
  - One-shot function that: loads data → preprocesses → applies drift → predicts → calculates metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import json

from .data import (
    load_raw_data, create_missingness_flags, temporal_train_test_split,
    identify_feature_types, PIMA_FEATURES, PIMA_TARGET, PIMA_COLS_WITH_MISSING
)
from .preprocessing import PreprocessingPipeline
from .drift import simulate_gradual_drift, simulate_multivariate_drift, DEFAULT_CLINICAL_RANGES
from .algorithms import fit_ocsvm, fit_isolation_forest, get_outlier_rate
from .evaluation import (
    calculate_outlier_rate, calculate_detection_ratio, validate_with_ks_test,
    check_monotonicity, validate_zero_drift
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'data_path': 'data/interim/pima_step1_clean.csv',
    'test_fraction': 0.30,
    'contamination': 0.05,
    'nu': 0.05,
    'ocsvm_gamma': 0.1,
    'isolation_forest_params': {
        'n_estimators': 100,
        'max_samples': 256,
        'contamination': 0.05,
        'random_state': 42
    },
    'drift_percentages': [0.0, 0.1, 0.2, 0.3, 0.4],
    'results_dir': 'results/'
}


# ============================================================================
# EXPERIMENT ORCHESTRATOR
# ============================================================================

def run_drift_detection_experiment(
    model: Union[OneClassSVM, IsolationForest],
    X_test_raw: pd.DataFrame,
    X_test_preprocessed: pd.DataFrame,
    pipeline: PreprocessingPipeline,
    feature: str,
    drift_percentage: float,
    feature_ranges: Dict[str, Tuple[float, float]] = None,
    baseline_outlier_rate: float = None,
    baseline_feature_values: np.ndarray = None,
    algorithm_name: str = "Model"
) -> Dict[str, Any]:
    """
    Run complete drift detection experiment on a single feature.

    Pipeline:
    1. Apply drift to test set (multiplicative)
    2. Preprocess drifted data
    3. Get anomaly predictions
    4. Calculate detection ratio
    5. Validate with K-S test
    6. Return comprehensive results

    Args:
        model: Fitted anomaly detection model
        X_test_raw: Raw test data (unimputed)
        X_test_preprocessed: Preprocessed test data (for baseline predictions)
        pipeline: PreprocessingPipeline (fitted on baseline)
        feature: Feature to apply drift to
        drift_percentage: Magnitude of drift to apply
        feature_ranges: Clinical ranges for clipping (default: DEFAULT_CLINICAL_RANGES)
        baseline_outlier_rate: Outlier rate on baseline (for detection ratio)
        baseline_feature_values: Baseline feature values (for K-S test)
        algorithm_name: Name of algorithm (for reporting)

    Returns:
        Dict with all results:
        - detection_ratio: Primary metric
        - outlier_rate_original: Baseline anomaly rate
        - outlier_rate_drifted: Drifted anomaly rate
        - ks_statistic, ks_p_value: K-S test results
        - drift_percentage: Applied drift magnitude
        - feature: Feature name
        - algorithm: Algorithm name

    Example:
        >>> results = run_drift_detection_experiment(
        ...     model=ocsvm,
        ...     X_test_raw=X_test_raw,
        ...     X_test_preprocessed=X_test_preprocessed,
        ...     pipeline=pipeline,
        ...     feature="Glucose",
        ...     drift_percentage=0.4,
        ...     baseline_outlier_rate=0.048
        ... )
        >>> print(f"Detection Ratio: {results['detection_ratio']:.2f}x")
    """
    if feature_ranges is None:
        feature_ranges = DEFAULT_CLINICAL_RANGES

    # Step 1: Apply drift
    X_test_drifted_raw = simulate_gradual_drift(
        X_test_raw,
        feature=feature,
        drift_percentage=drift_percentage,
        feature_ranges=feature_ranges
    )

    # Step 2: Preprocess drifted data
    X_test_drifted_preprocessed = pipeline.transform(X_test_drifted_raw)

    # Step 3: Get anomaly predictions
    drifted_outlier_rate = get_outlier_rate(model, X_test_drifted_preprocessed)

    # Step 4: Calculate detection ratio
    if baseline_outlier_rate is None:
        baseline_outlier_rate = get_outlier_rate(model, X_test_preprocessed)

    detection_ratio = calculate_detection_ratio(baseline_outlier_rate, drifted_outlier_rate)

    # Step 5: K-S validation
    if baseline_feature_values is not None:
        ks_stat, p_val, is_sig = validate_with_ks_test(
            baseline_feature_values,
            X_test_drifted_raw[feature].dropna().values
        )
    else:
        ks_stat, p_val, is_sig = np.nan, np.nan, None

    # Step 6: Return results
    return {
        'algorithm': algorithm_name,
        'feature': feature,
        'drift_percentage': drift_percentage,
        'outlier_rate_original': baseline_outlier_rate,
        'outlier_rate_drifted': drifted_outlier_rate,
        'detection_ratio': detection_ratio,
        'ks_statistic': ks_stat,
        'ks_p_value': p_val,
        'ks_is_significant': is_sig
    }


def run_univariate_drift_sweep(
    model: Union[OneClassSVM, IsolationForest],
    X_baseline_preprocessed: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_test_preprocessed: pd.DataFrame,
    pipeline: PreprocessingPipeline,
    feature: str,
    drift_percentages: List[float] = None,
    feature_ranges: Dict[str, Tuple[float, float]] = None,
    algorithm_name: str = "Model"
) -> pd.DataFrame:
    """
    Run drift detection experiments across multiple drift magnitudes (univariate).

    Args:
        model: Fitted anomaly detection model
        X_baseline_preprocessed: Baseline preprocessed data
        X_test_raw: Test data (raw)
        X_test_preprocessed: Test data (preprocessed)
        pipeline: PreprocessingPipeline
        feature: Feature to apply drift to
        drift_percentages: List of drift values to try (default: [0, 0.1, 0.2, 0.3, 0.4])
        feature_ranges: Clinical ranges
        algorithm_name: Algorithm name

    Returns:
        DataFrame with results for each drift percentage

    Example:
        >>> results_df = run_univariate_drift_sweep(
        ...     ocsvm, X_base_prep, X_test_raw, X_test_prep, pipeline,
        ...     feature="Glucose"
        ... )
        >>> print(results_df)
    """
    if drift_percentages is None:
        drift_percentages = [0.0, 0.1, 0.2, 0.3, 0.4]

    if feature_ranges is None:
        feature_ranges = DEFAULT_CLINICAL_RANGES

    baseline_outlier_rate = get_outlier_rate(model, X_baseline_preprocessed)
    baseline_feature = X_test_raw[feature].dropna().values

    results = []
    for drift_pct in drift_percentages:
        result = run_drift_detection_experiment(
            model=model,
            X_test_raw=X_test_raw,
            X_test_preprocessed=X_test_preprocessed,
            pipeline=pipeline,
            feature=feature,
            drift_percentage=drift_pct,
            feature_ranges=feature_ranges,
            baseline_outlier_rate=baseline_outlier_rate,
            baseline_feature_values=baseline_feature,
            algorithm_name=algorithm_name
        )
        results.append(result)

    return pd.DataFrame(results)


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def save_results(
    results_df: pd.DataFrame,
    output_path: str,
    file_format: str = 'csv'
) -> Path:
    """
    Save results to disk.

    Args:
        results_df: DataFrame with experimental results
        output_path: Path to save to
        file_format: 'csv' or 'json'

    Returns:
        Path to saved file

    Example:
        >>> save_results(results_df, 'results/ocsvm_gradual.csv')
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == 'csv':
        results_df.to_csv(path, index=False)
    elif file_format == 'json':
        results_df.to_json(path, orient='records')

    return path


def load_results(input_path: str) -> pd.DataFrame:
    """Load previously saved results."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from results DataFrame.

    Args:
        results_df: Results from run_univariate_drift_sweep or similar

    Returns:
        Dict with summary statistics
    """
    summary = {
        'n_experiments': len(results_df),
        'mean_detection_ratio': results_df['detection_ratio'].mean(),
        'max_detection_ratio': results_df['detection_ratio'].max(),
        'min_detection_ratio': results_df['detection_ratio'].min(),
        'n_significant_ks_tests': results_df['ks_is_significant'].sum(),
        'percent_significant_ks': (results_df['ks_is_significant'].sum() / len(results_df) * 100),
        'monotonicity_check': None
    }

    # Check monotonicity
    if 'drift_percentage' in results_df.columns and 'detection_ratio' in results_df.columns:
        rho, p_val, is_mono = check_monotonicity(
            results_df['drift_percentage'].tolist(),
            results_df['detection_ratio'].tolist()
        )
        summary['monotonicity_check'] = {
            'spearman_rho': rho,
            'p_value': p_val,
            'is_monotonic': is_mono
        }

    return summary
