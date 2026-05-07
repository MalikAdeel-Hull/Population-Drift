"""
Evaluation Metrics Module
==========================
Calculates drift detection performance metrics.

Primary Metric: Detection Ratio
- Measures how much the outlier rate increases under drift
- Formula: outlier_rate_drifted / outlier_rate_original
- Interpretation: 1.0x = no change, 2.0x = double the anomalies, 4.4x = very strong drift signal

Supporting Metrics:
- K-S Test: Statistical validation that drift actually occurred
- Monotonicity: Verification that detection ratio increases with drift magnitude
- Zero-Drift Control: Validation check (should give 1.0x)

Example Usage:
    >>> baseline_rate = calculate_outlier_rate(model, X_baseline)
    >>> drifted_rate = calculate_outlier_rate(model, X_drifted)
    >>> detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
    >>> ks_stat, p_value = validate_with_ks_test(X_baseline_feature, X_drifted_feature)
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import Tuple, Union, Dict, Any, List


# ============================================================================
# OUTLIER RATE CALCULATION
# ============================================================================

def calculate_outlier_rate(
    model: Union[OneClassSVM, IsolationForest],
    X: pd.DataFrame
) -> float:
    """
    Calculate percentage of samples flagged as outliers.

    Args:
        model: Fitted anomaly detection model
        X: Data to evaluate

    Returns:
        Outlier rate (0.0 to 1.0). Example: 0.05 means 5% are outliers

    Example:
        >>> baseline_rate = calculate_outlier_rate(model, X_baseline)
        >>> print(f"Baseline anomaly rate: {baseline_rate:.2%}")
    """
    predictions = model.predict(X)
    outlier_rate = (predictions == -1).sum() / len(predictions)
    return outlier_rate


# ============================================================================
# DETECTION RATIO (Primary Metric)
# ============================================================================

def calculate_detection_ratio(
    outlier_rate_original: float,
    outlier_rate_drifted: float,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate detection ratio: how much outlier rate changed due to drift.

    Detection Ratio = Outlier Rate (Drifted) / Outlier Rate (Original)

    Args:
        outlier_rate_original: Outlier rate before drift (baseline)
        outlier_rate_drifted: Outlier rate after drift
        tolerance: Minimum outlier rate to avoid division by zero (default 1e-6)

    Returns:
        Detection ratio (float >= 1.0)
        - 1.0: No change (no drift detected)
        - 2.0: Outlier rate doubled
        - 4.4: Outlier rate increased 4.4x (strong drift)

    Raises:
        ValueError: If outlier_rate_original is too close to 0

    Example:
        >>> baseline_rate = 0.05  # 5%
        >>> drifted_rate = 0.22   # 22% (after drift)
        >>> ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
        >>> print(f"Detection Ratio: {ratio:.2f}x")  # Output: 4.4x
    """
    if outlier_rate_original < tolerance:
        raise ValueError(
            f"Original outlier rate {outlier_rate_original:.6f} too close to 0. "
            "Cannot calculate meaningful detection ratio."
        )

    return outlier_rate_drifted / outlier_rate_original


# ============================================================================
# K-S TEST (Statistical Validation)
# ============================================================================

def validate_with_ks_test(
    baseline_feature: np.ndarray,
    drifted_feature: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Use Kolmogorov-Smirnov test to validate drift is statistically significant.

    K-S test compares two distributions. Large differences indicate statistical drift.

    Args:
        baseline_feature: Feature values from baseline data
        drifted_feature: Feature values from drifted data
        alpha: Significance level (default 0.05 = 95% confidence)

    Returns:
        Tuple of (ks_statistic, p_value, is_significant)
        - ks_statistic: How different the distributions are (0 = identical, 1 = completely different)
        - p_value: Probability that difference occurred by chance
        - is_significant: True if p_value < alpha (drift is real, not random)

    Example:
        >>> ks_stat, p_val, is_sig = validate_with_ks_test(
        ...     X_baseline['Glucose'], X_drifted['Glucose']
        ... )
        >>> print(f"K-S: {ks_stat:.3f}, p-value: {p_val:.6f}, Significant: {is_sig}")

    Interpretation:
        p_value < 0.05: REJECT H0 → Drift is statistically significant ✓
        p_value >= 0.05: DO NOT REJECT H0 → No significant difference
    """
    # Remove NaN values
    baseline_clean = baseline_feature[~np.isnan(baseline_feature)]
    drifted_clean = drifted_feature[~np.isnan(drifted_feature)]

    if len(baseline_clean) == 0 or len(drifted_clean) == 0:
        return np.nan, np.nan, False

    ks_stat, p_value = ks_2samp(baseline_clean, drifted_clean)
    is_significant = p_value < alpha

    return ks_stat, p_value, is_significant


def validate_multiple_features(
    X_baseline: pd.DataFrame,
    X_drifted: pd.DataFrame,
    features: List[str] = None,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Apply K-S test to multiple features.

    Args:
        X_baseline: Baseline DataFrame
        X_drifted: Drifted DataFrame
        features: Feature columns to test (default: all common columns)
        alpha: Significance level

    Returns:
        Dict mapping feature names to K-S results

    Example:
        >>> results = validate_multiple_features(X_baseline, X_drifted)
        >>> for feat, res in results.items():
        ...     print(f"{feat}: p-value={res['p_value']:.6f}, Sig: {res['is_significant']}")
    """
    if features is None:
        features = list(set(X_baseline.columns) & set(X_drifted.columns))

    results = {}
    for feature in features:
        ks_stat, p_val, is_sig = validate_with_ks_test(
            X_baseline[feature].values,
            X_drifted[feature].values,
            alpha=alpha
        )
        results[feature] = {
            'ks_statistic': ks_stat,
            'p_value': p_val,
            'is_significant': is_sig
        }

    return results


# ============================================================================
# MONOTONICITY CHECK
# ============================================================================

def check_monotonicity(
    drift_magnitudes: List[float],
    detection_ratios: List[float],
    min_correlation: float = 0.90
) -> Tuple[float, float, bool]:
    """
    Verify that detection ratios increase monotonically with drift magnitude.

    Uses Spearman correlation (rank-based, robust to outliers).

    Args:
        drift_magnitudes: List of drift percentages applied (e.g., [0.0, 0.1, 0.2, ..., 0.4])
        detection_ratios: Corresponding detection ratios (e.g., [1.0, 1.2, 1.5, ..., 4.4])
        min_correlation: Minimum acceptable Spearman correlation (default 0.90)

    Returns:
        Tuple of (spearman_rho, p_value, is_monotonic)
        - spearman_rho: Correlation between drift and detection (1.0 = perfect)
        - p_value: Statistical significance of correlation
        - is_monotonic: True if rho >= min_correlation

    Example:
        >>> drifts = [0, 10, 20, 30, 40]
        >>> ratios = [1.0, 1.5, 2.0, 3.0, 4.4]
        >>> rho, p_val, is_mono = check_monotonicity(drifts, ratios)
        >>> print(f"Spearman ρ: {rho:.3f}, p-value: {p_val:.6f}, Monotonic: {is_mono}")

    Notes:
        - Expects positive relationship (higher drift → higher detection)
        - Spearman ρ = 1.0: Perfect monotonic relationship
        - Spearman ρ < 0.90: Relationship not strong enough
    """
    if len(drift_magnitudes) < 3:
        return np.nan, np.nan, False

    rho, p_value = spearmanr(drift_magnitudes, detection_ratios)
    is_monotonic = rho >= min_correlation

    return rho, p_value, is_monotonic


# ============================================================================
# ZERO-DRIFT CONTROL (Validation Check)
# ============================================================================

def validate_zero_drift(
    outlier_rate_zero: float,
    expected_value: float = 1.0,
    tolerance: float = 0.01
) -> Tuple[float, bool]:
    """
    Verify zero-drift produces no change in outlier rate (quality check).

    When drift_percentage=0.0, the detection ratio should be 1.0x
    (no change). This validates that the pipeline doesn't introduce spurious changes.

    Args:
        outlier_rate_zero: Detection ratio when drift=0.0
        expected_value: Expected detection ratio for zero drift (default 1.0)
        tolerance: Acceptable deviation from expected (default 0.01 = ±1%)

    Returns:
        Tuple of (ratio, is_valid)
        - ratio: How close to expected value
        - is_valid: True if within tolerance

    Example:
        >>> # Run experiment with 0% drift
        >>> baseline_rate = 0.05
        >>> zero_drift_rate = 0.05  # Should be ~same as baseline
        >>> ratio = zero_drift_rate / baseline_rate  # Should be ~1.0
        >>> _, is_valid = validate_zero_drift(ratio)
        >>> assert is_valid, "Zero-drift control failed!"
    """
    deviation = abs(outlier_rate_zero - expected_value)
    is_valid = deviation <= tolerance

    return deviation, is_valid


# ============================================================================
# COMPREHENSIVE EVALUATION SUMMARY
# ============================================================================

def evaluate_drift_detection(
    model: Union[OneClassSVM, IsolationForest],
    X_baseline: pd.DataFrame,
    X_drifted: pd.DataFrame,
    feature_for_ks: str = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of drift detection model.

    Calculates: outlier rates, detection ratio, K-S validation, and summary.

    Args:
        model: Fitted anomaly detection model
        X_baseline: Baseline (pre-drift) data
        X_drifted: Drifted data
        feature_for_ks: Feature to validate with K-S test (default: first feature)
        alpha: K-S test significance level

    Returns:
        Dict with all metrics and results

    Example:
        >>> model = fit_ocsvm(X_base_preprocessed)
        >>> results = evaluate_drift_detection(model, X_base_preprocessed, X_drifted)
        >>> print(f"Detection Ratio: {results['detection_ratio']:.2f}x")
        >>> print(f"K-S p-value: {results['ks_p_value']:.6f}")
    """
    # Calculate outlier rates
    baseline_rate = calculate_outlier_rate(model, X_baseline)
    drifted_rate = calculate_outlier_rate(model, X_drifted)

    # Calculate detection ratio
    detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)

    # K-S test
    if feature_for_ks is None:
        feature_for_ks = X_baseline.columns[0]

    if feature_for_ks in X_baseline.columns:
        ks_stat, p_val, is_sig = validate_with_ks_test(
            X_baseline[feature_for_ks].values,
            X_drifted[feature_for_ks].values,
            alpha=alpha
        )
    else:
        ks_stat, p_val, is_sig = np.nan, np.nan, None

    return {
        'baseline_outlier_rate': baseline_rate,
        'drifted_outlier_rate': drifted_rate,
        'detection_ratio': detection_ratio,
        'ks_statistic': ks_stat,
        'ks_p_value': p_val,
        'ks_is_significant': is_sig,
        'detection_ratio_interpretation': (
            'Strong drift signal' if detection_ratio > 2.0
            else 'Moderate drift' if detection_ratio > 1.5
            else 'Weak drift' if detection_ratio > 1.1
            else 'No drift detected'
        )
    }


# ============================================================================
# METRICS SUMMARY TABLE GENERATION
# ============================================================================

def create_results_dataframe(
    drift_percentages: List[float],
    detection_ratios: List[float],
    ks_results: List[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Create summary DataFrame of drift detection results.

    Args:
        drift_percentages: List of drift magnitudes applied
        detection_ratios: Corresponding detection ratios
        ks_results: List of (ks_stat, p_value) tuples (optional)

    Returns:
        Formatted DataFrame for analysis and reporting

    Example:
        >>> results_df = create_results_dataframe(
        ...     drift_percentages=[0, 10, 20, 30, 40],
        ...     detection_ratios=[1.0, 1.5, 2.0, 3.0, 4.4]
        ... )
        >>> print(results_df)
    """
    data = {
        'Drift Percentage': [f"{d:.0%}" for d in drift_percentages],
        'Detection Ratio': [f"{r:.2f}x" for r in detection_ratios]
    }

    if ks_results:
        data['K-S Statistic'] = [f"{ks:.4f}" for ks, _ in ks_results]
        data['K-S p-value'] = [f"{p:.6f}" for _, p in ks_results]

    df = pd.DataFrame(data)
    return df
