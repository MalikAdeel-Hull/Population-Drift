"""
Drift Detection Algorithms Module
==================================
Implementations of One-Class SVM and Isolation Forest for drift detection.

Both algorithms are from scikit-learn (no custom implementations).

Key Design:
- Algorithms trained on preprocessed baseline data only
- Both use .fit() then .predict() standard sklearn interface
- Predict returns +1 (inlier) or -1 (outlier)
- Contamination/nu parameters set to 0.05 (5% expected anomaly rate)

Example Usage:
    >>> ocsvm = fit_ocsvm(X_base_preprocessed, gamma=0.1)
    >>> y_hat = ocsvm.predict(X_test_preprocessed)  # +1 or -1
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import Union, Tuple, Dict, Any


# ============================================================================
# ONE-CLASS SVM FOR DRIFT DETECTION
# ============================================================================

def fit_ocsvm(
    X_baseline: pd.DataFrame,
    kernel: str = 'rbf',
    nu: float = 0.05,
    gamma: float = 0.1
) -> OneClassSVM:
    """
    Train One-Class SVM on baseline data.

    One-Class SVM creates a decision boundary around the baseline data,
    then flags test samples outside this boundary as anomalies (drift).

    Args:
        X_baseline: Preprocessed baseline (training) data
        kernel: Kernel type (default 'rbf' for non-linear boundary)
        nu: Expected outlier fraction (0.05 = expect 5% anomalies in baseline)
        gamma: RBF kernel parameter controlling boundary tightness
               - Low values (0.001): smooth, loose boundary
               - High values (0.1): tight, sensitive boundary
               (default 0.1 chosen after grid search)

    Returns:
        Fitted OneClassSVM model

    Example:
        >>> from preprocessing import preprocess_baseline_and_test
        >>> X_base_prep, X_test_prep, pipeline = preprocess_baseline_and_test(...)
        >>> ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
        >>> y_hat = ocsvm.predict(X_test_prep)

    Notes:
        - Training outlier rate should be ~5% (nu parameter)
        - Gamma=0.1 provides good sensitivity for drift detection
        - No cross-validation in this wrapper (already tuned on notebooks)
    """
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_baseline)
    return model


def tune_ocsvm_gamma(
    X_baseline: pd.DataFrame,
    gamma_values: list = None,
    nu: float = 0.05,
    kernel: str = 'rbf'
) -> Tuple[OneClassSVM, float, Dict[str, Any]]:
    """
    Tune OCSVM gamma parameter via grid search on baseline.

    Selects gamma that minimizes |baseline_outlier_rate - nu|.

    Args:
        X_baseline: Preprocessed baseline data
        gamma_values: List of gamma values to try
                     (default: [0.001, 0.01, 0.1, 'scale'])
        nu: Target outlier fraction
        kernel: Kernel type

    Returns:
        Tuple of (best_model, best_gamma, results_dict)
        - best_model: OneClassSVM with optimal gamma
        - best_gamma: Selected gamma value
        - results_dict: Dict with outlier rates for each gamma

    Example:
        >>> best_model, best_gamma, results = tune_ocsvm_gamma(X_base_prep)
        >>> print(f"Best gamma: {best_gamma}")
    """
    if gamma_values is None:
        gamma_values = [0.001, 0.01, 0.1, 'scale']

    results = {}
    best_model = None
    best_gamma = None
    best_error = float('inf')

    for gamma in gamma_values:
        model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        model.fit(X_baseline)

        # Evaluate on baseline
        yhat = model.predict(X_baseline)
        outlier_rate = (yhat == -1).sum() / len(yhat)
        error = abs(outlier_rate - nu)

        results[str(gamma)] = {
            'outlier_rate': outlier_rate,
            'error': error
        }

        if error < best_error:
            best_error = error
            best_model = model
            best_gamma = gamma

    return best_model, best_gamma, results


# ============================================================================
# ISOLATION FOREST FOR DRIFT DETECTION
# ============================================================================

def fit_isolation_forest(
    X_baseline: pd.DataFrame,
    n_estimators: int = 100,
    max_samples: int = 256,
    contamination: float = 0.05,
    random_state: int = 42,
    n_jobs: int = -1
) -> IsolationForest:
    """
    Train Isolation Forest on baseline data.

    Isolation Forest isolates anomalies by random partitioning.
    Samples that are isolated quickly are anomalies; normal samples need more partitions.

    Args:
        X_baseline: Preprocessed baseline data
        n_estimators: Number of isolation trees (default 100)
        max_samples: Number of samples per tree (default 256, tuned)
        contamination: Expected fraction of anomalies (default 0.05)
        random_state: Random seed for reproducibility (default 42)
        n_jobs: Parallel processing (-1 = use all cores, default)

    Returns:
        Fitted IsolationForest model

    Example:
        >>> from preprocessing import preprocess_baseline_and_test
        >>> X_base_prep, X_test_prep, pipeline = preprocess_baseline_and_test(...)
        >>> if_model = fit_isolation_forest(X_base_prep)
        >>> y_hat = if_model.predict(X_test_prep)

    Notes:
        - Default parameters (n_est=100, max_samp=256) chosen after grid search
        - Isolation Forest often more sensitive to drift than OCSVM
        - Faster training than OCSVM, suitable for large datasets
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_baseline)
    return model


def tune_isolation_forest(
    X_baseline: pd.DataFrame,
    X_validation: pd.DataFrame,
    n_estimators_list: list = None,
    max_samples_list: list = None,
    contamination: float = 0.05,
    random_state: int = 42
) -> Tuple[IsolationForest, Dict[str, Any], Dict[str, Any]]:
    """
    Tune Isolation Forest hyperparameters via grid search on validation set.

    Selects parameters that minimize |val_outlier_rate - contamination|.

    Args:
        X_baseline: Preprocessed baseline (training) data
        X_validation: Preprocessed validation data (20% of baseline)
        n_estimators_list: Values to try (default: [100, 200, 300])
        max_samples_list: Values to try (default: [128, 256, 'auto'])
        contamination: Target outlier fraction
        random_state: Random seed

    Returns:
        Tuple of (best_model, best_params, results_dict)
        - best_model: Fitted IsolationForest with optimal hyperparameters
        - best_params: Dict with n_estimators and max_samples
        - results_dict: Performance metrics for all combinations

    Example:
        >>> split = int(0.8 * len(X_base_prep))
        >>> X_train, X_val = X_base_prep[:split], X_base_prep[split:]
        >>> best_model, best_params, results = tune_isolation_forest(X_train, X_val)
        >>> print(f"Best: n_est={best_params['n_estimators']}, max_samp={best_params['max_samples']}")
    """
    if n_estimators_list is None:
        n_estimators_list = [100, 200, 300]
    if max_samples_list is None:
        max_samples_list = [128, 256, 'auto']

    results = {}
    best_model = None
    best_params = None
    best_error = float('inf')

    for n_est in n_estimators_list:
        for max_samp in max_samples_list:
            # Train on baseline
            model = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_baseline)

            # Evaluate on validation set
            yhat_val = model.predict(X_validation)
            outlier_rate_val = (yhat_val == -1).sum() / len(yhat_val)
            error = abs(outlier_rate_val - contamination)

            key = f"n_est={n_est}_max_samp={max_samp}"
            results[key] = {
                'outlier_rate_validation': outlier_rate_val,
                'error': error,
                'n_estimators': n_est,
                'max_samples': max_samp
            }

            if error < best_error:
                best_error = error
                best_model = model
                best_params = {'n_estimators': n_est, 'max_samples': max_samp}

    return best_model, best_params, results


# ============================================================================
# UNIFIED PREDICTION INTERFACE
# ============================================================================

def predict_anomalies(
    model: Union[OneClassSVM, IsolationForest],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict anomalies using either OCSVM or Isolation Forest.

    Args:
        model: Fitted OneClassSVM or IsolationForest
        X: Data to predict on

    Returns:
        Array of predictions: +1 (inlier/normal) or -1 (outlier/anomaly)

    Example:
        >>> y_pred = predict_anomalies(model, X_test)
        >>> n_anomalies = (y_pred == -1).sum()
    """
    return model.predict(X)


def get_outlier_rate(
    model: Union[OneClassSVM, IsolationForest],
    X: pd.DataFrame
) -> float:
    """
    Calculate outlier rate as percentage.

    Args:
        model: Fitted model
        X: Data to evaluate

    Returns:
        Fraction of outliers (0.0 to 1.0)

    Example:
        >>> baseline_rate = get_outlier_rate(model, X_baseline)
        >>> drifted_rate = get_outlier_rate(model, X_drifted)
        >>> detection_ratio = drifted_rate / baseline_rate
    """
    yhat = model.predict(X)
    return (yhat == -1).sum() / len(yhat)


# ============================================================================
# ALGORITHM COMPARISON
# ============================================================================

ALGORITHM_SUMMARY = """
OCSVM vs Isolation Forest Comparison:

ONE-CLASS SVM:
  ✓ Creates explicit decision boundary
  ✓ Tunable parameter (gamma) controls sensitivity
  ✗ Slower for large datasets
  ✗ Memory intensive
  Best for: Precise boundary definition, small-medium datasets

ISOLATION FOREST:
  ✓ Fast, suitable for large datasets
  ✓ Multiple hyperparameters (n_est, max_samples) for tuning
  ✓ Often more sensitive to drift
  ✗ Random component (less reproducible)
  ✗ Less interpretable decision boundary
  Best for: Speed, large datasets, robust anomaly detection

In this study:
  - OCSVM: gamma=0.1 (after grid search)
  - IF: n_estimators=100, max_samples=256 (after grid search)
  - Both: contamination=0.05, nu=0.05 (5% baseline anomalies)
  - Result: IF ~2x more sensitive to drift than OCSVM
"""
