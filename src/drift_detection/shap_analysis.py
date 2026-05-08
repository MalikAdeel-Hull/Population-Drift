"""
SHAP Mechanistic Analysis Module
==================================
Explains anomaly detection decisions using SHAP (SHapley Additive exPlanations).

Provides interpretability for drift detection results by visualizing which features
contribute most to anomaly detection in baseline and drifted data.

Key Features:
- Initialize SHAP KernelExplainer for OCSVM or Isolation Forest
- Compute feature importance rankings
- Compare baseline vs drifted data feature contributions
- Generate publication-quality SHAP plots
- Validate that detected anomalies are driven by drifted features (mechanistic validation)

Usage:
    >>> from drift_detection import create_shap_explainer, get_feature_importance
    >>> explainer = create_shap_explainer(model, X_baseline)
    >>> importance = get_feature_importance(X_drifted, explainer)
    >>> # importance maps feature names to contribution scores

Example:
    >>> # Create explainer from trained OCSVM
    >>> explainer = create_shap_explainer(
    ...     oc_svm_model,
    ...     X_baseline_scaled,
    ...     n_background=100
    ... )
    >>> # Get feature importance for drifted data
    >>> shap_vals = explainer.shap_values(X_drifted_scaled)
    >>> importance = get_feature_importance(shap_vals, X_drifted_scaled)
    >>> print(f"Top feature: {max(importance, key=importance.get)}")
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Tuple, List, Optional
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import shap
import warnings


# ============================================================================
# SHAP EXPLAINER CREATION
# ============================================================================

def create_shap_explainer(
    model: Union[OneClassSVM, IsolationForest],
    X_background: pd.DataFrame,
    method: str = 'kmeans',
    n_background: int = 100,
    random_state: int = 42
) -> shap.KernelExplainer:
    """
    Create a SHAP KernelExplainer for anomaly detection model.

    Uses parametric KernelExplainer which works with any model type (black-box).
    Automatically selects the appropriate scoring function based on model type.

    Args:
        model: Fitted OneClassSVM or IsolationForest anomaly detector
        X_background: Background data for SHAP baseline (subset of training data)
        method: Background sampling method ('kmeans' or 'sample')
                - 'kmeans': Creates k-means cluster centers (faster, more stable)
                - 'sample': Random sample from background data
        n_background: Number of background samples for SHAP calculation (default 100)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Initialized SHAP KernelExplainer for model interpretation

    Raises:
        ValueError: If model type not supported or X_background invalid
        TypeError: If model is not OneClassSVM or IsolationForest

    Example:
        >>> from drift_detection import create_shap_explainer
        >>> # After training OCSVM on baseline
        >>> explainer = create_shap_explainer(
        ...     oc_svm_model,
        ...     X_baseline_scaled,
        ...     method='kmeans',
        ...     n_background=100
        ... )
        >>> print(f"Explainer created for {len(X_baseline_scaled)} samples")

    Notes:
        - KernelExplainer is model-agnostic but slower than TreeExplainer
        - Use n_background=100-200 for good balance of speed/accuracy
        - Larger n_background gives more stable SHAP values
        - KMeans background is recommended for stable results
    """
    # Validate model type
    if not isinstance(model, (OneClassSVM, IsolationForest)):
        raise TypeError(
            f"Model must be OneClassSVM or IsolationForest, got {type(model)}"
        )

    # Validate background data
    if not isinstance(X_background, pd.DataFrame):
        raise ValueError("X_background must be a pandas DataFrame")

    if len(X_background) == 0:
        raise ValueError("X_background cannot be empty")

    # Create background data summary
    if method == 'kmeans':
        background = shap.kmeans(X_background, min(n_background, len(X_background)))
    elif method == 'sample':
        np.random.seed(random_state)
        indices = np.random.choice(len(X_background), min(n_background, len(X_background)), replace=False)
        background = X_background.iloc[indices].values
    else:
        raise ValueError(f"method must be 'kmeans' or 'sample', got {method}")

    # Select scoring function based on model type
    if isinstance(model, OneClassSVM):
        # For OCSVM, use decision_function (distance to hyperplane)
        scoring_function = model.decision_function
        model_name = "OneClassSVM"
    else:  # IsolationForest
        # For Isolation Forest, use decision_function (anomaly score)
        scoring_function = model.decision_function
        model_name = "IsolationForest"

    # Create explainer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        explainer = shap.KernelExplainer(scoring_function, background)

    return explainer


# ============================================================================
# SHAP VALUE COMPUTATION
# ============================================================================

def compute_shap_values(
    explainer: shap.KernelExplainer,
    X_data: pd.DataFrame,
    check_additivity: bool = False,
    suppress_warnings: bool = True
) -> np.ndarray:
    """
    Compute SHAP values for a dataset using pre-fitted explainer.

    SHAP values measure each feature's contribution to pushing the model's
    prediction away from the baseline (background) value.

    Args:
        explainer: SHAP KernelExplainer created with create_shap_explainer()
        X_data: Data to explain (pandas DataFrame)
        check_additivity: Verify SHAP additivity property (slower, rarely needed)
        suppress_warnings: Suppress feature name conversion warnings (default True)

    Returns:
        SHAP values array (n_samples, n_features)
        - Positive values: Feature pushes prediction toward anomaly
        - Negative values: Feature pushes prediction toward normal
        - Magnitude: Strength of feature's contribution

    Example:
        >>> shap_values = compute_shap_values(explainer, X_drifted)
        >>> print(f"Shape: {shap_values.shape}")  # (n_samples, n_features)
        >>> print(f"Mean SHAP value: {shap_values.mean():.3f}")

    Notes:
        - Computation may take time for large datasets
        - SHAP values sum to model prediction (by design)
        - NaN values in X_data will cause errors (handle beforehand)
        - Feature name conversion warnings are common and safe to suppress
    """
    if not isinstance(X_data, pd.DataFrame):
        raise ValueError("X_data must be a pandas DataFrame")

    if len(X_data) == 0:
        raise ValueError("X_data cannot be empty")

    # Compute SHAP values (suppress feature name warnings)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            shap_values = explainer.shap_values(X_data)
    else:
        shap_values = explainer.shap_values(X_data)

    return np.array(shap_values)


# ============================================================================
# FEATURE IMPORTANCE EXTRACTION
# ============================================================================

def get_feature_importance(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    aggregation: str = 'mean',
    absolute: bool = True
) -> Dict[str, float]:
    """
    Extract feature importance from SHAP values.

    Aggregates SHAP values across samples to rank features by importance.
    Higher scores = more important for anomaly detection.

    Args:
        shap_values: SHAP values array from compute_shap_values()
        X_data: Original data used to compute SHAP values (for feature names)
        aggregation: How to aggregate SHAP values ('mean', 'median', 'std')
                    - 'mean': Average absolute SHAP value across samples
                    - 'median': Median absolute SHAP value
                    - 'std': Standard deviation of absolute SHAP values
        absolute: Use absolute SHAP values (default True) for magnitude-based ranking

    Returns:
        Dict mapping feature names to importance scores
        - Sorted by importance (highest first)
        - Scores are non-negative and comparable across features

    Example:
        >>> importance = get_feature_importance(shap_values, X_drifted)
        >>> for feature, score in importance.items():
        ...     print(f"{feature}: {score:.4f}")

    Notes:
        - Absolute=True is recommended for overall importance ranking
        - Absolute=False preserves direction (positive/negative contribution)
        - Use 'mean' for most consistent results
        - Use 'std' to find highly variable features
    """
    if shap_values.shape[0] != len(X_data):
        raise ValueError("SHAP values and X_data must have same number of samples")

    # Aggregate SHAP values
    if absolute:
        shap_values_agg = np.abs(shap_values)
    else:
        shap_values_agg = shap_values

    if aggregation == 'mean':
        importance_scores = np.mean(shap_values_agg, axis=0)
    elif aggregation == 'median':
        importance_scores = np.median(shap_values_agg, axis=0)
    elif aggregation == 'std':
        importance_scores = np.std(shap_values_agg, axis=0)
    else:
        raise ValueError(f"aggregation must be 'mean', 'median', or 'std', got {aggregation}")

    # Create feature importance dictionary
    feature_importance = {
        feature: score
        for feature, score in zip(X_data.columns, importance_scores)
    }

    # Sort by importance (highest first)
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return feature_importance


# ============================================================================
# BASELINE VS DRIFT COMPARISON
# ============================================================================

def compare_baseline_vs_drift(
    explainer: shap.KernelExplainer,
    X_baseline: pd.DataFrame,
    X_drifted: pd.DataFrame,
    drifted_features: Optional[List[str]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Compare SHAP feature importance between baseline and drifted data.

    Validates mechanistic drift detection: anomalies should be driven by
    the features that were intentionally drifted.

    Args:
        explainer: SHAP KernelExplainer
        X_baseline: Baseline (pre-drift) data
        X_drifted: Drifted data
        drifted_features: List of features that were intentionally drifted
                         (for validation)
        top_k: Number of top features to report (default 5)

    Returns:
        Dict with:
        - baseline_importance: Feature importance in baseline
        - drifted_importance: Feature importance when drifted
        - importance_change: How much each feature's importance changed
        - top_baseline: Top K features in baseline
        - top_drifted: Top K features after drift
        - mechanistic_valid: Whether drifted features drive anomalies
        - validation_score: How well drifted features explain anomalies (0-1)

    Example:
        >>> comparison = compare_baseline_vs_drift(
        ...     explainer, X_base_scaled, X_drifted_scaled,
        ...     drifted_features=['Glucose', 'BMI', 'Age']
        ... )
        >>> if comparison['mechanistic_valid']:
        ...     print("✓ SHAP validates drift detection is mechanistically sound")
        >>> print(f"Drifted features rank: {comparison['validation_score']:.1%}")

    Notes:
        - Mechanistic validation confirms model uses drifted features
        - High validation_score (>70%) indicates sound experimental design
        - If drifted_features not provided, relies on top features comparison
        - Comparison helps ensure anomalies are real, not artifacts
    """
    if len(X_baseline) != len(X_drifted):
        warnings.warn(
            "X_baseline and X_drifted have different lengths. "
            "Results may not be directly comparable."
        )

    # Compute SHAP values for both
    shap_baseline = compute_shap_values(explainer, X_baseline)
    shap_drifted = compute_shap_values(explainer, X_drifted)

    # Extract feature importance
    importance_baseline = get_feature_importance(shap_baseline, X_baseline)
    importance_drifted = get_feature_importance(shap_drifted, X_drifted)

    # Calculate importance change
    importance_change = {}
    for feature in X_baseline.columns:
        base_score = importance_baseline.get(feature, 0)
        drift_score = importance_drifted.get(feature, 0)
        change = drift_score - base_score
        pct_change = (change / base_score * 100) if base_score > 0 else 0
        importance_change[feature] = {
            'absolute_change': change,
            'percent_change': pct_change,
            'baseline_score': base_score,
            'drifted_score': drift_score
        }

    # Mechanistic validation
    mechanistic_valid = False
    validation_score = 0.0

    if drifted_features is not None:
        # Check if drifted features are in top K
        top_drifted_features = list(importance_drifted.keys())[:top_k]
        drifted_in_top = sum(1 for f in drifted_features if f in top_drifted_features)
        validation_score = drifted_in_top / len(drifted_features) if drifted_features else 0
        mechanistic_valid = validation_score > 0.6  # >60% in top 5

    return {
        'baseline_importance': importance_baseline,
        'drifted_importance': importance_drifted,
        'importance_change': importance_change,
        'top_baseline': dict(list(importance_baseline.items())[:top_k]),
        'top_drifted': dict(list(importance_drifted.items())[:top_k]),
        'drifted_features': drifted_features,
        'mechanistic_valid': mechanistic_valid,
        'validation_score': validation_score
    }


# ============================================================================
# SHAP PLOTTING UTILITIES
# ============================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    title: str = "SHAP Summary Plot",
    plot_type: str = 'dot',
    show_plot: bool = True
) -> None:
    """
    Generate SHAP summary plot for model interpretation.

    Visualizes feature contributions to model predictions. Shows which features
    are most important and how they affect predictions.

    Args:
        shap_values: SHAP values array from compute_shap_values()
        X_data: Original data used to compute SHAP values
        title: Plot title (default "SHAP Summary Plot")
        plot_type: 'dot' (default) or 'bar'
                  - 'dot': Show individual sample contributions
                  - 'bar': Show aggregate importance
        show_plot: Display plot immediately (default True)

    Returns:
        None (displays plot using matplotlib)

    Example:
        >>> plot_shap_summary(shap_values, X_drifted, title="Drift Detection SHAP")

    Notes:
        - 'dot' plot is more informative (shows sample-level detail)
        - 'bar' plot is simpler and good for presentations
        - Title helps distinguish baseline vs drifted plots
        - Requires matplotlib to be configured for display
    """
    if plot_type not in ['dot', 'bar']:
        raise ValueError(f"plot_type must be 'dot' or 'bar', got {plot_type}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shap.summary_plot(
            shap_values,
            X_data,
            plot_type=plot_type,
            title=title,
            show=show_plot
        )


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    sample_index: int = 0,
    title: str = "SHAP Waterfall Plot",
    show_plot: bool = True
) -> None:
    """
    Generate SHAP waterfall plot for individual sample explanation.

    Shows how each feature contributes to a single sample's anomaly score.
    Useful for understanding why specific samples are flagged as anomalies.

    Args:
        shap_values: SHAP values array from compute_shap_values()
        X_data: Original data used to compute SHAP values
        sample_index: Which sample to explain (default 0)
        title: Plot title
        show_plot: Display plot immediately (default True)

    Returns:
        None (displays plot)

    Example:
        >>> # Explain why sample 0 is flagged as anomaly
        >>> plot_shap_waterfall(shap_values, X_drifted, sample_index=0,
        ...                     title="Sample 0 Anomaly Explanation")

    Notes:
        - Waterfall shows cumulative feature contributions
        - Helpful for understanding individual predictions
        - Use for representative examples in publications
    """
    if sample_index < 0 or sample_index >= len(shap_values):
        raise ValueError(f"sample_index {sample_index} out of bounds (0-{len(shap_values)-1})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_index],
                base_values=shap_values.mean(),
                data=X_data.iloc[sample_index]
            ),
            show=show_plot
        )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_mechanistic_consistency(
    comparison_results: Dict[str, Any],
    min_validation_score: float = 0.6,
    verbose: bool = True
) -> bool:
    """
    Validate that drift detection is mechanistically sound.

    Confirms that anomalies are driven by intentionally drifted features,
    not by spurious correlations or model artifacts.

    Args:
        comparison_results: Output from compare_baseline_vs_drift()
        min_validation_score: Minimum acceptable validation score (default 0.6 = 60%)
        verbose: Print validation report (default True)

    Returns:
        bool: True if mechanistic validation passes, False otherwise

    Example:
        >>> comparison = compare_baseline_vs_drift(...)
        >>> is_valid = validate_mechanistic_consistency(comparison)
        >>> if is_valid:
        ...     print("✓ Drift detection is mechanistically sound")

    Notes:
        - Validation score = fraction of drifted features in top K
        - Score >60% indicates strong mechanistic support
        - Score >80% indicates excellent consistency
        - Can adjust min_validation_score based on requirements
    """
    validation_score = comparison_results.get('validation_score', 0)
    mechanistic_valid = comparison_results.get('mechanistic_valid', False)
    drifted_features = comparison_results.get('drifted_features', [])

    is_valid = validation_score >= min_validation_score

    if verbose:
        print("\n" + "="*70)
        print("MECHANISTIC CONSISTENCY VALIDATION")
        print("="*70)
        if drifted_features:
            print(f"\nDrifted Features: {drifted_features}")
        print(f"Validation Score: {validation_score:.1%}")
        print(f"Required Score: {min_validation_score:.1%}")
        print(f"\nResult: {'✓ PASS' if is_valid else '✗ FAIL'}")

        if is_valid and validation_score > 0.8:
            print("  → Excellent mechanistic support (>80%)")
        elif is_valid and validation_score > 0.6:
            print("  → Good mechanistic support (60-80%)")
        else:
            print("  → Weak mechanistic support (<60%)")

        print("="*70 + "\n")

    return is_valid
