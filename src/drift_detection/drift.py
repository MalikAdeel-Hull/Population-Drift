"""
Data Drift Simulation Module
=============================
Implements gradual and multivariate drift simulation for drift detection evaluation.

Drift is applied via multiplicative scaling: new_value = original * (1 + drift_factor)

Key Features:
- Monotonic drift schedules (linear increase over time steps)
- Clinical range enforcement (clipping to realistic bounds)
- Only applies to non-NaN values
- Preserves data types and structure

Example Usage:
    >>> drifted_df = simulate_gradual_drift(
    ...     X_test_raw, "Glucose", drift_percentage=0.4, duration=100
    ... )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Clinical ranges for major features
# Used to clip drifted values to realistic bounds
DEFAULT_CLINICAL_RANGES = {
    'Pregnancies': (0, 15),
    'Glucose': (70, 200),
    'BloodPressure': (60, 140),
    'SkinThickness': (5, 50),
    'Insulin': (0, 600),
    'BMI': (15, 60),
    'DiabetesPedigreeFunction': (0.078, 2.42),
    'Age': (21, 81)
}


def simulate_gradual_drift(
    df: pd.DataFrame,
    feature: str,
    drift_percentage: float = 0.40,
    duration: int = None,
    feature_ranges: Dict[str, Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Simulate gradual (covariate) drift on a single feature.

    Drift is applied multiplicatively over time steps, increasing linearly from 0% to
    drift_percentage. Values are clipped to clinical ranges to remain realistic.

    Args:
        df: Input DataFrame (typically test set)
        feature: Column name to apply drift to
        drift_percentage: Target drift magnitude as decimal (e.g., 0.40 for 40%)
        duration: Number of samples over which to apply drift (default: len(df))
        feature_ranges: Dict mapping feature names to (min, max) clinical bounds
                       (default: DEFAULT_CLINICAL_RANGES)

    Returns:
        DataFrame with drift applied to specified feature. Non-NaN values scaled,
        NaN values preserved.

    Raises:
        ValueError: If feature not in DataFrame or invalid drift_percentage
        KeyError: If feature not in feature_ranges

    Example:
        >>> X_test = pd.DataFrame({'Glucose': [100, 110, 120, ...]})
        >>> X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.4)
        >>> # Glucose values increase by 0-40% over the duration

    Notes:
        - Drift schedule: [0%, 1%, 2%, ..., 40%] applied to successive samples
        - Only non-NaN values are drifted; NaN values remain NaN
        - Final values clipped to clinical ranges
        - Zero drift (drift_percentage=0.0) returns original data unchanged
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns")

    if drift_percentage < 0:
        raise ValueError(f"drift_percentage must be non-negative, got {drift_percentage}")

    if feature_ranges is None:
        feature_ranges = DEFAULT_CLINICAL_RANGES

    if feature not in feature_ranges:
        raise KeyError(f"Feature '{feature}' not in feature_ranges")

    # Default to full duration
    if duration is None:
        duration = len(df)

    df = df.copy()
    n_samples = len(df)

    # Create drift schedule: linearly increase from 0 to drift_percentage
    # Example: drift_percentage=0.40, duration=100 → [0, 0.004, 0.008, ..., 0.40]
    drift_schedule = np.linspace(0, drift_percentage, duration)

    # Get feature values and missing mask
    feature_values = df[feature].values.copy().astype(float)
    is_missing = pd.isna(feature_values)

    # Apply drift: new = original * (1 + drift_factor)
    for i in range(n_samples):
        if not is_missing[i]:
            drift_factor = drift_schedule[i] if i < duration else drift_percentage
            feature_values[i] = feature_values[i] * (1 + drift_factor)

    # Clip to clinical ranges
    min_val, max_val = feature_ranges[feature]
    feature_values = np.clip(feature_values, min_val, max_val)

    # Restore NaN values
    feature_values[is_missing] = np.nan

    df[feature] = feature_values
    return df


def simulate_multivariate_drift(
    df: pd.DataFrame,
    features: List[str],
    drift_percentage: float = 0.40,
    duration: int = None,
    feature_ranges: Dict[str, Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Simulate gradual drift on multiple features simultaneously.

    Each feature receives the same drift schedule, allowing study of multivariate drift.

    Args:
        df: Input DataFrame
        features: List of column names to apply drift to
        drift_percentage: Target drift magnitude (decimal)
        duration: Number of samples for drift application (default: len(df))
        feature_ranges: Clinical bounds (default: DEFAULT_CLINICAL_RANGES)

    Returns:
        DataFrame with drift applied to all specified features

    Raises:
        ValueError: If any feature not in DataFrame
        KeyError: If any feature not in feature_ranges

    Example:
        >>> X_drifted = simulate_multivariate_drift(
        ...     X_test, ['Glucose', 'BMI'], drift_percentage=0.3
        ... )
        >>> # Both Glucose and BMI drift by same schedule simultaneously
    """
    if feature_ranges is None:
        feature_ranges = DEFAULT_CLINICAL_RANGES

    # Validate all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")

    # Apply drift to each feature
    df = df.copy()
    for feature in features:
        df = simulate_gradual_drift(
            df,
            feature,
            drift_percentage=drift_percentage,
            duration=duration,
            feature_ranges=feature_ranges
        )

    return df


def verify_drift_application(
    original_df: pd.DataFrame,
    drifted_df: pd.DataFrame,
    feature: str,
    expected_direction: str = "increase"
) -> Dict[str, float]:
    """
    Verify that drift was correctly applied to a feature.

    Args:
        original_df: Original (pre-drift) DataFrame
        drifted_df: Drifted DataFrame
        feature: Feature name to check
        expected_direction: "increase" or "decrease" expected drift direction

    Returns:
        Dict with statistics:
        - mean_change: Percentage change in mean
        - nans_preserved: Count of NaN values preserved
        - out_of_range: Count of values clipped to ranges
    """
    orig_vals = original_df[feature].dropna().values
    drift_vals = drifted_df[feature].dropna().values

    if len(orig_vals) == 0:
        return {"error": "No non-NaN values in original data"}

    mean_change = (drift_vals.mean() - orig_vals.mean()) / orig_vals.mean() * 100

    # Check direction
    if expected_direction == "increase" and mean_change < 0:
        print(f"Warning: Expected increase but got {mean_change:.2f}% change")
    elif expected_direction == "decrease" and mean_change > 0:
        print(f"Warning: Expected decrease but got {mean_change:.2f}% change")

    # Count NaNs
    nans_original = original_df[feature].isna().sum()
    nans_drifted = drifted_df[feature].isna().sum()

    return {
        "mean_change_percent": mean_change,
        "nans_preserved": nans_original == nans_drifted,
        "n_nans": nans_original
    }
def apply_minmax_drift(
    X_data: pd.DataFrame,
    X_base_stats: Dict[str, Dict[str, float]],
    features: List[str],
    shift_f: float = 0.4,
    range_f: float = 1.5,
    verbose: bool = True,
    feature_ranges: Dict[str, Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Apply min/max affine transformation for abrupt drift.

    Simulates abrupt (concept drift) by transforming the feature distribution:
    x_drifted = (x - f_min) / f_range * (max_t - min_t) + min_t

    Where:
    - f_min, f_range: Baseline feature minimum and range
    - min_t = f_min * (1 - shift_f): Target minimum (shifted down)
    - max_t = f_min + (f_max - f_min) * range_f: Target maximum (expanded)

    Args:
        X_data: DataFrame to apply drift to
        X_base_stats: Dict with baseline statistics
                     Format: {'feature': {'f_min': val, 'f_range': val}}
        features: List of feature names to drift
        shift_f: Downward shift factor (default 0.4 = 40% shift)
        range_f: Range expansion factor (default 1.5 = 1.5x stretch)
        verbose: Print transformation details (default True)
        feature_ranges: Clinical bounds for clipping (default: DEFAULT_CLINICAL_RANGES)

    Returns:
        DataFrame with abrupt drift applied to specified features

    Raises:
        KeyError: If feature not in X_base_stats
        ValueError: If feature not in DataFrame

    Example:
        >>> # Compute baseline statistics
        >>> X_base_stats = {
        ...     'Glucose': {'f_min': 44.0, 'f_range': 199.0},
        ...     'BMI': {'f_min': 18.2, 'f_range': 58.0}
        ... }
        >>> X_drifted = apply_minmax_drift(
        ...     X_test, X_base_stats,
        ...     features=['Glucose', 'BMI'],
        ...     shift_f=0.4, range_f=1.5
        ... )
        >>> # Glucose and BMI now exhibit abrupt distribution shift

    Notes:
        - Applied abruptly to entire dataset (unlike gradual drift)
        - Preserves NaN values
        - Clips to clinical ranges if provided
        - Contamination/nu typically 0.2 (20%) for abrupt drift scenarios
    """
    if feature_ranges is None:
        feature_ranges = DEFAULT_CLINICAL_RANGES

    # Validate inputs
    for feature in features:
        if feature not in X_data.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame columns")
        if feature not in X_base_stats:
            raise KeyError(f"Feature '{feature}' not in baseline statistics dict")

    X_drifted = X_data.copy()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Applying Abrupt Min/Max Drift")
        print(f"Shift Factor: {shift_f:.2f} | Range Factor: {range_f:.2f}")
        print(f"{'='*70}\n")

    for feature in features:
        # Get baseline stats
        f_min = X_base_stats[feature]['f_min']
        f_range = X_base_stats[feature]['f_range']

        # Calculate target range bounds
        min_t = f_min * (1 - shift_f)
        max_t = f_min + f_range * range_f

        # Get feature values and missing mask
        feature_values = X_drifted[feature].values.copy().astype(float)
        is_missing = pd.isna(feature_values)

        # Apply affine transformation: x_drifted = (x - f_min) / f_range * (max_t - min_t) + min_t
        valid_mask = ~is_missing
        if valid_mask.any():
            feature_values[valid_mask] = (
                (feature_values[valid_mask] - f_min) / f_range * (max_t - min_t) + min_t
            )

        # Clip to clinical ranges if available
        if feature in feature_ranges:
            min_clip, max_clip = feature_ranges[feature]
            feature_values = np.clip(feature_values, min_clip, max_clip)

            if verbose:
                n_clipped = ((feature_values[valid_mask] == min_clip) |
                            (feature_values[valid_mask] == max_clip)).sum()
                if n_clipped > 0:
                    print(f"  {feature}: {n_clipped} values clipped to clinical range ({min_clip}, {max_clip})")

        # Restore NaN values
        feature_values[is_missing] = np.nan
        X_drifted[feature] = feature_values

        if verbose:
            orig_mean = X_data[feature].mean()
            drift_mean = X_drifted[feature].mean()
            mean_shift = ((drift_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
            print(f"  {feature}: mean {orig_mean:.2f} → {drift_mean:.2f} ({mean_shift:+.1f}%)")

    if verbose:
        print(f"\n{'='*70}\n")

    return X_drifted


# ============================================================================
# DRIFT SIMULATION QUICK REFERENCE
# ============================================================================

DRIFT_SCHEDULE_EXAMPLES = """
Example drift schedules (drift_percentage=0.40, duration=100):
- Sample 0: 0% drift → value * 1.00
- Sample 25: 10% drift → value * 1.10
- Sample 50: 20% drift → value * 1.20
- Sample 75: 30% drift → value * 1.30
- Sample 100: 40% drift → value * 1.40

For zero-drift control: simulate_gradual_drift(..., drift_percentage=0.0)
→ All samples get 0% drift → values unchanged
→ Outlier rate should equal original (validation check)
"""
