"""
Test script for abrupt drift implementation
Validates apply_minmax_drift() function with both OCSVM and Isolation Forest
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from drift_detection import (
    load_raw_data,
    create_missingness_flags,
    temporal_train_test_split,
    identify_feature_types,
    PIMA_COLS_WITH_MISSING,
    PreprocessingPipeline,
    fit_ocsvm,
    fit_isolation_forest,
    simulate_gradual_drift,
    apply_minmax_drift,
    get_outlier_rate,
    calculate_detection_ratio,
    validate_with_ks_test,
    DEFAULT_CLINICAL_RANGES
)


def compute_baseline_stats(X_baseline, features):
    """Compute min and range statistics for baseline features."""
    stats = {}
    X_clean = X_baseline.dropna()
    for feature in features:
        if feature in X_clean.columns:
            f_min = X_clean[feature].min()
            f_max = X_clean[feature].max()
            f_range = f_max - f_min
            stats[feature] = {'f_min': f_min, 'f_range': f_range}
    return stats


def test_abrupt_drift_single_feature():
    """Test abrupt drift on single feature"""
    print("\n" + "="*70)
    print("TEST 1: Abrupt Drift - Single Feature")
    print("="*70)

    # Load data
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    features = [col for col in df.columns if col != 'Outcome']

    X_base_raw, X_test_raw, _, _, _ = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )

    # Preprocess
    continuous_cols, indicator_cols = identify_feature_types(X_base_raw)
    pipeline = PreprocessingPipeline()
    X_base_prep = pipeline.fit_transform(X_base_raw, continuous_cols, indicator_cols)
    X_test_prep = pipeline.transform(X_test_raw)

    # Train OCSVM
    ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
    baseline_rate = get_outlier_rate(ocsvm, X_base_prep)
    print(f"\nBaseline outlier rate (OCSVM): {baseline_rate:.4f}")

    # Compute baseline stats
    X_base_stats = compute_baseline_stats(X_base_raw, ['Glucose'])
    print(f"Baseline stats: {X_base_stats}")

    # Apply abrupt drift to Glucose
    X_test_drifted_raw = apply_minmax_drift(
        X_test_raw,
        X_base_stats,
        features=['Glucose'],
        shift_f=0.4,
        range_f=1.5,
        verbose=True
    )

    # Preprocess drifted data
    X_test_drifted_prep = pipeline.transform(X_test_drifted_raw)
    drifted_rate = get_outlier_rate(ocsvm, X_test_drifted_prep)

    # Calculate metrics
    detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)

    # K-S test
    ks_stat, p_val, is_sig = validate_with_ks_test(
        X_base_raw['Glucose'].dropna().values,
        X_test_drifted_raw['Glucose'].dropna().values
    )

    print(f"\nDrifted outlier rate (OCSVM): {drifted_rate:.4f}")
    print(f"Detection Ratio: {detection_ratio:.2f}x")
    print(f"K-S Test: stat={ks_stat:.4f}, p={p_val:.6f}, significant={is_sig}")

    # Verify results are reasonable
    assert detection_ratio >= 1.0, "Detection ratio should be >= 1.0"
    assert X_test_drifted_raw.shape == X_test_raw.shape, "Shape mismatch"
    assert X_test_drifted_raw['Glucose'].notna().sum() > 0, "All values are NaN"

    print("✓ Test 1 PASSED")
    return True


def test_abrupt_drift_multivariate():
    """Test abrupt drift on multiple features"""
    print("\n" + "="*70)
    print("TEST 2: Abrupt Drift - Multivariate")
    print("="*70)

    # Load data
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    features = [col for col in df.columns if col != 'Outcome']

    X_base_raw, X_test_raw, _, _, _ = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )

    # Preprocess
    continuous_cols, indicator_cols = identify_feature_types(X_base_raw)
    pipeline = PreprocessingPipeline()
    X_base_prep = pipeline.fit_transform(X_base_raw, continuous_cols, indicator_cols)
    X_test_prep = pipeline.transform(X_test_raw)

    # Train Isolation Forest
    iforest = fit_isolation_forest(
        X_base_prep,
        n_estimators=100,
        max_samples=256,
        contamination=0.05
    )
    baseline_rate = get_outlier_rate(iforest, X_base_prep)
    print(f"\nBaseline outlier rate (IForest): {baseline_rate:.4f}")

    # Compute baseline stats for 4 features
    drift_features = ['Glucose', 'BMI', 'BloodPressure', 'Insulin']
    X_base_stats = compute_baseline_stats(X_base_raw, drift_features)
    print(f"Baseline stats for {len(drift_features)} features computed")

    # Apply abrupt drift to all 4 features
    X_test_drifted_raw = apply_minmax_drift(
        X_test_raw,
        X_base_stats,
        features=drift_features,
        shift_f=0.4,
        range_f=1.5,
        verbose=True
    )

    # Preprocess drifted data
    X_test_drifted_prep = pipeline.transform(X_test_drifted_raw)
    drifted_rate = get_outlier_rate(iforest, X_test_drifted_prep)

    # Calculate metrics
    detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)

    print(f"\nDrifted outlier rate (IForest): {drifted_rate:.4f}")
    print(f"Detection Ratio: {detection_ratio:.2f}x")

    # Verify results
    assert detection_ratio >= 1.0, "Detection ratio should be >= 1.0"
    assert X_test_drifted_raw.shape == X_test_raw.shape, "Shape mismatch"

    # Check that all features were drifted
    for feature in drift_features:
        orig_mean = X_test_raw[feature].mean()
        drift_mean = X_test_drifted_raw[feature].mean()
        mean_change = abs(drift_mean - orig_mean) / orig_mean * 100
        print(f"  {feature}: {orig_mean:.2f} to {drift_mean:.2f} (change: {mean_change:+.1f}%)")
        assert mean_change > 5, f"{feature} should show significant change"

    print("✓ Test 2 PASSED")
    return True


def test_nan_preservation():
    """Test that NaN values are preserved"""
    print("\n" + "="*70)
    print("TEST 3: NaN Preservation")
    print("="*70)

    # Load data
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    features = [col for col in df.columns if col != 'Outcome']

    X_base_raw, X_test_raw, _, _, _ = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )

    # Count original NaNs
    orig_nans = X_test_raw.isnull().sum()

    # Compute baseline stats
    X_base_stats = compute_baseline_stats(X_base_raw, ['Glucose'])

    # Apply drift
    X_test_drifted_raw = apply_minmax_drift(
        X_test_raw,
        X_base_stats,
        features=['Glucose'],
        shift_f=0.4,
        range_f=1.5,
        verbose=False
    )

    # Count drifted NaNs
    drift_nans = X_test_drifted_raw.isnull().sum()

    # Check NaNs preserved
    assert (orig_nans == drift_nans).all(), "NaN counts do not match"
    print(f"NaNs in Glucose: {orig_nans['Glucose']} to {drift_nans['Glucose']} (preserved) ✓")

    print("✓ Test 3 PASSED")
    return True


def test_clinical_range_clipping():
    """Test that values are clipped to clinical ranges"""
    print("\n" + "="*70)
    print("TEST 4: Clinical Range Clipping")
    print("="*70)

    # Load data
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    features = [col for col in df.columns if col != 'Outcome']

    X_base_raw, X_test_raw, _, _, _ = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )

    # Compute baseline stats
    X_base_stats = compute_baseline_stats(X_base_raw, ['Glucose', 'BMI'])

    # Apply aggressive drift (high shift and range factors)
    X_test_drifted_raw = apply_minmax_drift(
        X_test_raw,
        X_base_stats,
        features=['Glucose', 'BMI'],
        shift_f=0.8,
        range_f=3.0,
        feature_ranges=DEFAULT_CLINICAL_RANGES,
        verbose=True
    )

    # Check that values respect clinical ranges
    glucose_min, glucose_max = DEFAULT_CLINICAL_RANGES['Glucose']
    bmi_min, bmi_max = DEFAULT_CLINICAL_RANGES['BMI']

    glucose_vals = X_test_drifted_raw['Glucose'].dropna()
    bmi_vals = X_test_drifted_raw['BMI'].dropna()

    assert glucose_vals.min() >= glucose_min, "Glucose below minimum"
    assert glucose_vals.max() <= glucose_max, "Glucose above maximum"
    assert bmi_vals.min() >= bmi_min, "BMI below minimum"
    assert bmi_vals.max() <= bmi_max, "BMI above maximum"

    print(f"\nGlucose range: [{glucose_vals.min():.1f}, {glucose_vals.max():.1f}] (OK) ✓")
    print(f"BMI range: [{bmi_vals.min():.1f}, {bmi_vals.max():.1f}] (OK) ✓")

    print("✓ Test 4 PASSED")
    return True


def test_comparison_gradual_vs_abrupt():
    """Compare gradual vs abrupt drift detection ratios"""
    print("\n" + "="*70)
    print("TEST 5: Gradual vs Abrupt Drift Comparison")
    print("="*70)

    # Load data
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    features = [col for col in df.columns if col != 'Outcome']

    X_base_raw, X_test_raw, _, _, _ = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )

    # Preprocess
    continuous_cols, indicator_cols = identify_feature_types(X_base_raw)
    pipeline = PreprocessingPipeline()
    X_base_prep = pipeline.fit_transform(X_base_raw, continuous_cols, indicator_cols)
    X_test_prep = pipeline.transform(X_test_raw)

    # Train both models
    ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
    iforest = fit_isolation_forest(X_base_prep, contamination=0.05)

    baseline_rate_ocsvm = get_outlier_rate(ocsvm, X_base_prep)
    baseline_rate_iforest = get_outlier_rate(iforest, X_base_prep)

    # Gradual drift
    X_gradual_raw = simulate_gradual_drift(
        X_test_raw, 'Glucose',
        drift_percentage=0.40,
        duration=len(X_test_raw)
    )
    X_gradual_prep = pipeline.transform(X_gradual_raw)

    gradual_ocsvm_rate = get_outlier_rate(ocsvm, X_gradual_prep)
    gradual_iforest_rate = get_outlier_rate(iforest, X_gradual_prep)

    gradual_ocsvm_ratio = calculate_detection_ratio(baseline_rate_ocsvm, gradual_ocsvm_rate)
    gradual_iforest_ratio = calculate_detection_ratio(baseline_rate_iforest, gradual_iforest_rate)

    # Abrupt drift
    X_base_stats = compute_baseline_stats(X_base_raw, ['Glucose'])
    X_abrupt_raw = apply_minmax_drift(
        X_test_raw,
        X_base_stats,
        features=['Glucose'],
        shift_f=0.4,
        range_f=1.5,
        verbose=False
    )
    X_abrupt_prep = pipeline.transform(X_abrupt_raw)

    abrupt_ocsvm_rate = get_outlier_rate(ocsvm, X_abrupt_prep)
    abrupt_iforest_rate = get_outlier_rate(iforest, X_abrupt_prep)

    abrupt_ocsvm_ratio = calculate_detection_ratio(baseline_rate_ocsvm, abrupt_ocsvm_rate)
    abrupt_iforest_ratio = calculate_detection_ratio(baseline_rate_iforest, abrupt_iforest_rate)

    # Print comparison
    print(f"\nOCSVM:")
    print(f"  Gradual: {gradual_ocsvm_ratio:.2f}x")
    print(f"  Abrupt:  {abrupt_ocsvm_ratio:.2f}x")

    print(f"\nIsolation Forest:")
    print(f"  Gradual: {gradual_iforest_ratio:.2f}x")
    print(f"  Abrupt:  {abrupt_iforest_ratio:.2f}x")

    # Verify both detect drift
    assert gradual_ocsvm_ratio > 1.5, "Gradual drift not detected by OCSVM"
    assert gradual_iforest_ratio > 1.0, "Gradual drift not detected by IF"
    assert abrupt_ocsvm_ratio > 1.2, "Abrupt drift not detected by OCSVM"
    assert abrupt_iforest_ratio > 1.0, "Abrupt drift not detected by IF"

    print("✓ Test 5 PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TESTING ABRUPT DRIFT IMPLEMENTATION")
    print("="*70)

    try:
        test_abrupt_drift_single_feature()
        test_abrupt_drift_multivariate()
        test_nan_preservation()
        test_clinical_range_clipping()
        test_comparison_gradual_vs_abrupt()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
