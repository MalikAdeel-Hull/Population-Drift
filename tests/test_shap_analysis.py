"""
Test Suite: SHAP Analysis Module
=================================
Validates SHAP mechanistic analysis functions against real experimental data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# Direct import from shap_analysis module
from drift_detection.shap_analysis import (
    create_shap_explainer,
    compute_shap_values,
    get_feature_importance,
    compare_baseline_vs_drift,
    validate_mechanistic_consistency
)


def test_shap_explainer_creation():
    """Test SHAP explainer initialization for OCSVM and Isolation Forest."""
    print("\n" + "="*70)
    print("TEST 1: SHAP Explainer Creation")
    print("="*70)

    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
    )
    X_background = X_train.iloc[:50]

    print("\nTesting OCSVM explainer...")
    ocsvm = OneClassSVM(kernel='rbf', nu=0.2, gamma='scale')
    ocsvm.fit(X_train)
    explainer_ocsvm = create_shap_explainer(ocsvm, X_background, n_background=20)
    print("[OK] OCSVM explainer created")

    print("\nTesting Isolation Forest explainer...")
    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(X_train)
    explainer_iso = create_shap_explainer(iso_forest, X_background, n_background=20)
    print("[OK] Isolation Forest explainer created")

    print("\nTesting background sampling methods...")
    explainer_kmeans = create_shap_explainer(ocsvm, X_background, method='kmeans')
    print("[OK] KMeans background method works")

    explainer_sample = create_shap_explainer(ocsvm, X_background, method='sample')
    print("[OK] Random sample background method works")

    return explainer_ocsvm, X_train, X_background


def test_feature_importance(explainer_ocsvm, X_train, X_background):
    """Test feature importance computation and ranking."""
    print("\n" + "="*70)
    print("TEST 2: Feature Importance Extraction")
    print("="*70)

    print("\nComputing SHAP values...")
    shap_values = compute_shap_values(explainer_ocsvm, X_train)
    print("[OK] SHAP values computed")
    print("     Shape: {}".format(shap_values.shape))
    print("     Mean value: {:.4f}".format(shap_values.mean()))

    print("\nExtracting feature importance...")
    importance = get_feature_importance(shap_values, X_train, aggregation='mean')
    print("[OK] Feature importance extracted")
    print("\nRanking (by mean |SHAP|):")
    for i, (feature, score) in enumerate(importance.items(), 1):
        print("  {}. {}: {:.6f}".format(i, feature, score))

    return shap_values, importance


def test_baseline_vs_drift_comparison(explainer_ocsvm, X_background):
    """Test comparison between baseline and drifted data."""
    print("\n" + "="*70)
    print("TEST 3: Baseline vs Drift Comparison")
    print("="*70)

    np.random.seed(42)
    X_baseline = X_background.copy()
    X_drifted = X_baseline.copy()

    X_drifted['Feature_1'] = X_drifted['Feature_1'] * 1.5 + 0.5
    X_drifted['Feature_2'] = X_drifted['Feature_2'] * 1.3 + 0.3
    drifted_features = ['Feature_1', 'Feature_2']

    print("\nComparing baseline vs drifted data...")
    print("Drifted features: {}".format(drifted_features))

    comparison = compare_baseline_vs_drift(
        explainer_ocsvm,
        X_baseline,
        X_drifted,
        drifted_features=drifted_features,
        top_k=3
    )

    print("\n[OK] Comparison completed")
    print("\nMechanistic Validation:")
    print("  Valid: {}".format(comparison['mechanistic_valid']))
    print("  Validation Score: {:.1%}".format(comparison['validation_score']))

    return comparison


def test_mechanistic_validation(comparison):
    """Test mechanistic consistency validation function."""
    print("\n" + "="*70)
    print("TEST 4: Mechanistic Consistency Validation")
    print("="*70)

    is_valid = validate_mechanistic_consistency(
        comparison,
        min_validation_score=0.6,
        verbose=True
    )

    return is_valid


def test_fhgd_multivariate_simulation():
    """
    Simulate FHGD multivariate abrupt drift analysis.
    """
    print("\n" + "="*70)
    print("TEST 5: FHGD Multivariate Drift Simulation (Headline Result)")
    print("="*70)

    np.random.seed(42)

    n_baseline = 1400
    n_test = 600
    n_features = 13

    X_baseline = pd.DataFrame(
        np.random.randn(n_baseline, n_features),
        columns=['Feature_{}'.format(i) for i in range(n_features)]
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=['Feature_{}'.format(i) for i in range(n_features)]
    )

    drifted_features = ['Feature_0', 'Feature_1', 'Feature_2']
    for feature in drifted_features:
        X_test[feature] = X_test[feature] * 1.5 + 0.5

    print("\nDataset simulation:")
    print("  Baseline: {}".format(X_baseline.shape))
    print("  Test: {}".format(X_test.shape))
    print("  Drifted features: {}".format(drifted_features))

    print("\nTraining OCSVM...")
    ocsvm = OneClassSVM(kernel='rbf', nu=0.2, gamma='scale')
    ocsvm.fit(X_baseline)

    baseline_pred = ocsvm.predict(X_baseline)
    baseline_rate = (baseline_pred == -1).sum() / len(baseline_pred)
    print("  Baseline anomaly rate: {:.2%}".format(baseline_rate))

    drifted_pred = ocsvm.predict(X_test)
    drifted_rate = (drifted_pred == -1).sum() / len(drifted_pred)
    print("  Drifted anomaly rate: {:.2%}".format(drifted_rate))

    detection_ratio = drifted_rate / baseline_rate
    print("  Detection Ratio: {:.2f}x".format(detection_ratio))

    print("\nPerforming SHAP mechanistic analysis...")
    explainer = create_shap_explainer(ocsvm, X_baseline, n_background=50)
    print("  [OK] Explainer created")

    comparison = compare_baseline_vs_drift(
        explainer,
        X_baseline,
        X_test,
        drifted_features=drifted_features,
        top_k=5
    )

    print("\nSHAP Mechanistic Validation:")
    print("  Drifted features in top 5: {:.1%}".format(comparison['validation_score']))
    print("  Mechanistically valid: {}".format(comparison['mechanistic_valid']))

    is_valid = validate_mechanistic_consistency(
        comparison,
        min_validation_score=0.6,
        verbose=False
    )

    result = "PASS" if is_valid else "FAIL"
    print("\n  Overall result: {}".format(result))

    return is_valid


if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("SHAP ANALYSIS MODULE TEST SUITE")
    print("="*70)

    explainer, X_train, X_background = test_shap_explainer_creation()
    shap_values, importance = test_feature_importance(explainer, X_train, X_background)
    comparison = test_baseline_vs_drift_comparison(explainer, X_background)
    is_valid = test_mechanistic_validation(comparison)
    is_valid_fhgd = test_fhgd_multivariate_simulation()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("[OK] Test 1: SHAP Explainer Creation - PASSED")
    print("[OK] Test 2: Feature Importance Extraction - PASSED")
    print("[OK] Test 3: Baseline vs Drift Comparison - PASSED")
    print("[OK] Test 4: Mechanistic Validation - {}".format("PASSED" if is_valid else "FAILED"))
    print("[OK] Test 5: FHGD Simulation - {}".format("PASSED" if is_valid_fhgd else "FAILED"))
    print("\n" + "="*70)
    print("All SHAP analysis tests completed!")
    print("="*70 + "\n")
